from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exp4c_repeated_execution'))
from fingerprint import code_md5
import argparse, hashlib, json, os, sys, time
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import instrument as I
DT = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}

def build(model_name, dtype, device, data_path, batch_size, n_train, hf_model=None, tokenizer=None, cfg=None):
    import transformer_lens as tl
    from auto_circuit.data import load_datasets_from_json
    from auto_circuit.utils.graph_utils import patchable_model
    if cfg is not None:
        model = tl.HookedTransformer(cfg, tokenizer=tokenizer).to(dtype).to(device)
    else:
        model = tl.HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype=dtype)
    model.cfg.use_attn_result = True
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    train, test = load_datasets_from_json(model=model, path=data_path, device=device, prepend_bos=True, batch_size=batch_size, train_test_size=(n_train, n_train), return_seq_length=False, tail_divergence=True, pad=True)
    pm = patchable_model(model=model, factorized=True, slice_output='last_seq', seq_len=train.seq_len, separate_qkv=True, kv_caches=(train.kv_cache, test.kv_cache), device=device)
    return (pm, train)

def run_acdc(pm, loader, mode, tao_exp, stats):
    import auto_circuit.prune_algos.ACDC as A
    from auto_circuit.utils.graph_utils import set_all_masks
    from auto_circuit.utils.tensor_ops import multibatch_kl_div as lib_kl
    orig = (A.log_softmax, A.multibatch_kl_div)
    set_all_masks(pm, val=0.0)
    batch = next(iter(loader))
    with torch.inference_mode():
        clean = pm(batch.clean)[pm.out_slice]
    stats.update(clean_finite=bool(torch.isfinite(clean).all()), rows=int(clean.reshape(-1, clean.shape[-1]).shape[0]), vocab=int(clean.shape[-1]))
    if not stats['clean_finite']:
        return None
    clean_lp = F.log_softmax(clean, dim=-1)
    c2 = clean.reshape(-1, clean.shape[-1])
    calls = dict(n=0, neg=0, nonfinite=0, checked=False)

    def check(clean_logprobs):
        if not calls['checked']:
            assert torch.equal(clean_logprobs, clean_lp), "clean logits differ from the library's"
            calls['checked'] = True

    def kl_official(out_lp, clean_logprobs):
        check(clean_logprobs)
        v = lib_kl(out_lp, clean_logprobs)
        calls['n'] += 1
        x = float(v)
        calls['neg'] += x < 0
        calls['nonfinite'] += x != x or abs(x) == float('inf')
        return v

    def kl_fp32(out_logits, clean_logprobs):
        check(clean_logprobs)
        calls['n'] += 1
        return lib_kl(F.log_softmax(out_logits.float(), dim=-1), F.log_softmax(clean.float(), dim=-1))

    def kl_logits(out_logits, clean_logprobs):
        check(clean_logprobs)
        calls['n'] += 1
        return torch.tensor(I.logits_path(c2, out_logits.reshape(-1, out_logits.shape[-1])))
    try:
        if mode == 'official':
            A.multibatch_kl_div = kl_official
        else:
            A.log_softmax = lambda x, dim=-1: x
            A.multibatch_kl_div = kl_fp32 if mode == 'fp32cast' else kl_logits
        ps = A.acdc_prune_scores(pm, loader, None, tao_exps=[tao_exp], tao_bases=[1])
    finally:
        A.log_softmax, A.multibatch_kl_div = orig
    stats.update(kl_calls=calls['n'], official_negative=calls['neg'], official_nonfinite=calls['nonfinite'])
    return ps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2-small')
    ap.add_argument('--dtype', required=True)
    ap.add_argument('--modes', default='official,fp32cast,logits')
    ap.add_argument('--tao-exps', default='-6,-5,-4,-3')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--n-train', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    from pathlib import Path
    pm, loader = build(a.model, DT[a.dtype], dev, Path(a.data), a.batch, a.n_train)
    from auto_circuit.metrics.official_circuits.circuits.ioi_official import ioi_true_edges
    mods = sorted((m.module_name for m in pm.dest_wrappers))
    n_edges = len(pm.edges)
    try:
        true_edges = ioi_true_edges(pm)
    except Exception as e:
        true_edges = None
        print('official IOI edges not available:', e)

    def mask_of(edges):
        d = {m: torch.zeros_like(pm.new_prune_scores()[m], dtype=torch.bool) for m in mods}
        for e in edges:
            d[e.dest.module_name][e.patch_idx] = True
        return torch.cat([d[m].flatten().cpu() for m in mods])
    meta = dict(args=vars(a), n_edges=n_edges, modules=mods, torch=torch.__version__, transformer_lens=__import__('transformer_lens').__version__ if hasattr(__import__('transformer_lens'), '__version__') else '?', code={f: code_md5(f, HERE) for f in ('acdc_run.py', 'instrument.py', 'logits_kl.py', 'gkd.py')}, data_sha256=hashlib.sha256(open(a.data, 'rb').read()).hexdigest(), device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu')
    json.dump(meta, open(os.path.join(a.out, f'meta_{a.dtype}.json'), 'w'), indent=1)
    print(meta, flush=True)
    torch.save(dict(edge_exists=mask_of(pm.edges), true_edges=mask_of(true_edges) if true_edges is not None else None), os.path.join(a.out, f'edges_{a.dtype}.pt'))
    for mode in a.modes.split(','):
        if a.dtype == 'fp32' and mode == 'fp32cast':
            continue
        for te in (int(x) for x in a.tao_exps.split(',')):
            name = f'{a.dtype}_{mode}_tao1e{te}'
            path = os.path.join(a.out, name + '.pt')
            if os.path.exists(path):
                print(name, 'exists, skipped', flush=True)
                continue
            st = {}
            t0 = time.time()
            ps = run_acdc(pm, loader, mode, te, st)
            st['seconds'] = time.time() - t0
            if ps is None:
                st['status'] = 'clean logits not finite'
                removed = None
            else:
                removed = torch.cat([torch.isfinite(ps[m]).flatten().cpu() for m in mods])
                st.update(status='complete', removed=int(removed.sum()), kept=n_edges - int(removed.sum()))
            torch.save(dict(removed=removed, stats=st, dtype=a.dtype, mode=mode, tao_exp=te), path + '.tmp')
            os.replace(path + '.tmp', path)
            print(name, st, flush=True)
    print('done', flush=True)
if __name__ == '__main__':
    main()
