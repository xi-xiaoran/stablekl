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
import acdc_run as R
from stablekl.logits import _phi, _round_once

def logits_mean64(clean2, out2, sub=128):
    rows = []
    for r in range(0, clean2.shape[0], sub):
        A, C = (clean2[r:r + sub].double(), out2[r:r + sub].double())
        u = A - C
        qq = torch.softmax(C, -1)
        w = u - torch.log1p((qq * torch.expm1(u)).sum(-1, keepdim=True))
        rows.append((qq * _phi(w)).sum(-1))
    return torch.cat(rows).mean()

def truth64(clean2, out2, sub=128):
    tot = torch.zeros((), dtype=torch.float64, device=clean2.device)
    for r in range(0, clean2.shape[0], sub):
        lc, lo = (F.log_softmax(clean2[r:r + sub].double(), -1), F.log_softmax(out2[r:r + sub].double(), -1))
        tot = tot + (lc.exp() * (lc - lo)).sum()
    return float(tot / clean2.shape[0])

def score_edges(pm, loader, half, limit=None, progress=2000):
    from auto_circuit.utils.ablation_activations import src_ablations
    from auto_circuit.utils.graph_utils import patch_mode, set_all_masks
    from auto_circuit.utils.tensor_ops import multibatch_kl_div
    out_slice = pm.out_slice
    edges = sorted(pm.edges, key=lambda x: x.dest.layer, reverse=True)
    if limit:
        edges = edges[:limit]
    batch = next(iter(loader))
    clean_batch, corrupt_batch = (batch.clean, batch.corrupt)
    patch_outs = src_ablations(pm, corrupt_batch)
    src_outs = src_ablations(pm, clean_batch)
    with torch.inference_mode():
        clean = pm(clean_batch)[out_slice]
        _, toks, short_embd, attn_mask = pm.input_to_embed(clean_batch)
        _, cache = pm.run_with_cache(clean_batch)
        resids = [cache[f'blocks.{i}.hook_resid_pre'].clone() for i in range(pm.cfg.n_layers)]
        del cache
    V = clean.shape[-1]
    c2 = clean.reshape(-1, V)
    clean_lp = F.log_softmax(clean, dim=-1)
    clean_lp32 = F.log_softmax(clean.float(), dim=-1)
    info = dict(clean_finite=bool(torch.isfinite(clean).all()), rows=int(c2.shape[0]), vocab=int(V), n_scored=len(edges))
    n = len(edges)
    S = {k: torch.full((n,), float('nan'), dtype=torch.float64) for k in ('official', 'logits', 'truth') + (('fp32cast', 'logits32') if half else ())}
    fwd = lambda L: pm(resids[L], start_at_layer=L, tokens=toks, shortformer_pos_embed=short_embd, attention_mask=attn_mask)[out_slice]
    set_all_masks(pm, val=0.0)
    null = []
    t0 = time.time()
    with patch_mode(pm, patch_outs, curr_src_outs=src_outs):
        with torch.inference_mode():
            for L in range(pm.cfg.n_layers):
                null.append(truth64(c2, fwd(L).reshape(-1, V)))
        for i, e in enumerate(edges):
            e.patch_mask(pm).data[e.patch_idx] = 1.0
            with torch.inference_mode():
                out = fwd(int(e.dest.module_name.split('.')[1]))
                o2 = out.reshape(-1, V)
                S['official'][i] = multibatch_kl_div(F.log_softmax(out, dim=-1), clean_lp).item()
                if half:
                    S['fp32cast'][i] = multibatch_kl_div(F.log_softmax(out.float(), dim=-1), clean_lp32).item()
                m = logits_mean64(c2, o2)
                S['logits'][i] = float(_round_once(m, c2.dtype))
                S['truth'][i] = truth64(c2, o2)
                if half:
                    S['logits32'][i] = float(_round_once(m, torch.float32))
            e.patch_mask(pm).data[e.patch_idx] = 0.0
            if progress and (i + 1) % progress == 0:
                print(f'   {i + 1} / {n} edges, {time.time() - t0:.0f} s', flush=True)
    info.update(null=null, seconds=time.time() - t0)
    return (edges, S, info)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2-small')
    ap.add_argument('--dtype', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--n-train', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'scores_{a.dtype}.pt')
    if os.path.exists(path):
        print(path, 'exists, skipped')
        return
    from pathlib import Path
    pm, loader = R.build(a.model, R.DT[a.dtype], dev, Path(a.data), a.batch, a.n_train)
    mods = sorted((m.module_name for m in pm.dest_wrappers))
    off, k = ({}, 0)
    for m in mods:
        sz = pm.new_prune_scores()[m].numel()
        off[m] = torch.arange(k, k + sz).reshape(pm.new_prune_scores()[m].shape)
        k += sz
    flat = lambda es: torch.tensor([int(off[e.dest.module_name][e.patch_idx]) for e in es])
    try:
        from auto_circuit.metrics.official_circuits.circuits.ioi_official import ioi_true_edges
        true_idx = flat(ioi_true_edges(pm))
    except Exception as ex:
        true_idx = None
        print('official IOI edges not available:', ex)
    meta = dict(args=vars(a), n_edges=len(pm.edges), torch=torch.__version__, device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', code={f: code_md5(f, HERE) for f in ('edge_scores.py', 'acdc_run.py', 'instrument.py', 'logits_kl.py', 'gkd.py')}, data_sha256=hashlib.sha256(open(a.data, 'rb').read()).hexdigest())
    json.dump(meta, open(os.path.join(a.out, f'meta_{a.dtype}.json'), 'w'), indent=1)
    print(meta, flush=True)
    edges, S, info = score_edges(pm, loader, half=a.dtype != 'fp32', limit=a.limit or None)
    torch.save(dict(dtype=a.dtype, edge_index=flat(edges), n_edges=len(pm.edges), true_index=true_idx, scores=S, info=info, names=[str(e) for e in edges]), path + '.tmp')
    os.replace(path + '.tmp', path)
    print(a.dtype, {k: v for k, v in info.items() if k != 'null'}, 'null max', max(info['null']), flush=True)
    print('done', flush=True)
if __name__ == '__main__':
    main()
