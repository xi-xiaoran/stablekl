from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'no_effect' / 'llm_distillation'))
from fingerprint import code_md5
import argparse, hashlib, json, math, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import gkd as P
from trl_gkd_loss import generalized_jsd_loss, SOURCE_MD5
from stablekl.logits import logits_kl
from rtn import START, quantise_decoder_linears
SCEN = {'S0': None, 'S1': (8, None), 'S2': (4, 128), 'ANCHOR': None}
PATHS = ('bf16', 'fp32', 'logits', 'none')
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = ('lora_strong.py', 'gkd.py', 'trl_gkd_loss.py', 'logits_kl.py', 'rtn.py')

class LoRALinear(nn.Module):

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))

    def reset(self, gen, b_std=0.0):
        a = torch.empty(self.A.shape, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=gen)
        with torch.no_grad():
            self.A.copy_(a)
            self.B.zero_()
            if b_std > 0:
                self.B.copy_(torch.randn(self.B.shape, dtype=torch.float32, generator=gen) * b_std)

    def forward(self, x):
        y = self.base(x)
        return y + (self.scale * F.linear(F.linear(x.float(), self.A), self.B)).to(y.dtype)

def inject_lora(model, r, alpha):
    mods = []
    for name, m in list(model.named_modules()):
        if name.startswith('model.layers.') and isinstance(m, nn.Linear):
            parent = model.get_submodule(name.rsplit('.', 1)[0])
            lm = LoRALinear(m, r, alpha)
            setattr(parent, name.rsplit('.', 1)[1], lm)
            mods.append(lm)
    if not mods or len(mods) % 7:
        raise SystemExit(f'expected 7 linear modules per decoder layer, found {len(mods)}')
    return mods

def kl_term(path, x, t, N):
    if path == 'bf16':
        return generalized_jsd_loss(x, t, beta=0.0, num_items_in_batch=N)
    if path == 'fp32':
        return generalized_jsd_loss(x.float(), t.float(), beta=0.0, num_items_in_batch=N)
    if path == 'logits':
        return logits_kl(t, x) * (x.shape[0] / N)
    raise ValueError(path)

def step_grads(student, teacher, ids, path, N, lam, chunk):
    with torch.no_grad():
        t = teacher(input_ids=ids).logits[:, :-1]
    s = student(input_ids=ids).logits[:, :-1]
    y = ids[:, 1:]
    V = s.shape[-1]
    sf, tf, yf = (s.reshape(-1, V), t.reshape(-1, V), y.reshape(-1))
    g = torch.empty_like(sf)
    rep = ex = ce = 0.0
    for r0 in range(0, sf.shape[0], chunk):
        r1 = r0 + chunk
        x = sf[r0:r1].detach().clone().requires_grad_(True)
        kl = kl_term(path, x, tf[r0:r1], N) if path != 'none' else None
        if lam is None:
            loss = kl
        else:
            c = F.cross_entropy(x.float(), yf[r0:r1], reduction='sum') / N
            ce += float(c.detach())
            loss = c if kl is None else c + lam * kl
        loss.backward()
        g[r0:r1] = x.grad
        rep += float(kl.detach().double()) if kl is not None else float('nan')
        ex += float(P.exact_kl64(tf[r0:r1], sf[r0:r1].detach()).sum()) / N
    ident = bool(torch.equal(s.detach(), t))
    s.backward(g.view_as(s))
    return (rep, ex, ce, ident)

def sha(ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scenario', choices=list(SCEN), required=True)
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--paths', default='bf16,fp32,logits')
    ap.add_argument('--seeds', default='0-2')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--n-train', type=int, default=256)
    ap.add_argument('--n-eval', type=int, default=32)
    ap.add_argument('--seq', type=int, default=257)
    ap.add_argument('--micro', type=int, default=8)
    ap.add_argument('--chunk-rows', type=int, default=128)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=float, default=32.0)
    ap.add_argument('--anchor-lambda', type=float, default=1.0)
    ap.add_argument('--b-init-std', type=float, default=0.0)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--max-grad-norm', type=float, default=1.0)
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--text-file', default=None)
    ap.add_argument('--eval-bs', type=int, default=8)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split('-')) if '-' in a.seeds else (int(a.seeds),) * 2
    seeds = list(range(lo, hi + 1))
    paths = a.paths.split(',')
    assert all((p in PATHS for p in paths))
    assert 'none' not in paths or a.scenario == 'ANCHOR', 'the CE-only control exists only for ANCHOR'
    assert a.b_init_std >= 0 and (a.b_init_std == 0 or a.scenario == 'S0'), 'the perturbed start is defined for S0 only'
    dev = torch.device(a.device)
    dt = getattr(torch, a.dtype)
    torch.use_deterministic_algorithms(True, warn_only=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    T, E = P.load_tokens(tok, a.text_file, a.n_train, a.n_eval, a.seq)
    rows = a.seq - 1
    N = a.micro * rows
    chunk = min(a.chunk_rows, N)
    assert a.n_train % a.micro == 0 and N % chunk == 0 and (N // chunk & N // chunk - 1 == 0)
    spe = a.n_train // a.micro
    total = spe * a.epochs
    lam = a.anchor_lambda if a.scenario == 'ANCHOR' else None
    os.makedirs(a.out, exist_ok=True)
    fp = dict(code={f: code_md5(f, HERE) for f in CODE}, trl_source_md5=SOURCE_MD5, model=os.path.realpath(a.model), tokens_sha256=hashlib.sha256(T.numpy().tobytes() + E.numpy().tobytes()).hexdigest(), torch=torch.__version__, args={k: v for k, v in vars(a).items() if k not in ('seeds', 'out', 'paths')})
    fpp = os.path.join(a.out, 'fingerprint.json')
    if os.path.exists(fpp):
        if json.load(open(fpp)) != fp:
            raise SystemExit(f'RESUME REFUSED: {fpp} differs from this run')
    else:
        json.dump(fp, open(fpp, 'w'), indent=1)
    runs_path = os.path.join(a.out, 'runs.jsonl')
    done = {(r['path'], r['seed']) for r in P.read_jsonl(runs_path) if r.get('status') == 'complete'}
    load = lambda: AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt, attn_implementation='eager', local_files_only=True).to(dev)
    teacher = load().eval()
    [p.requires_grad_(False) for p in teacher.parameters()]
    student = load()
    student.config.use_cache = False
    if SCEN[a.scenario] is not None:
        quantise_decoder_linears(student, *SCEN[a.scenario])
    [p.requires_grad_(False) for p in student.parameters()]
    mods = inject_lora(student, a.rank, a.alpha)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.train()
    lora = [p for m in mods for p in (m.A, m.B)]
    base_sha = sha([p for n, p in student.named_parameters() if not (n.endswith('.A') or n.endswith('.B'))])
    env = dict(device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', torch=torch.__version__, transformers=__import__('transformers').__version__, lora_modules=len(mods), lora_params=sum((p.numel() for p in lora)), base_equals_teacher=base_sha == sha(list(teacher.parameters())), N_tokens_per_step=N, steps=total, anchor_lambda=lam, b_init_std=a.b_init_std, tag=a.tag)
    json.dump(env, open(os.path.join(a.out, 'env.json'), 'w'), indent=1)
    print(f'[{a.tag}: {a.scenario} lr {a.lr:g}] {env}', flush=True)
    if a.scenario in ('S0', 'ANCHOR') and (not env['base_equals_teacher']):
        raise SystemExit('identical-start base differs from the teacher')
    for seed in seeds:
        for path in paths:
            if (path, seed) in done:
                print(f'{path} seed {seed}: already complete', flush=True)
                continue
            t0 = time.time()
            gen = torch.Generator().manual_seed(seed)
            for m in mods:
                m.reset(gen, a.b_init_std)
            init_sha = sha(lora)
            init_lora = [p.detach().clone() for p in lora]
            nB0 = math.sqrt(sum((float((m.B.double() ** 2).sum()) for m in mods)))
            opt = torch.optim.AdamW(lora, lr=a.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, fused=dev.type == 'cuda')
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: max(0.0, (total - st) / max(1, total)))
            if dev.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            evals = {'0': P.evaluate(student, teacher, E, a.eval_bs, dev, with_teacher_ppl=True)}
            steps = []
            status = 'complete'
            first_ident = None
            for ep in range(a.epochs):
                perm = torch.randperm(a.n_train, generator=torch.Generator().manual_seed(seed * 100003 + ep))
                for b in range(spe):
                    ids = T[perm[b * a.micro:(b + 1) * a.micro]].to(dev)
                    rep, ex, ce, ident = step_grads(student, teacher, ids, path, N, lam, chunk)
                    if first_ident is None:
                        first_ident = ident
                    gn = float(torch.nn.utils.clip_grad_norm_(lora, a.max_grad_norm))
                    lr_now = sched.get_last_lr()[0]
                    if not ((math.isfinite(rep) or path == 'none') and math.isfinite(gn) and math.isfinite(ce)):
                        status = 'nonfinite'
                    else:
                        opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    steps.append([rep, ex, ce, gn, lr_now])
                    if status != 'complete':
                        break
                if status != 'complete':
                    break
                evals[str(ep + 1)] = P.evaluate(student, teacher, E, a.eval_bs, dev)
            dB = math.sqrt(sum((float((m.B.double() ** 2).sum()) for m in mods)))
            dA = math.sqrt(sum((float(((p.double() - q.double()) ** 2).sum()) for p, q in zip(lora[0::2], init_lora[0::2]))))
            fin = sha(lora)
            e = evals[max(evals, key=int)]
            rec = dict(tag=a.tag, b_init_std=a.b_init_std, anchor_lambda=lam, norm_B_init=nB0, scenario=a.scenario, lr=a.lr, path=path, seed=seed, status=status, steps_done=len(steps), steps_planned=total, first_step_logits_identical=first_ident, evals=evals, steps=steps, lora_unchanged=fin == init_sha, norm_B=dB, norm_dA=dA, mem_gib_peak=torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == 'cuda' else None, seconds=time.time() - t0)
            with open(runs_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"{path} seed {seed}: {status}, eval KL {e['kl']:.4e}, top-1 {e['top1_disagree']:.4f}, ppl {e['ppl_student']:.4f}, KL start {evals['0']['kl']:.4e}, |B| {nB0:.2e} -> {dB:.2e}, unchanged {fin == init_sha}, {time.time() - t0:.0f} s, mem {rec['mem_gib_peak']}", flush=True)
            del opt, sched
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
    print('done', flush=True)
if __name__ == '__main__':
    main()
