from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'common'))
from fingerprint import code_md5
import argparse, hashlib, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from trl_gkd_loss import generalized_jsd_loss, SOURCE_MD5
from stablekl.logits import logits_kl
from rtn import START, quantise_decoder_linears
LOSSES = ('bf16', 'fp32', 'logits')
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = ('gkd.py', 'trl_gkd_loss.py', 'logits_kl.py', 'rtn.py')

def load_tokens(tok, text_file, n_train, n_eval, L):
    if text_file:
        ids = tok(open(text_file, encoding='utf-8').read())['input_ids']
        tr, ev = (ids[:n_train * L], ids[n_train * L:(n_train + n_eval) * L])
    else:
        from datasets import load_dataset

        def text(split):
            last = None
            for name in ('Salesforce/wikitext', 'wikitext'):
                try:
                    return '\n\n'.join(load_dataset(name, 'wikitext-2-raw-v1', split=split)['text'])
                except Exception as e:
                    last = e
            raise SystemExit(f'WikiText-2 {split} not available offline: {last}')
        tr, ev = (tok(text('train'))['input_ids'], tok(text('validation'))['input_ids'])
    if len(tr) < n_train * L or len(ev) < n_eval * L:
        raise SystemExit('text too short')
    return (torch.tensor(tr[:n_train * L]).view(n_train, L), torch.tensor(ev[:n_eval * L]).view(n_eval, L))

def exact_kl64(a, c):
    A, C = (a.double(), c.double())
    u = A - C
    q = torch.softmax(C, -1)
    delta = torch.log1p((q * torch.expm1(u)).sum(-1, keepdim=True))
    kl = (torch.softmax(A, -1) * (u - delta)).sum(-1)
    return torch.where((a == c).all(-1), torch.zeros_like(kl), kl)

def exact_grad(a, c, N):
    return (torch.softmax(c.double(), -1) - torch.softmax(a.double(), -1)) / N

def chunk_loss_grad(kind, s, t, N):
    x = s.detach().clone().requires_grad_(True)
    if kind == 'bf16':
        v = generalized_jsd_loss(x, t, beta=0.0, num_items_in_batch=N)
    elif kind == 'fp32':
        v = generalized_jsd_loss(x.float(), t.float(), beta=0.0, num_items_in_batch=N)
    elif kind == 'logits':
        v = logits_kl(t, x) * (x.shape[0] / N)
    else:
        raise ValueError(kind)
    v.backward()
    return (float(v.detach().double()), x.grad)
CHUNK_ROWS = 128

def backward_batch(student, teacher, ids, kind, N):
    with torch.no_grad():
        t = teacher(input_ids=ids).logits[:, :-1]
    s = student(input_ids=ids).logits[:, :-1]
    V = s.shape[-1]
    sf, tf = (s.reshape(-1, V), t.reshape(-1, V))
    g = torch.empty_like(sf)
    rep = ex = 0.0
    for r0 in range(0, sf.shape[0], CHUNK_ROWS):
        r1 = r0 + CHUNK_ROWS
        v, gi = chunk_loss_grad(kind, sf[r0:r1], tf[r0:r1], N)
        g[r0:r1] = gi
        rep += v
        ex += float(exact_kl64(tf[r0:r1], sf[r0:r1].detach()).sum()) / N
    ident = bool(torch.equal(s.detach(), t))
    s.backward(g.view_as(s))
    return (rep, ex, ident)

def params_of(model):
    return [p for p in model.parameters() if p.requires_grad]

def sha_params(params):
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()
BLOCK = 1 << 24

def blocks(*ts):
    fl = [t.detach().reshape(-1) for t in ts]
    for i in range(0, fl[0].numel(), BLOCK):
        yield [f[i:i + BLOCK].to(fl[0].device).double() for f in fl]

def drift(params, init_cpu, den):
    num, mx = (0.0, 0.0)
    for p, p0 in zip(params, init_cpu):
        for a, b in blocks(p, p0):
            d = a - b
            num += float((d * d).sum())
            mx = max(mx, float(d.abs().max()))
    return (math.sqrt(num) / den, mx)

@torch.no_grad()
def evaluate(student, teacher, E, bs, dev, with_teacher_ppl=False):
    student.eval()
    kl = dis = nll_s = nll_t = 0.0
    n = 0
    for b in range(0, len(E), bs):
        ids = E[b:b + bs].to(dev)
        y = ids[:, 1:]
        t = teacher(input_ids=ids).logits[:, :-1]
        s = student(input_ids=ids).logits[:, :-1]
        for i in range(ids.shape[0]):
            kl += float(exact_kl64(t[i], s[i]).sum())
            dis += int((t[i].argmax(-1) != s[i].argmax(-1)).sum())
            nll_s -= float(torch.log_softmax(s[i].double(), -1).gather(-1, y[i, :, None]).sum())
            if with_teacher_ppl:
                nll_t -= float(torch.log_softmax(t[i].double(), -1).gather(-1, y[i, :, None]).sum())
        n += y.numel()
    student.train()
    r = dict(kl=kl / n, top1_disagree=dis / n, ppl_student=math.exp(nll_s / n))
    if with_teacher_ppl:
        r['ppl_teacher'] = math.exp(nll_t / n)
    return r

def grad_probe(student, teacher, ids, kind, N, params):
    for p in params:
        p.grad = None
    backward_batch(student, teacher, ids, kind, N)
    gm = [p.grad.detach().to('cpu', copy=True) if p.grad is not None else torch.zeros_like(p, device='cpu') for p in params]
    for p in params:
        p.grad = None
    with torch.no_grad():
        t = teacher(input_ids=ids).logits[:, :-1]
    s = student(input_ids=ids).logits[:, :-1]
    gref = torch.empty_like(s)
    for i in range(s.shape[0]):
        for r0 in range(0, s.shape[1], CHUNK_ROWS):
            gref[i, r0:r0 + CHUNK_ROWS] = exact_grad(t[i, r0:r0 + CHUNK_ROWS], s[i, r0:r0 + CHUNK_ROWS].detach(), N).to(s.dtype)
    s.backward(gref)
    dot = nm = nr = nd = 0.0
    for p, a in zip(params, gm):
        r0 = p.grad if p.grad is not None else torch.zeros_like(p)
        for r, m in blocks(r0, a):
            dot += float((m * r).sum())
            nm += float((m * m).sum())
            nr += float((r * r).sum())
            nd += float(((m - r) ** 2).sum())
    for p in params:
        p.grad = None
    return dict(method_norm=math.sqrt(nm), ref_norm=math.sqrt(nr), cosine=dot / math.sqrt(nm * nr) if nm > 0 and nr > 0 else None, rel_err=math.sqrt(nd) / math.sqrt(nr) if nr > 0 else None)

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    raw = open(path).read()
    lines = raw.split('\n')
    rows = []
    if lines and lines[-1] == '':
        lines = lines[:-1]
    for i, l in enumerate(lines):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            print(f'{path}: dropped a truncated last line', flush=True)
            with open(path, 'w') as f:
                f.write(''.join((json.dumps(r) + '\n' for r in rows)))
            return rows
    if raw and (not raw.endswith('\n')):
        with open(path, 'a') as f:
            f.write('\n')
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', choices=list(START), required=True)
    ap.add_argument('--loss', choices=LOSSES, required=True)
    ap.add_argument('--seeds', default='0-9')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--n-train', type=int, default=256)
    ap.add_argument('--n-eval', type=int, default=32)
    ap.add_argument('--seq', type=int, default=257)
    ap.add_argument('--micro', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-05)
    ap.add_argument('--max-grad-norm', type=float, default=1.0)
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--text-file', default=None)
    ap.add_argument('--eval-bs', type=int, default=8)
    ap.add_argument('--chunk-rows', type=int, default=128)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split('-')) if '-' in a.seeds else (int(a.seeds),) * 2
    seeds = list(range(lo, hi + 1))
    dev = torch.device(a.device)
    dt = getattr(torch, a.dtype)
    torch.use_deterministic_algorithms(True, warn_only=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    T, E = load_tokens(tok, a.text_file, a.n_train, a.n_eval, a.seq)
    global CHUNK_ROWS
    rows = a.seq - 1
    N = a.micro * rows
    CHUNK_ROWS = min(a.chunk_rows, N)
    assert a.n_train % a.micro == 0 and N % CHUNK_ROWS == 0 and (N // CHUNK_ROWS & N // CHUNK_ROWS - 1 == 0), 'chunk rows / N must be 2^-k'
    steps_per_epoch = a.n_train // a.micro
    total = steps_per_epoch * a.epochs
    os.makedirs(a.out, exist_ok=True)
    tok_sha = hashlib.sha256(T.numpy().tobytes() + E.numpy().tobytes()).hexdigest()
    fp = dict(code={f: code_md5(f, HERE) for f in CODE}, trl_source_md5=SOURCE_MD5, model=os.path.realpath(a.model), tokens_sha256=tok_sha, torch=torch.__version__, args={k: v for k, v in vars(a).items() if k not in ('seeds', 'out')})
    fpp = os.path.join(a.out, 'fingerprint.json')
    if os.path.exists(fpp):
        old = json.load(open(fpp))
        if old != fp:
            raise SystemExit(f'RESUME REFUSED: {fpp} differs from this run (code, data, model or arguments changed)')
    else:
        json.dump(fp, open(fpp, 'w'), indent=1)
    runs_path = os.path.join(a.out, 'runs.jsonl')
    done = {r['seed'] for r in read_jsonl(runs_path) if r.get('status') == 'complete'}
    load = lambda: AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt, attn_implementation='eager', local_files_only=True).to(dev)
    teacher = load().eval()
    [p.requires_grad_(False) for p in teacher.parameters()]
    student = load()
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.train()
    params = params_of(student)
    teacher_sha = sha_params(list(teacher.parameters()))
    if START[a.start] is not None:
        nq = quantise_decoder_linears(student, *START[a.start])
    else:
        nq = 0
    init_cpu = [p.detach().to('cpu', copy=True) for p in params]
    init_sha = sha_params(params)
    den = math.sqrt(sum((float((p0.double() ** 2).sum()) for p0 in init_cpu)))
    env = dict(device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', torch=torch.__version__, transformers=__import__('transformers').__version__, n_params=sum((p.numel() for p in params)), quantised_linears=nq, init_equals_teacher=init_sha == teacher_sha, N_tokens_per_step=N, steps=total)
    try:
        env['accelerate'] = __import__('accelerate').__version__
    except Exception:
        env['accelerate'] = None
    json.dump(env, open(os.path.join(a.out, 'env.json'), 'w'), indent=1)
    print(f'[{a.start}/{a.loss}] {env}', flush=True)
    if a.start == 'S0' and (not env['init_equals_teacher']):
        raise SystemExit('S0 student does not equal the teacher')
    probe_ids = T[:a.micro].to(dev)
    for seed in seeds:
        if seed in done:
            print(f'seed {seed}: already complete', flush=True)
            continue
        t0 = time.time()
        with torch.no_grad():
            for p, p0 in zip(params, init_cpu):
                p.copy_(p0)
        assert sha_params(params) == init_sha
        master = [p.detach().float().clone() for p in params]
        opt = torch.optim.AdamW(master, lr=a.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, fused=dev.type == 'cuda')
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: max(0.0, (total - st) / max(1, total)))
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        rec = dict(start=a.start, loss=a.loss, seed=seed, steps_planned=total, status='running')
        evals = {'0': evaluate(student, teacher, E, a.eval_bs, dev, with_teacher_ppl=True)}
        probes = {'0': grad_probe(student, teacher, probe_ids, a.loss, N, params)}
        drifts = {'0': [0.0, 0.0]}
        steps = []
        status = 'complete'
        first_ident = None
        mem_first = None
        for ep in range(a.epochs):
            perm = torch.randperm(a.n_train, generator=torch.Generator().manual_seed(seed * 100003 + ep))
            for b in range(steps_per_epoch):
                ids = T[perm[b * a.micro:(b + 1) * a.micro]].to(dev)
                rep, ex, ident = backward_batch(student, teacher, ids, a.loss, N)
                if first_ident is None:
                    first_ident = ident
                for p, m in zip(params, master):
                    m.grad = p.grad.float() if p.grad is not None else torch.zeros_like(m)
                    p.grad = None
                gn = float(torch.nn.utils.clip_grad_norm_(master, a.max_grad_norm))
                lr_now = sched.get_last_lr()[0]
                if not (math.isfinite(rep) and math.isfinite(gn)):
                    status = 'nonfinite'
                else:
                    opt.step()
                    with torch.no_grad():
                        for p, m in zip(params, master):
                            p.copy_(m)
                sched.step()
                for m in master:
                    m.grad = None
                steps.append([rep, ex, gn, lr_now])
                if mem_first is None and dev.type == 'cuda':
                    mem_first = [torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30]
                if status != 'complete':
                    break
            if status != 'complete':
                break
            evals[str(ep + 1)] = evaluate(student, teacher, E, a.eval_bs, dev)
            probes[str(ep + 1)] = grad_probe(student, teacher, probe_ids, a.loss, N, params)
            drifts[str(ep + 1)] = list(drift(params, init_cpu, den))
            e = evals[str(ep + 1)]
            print(f"seed {seed} epoch {ep + 1}: eval KL {e['kl']:.4e}, top-1 {e['top1_disagree']:.4f}, ppl {e['ppl_student']:.4f}, rel dW {drifts[str(ep + 1)][0]:.2e}, {time.time() - t0:.0f} s", flush=True)
        fin = sha_params(params)
        rec.update(status=status, steps_done=len(steps), first_step_logits_identical=first_ident, evals=evals, grad_probes=probes, drift=drifts, steps=steps, init_sha256=init_sha, final_sha256=fin, weights_unchanged=fin == init_sha, mem_gib_first_step=mem_first, mem_gib_peak=[torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30] if dev.type == 'cuda' else None, seconds=time.time() - t0)
        with open(runs_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        del opt, sched, master
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"seed {seed}: {status}, {len(steps)} steps, weights unchanged {fin == init_sha}, {time.time() - t0:.0f} s, peak mem {rec['mem_gib_peak']}", flush=True)
    print('done', flush=True)
if __name__ == '__main__':
    main()
