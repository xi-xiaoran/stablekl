import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from fingerprint import code_md5
import argparse, hashlib, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_gpu as D
VS = (1024, 32768, 151936)
DTS = ('fp32', 'fp16', 'bf16')
OPT_LR = (('adam', 0.0001), ('adam', 0.0003), ('adam', 0.001), ('adam', 0.003), ('adam', 0.01), ('sgd', 0.01), ('sgd', 0.1), ('sgd', 1.0))
METHODS = ('kl_div', 'kl_div_lt', 'categorical', 'logitsA')
SEEDS = range(10)
STEPS = 300
LOGS = (1, 10, 30, 50, 100, 300)

def plan(smoke):
    vs, seeds, steps, logs = ((16, 64), range(2), 20, (1, 10, 20)) if smoke else (VS, SEEDS, STEPS, LOGS)
    return [dict(V=V, dtn=dt, method=m, opt_name=o, seed=s, steps=steps, lr=lr, log_at=logs) for V in vs for dt in DTS for o, lr in OPT_LR for m in METHODS for s in seeds]
key_p = lambda p: (p['V'], p['dtn'], p['method'], p['opt_name'], float(p['lr']), p['seed'], p['steps'])
key_r = lambda r: (r['V'], r['dt'], r['method'], r['opt'], float(r['lr']), r['seed'], r['steps'])

def fingerprint(dev, smoke):
    here = os.path.dirname(os.path.abspath(__file__))
    return dict(code={f: code_md5(f, here) for f in ('distill_grid.py', 'distill_gpu.py', 'logits_kl.py')}, torch=torch.__version__, device=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', smoke=smoke, grid=dict(V=VS, dt=DTS, opt_lr=OPT_LR, methods=METHODS, seeds=len(SEEDS), steps=STEPS))

def compare(rows, frozen_path):
    F = {key_r(r): r for r in D.read_jsonl(frozen_path) if r.get('kick', 0) == 0}
    common = [r for r in rows if key_r(r) in F]
    flds = ('log', 'gnorm0', 'loss0', 'W_final_sha256', 'status', 'steps_done')
    same = sum((all((r.get(f) == F[key_r(r)].get(f) for f in flds)) for r in common))
    return dict(frozen=frozen_path, common=len(common), identical=same)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--compare', default=None, help='runs.jsonl of an earlier run: report whether the new runs are bit-identical')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fp = fingerprint(dev, a.smoke)
    fpath = os.path.join(a.out, 'fingerprint.json')
    if os.path.exists(fpath):
        old = json.load(open(fpath))
        if old != json.loads(json.dumps(fp)):
            sys.exit(f'RESUME REFUSED: {a.out} was written by a different code / environment / grid\n old {old}\n now {fp}')
        print('resume: fingerprint matches', flush=True)
    elif any((f.endswith('.jsonl') for f in os.listdir(a.out))):
        sys.exit(f'RESUME REFUSED: {a.out} has results but no fingerprint.json')
    else:
        json.dump(fp, open(fpath, 'w'), indent=1)
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, device=fp['device'])
    json.dump(env, open(os.path.join(a.out, 'env.json'), 'w'), indent=1)
    print(env, flush=True)
    P = plan(a.smoke)
    rpath = os.path.join(a.out, 'runs.jsonl')
    done = {key_r(r) for r in D.read_jsonl(rpath)}
    t0 = time.time()
    n = 0
    with open(rpath, 'a') as f:
        for p in P:
            if key_p(p) in done:
                continue
            r = D.run(dev, **p)
            f.write(json.dumps(r) + '\n')
            f.flush()
            n += 1
            if n % 100 == 0:
                print(f'{n} runs, {time.time() - t0:.0f} s', flush=True)
    rows = D.read_jsonl(rpath)
    st = {}
    for r in rows:
        st[r['status']] = st.get(r['status'], 0) + 1
    planned = {key_p(p) for p in P}
    got = {key_r(r) for r in rows}
    status = dict(planned=len(P), recorded=len(rows), missing=len(planned - got), unexpected=len(got - planned), by_status=st)
    if a.compare:
        status['compare_with_frozen'] = compare(rows, a.compare)
    json.dump(status, open(os.path.join(a.out, 'status.json'), 'w'), indent=1)
    print(f'{status}; {n} run in this job, {time.time() - t0:.0f} s', flush=True)
    if status['missing'] or status['unexpected'] or len(rows) != len(P):
        sys.exit('recorded runs do not match the plan')
    c = status.get('compare_with_frozen')
    if c and c['identical'] != c['common']:
        sys.exit(f'runs shared with the frozen job differ: {c}')
if __name__ == '__main__':
    main()
