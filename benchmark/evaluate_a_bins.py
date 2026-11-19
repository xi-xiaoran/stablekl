from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import evaluate_a as G

def unit(task):
    fpath, rawdir, gdir, dt, comps, name, smoke = task
    from klbench.fp import get_format, round_fraction
    from klbench.score.runner import score_one
    load_unit = G.loader()
    fmt = get_format(dt)
    arr, _ = load_unit(fpath)
    n = len(arr['sample_id'])
    C = Counter()
    stem = Path(fpath).stem
    if n == 0:
        return (stem, dt, {})
    if smoke:
        n = min(n, smoke)
    widths = {k[:-6]: arr[k].shape[1] for k in arr if k.startswith('grad__') and k.endswith('__bits') and (k.count('__') == 3)}
    raws = {}
    for m in G.OFFICIAL[dt]:
        p = Path(rawdir) / dt / f'{stem}__{m}.npz'
        if p.exists():
            ra, _ = load_unit(p)
            assert list(ra['sample_id'])[:n] == list(arr['sample_id'])[:n]
            raws[m + '_cpu'] = ra
        g = Path(gdir) / dt / f'{stem}__{m}.npz'
        if g.exists():
            raws[m] = dict(np.load(g, allow_pickle=False))
        else:
            C[f'{m}|missing_unit'] += 1
    for m in G.REWRITE[dt]:
        g = Path(gdir) / dt / f'{stem}__{m}.npz'
        if g.exists():
            raws[m] = G.rawlike(n, dt, dict(np.load(g, allow_pickle=False)), comps, m != 'C', widths, fmt, round_fraction)
        else:
            C[f'{m}|missing_unit'] += 1
    for j in range(n):
        for m, raw in raws.items():
            metrics, _, _ = score_one(arr, raw, j, dt)
            C[f'{m}|rows'] += 1
            sig, notcr, app = G.judge_kl(metrics)
            if app:
                C[f'{m}|SIG|n'] += 1
                C[f'{m}|SIG|e'] += sig
            for k, v in metrics.items():
                if k.startswith('KL_') and v is not None:
                    C[f'{m}|{k}|n'] += 1
                    C[f'{m}|{k}|e'] += v is True
    return (stem, dt, dict(C))

def main():
    from multiprocessing import get_context
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--root', default='data', help='root of the released data (contains interface_a/)')
    ap.add_argument('--gdir', required=True, help='outputs/<config> directory written by evaluate_a.py compute')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--max-units', type=int, default=0)
    a = ap.parse_args()
    M = G.load_fix(a.name)
    FROZEN, RAW = G.paths(a.root, a.name)
    comps = list(M.COMPS)
    t0 = time.time()
    res = {}
    tasks = [(str(f), str(RAW), a.gdir, dt, comps, a.name, a.smoke) for dt in ('fp16', 'fp32') if dt not in (getattr(M, 'SKIP_DTYPES', {}) or {}) for f in sorted((FROZEN / dt).glob('*.npz'))[:a.max_units or None]]
    if a.workers > 1:
        with get_context('spawn').Pool(a.workers) as pool:
            out = list(pool.imap_unordered(unit, tasks, chunksize=1))
    else:
        out = [unit(t) for t in tasks]
    for stem, dt, c in out:
        res[f'{dt}/{stem}'] = c
    miss = sum((v for c in res.values() for k, v in c.items() if k.endswith('missing_unit')))
    dst = Path(a.out) / 'bins'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f'{a.name}.json').write_text(json.dumps(dict(name=a.name, module_md5=G.md5(M.__file__), tool_md5=G.md5(__file__), gdir=a.gdir, smoke=a.smoke, units=res, seconds=time.time() - t0, missing_units=miss)))
    tot = Counter()
    for u, c in res.items():
        for k, v in c.items():
            tot[u.split('/')[0] + '|' + k] += v
    for dt in ('fp16', 'fp32'):
        for m in sorted({k.split('|')[1] for k in tot if k.startswith(dt)}):
            if tot[f'{dt}|{m}|KL_REL1|n']:
                print(f"{a.name} {dt} {m:8s} KL_REL1 {tot[f'{dt}|{m}|KL_REL1|e']} / {tot[f'{dt}|{m}|KL_REL1|n']}   significant {tot[f'{dt}|{m}|SIG|e']} / {tot[f'{dt}|{m}|SIG|n']}", flush=True)
    print(f'{a.name}: {len(tasks)} units in {time.time() - t0:.0f} s; unit files missing from the compute stage: {miss}', flush=True)
    return 1 if miss or not tasks else 0
if __name__ == '__main__':
    sys.exit(main())
