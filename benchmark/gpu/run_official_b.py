from __future__ import annotations
import argparse, json, os, sys, warnings
from collections import Counter
from fractions import Fraction as Fr
from pathlib import Path
import numpy as np

def cfg_name(lt, dt, K):
    return f"lt{('T' if lt else 'F')}_{dt}_K{K}"

def link(root, root3, cfg):
    d = Path(root3) / cfg
    d.mkdir(parents=True, exist_ok=True)
    for sub in ('inputs', 'truth'):
        src = (Path(root) / cfg / sub).resolve()
        dst = d / sub
        if not src.is_dir():
            raise SystemExit(f'frozen directory missing: {src}')
        if not dst.is_symlink():
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass
        if not dst.is_symlink() or dst.resolve() != src:
            raise SystemExit(f'{dst} is not a link to {src}')
    return d

def patch(device):
    import torch, torch.nn.functional as F
    import kldivbench.torchrun as R
    from klbench.fp import round_fraction, get_format
    dev = torch.device(device)
    NP, TD, REDS, METHODS, CDT, _bits = (R.NP, R.TD, R.REDS, R.METHODS, R.CDT, R._bits)

    def run_batch(xbits, tbits, dt, log_target, kl=F.kl_div, methods=None):
        f, u = NP[dt]
        xv = np.asarray(xbits, dtype=u).view(f)
        tv = np.asarray(tbits, dtype=u).view(f)
        out = {}
        fmt = get_format(dt)
        for m in methods or METHODS[dt]:
            cdt = CDT[m] or dt
            res = {'call_status': {}, 'grad_status': {}, 'errors': {}, 'dtype_ok': True}
            for red in REDS:
                res[red] = res[red + '_gx'] = res[red + '_gt'] = None
                try:
                    x = torch.from_numpy(xv.copy()).to(dev).to(TD[cdt]).requires_grad_(True)
                    t = torch.from_numpy(tv.copy()).to(dev).to(TD[cdt]).requires_grad_(True)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        y = kl(x, t, reduction=red, log_target=log_target)
                    yb = _bits(y.detach().cpu().numpy(), dt)
                    res[red] = yb.reshape(-1).tolist() if red == 'none' else int(yb)
                    res['dtype_ok'] &= y.dtype == TD[cdt] and y.device.type == dev.type
                    res['call_status'][red] = 'OK'
                except Exception as e:
                    res['call_status'][red] = 'CALL_ERROR'
                    res['grad_status'][red] = 'NO_CALL'
                    res['errors'][red] = f'{type(e).__name__}: {e}'[:200]
                    continue
                try:
                    (y.sum() if red == 'none' else y).backward()
                    if x.grad is None or t.grad is None:
                        res['grad_status'][red] = 'GRAD_MISSING'
                    else:
                        res[red + '_gx'] = _bits(x.grad.cpu().numpy(), dt).reshape(-1).tolist()
                        res[red + '_gt'] = _bits(t.grad.cpu().numpy(), dt).reshape(-1).tolist()
                        res['grad_status'][red] = 'OK'
                except Exception as e:
                    res['grad_status'][red] = 'BACKWARD_ERROR'
                    res['errors'][red + '_backward'] = f'{type(e).__name__}: {e}'[:200]
                if m == 'A' and red == 'none':
                    p = y.detach().cpu().numpy().reshape(-1)
                    s = f(0)
                    with np.errstate(all='ignore'):
                        for v in p:
                            s = f(s + v)
                    res['Y_seq'] = int(np.asarray(s, dtype=f).view(u))
                    res['Y_exact'] = int(round_fraction(sum((Fr(float(v)) for v in p), Fr(0)), fmt)) if np.isfinite(p).all() else None
            if m == 'A':
                res['sum_noncontig'] = None
                try:
                    wide_x = np.zeros((xv.shape[0], 2 * xv.shape[1]), dtype=f)
                    wide_t = np.ones_like(wide_x)
                    wide_x[:, ::2] = xv
                    wide_t[:, ::2] = tv
                    xs = torch.from_numpy(wide_x).to(dev)[:, ::2]
                    ts = torch.from_numpy(wide_t).to(dev)[:, ::2]
                    assert not xs.is_contiguous()
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        res['sum_noncontig'] = int(_bits(kl(xs, ts, reduction='sum', log_target=log_target).cpu().numpy(), dt))
                except Exception as e:
                    res['errors']['noncontig'] = f'{type(e).__name__}: {e}'[:200]
            out[m] = res
        return out
    env0 = R.environment

    def environment():
        e = dict(env0())
        e['device'] = str(dev)
        e['device_name'] = torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'
        e['runner'] = 'b_off_gpu.py'
        return e
    R.run_batch = run_batch
    R.environment = environment
    if dev.type == 'cuda':
        assert torch.cuda.is_available(), 'cuda requested but not available'

def compare(root, root3, cfg):
    from kldivbench import store
    c = Counter()
    files = sorted((Path(root3) / cfg / 'out').glob('*.npz'))
    ex = []
    for p in files:
        g, gm = store.load(p)
        o, om = store.load(Path(root) / cfg / 'out' / p.name)
        if gm['inputs_sha256'] != om['inputs_sha256'] or not (g['rows'] == o['rows']).all():
            raise SystemExit(f'outputs do not belong to the same inputs: {p}')
        c['files'] += 1
        for m in gm['methods']:
            for j, red in enumerate(('none', 'sum', 'mean', 'batchmean')):
                ok = (g[m + '__cs'][:, j] == 0) & (o[m + '__cs'][:, j] == 0)
                c[f'{m}|{red}|call_status_differs'] += int((g[m + '__cs'][:, j] != o[m + '__cs'][:, j]).sum())
                a, b = (g[f'{m}__{red}'][ok], o[f'{m}__{red}'][ok])
                d = a != b
                c[f'{m}|{red}|fwd|n'] += int(a.size)
                c[f'{m}|{red}|fwd|differ'] += int(d.sum())
                c[f'{m}|{red}|fwd|batches'] += int(ok.sum())
                c[f'{m}|{red}|fwd|batches_differ'] += int(d.reshape(len(a), -1).any(axis=1).sum()) if a.size else 0
                okg = ok & (g[m + '__gs'][:, j] == 0) & (o[m + '__gs'][:, j] == 0)
                for side in ('gx', 'gt'):
                    a, b = (g[f'{m}__{red}_{side}'][okg], o[f'{m}__{red}_{side}'][okg])
                    c[f'{m}|{red}|{side}|n'] += int(a.size)
                    c[f'{m}|{red}|{side}|differ'] += int((a != b).sum())
                if d.any() and len(ex) < 10:
                    k = int(np.argwhere(d.reshape(len(a), -1).any(axis=1))[0, 0])
                    ex.append(dict(file=p.name, method=m, reduction=red, gpu=np.asarray(a[k]).reshape(-1)[:4].tolist(), cpu=np.asarray(b[k]).reshape(-1)[:4].tolist()))
    res = dict(cfg=cfg, counts=dict(c), examples=ex, gpu_environment=gm.get('environment') if files else None, cpu_environment=om.get('environment') if files else None)
    (Path(root3) / cfg / 'compare.json').write_text(json.dumps(res, indent=1, default=str))
    for m in ('A', 'B32', 'B64'):
        if f'{m}|sum|fwd|n' in c:
            print(f'compare {cfg} {m}: ' + '  '.join((f"{red} {c[f'{m}|{red}|fwd|differ']}/{c[f'{m}|{red}|fwd|n']}" for red in ('none', 'sum', 'mean', 'batchmean'))), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['run', 'score', 'compare'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--root3', required=True)
    ap.add_argument('--lt', type=int, required=True)
    ap.add_argument('--dt', required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--shard', default=None)
    ap.add_argument('--log-prefix', default='')
    a = ap.parse_args()
    cfg = cfg_name(bool(a.lt), a.dt, a.K)
    link(a.root, a.root3, cfg)
    if a.stage == 'compare':
        return compare(a.root, a.root3, cfg)
    from kldivbench import pipeline as P
    if a.stage == 'run':
        patch(a.device)
    argv = [a.stage, '--root', a.root3, '--lt', str(a.lt), '--dt', a.dt, '--K', str(a.K), '--log-prefix', a.log_prefix] + (['--shard', a.shard] if a.shard else [])
    P.main(argv)
if __name__ == '__main__':
    main()
