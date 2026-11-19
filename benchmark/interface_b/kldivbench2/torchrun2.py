from __future__ import annotations
import os, warnings
import numpy as np, torch
from stablekl import interface_b as KF
NP = {'fp16': (np.float16, np.uint16), 'fp32': (np.float32, np.uint32)}
TD = {'fp16': torch.float16, 'fp32': torch.float32}
REDS = ('none', 'sum', 'mean', 'batchmean')
METHODS = {'fp16': ('C', 'D32', 'D64'), 'fp32': ('C', 'D64')}

def run_batch(xbits, tbits, dt, log_target, methods=None, device='cpu', want_pre=None, kl=None):
    f, u = NP[dt]
    xv = np.asarray(xbits, dtype=u).view(f)
    tv = np.asarray(tbits, dtype=u).view(f)
    out = {}
    kl = kl or KF.kl_div
    if os.environ.get('KLDIV2_INJECT') == 'uncertified':

        def kl(x, t, method=None, **kw):
            if method == 'D64':
                raise KF.UncertifiedRounding('injected')
            return KF.kl_div(x, t, method=method, **kw)
    for m in methods or METHODS[dt]:
        res = {'call_status': {}, 'grad_status': {}, 'errors': {}, 'dtype_ok': True, 'pre': {}}
        for red in REDS:
            res[red] = res[red + '_gx'] = res[red + '_gt'] = None
            tap = {} if want_pre and (m, red) in want_pre else None
            try:
                x = torch.from_numpy(xv.copy()).to(device).requires_grad_(True)
                t = torch.from_numpy(tv.copy()).to(device).requires_grad_(True)
                KF.TAP = tap
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    y = kl(x, t, reduction=red, log_target=log_target, method=m)
                yb = y.detach().cpu().numpy().view(u)
                res[red] = yb.reshape(-1).tolist() if red == 'none' else int(yb)
                res['dtype_ok'] &= y.dtype == TD[dt]
                res['call_status'][red] = 'OK'
            except Exception as e:
                KF.TAP = None
                res['call_status'][red] = 'CALL_ERROR'
                res['grad_status'][red] = 'NO_CALL'
                res['errors'][red] = f'{type(e).__name__}: {e}'[:200]
                continue
            try:
                (y.sum() if red == 'none' else y).backward()
                if x.grad is None or t.grad is None:
                    res['grad_status'][red] = 'GRAD_MISSING'
                else:
                    res[red + '_gx'] = x.grad.cpu().numpy().view(u).reshape(-1).tolist()
                    res[red + '_gt'] = t.grad.cpu().numpy().view(u).reshape(-1).tolist()
                    res['grad_status'][red] = 'OK'
            except Exception as e:
                res['grad_status'][red] = 'BACKWARD_ERROR'
                res['errors'][red + '_backward'] = f'{type(e).__name__}: {e}'[:200]
            finally:
                KF.TAP = None
            if tap is not None:
                for part, key in (('out', 'out_pre'), ('gx', 'gx_pre'), ('gt', 'gt_pre')):
                    if key in tap:
                        res['pre'][red, part] = tap[key].numpy().reshape(-1)
        out[m] = res
    inj = os.environ.get('KLDIV2_INJECT')
    if inj and 'D64' in out:
        r = out['D64']
        one = lambda b, k: int(np.nextafter(np.asarray(b, dtype=u).view(f), f(np.inf)).view(u)) if k == 1 else one(one(b, 1), k - 1)
        if inj == 'bump3' and r['sum'] is not None:
            r['sum'] = one(r['sum'], 3)
        if inj == 'nan_gt' and r['mean_gt'] is not None:
            r['mean_gt'][0] = int(np.asarray(np.nan, dtype=f).view(u))
        if inj == 'scale_gt' and r['sum_gt'] is not None:
            r['sum_gt'] = (np.asarray(r['sum_gt'], dtype=u).view(f) * f(1.5)).view(u).tolist()
        if inj == 'false_inf' and r['none'] is not None:
            r['none'][0] = int(np.asarray(np.inf, dtype=f).view(u))
    return out

def environment(device='cpu'):
    e = {'torch': torch.__version__, 'cpu_capability': torch.backends.cpu.get_cpu_capability(), 'threads': torch.get_num_threads(), 'numpy': np.__version__, 'kldivfix': KF.__version__, 'device': device}
    if device != 'cpu':
        e['cuda'] = torch.version.cuda
        e['gpu'] = torch.cuda.get_device_name(0)
    try:
        import mpmath
        e['mpmath'] = mpmath.__version__
    except Exception as ex:
        e['mpmath'] = f'MISSING: {ex}'
    return e
