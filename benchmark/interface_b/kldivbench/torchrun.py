from __future__ import annotations
import warnings
from fractions import Fraction as Fr
import numpy as np, torch, torch.nn.functional as F
from klbench.fp import round_fraction, get_format
NP = {'fp16': (np.float16, np.uint16), 'fp32': (np.float32, np.uint32), 'fp64': (np.float64, np.uint64)}
TD = {'fp16': torch.float16, 'fp32': torch.float32, 'fp64': torch.float64}
REDS = ('none', 'sum', 'mean', 'batchmean')
METHODS = {'fp16': ('A', 'B32', 'B64'), 'fp32': ('A', 'B64')}
CDT = {'A': None, 'B32': 'fp32', 'B64': 'fp64'}

def _bits(a, dt):
    f, u = NP[dt]
    with np.errstate(all='ignore'):
        return np.asarray(a).astype(f).view(u)

def torch_inputs(zp_bits, zq_bits, dt, log_target):
    f, u = NP[dt]
    zp = torch.from_numpy(np.asarray(zp_bits, dtype=u).view(f).copy())
    zq = torch.from_numpy(np.asarray(zq_bits, dtype=u).view(f).copy())
    x = torch.log_softmax(zq, -1)
    t = torch.log_softmax(zp, -1) if log_target else torch.softmax(zp, -1)
    return (x.numpy().view(u).tolist(), t.numpy().view(u).tolist())

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
                x = torch.from_numpy(xv.copy()).to(TD[cdt]).requires_grad_(True)
                t = torch.from_numpy(tv.copy()).to(TD[cdt]).requires_grad_(True)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    y = kl(x, t, reduction=red, log_target=log_target)
                yb = _bits(y.detach().numpy(), dt)
                res[red] = yb.reshape(-1).tolist() if red == 'none' else int(yb)
                res['dtype_ok'] &= y.dtype == TD[cdt]
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
                    res[red + '_gx'] = _bits(x.grad.numpy(), dt).reshape(-1).tolist()
                    res[red + '_gt'] = _bits(t.grad.numpy(), dt).reshape(-1).tolist()
                    res['grad_status'][red] = 'OK'
            except Exception as e:
                res['grad_status'][red] = 'BACKWARD_ERROR'
                res['errors'][red + '_backward'] = f'{type(e).__name__}: {e}'[:200]
            if m == 'A' and red == 'none':
                p = y.detach().numpy().reshape(-1)
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
                xs = torch.from_numpy(wide_x)[:, ::2]
                ts = torch.from_numpy(wide_t)[:, ::2]
                assert not xs.is_contiguous()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    res['sum_noncontig'] = int(_bits(kl(xs, ts, reduction='sum', log_target=log_target).numpy(), dt))
            except Exception as e:
                res['errors']['noncontig'] = f'{type(e).__name__}: {e}'[:200]
        out[m] = res
    return out

def environment():
    return {'torch': torch.__version__, 'cpu_capability': torch.backends.cpu.get_cpu_capability(), 'threads': torch.get_num_threads(), 'numpy': np.__version__}
