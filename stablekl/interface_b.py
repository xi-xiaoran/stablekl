"""Repaired torch.nn.functional.kl_div (methods C, D32, D64); use it through stablekl.kl_div."""
from __future__ import annotations
import math, warnings
from fractions import Fraction as Fr
import torch
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
__version__ = '1.1'
METHODS = ('C', 'D32', 'D64')
_FMT = {torch.float16: (11, -14, 15), torch.float32: (24, -126, 127), torch.float64: (53, -1022, 1023)}
_MEAN_WARNING = "reduction: 'mean' divides the total loss by both the batch size and the support size.'batchmean' divides only by the batch size, and aligns with the KL div math definition.'mean' will be changed to behave the same as 'batchmean' in the next major release."
TAP = None
EXT_LADDER = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
_U = {torch.float16: 2.0 ** (-11), torch.float32: 2.0 ** (-24), torch.float64: 2.0 ** (-53)}
_TINY = {torch.float16: 2.0 ** (-24), torch.float32: 2.0 ** (-149), torch.float64: 2.0 ** (-1074)}
STATS = {'ext_calls': 0, 'ext_elements': 0, 'guard_calls': 0}

class UncertifiedRounding(ArithmeticError):
    pass

def _wide(method, d):
    if method == 'C':
        return d
    if method == 'D32':
        return torch.float32 if d == torch.float16 else d
    return torch.float64

def _two_sum(a, b):
    s = a + b
    bv = s - a
    return (s, a - (s - bv) + (b - bv))

def _sum3(a, b, c):
    s1, e1 = _two_sum(a, b)
    s2, e2 = _two_sum(s1, c)
    return (s2 + (e1 + e2), (s2 == 0) & (e1 == 0) & (e2 == 0))

def _csum(v):
    errs = []
    while v.shape[-1] > 1:
        if v.shape[-1] % 2:
            v = torch.cat([v, torch.zeros_like(v[..., :1])], -1)
        s, e = _two_sum(v[..., 0::2], v[..., 1::2])
        errs.append(e)
        v = s
    r = v[..., 0]
    return r + torch.cat(errs, -1).sum(-1) if errs else r

def _round(v, d):
    if v.dtype == d:
        return v
    if not (v.dtype == torch.float64 and d == torch.float16):
        return v.to(d)
    f = v.to(torch.float32)
    b = f.view(torch.int32)
    fix = (f.double() != v) & ~torch.isnan(v) & (b & 1 == 0)
    up = v.abs() > f.double().abs()
    return torch.where(fix, torch.where(up, b + 1, b - 1), b).view(torch.float32).to(d)
_POW2 = {}

def _pow2(d, device):
    k = (d, str(device))
    if k not in _POW2:
        p, emin, emax = _FMT[d]
        lo = emin - p + 1
        t = torch.ones(emax - lo + 1, dtype=torch.float64, device=device)
        t = torch.ldexp(t, torch.arange(lo, emax + 1, device=device)).to(d)
        _POW2[k] = (t, lo, emax)
    return _POW2[k]

def _scale2(m, e, d):
    tbl, lo, emax = _pow2(d, m.device)
    e = e.clamp(2 * lo, 2 * emax).long()
    h = torch.div(e, 2, rounding_mode='floor')
    return m * tbl[h - lo] * tbl[e - h - lo]

def _prod_div(factors, den, d):
    naive = factors[0]
    for f in factors[1:]:
        naive = naive * f
    ok = torch.ones_like(naive, dtype=torch.bool)
    for f in factors:
        ok &= torch.isfinite(f) & (f != 0)
    one = torch.ones_like(naive)
    m, e = torch.frexp(torch.where(ok, factors[0], one))
    e = e.long()
    for f in factors[1:]:
        mi, ei = torch.frexp(torch.where(ok, f, one))
        m, e2 = torch.frexp(m * mi)
        e = e + ei + e2
    md, ed = math.frexp(den)
    m, e2 = torch.frexp(m / torch.tensor(md, dtype=d, device=m.device))
    e = e + e2 - ed
    return torch.where(ok, _scale2(m, e, d), naive)
_LN2 = {torch.float16: (0.6875, math.log(2) - 0.6875), torch.float32: (45426 / 65536, math.log(2) - 45426 / 65536), torch.float64: (762123384785 / 2 ** 40, float.fromhex('0x1.9ef35793c7673p-41'))}

def _logdiff(ts, xs, d, plus_one=False):
    m, e = torch.frexp(ts)
    low = m < 0.7071067811865476
    m = torch.where(low, 2 * m, m)
    ef = (e - low.to(e.dtype)).to(d)
    hi, lo = _LN2[d]
    s, err = _two_sum(ef * hi, -xs)
    if plus_one:
        s, e2 = _two_sum(s, torch.ones_like(ts))
        err = err + e2
    s, e3 = _two_sum(s, torch.log(m))
    s, e4 = _two_sum(s, ef * lo)
    return s + (err + (e3 + e4))

def _exp_parts(tw, cof, d):
    p, emin, emax = _FMT[d]
    e = torch.exp(tw)
    split = (e < 2.0 ** emin) & torch.isfinite(tw) & (cof.abs() > 1)
    q = torch.exp(tw / 4)
    one = torch.ones_like(tw)
    return ([torch.where(split, q, e)] + [torch.where(split, q, one)] * 3, split)

def _rn_frac(q, d):
    if q == 0:
        return 0.0
    p, emin, emax = _FMT[d]
    s = -1.0 if q < 0 else 1.0
    a = abs(q)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fr(2) ** e > a:
        e -= 1
    e = max(e, emin)
    quantum = Fr(2) ** (e - p + 1)
    n, r = divmod(a, quantum)
    if r * 2 > quantum or (r * 2 == quantum and n % 2):
        n += 1
    v = n * quantum
    return s * math.inf if v >= Fr(2) ** (emax + 1) else s * float(v)

def _rn_raw(raw, d):
    sign, man, exp, bc = raw
    if man == 0:
        if exp == 0:
            return 0.0
        from mpmath.libmp import finf, fninf
        if raw == finf:
            return math.inf
        if raw == fninf:
            return -math.inf
        return math.nan
    p, emin, emax = _FMT[d]
    top = exp + bc
    s = -1.0 if sign else 1.0
    if top - 1 >= emax + 1:
        return s * math.inf
    if top <= emin - p:
        return s * 0.0
    return _rn_frac((-1 if sign else 1) * Fr(man) * Fr(2) ** exp, d)

def _ext(kind, lt, xs, ts, Gs, den, d):
    import mpmath
    iv = mpmath.iv
    STATS['ext_calls'] += 1
    STATS['ext_elements'] += len(xs)
    R = Fr(0)
    coef = {}
    for x, t, G in zip(xs, ts, Gs):
        X, T, Gf = (Fr(x), Fr(t), Fr(G))
        if lt:
            c = T - X if kind == 'L' else -Gf if kind == 'gx' else Gf * (T - X + 1)
            coef[t] = coef.get(t, Fr(0)) + c
        elif kind == 'L':
            coef[t] = coef.get(t, Fr(0)) + T
            R -= T * X
        elif kind == 'gx':
            R -= Gf * T
        else:
            coef[t] = coef.get(t, Fr(0)) + Gf
            R += Gf * (1 - X)
    neutral = 0.0 if lt else 1.0
    if lt:
        R += coef.pop(neutral, Fr(0))
    else:
        coef.pop(neutral, None)
    coef = {k: c for k, c in coef.items() if c != 0}
    if not coef:
        return _rn_frac(R / den, d)
    old = iv.prec
    try:
        for prec in EXT_LADDER:
            iv.prec = prec
            tot = iv.mpf(R.numerator) / R.denominator
            for tau, c in coef.items():
                f = iv.exp(iv.mpf(tau)) if lt else iv.log(iv.mpf(tau))
                tot += f * iv.mpf(c.numerator) / c.denominator
            tot = tot / den
            lo, hi = tot._mpi_
            a, b = (_rn_raw(lo, d), _rn_raw(hi, d))
            if a == b:
                return a
    finally:
        iv.prec = old
    raise UncertifiedRounding(f'kind={kind} log_target={lt} n={len(xs)}: the rounding is undecided at {EXT_LADDER[-1]} bits')

def _near_boundary(v, bound, d):
    p, emin, emax = _FMT[d]
    it = {torch.float16: torch.int16, torch.float32: torch.int32}[d]
    a = v.abs()
    r = _round(a, d)
    big = 2.0 ** (emax + 1)
    bits = r.view(it).long()
    isinf = torch.isinf(r)
    r64 = torch.where(isinf, torch.full_like(a, big), r.double())
    up = (bits + 1).to(it).view(d).double()
    up = torch.where(torch.isinf(up) | isinf, torch.full_like(a, big), up)
    dn = torch.where(bits > 0, (bits - 1).clamp(min=0).to(it).view(d).double(), -up)
    dist = torch.minimum((a - (r64 + up) / 2).abs(), (a - (r64 + dn) / 2).abs())
    dist = torch.where(isinf, (a - (dn + big) / 2).abs(), dist)
    return (dist <= bound) & torch.isfinite(v)

def _outside(xd, td, lt):
    bad = torch.isnan(xd) | torch.isnan(td) | (xd == math.inf) | (td == math.inf)
    return bad if lt else bad | (td < 0)

def _official(xb, tb, lt, R0, extra, G=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if G is None:
            y = F.kl_div(xb, tb, reduction=R0, log_target=lt)
            return y / extra if extra != 1 else y
        with torch.enable_grad():
            xc = xb.detach().clone().requires_grad_(True)
            tc = tb.detach().clone().requires_grad_(True)
            y = F.kl_div(xc, tc, reduction=R0, log_target=lt)
            if extra != 1:
                y = y / extra
            return torch.autograd.grad(y, [xc, tc], grad_outputs=G.to(y.dtype).expand(y.shape))

def _masks(xw, tw, lt):
    xn = xw == -math.inf
    if lt:
        tn = tw == -math.inf
        reg = ~tn & ~xn
        return (reg, dict(pinf=xn & ~tn, zero=tn, undef=tn & xn))
    pos = tw > 0
    reg = pos & ~xn
    return (reg, dict(pinf=pos & xn, zero=~pos, undef=~pos & xn))

def _lists(idx, *tensors):
    return [t[idx].double().tolist() for t in tensors]

def _forward_core(x, t, lt, method, R0, n, extra):
    den = (n if R0 == 'mean' else 1) * extra
    d = x.dtype
    W = _wide(method, d)
    xb, tb = torch.broadcast_tensors(x, t)
    shape = xb.shape
    xw = xb.reshape(-1).to(W)
    tw = tb.reshape(-1).to(W)
    bad = _outside(xw, tw, lt)
    anybad = bool(bad.any())
    if anybad:
        if R0 != 'none':
            out = _official(xb, tb, lt, R0, extra)
            return (out, out.double())
        xw = torch.where(bad, torch.zeros_like(xw), xw)
        tw = torch.where(bad, torch.zeros_like(tw) if lt else torch.ones_like(tw), tw)
    reg, mk = _masks(xw, tw, lt)
    zero = torch.zeros((), dtype=W, device=xw.device)
    pinf = torch.full((), math.inf, dtype=W, device=xw.device)
    ts = torch.where(reg, tw, torch.ones_like(tw) if not lt else zero)
    xs = torch.where(reg, xw, zero)
    if lt:
        m = ts - xs
        Lw = torch.exp(ts) * m
        live = reg & (m != 0)
        if method == 'C':
            parts, split = _exp_parts(ts, m, d)
            Lw = torch.where(split, _prod_div(parts + [m], 1, d), Lw)
        Lw = torch.where(m == 0, zero, Lw)
    elif method == 'C':
        Lw = torch.xlogy(ts, ts) - ts * xs
        live = reg
    else:
        Lw = ts * _logdiff(ts, xs, W)
        live = reg
    Lw = torch.where(reg, Lw, torch.where(mk['pinf'], pinf, zero))
    flag = live & ~torch.isfinite(Lw)
    ones = [1.0]
    guard = method != 'C' and W != d
    full = 1.0 if method == 'D64' else 0.0
    if guard:
        u = _U[W]
        L64 = torch.where(torch.isfinite(Lw), Lw, zero).double().abs()
        x64, t64 = (xs.double().abs(), ts.double())
        if lt:
            lost = (torch.exp(ts) < 2.0 ** _FMT[W][1]) & live
            bel = full * 4.2 * u * L64 + torch.where(lost, 2 * _TINY[W] * (t64 - xs.double()).abs(), torch.zeros_like(L64))
        else:
            bel = full * (t64 * (0.81 * u + 1.2 * u * u * (torch.log(t64).abs() + x64 + 2)) + 2.2 * u * L64)
        bel = torch.where(live, bel, torch.zeros_like(bel))

    def ext_sum(idx):
        xs_, ts_ = _lists(idx, xw, tw)
        return _ext('L', lt, xs_, ts_, ones * len(xs_), den, d)
    if R0 == 'none':
        if den == 1:
            out = _round(Lw, d)
            pre = Lw
        elif method == 'C':
            out = Lw / den
            pre = out
        else:
            pre = Lw / den
            out = _round(pre, d)
        if guard:
            g_ = live & ~flag & _near_boundary(pre.double(), bel / den + full * 1.2 * _U[W] * pre.double().abs() * (den != 1), d)
            STATS['guard_calls'] += int(g_.sum())
            flag = flag | g_
        if flag.any():
            out = out.clone()
            pre = pre.clone().double()
            for i in flag.nonzero().flatten().tolist():
                v = ext_sum(torch.tensor([i], device=xw.device))
                out[i] = v
                pre[i] = v
        if anybad:
            out = torch.where(bad, _official(xb, tb, lt, R0, extra).reshape(-1), out)
            pre = torch.where(bad, out.to(pre.dtype), pre)
        return (out.reshape(shape), pre.reshape(shape))
    if mk['pinf'].any():
        out = torch.full((), math.inf, dtype=d, device=xw.device)
        return (out, out.double())
    S = None
    if not flag.any():
        if method == 'C':
            S = Lw.reshape(shape).mean() if R0 == 'mean' else Lw.reshape(shape).sum()
            if extra != 1:
                S = S / extra
        else:
            S = _csum(Lw)
        if not torch.isfinite(S):
            S = None
    if S is None:
        v = ext_sum(reg.nonzero().flatten())
        out = torch.tensor(v, dtype=d, device=xw.device)
        return (out, out.double())
    if method == 'C':
        return (S, S.double())
    pre = S / den
    if guard:
        nn = Lw.numel()
        lg = max(1, (2 * nn - 1).bit_length())
        u = _U[W]
        bs = (bel.sum() * 1.01 + full * 1.02 * nn * lg * u * u * L64.sum()) / den + full * 2.3 * u * pre.double().abs()
        if bool(_near_boundary(pre.double().reshape(1), bs.reshape(1), d)):
            STATS['guard_calls'] += 1
            v = ext_sum(reg.nonzero().flatten())
            out = torch.tensor(v, dtype=d, device=xw.device)
            return (out, out.double())
    return (_round(pre, d), pre)

def _reduce_to(vw, shape, tshape, flag, W, d, method, ext_group, bnd=None):
    if tuple(tshape) == tuple(shape):
        return (_round(vw, d).reshape(shape), vw.reshape(shape))
    nd = len(shape)
    lead = nd - len(tshape)
    dims = list(range(lead)) + [lead + i for i, (a, b) in enumerate(zip(tshape, shape[lead:])) if a == 1 and b != 1]
    keep = [i for i in range(nd) if i not in dims]
    perm = keep + dims
    ng = 1
    for i in keep:
        ng *= shape[i]
    vp = vw.reshape(shape).permute(perm).reshape(ng, -1)
    fp = flag.reshape(shape).permute(perm).reshape(ng, -1)
    ip = torch.arange(vw.numel(), device=vw.device).reshape(shape).permute(perm).reshape(ng, -1)
    fin = torch.isfinite(vp).all(-1)
    plain = vp.sum(-1)
    s = plain if method == 'C' else torch.where(fin, _csum(torch.where(torch.isfinite(vp), vp, torch.zeros_like(vp))), plain)
    pre = s.clone().double()
    out = _round(s, d)
    need = fp.any(-1) | fin & ~torch.isfinite(s)
    if bnd is not None:
        full = 1.0 if method == 'D64' else 0.0
        u = _U[W]
        nn = vp.shape[-1]
        lg = max(1, (2 * nn - 1).bit_length())
        av = torch.where(torch.isfinite(vp), vp, torch.zeros_like(vp)).double().abs().sum(-1)
        bs = bnd.clamp(min=0).reshape(shape).permute(perm).reshape(ng, -1).sum(-1) * 1.01 + full * (1.02 * nn * lg * u * u * av + 2.3 * u * s.double().abs())
        g_ = fin & ~need & _near_boundary(s.double(), bs, d)
        STATS['guard_calls'] += int(g_.sum())
        need = need | g_
    if need.any():
        out = out.clone()
        for g in need.nonzero().flatten().tolist():
            v = ext_group(ip[g])
            out[g] = v
            pre[g] = v
    return (out.reshape(tshape), pre.reshape(tshape))

def _grad_core(x, t, G, lt, method, R0, n, extra):
    den = (n if R0 == 'mean' else 1) * extra
    d = x.dtype
    W = _wide(method, d)
    xb, tb = torch.broadcast_tensors(x, t)
    shape = xb.shape
    dev = x.device
    xd = xb.reshape(-1)
    td = tb.reshape(-1)
    Gd = G.to(d).expand(shape).reshape(-1)
    xw, tw, Gw = (xd.to(W), td.to(W), Gd.to(W))
    bad = _outside(xw, tw, lt)
    anybad = bool(bad.any())
    if anybad:
        xw = torch.where(bad, torch.zeros_like(xw), xw)
        tw = torch.where(bad, torch.zeros_like(tw) if lt else torch.ones_like(tw), tw)
    reg, mk = _masks(xw, tw, lt)
    g0 = Gd == 0
    gfin = torch.isfinite(Gd)
    zero = torch.zeros((), dtype=W, device=dev)
    nan = torch.full((), math.nan, dtype=W, device=dev)
    ts = torch.where(reg, tw, torch.ones_like(tw) if not lt else zero)
    xs = torch.where(reg, xw, zero)
    sgn_inf = torch.where(Gw > 0, math.inf, -math.inf).to(W)
    C = method == 'C'
    if lt:
        tfin = ~mk['zero']
        e = torch.exp(torch.where(tfin, tw, zero))
        a, az = _sum3(ts, -xs, torch.ones_like(ts))
        if C:
            te = torch.where(tfin, tw, zero)
            px_, _ = _exp_parts(te, _prod_div([Gw], den, d), d)
            pt_, _ = _exp_parts(te, _prod_div([Gw, a], den, d), d)
            gx = -_prod_div([Gw] + px_, den, d)
            gt = _prod_div([Gw] + pt_ + [a], den, d)
        else:
            gx = -(Gw * e) / den
            gt = Gw * (e * a) / den
        gt = torch.where(az, zero, gt)
        liv_x = tfin
        liv_t = reg & ~az
        gx = torch.where(tfin, gx, zero)
        gt = torch.where(reg, gt, torch.where(mk['undef'], nan, torch.where(mk['pinf'], sgn_inf, zero)))
    else:
        b = _logdiff(ts, xs, W, plus_one=True)
        if C:
            gx = -_prod_div([Gw, tw], den, d)
            gt = _prod_div([Gw, b], den, d)
        else:
            gx = -(Gw * tw) / den
            gt = Gw * b / den
        liv_x = ~mk['zero']
        liv_t = reg
        gx = torch.where(mk['zero'], zero, gx)
        gt = torch.where(reg, gt, torch.where(mk['undef'], nan, torch.where(mk['pinf'], sgn_inf, -sgn_inf)))
    flag_x = liv_x & gfin & ~g0 & ~torch.isfinite(gx)
    flag_t = liv_t & gfin & ~g0 & ~torch.isfinite(gt)
    gx = torch.where(g0, zero, gx)
    gt = torch.where(g0, zero, gt)
    bx = bt = None
    if not C and W != d:
        full = 1.0 if method == 'D64' else 0.0
        u = _U[W]
        aG = Gw.double().abs() / den
        f64 = lambda v: torch.where(torch.isfinite(v), v, zero).double().abs()
        x64, t64 = (xs.double().abs(), ts.double())
        z64 = torch.zeros_like(aG)
        if lt:
            lost = e < 2.0 ** _FMT[W][1]
            bx = full * 4.2 * u * f64(gx) + torch.where(lost, 2 * _TINY[W] * aG, z64)
            bt = full * (aG * e.double() * 1.2 * u * u * (t64.abs() + x64 + 1) + 5.3 * u * f64(gt)) + torch.where(lost, 2 * _TINY[W] * aG * a.double().abs(), z64)
        else:
            bx = full * 2.2 * u * f64(gx)
            if den & den - 1 == 0 and 2 * _FMT[d][0] <= _FMT[W][0]:
                bx = bx - 1.0
            bt = full * (aG * (0.81 * u + 1.2 * u * u * (torch.log(t64).abs() + x64 + 2)) + 3.3 * u * f64(gt))
        okx = liv_x & gfin & ~g0 & ~flag_x
        okt = liv_t & gfin & ~g0 & ~flag_t
        bx = torch.where(okx | (bx < 0), bx, z64)
        bt = torch.where(okt, bt, z64)
        if tuple(x.shape) == tuple(shape):
            g_ = okx & _near_boundary(gx.double(), bx, d)
            STATS['guard_calls'] += int(g_.sum())
            flag_x = flag_x | g_
        if tuple(t.shape) == tuple(shape):
            g_ = okt & _near_boundary(gt.double(), bt, d)
            STATS['guard_calls'] += int(g_.sum())
            flag_t = flag_t | g_

    def patch(v, flag, kind):
        if not flag.any():
            return v
        v = v.clone()
        for i in flag.nonzero().flatten().tolist():
            xs_, ts_, Gs_ = _lists(torch.tensor([i], device=dev), xw, tw, Gw)
            if kind == 'gx' and (not math.isfinite(xs_[0])):
                xs_ = [0.0]
            v[i] = _ext(kind, lt, xs_, ts_, Gs_, den, d)
        return v

    def group(kind, liv):

        def f(idx):
            idx = idx[(liv & ~g0)[idx]]
            xs_, ts_, Gs_ = _lists(idx, xw, tw, Gw)
            if kind == 'gx':
                xs_ = [0.0] * len(ts_)
            return _ext(kind, lt, xs_, ts_, Gs_, den, d)
        return f
    gx = patch(gx, flag_x, 'gx')
    gt = patch(gt, flag_t, 'gt')
    if anybad:
        ox_, ot_ = _official(xb, tb, lt, R0, extra, G)
        gx = torch.where(bad, ox_.reshape(-1).to(W), gx)
        gt = torch.where(bad, ot_.reshape(-1).to(W), gt)
    ox, px = _reduce_to(gx, shape, x.shape, flag_x, W, d, method, group('gx', liv_x), bx)
    ot, pt = _reduce_to(gt, shape, t.shape, flag_t, W, d, method, group('gt', liv_t), bt)
    return (ox, ot, px, pt)

def _to_shape(v, bshape, shape):
    v = v.expand(bshape)
    return v.sum_to_size(shape) if tuple(bshape) != tuple(shape) else v

class _KLGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, t, G, cfg):
        lt, method, R0, n, extra = cfg
        gx, gt, px, pt = _grad_core(x, t, G, lt, method, R0, n, extra)
        if TAP is not None:
            TAP['gx_pre'] = px.detach().double().cpu()
            TAP['gt_pre'] = pt.detach().double().cpu()
        ctx.save_for_backward(x, t, G)
        ctx.cfg = cfg
        return (gx, gt)

    @staticmethod
    def backward(ctx, Hx, Ht):
        x, t, G = ctx.saved_tensors
        lt, method, R0, n, extra = ctx.cfg
        s = 1.0 / ((n if R0 == 'mean' else 1) * extra)
        if lt:
            e = torch.exp(t)
            u = t - x
            ddx = -(Ht * G) * s * e
            ddt = -(Hx * G) * s * e + Ht * G * s * e * (u + 2)
            ddG = -(Hx * e) * s + Ht * s * e * (u + 1)
        else:
            ddx = -(Ht * G) * s
            ddt = -(Hx * G) * s + Ht * G * s / t
            ddG = -(Hx * t) * s + Ht * s * (torch.log(t) + 1 - x)
        bs = torch.broadcast_shapes(x.shape, t.shape)
        return (_to_shape(ddx, bs, x.shape), _to_shape(ddt, bs, t.shape), _to_shape(ddG, bs, G.shape), None)

class _KL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, t, cfg):
        lt, method, R0, n, extra = cfg
        out, pre = _forward_core(x, t, lt, method, R0, n, extra)
        if TAP is not None:
            TAP['out_pre'] = pre.detach().double().cpu()
        ctx.save_for_backward(x, t)
        ctx.cfg = cfg
        return out

    @staticmethod
    def backward(ctx, G):
        x, t = ctx.saved_tensors
        gx, gt = _KLGrad.apply(x, t, G, ctx.cfg)
        return (gx, gt, None)

def kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False, method='D64'):
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')

    def official():
        return F.kl_div(input, target, size_average=size_average, reduce=reduce, reduction=reduction, log_target=log_target)
    if not (isinstance(input, torch.Tensor) and isinstance(target, torch.Tensor)):
        return official()
    if input.dtype != target.dtype or input.dtype not in _FMT or input.layout != torch.strided or (target.layout != torch.strided):
        return official()
    if size_average is None and reduce is None and (reduction not in ('none', 'sum', 'mean', 'batchmean')):
        return official()
    try:
        shape = torch.broadcast_shapes(input.shape, target.shape)
    except RuntimeError:
        return official()
    n = 1
    for k in shape:
        n *= k
    if n == 0:
        return official()
    if size_average is not None or reduce is not None:
        R0 = _Reduction.legacy_get_string(size_average, reduce)
    else:
        if reduction == 'mean':
            warnings.warn(_MEAN_WARNING, stacklevel=2)
        R0 = 'sum' if reduction == 'batchmean' else reduction
    extra = input.size(0) if reduction == 'batchmean' and input.dim() != 0 else 1
    return _KL.apply(input, target, (bool(log_target), method, R0, n, extra))
