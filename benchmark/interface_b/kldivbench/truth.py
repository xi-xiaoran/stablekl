from __future__ import annotations
from fractions import Fraction as Fr
from flint import arb, ctx
from klbench.fp import get_format, bits_to_fraction, round_fraction
from klbench.ref.arbutil import fraction_to_arb, arb_bounds
LADDER = (128, 256, 512, 1024, 2048, 4096)
CERT = 'CERTIFIED'
EXZ = 'EXACT_ZERO'
EXGZ = 'EXACT_GRAD_ZERO'
CONVZ = 'CONVENTION_ZERO'
MINF = 'MATH_INF'
ONESIDED = 'ONE_SIDED_INF'
LIMIT = 'LIMIT_GRAD'
UNDEF = 'UNDEFINED'
DOMAIN = 'DOMAIN_UNDEFINED'
UNCERT_V = 'UNCERT_VALUE'
UNCERT_G = 'UNCERT_GRAD'
PINF = '+inf'
NINF = '-inf'

class Unsupported(ValueError):
    pass

def decode(bits, fmt):
    d = bits_to_fraction(int(bits), fmt)
    if d.kind == 'nan':
        raise Unsupported('NaN input')
    if d.kind == 'infinity':
        if d.sign > 0:
            raise Unsupported('+inf input')
        return NINF
    return d.value

def rn(value, fmt):
    if value == PINF:
        return fmt.infinity_bits
    if value == NINF:
        return fmt.sign_mask | fmt.infinity_bits
    return round_fraction(value, fmt)

def _same(a, b, fmt):
    m = fmt.sign_mask - 1
    return a == b or (a & m == 0 and b & m == 0)
Q_CERT, Q_OVF, Q_RZ = (1, 2, 4)
STATUS_CODES = (CERT, EXZ, EXGZ, CONVZ, MINF, ONESIDED, LIMIT, UNDEF, DOMAIN)

class Val:
    __slots__ = ('status', 'exact', 'fn', 'lo', 'hi', 'prec', '_c')

    def __init__(self, status, exact=None, fn=None):
        self.status = status
        self.exact = exact
        self.fn = fn
        self.lo = self.hi = None
        self.prec = None
        self._c = {}

    @property
    def scored(self):
        return self.status not in (UNDEF, DOMAIN)

    @property
    def infinite(self):
        return self.exact in (PINF, NINF)

    def ball(self):
        if not self.scored or self.infinite:
            return None
        return fraction_to_arb(self.exact) if self.exact is not None else self.fn()

    def certify(self, fmt, scales=(Fr(1),)):
        if not self.scored:
            return [(None, 0)] * len(scales)
        m = fmt.sign_mask - 1
        res = {}

        def flag(b, nonzero):
            q = Q_CERT
            if b & m == fmt.infinity_bits:
                q |= Q_OVF
            if b & m == 0 and nonzero:
                q |= Q_RZ
            return q
        todo = [s for s in scales if (fmt.name, s) not in self._c]
        if todo and self.exact is not None:
            for s in todo:
                if self.infinite:
                    self._c[fmt.name, s] = (rn(self.exact, fmt), Q_CERT)
                else:
                    b = round_fraction(self.exact * s, fmt)
                    self._c[fmt.name, s] = (b, flag(b, self.exact != 0))
        elif todo:
            for prec in LADDER:
                old = ctx.prec
                ctx.prec = prec
                try:
                    ball = self.fn()
                    if not ball.is_finite():
                        continue
                    lo, hi = arb_bounds(ball)
                finally:
                    ctx.prec = old
                self.lo, self.hi, self.prec = (lo, hi, prec)
                for s in list(todo):
                    a, b = (round_fraction(lo * s, fmt), round_fraction(hi * s, fmt))
                    if _same(a, b, fmt):
                        self._c[fmt.name, s] = (a, flag(a, self.status not in (EXZ, EXGZ, CONVZ)))
                        todo.remove(s)
                if not todo:
                    break
            for s in todo:
                self._c[fmt.name, s] = (None, 0)
        return [self._c[fmt.name, s] for s in scales]

    def rounded(self, fmt, scales=(Fr(1),)):
        return [b for b, _ in self.certify(fmt, scales)]

def _A(v):
    return fraction_to_arb(v)

def element(xb, tb, fmt, log_target):
    x = decode(xb, fmt)
    t = decode(tb, fmt)
    if not log_target:
        if t == NINF or t < 0:
            return (Val(DOMAIN), Val(DOMAIN), Val(DOMAIN))
        if t == 0:
            if x == NINF:
                return (Val(CONVZ, Fr(0)), Val(LIMIT, Fr(0)), Val(UNDEF))
            return (Val(EXZ, Fr(0)), Val(EXGZ, Fr(0)), Val(ONESIDED, NINF))
        if x == NINF:
            return (Val(MINF, PINF), Val(LIMIT, -t), Val(LIMIT, PINF))
        gx = Val(CERT, -t)
        if t == 1 and x == 0:
            L = Val(EXZ, Fr(0))
        else:
            L = Val(CERT, fn=lambda: _A(t) * (_A(t).log() - _A(x)))
        if t == 1 and x == 1:
            gt = Val(EXGZ, Fr(0))
        elif t == 1:
            gt = Val(CERT, 1 - x)
        else:
            gt = Val(CERT, fn=lambda: _A(t).log() + 1 - _A(x))
        return (L, gx, gt)
    if t == NINF:
        if x == NINF:
            return (Val(CONVZ, Fr(0)), Val(LIMIT, Fr(0)), Val(UNDEF))
        return (Val(CONVZ, Fr(0)), Val(LIMIT, Fr(0)), Val(LIMIT, Fr(0)))
    if x == NINF:
        return (Val(MINF, PINF), Val(LIMIT, fn=lambda: -_A(t).exp()), Val(LIMIT, PINF))
    one = (lambda: arb(1)) if t == 0 else None
    ex = lambda: _A(t).exp()
    L = Val(EXZ, Fr(0)) if t == x else Val(CERT, t - x) if t == 0 else Val(CERT, fn=lambda: ex() * _A(t - x))
    gx = Val(CERT, Fr(-1)) if t == 0 else Val(CERT, fn=lambda: -ex())
    if x == t + 1:
        gt = Val(EXGZ, Fr(0))
    elif t == 0:
        gt = Val(CERT, t - x + 1)
    else:
        gt = Val(CERT, fn=lambda: ex() * _A(t - x + 1))
    return (L, gx, gt)

def reduce_sum(Ls):
    if any((v.status == DOMAIN for v in Ls)):
        return Val(DOMAIN)
    if any((v.exact == PINF for v in Ls)):
        return Val(MINF, PINF)
    if all((v.status in (EXZ, CONVZ) for v in Ls)):
        return Val(EXZ, Fr(0))
    rat = sum((v.exact for v in Ls if v.exact is not None), Fr(0))
    fns = [v.fn for v in Ls if v.exact is None]
    if not fns:
        return Val(CERT, rat)

    def total():
        s = _A(rat) if rat else arb(0)
        for f in fns:
            s = s + f()
        return s
    return Val(CERT, fn=total)

def row_vals(xrow, trow, fmt, log_target):
    return [element(xb, tb, fmt, log_target) for xb, tb in zip(xrow, trow)]

def batch_scales(B, K):
    return {'sum': Fr(1), 'mean': Fr(1, B * K), 'batchmean': Fr(1, B)}

def certify_batch(xbits, tbits, fmt, log_target, vals=None, zp=None, zq=None):
    fmt = get_format(fmt)
    B = len(xbits)
    K = len(xbits[0])
    scales = batch_scales(B, K)
    sc = list(scales.values())
    if vals is None:
        vals = [row_vals(xr, tr, fmt, log_target) for xr, tr in zip(xbits, tbits)]
    out = {'L_st': [], 'L_bits': [], 'L_q': [], 'gx_st': [], 'gt_st': [], 'gx_bits': {k: [] for k in scales}, 'gt_bits': {k: [] for k in scales}, 'gx_q': {k: [] for k in scales}, 'gt_q': {k: [] for k in scales}}
    Ls = []
    for row in vals:
        for L, gx, gt in row:
            Ls.append(L)
            lb, lq = L.certify(fmt)[0]
            out['L_st'].append(L.status)
            out['L_bits'].append(lb)
            out['L_q'].append(lq)
            for nm, g in (('gx', gx), ('gt', gt)):
                out[nm + '_st'].append(g.status)
                for k, (b, q) in zip(scales, g.certify(fmt, sc)):
                    out[nm + '_bits'][k].append(b)
                    out[nm + '_q'][k].append(q)
    S = reduce_sum(Ls)
    r = S.certify(fmt, sc)
    out['S_st'] = S.status
    out['S_bits'] = {k: b for k, (b, _) in zip(scales, r)}
    out['S_q'] = {k: q for k, (_, q) in zip(scales, r)}
    out['S_val'] = S
    if zp is not None:
        E = e2e_batch(zp, zq)
        r = E.certify(fmt, sc)
        out['E2E_st'] = E.status
        out['E2E_bits'] = {k: b for k, (b, _) in zip(scales, r)}
        out['E2E_q'] = {k: q for k, (_, q) in zip(scales, r)}
        out['E2E_val'] = E
    return out

def _lsm(z):
    zs = [_A(v) for v in z]
    m = max(z)
    am = _A(m)
    lse = am + sum(((v - am).exp() for v in zs), arb(0)).log()
    return [v - lse for v in zs]

def e2e_row(zp, zq):
    if list(zp) == list(zq):
        return Val(EXZ, Fr(0))

    def f():
        lp = _lsm(zp)
        lq = _lsm(zq)
        return sum((a.exp() * (a - b) for a, b in zip(lp, lq)), arb(0))
    return Val(CERT, fn=f)

def e2e_row_alt(zp, zq):

    def f():
        lp = _lsm(zp)
        a = [_A(v) for v in zp]
        b = [_A(v) for v in zq]
        lsep = a[0] - lp[0]
        lseq = b[0] - _lsm(zq)[0]
        return sum((p.exp() * (u - v) for p, u, v in zip(lp, a, b)), arb(0)) - lsep + lseq
    return Val(CERT, fn=f)

def e2e_batch(zps, zqs):
    rows = [e2e_row(zp, zq) for zp, zq in zip(zps, zqs)]
    fns = [r.fn for r in rows if r.exact is None]
    if not fns:
        return Val(EXZ, Fr(0))
    return Val(CERT, fn=lambda: sum((f() for f in fns), arb(0)))

def interval(val):
    if val.exact is not None:
        return (val.exact, val.exact)
    for prec in LADDER:
        old = ctx.prec
        ctx.prec = prec
        try:
            ball = val.fn()
            ok = ball.is_finite()
            if ok:
                lo, hi = arb_bounds(ball)
        finally:
            ctx.prec = old
        if ok:
            if lo > 0 and hi - lo <= lo / 10 ** 9 or (hi < 0 and hi - lo <= -hi / 10 ** 9):
                return (lo, hi)
    return (lo, hi)

def rounded_vector(fn_list, fmt):
    fmt = get_format(fmt)
    out = []
    for i in range(len(fn_list)):
        v = Val(CERT, fn=fn_list[i])
        b = v.rounded(fmt)[0]
        if b is None:
            raise ArithmeticError('input rounding not certified at 4096 bits')
        out.append(b)
    return out

def rn_log_softmax(z, fmt):
    return rounded_vector([lambda i=i: _lsm(z)[i] for i in range(len(z))], fmt)

def rn_softmax(z, fmt):
    return rounded_vector([lambda i=i: _lsm(z)[i].exp() for i in range(len(z))], fmt)

def rn_log(tbits, fmt):
    f = get_format(fmt)
    out = []
    for b in tbits:
        t = decode(b, f)
        out.append(round_fraction(Fr(0), f) if t == 1 else Val(CERT, fn=lambda t=t: _A(t).log()).rounded(f)[0])
    return out
