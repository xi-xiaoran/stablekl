"""Repaired KL divergence between two Gumbel distributions: formula, analytic gradients, and the
entry point run_rewrite_torch used by the benchmark (dtype in, compute dtype, outputs per component).
The derivation is in the paper's appendix on interface A."""
from __future__ import annotations
import math
from decimal import Decimal
from fractions import Fraction as Fr
import numpy as np
FMT = {'fp16': dict(np=np.float16, u=2.0 ** (-11), eta=2.0 ** (-24), minnorm=2.0 ** (-14), emax=15), 'fp32': dict(np=np.float32, u=2.0 ** (-24), eta=2.0 ** (-149), minnorm=2.0 ** (-126), emax=127), 'fp64': dict(np=np.float64, u=2.0 ** (-53), eta=2.0 ** (-1074), minnorm=2.0 ** (-1022), emax=1023)}
SAFETY = 1.02
KTERMS = {'fp16': 9, 'fp32': 17, 'fp64': 36}
KS2 = {'fp16': 18, 'fp32': 31, 'fp64': 60}
KST = {'fp16': 2, 'fp32': 4, 'fp64': 8}
SEG_LO, SEG_HI = (-0.75, 3.0)
R_BIG = 9.5
OFFICIAL_FN = '_kl_gumbel_gumbel'
GRAD_TRUTH = 'frozen_interval'
ZETA_STR = ('1.644934066848226436472415166646025189218949', '1.202056903159594285399738161511449990764986', '1.082323233711138191516003696541167902774750', '1.036927755143369926331365486457034168057080', '1.017343061984449139714517929790920527901817', '1.008349277381922826839797549849796759599863', '1.004077356197944339378685238508652465258960', '1.002008392826082214417852769232412060485605', '1.000994575127818085337145958900319017006019', '1.000494188604119464558702282526469936468606', '1.000246086553308048298637998047739670960416', '1.000122713347578489146751836526357395714275', '1.000061248135058704829258545105135333747481', '1.000030588236307020493551728510645062587627', '1.000015282259408651871732571487636722023237', '1.000007637197637899762273600293563029213088', '1.000003817293264999839856461644621939730454', '1.000001908212716553938925656957795101353258', '1.000000953962033872796113152038683449345943', '1.000000476932986787806463116719604373045966', '1.000000238450502727732990003648186752994935', '1.000000119219925965311073067788718882326387', '1.000000059608189051259479612440207935801227', '1.000000029803503514652280186063705069366011', '1.000000014901554828365041234658506630698628', '1.000000007450711789835429491981004170604119', '1.000000003725334024788457054819204018402423', '1.000000001862659723513049006403909945416948', '1.000000000931327432419668182871764735021219', '1.000000000465662906503378407298923325122007', '1.000000000232831183367650549200145597594049', '1.000000000116415501727005197759297383545630', '1.000000000058207720879027008892436859891063', '1.000000000029103850444970996869294252278840', '1.000000000014551921891041984235929632245318', '1.000000000007275959835057481014520869012338', '1.000000000003637979547378651190237236355873', '1.000000000001818989650307065947584832100730', '1.000000000000909494784026388928253311838694', '1.000000000000454747378304215402679911202948', '1.000000000000227373684582465251522682157797', '1.000000000000113686840768022784934910483802', '1.000000000000056843419876275856092771829675', '1.000000000000028421709768893018554550737049', '1.000000000000014210854828031606769834307141', '1.000000000000007105427395210852712877354479', '1.000000000000003552713691337113673298469534', '1.000000000000001776356843579120327473349014', '1.000000000000000888178421093081590309609138', '1.000000000000000444089210314381336419777094', '1.000000000000000222044605079804198399932009', '1.000000000000000111022302514106613372054456', '1.000000000000000055511151248454812437237365', '1.000000000000000027755575621361241725816324', '1.000000000000000013877787809725232762839094', '1.000000000000000006938893904544153697446085', '1.000000000000000003469446952165922624744271', '1.000000000000000001734723476047576572048972', '1.000000000000000000867361738011993372834205', '1.000000000000000000433680869002065048749702', '1.000000000000000000216840434499721978501391')
EULER_STR = '0.577215664901532860606512090082402431042159'
HLN2PI_STR = '0.918938533204672741780329736405617639861397'
_FR = lambda s: Fr(Decimal(s))
ZETA_FR = tuple((_FR(z) for z in ZETA_STR))
EULER_FR, HLN2PI_FR = (_FR(EULER_STR), _FR(HLN2PI_STR))
_BERN = (Fr(1, 6), Fr(-1, 30), Fr(1, 42), Fr(-1, 30), Fr(5, 66), Fr(-691, 2730), Fr(7, 6), Fr(-3617, 510), Fr(43867, 798))
ST_LG_FR = tuple((_BERN[k - 1] / (2 * k * (2 * k - 1)) for k in range(1, 10)))
ST_PS_FR = tuple((_BERN[k - 1] / (2 * k) for k in range(1, 10)))

def P_FR(K):
    return tuple((Fr((-1) ** k) * ZETA_FR[k - 2] / k for k in range(2, K + 1)))

def Q_FR(K):
    return tuple((Fr((-1) ** k) * ZETA_FR[k - 2] for k in range(2, K + 1)))

def q_tail_bound(z, K):
    return z ** K / ((2 * K + 3) * (1 - z))

def s2_tails(x, K):
    a = abs(x)
    zk1 = float(ZETA_FR[K - 1])
    return (zk1 * a ** (K + 1) / ((K + 1) * (1 - a)), zk1 * a ** K / (1 - a))
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    _DT = {torch.float16: 'fp16', torch.float32: 'fp32', torch.float64: 'fp64'}

    def _phiS(xs, s, K):
        z = s * s
        q = torch.full_like(s, 1.0 / (2 * K + 1))
        for k in range(K - 1, 0, -1):
            q = q * z + torch.full_like(s, 1.0 / (2 * k + 1))
        return xs * s - torch.full_like(s, 2.0) * (s * z) * q

    def _horner(x, coefs):
        p = torch.full_like(x, float(coefs[-1]))
        for c in coefs[-2::-1]:
            p = p * x + float(c)
        return p

    def _mz(a, v, TL):
        zero = torch.zeros_like(a)
        m1 = a * torch.exp(v)
        h2 = torch.exp(v * 0.5)
        m2 = a * h2 * h2
        h4 = torch.exp(v * 0.25)
        m4 = a * h4 * h4 * h4 * h4
        return torch.where(a == 0, zero, torch.where(v <= TL, m1, torch.where(v <= 2 * TL, m2, m4)))

    def _xfc(x, f, c):
        return torch.where(f.abs() <= 1, x * f / c, torch.where(c >= 1, x / c * f, torch.where(x.abs() >= 1, x * (f / c), x * f / c)))

    class GumbelKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, m, b, n, c):
            dt = _DT[m.dtype]
            K = KTERMS[dt]
            K2 = KS2[dt]
            KS = KST[dt]
            one = torch.ones_like(m)
            zero = torch.zeros_like(m)
            half = torch.full_like(m, 0.5)
            g = float(EULER_FR)
            fmax = float(np.finfo(FMT[dt]['np']).max)
            dl = m - n
            t = torch.where(torch.isinf(dl), m / c - n / c, dl / c)
            rho = (b - c) / c
            r = b / c
            mid = rho.abs() <= 0.5
            lo = rho < -0.5
            hi = rho > 0.5
            big = hi & (r >= R_BIG)
            hmod = hi & ~big
            nsh = torch.floor(torch.where(hmod, r, one) - half)
            rp = torch.where(hmod, r, one) - nsh
            rhop = rp - one
            x = torch.where(mid, rho, torch.where(lo, r, torch.where(hmod, rhop, zero)))
            S2 = x * x * _horner(x, P_FR(K2))
            S2p = x * _horner(x, Q_FR(K2))
            y0 = torch.where(mid, rho, rhop)
            Gm = torch.log1p(y0) - y0 * g + S2
            Pm = one / (one + y0) - g + S2p
            sl = zero
            si = zero
            for j in range(1, 9):
                on = nsh >= j
                sl = sl + torch.where(on, torch.log(rp + j), zero)
                si = si + torch.where(on, one / (rp + j), zero)
            y = one + torch.where(big, torch.clamp_max(r, fmax), torch.full_like(m, 10.0))
            ly = torch.log(y)
            iy = one / y
            iy2 = iy * iy
            Gb = (y - half) * (ly - one) - half + float(HLN2PI_FR) + iy * _horner(iy2, ST_LG_FR[:KS])
            Pb = ly - iy * half - iy2 * _horner(iy2, ST_PS_FR[:KS])
            G = torch.where(mid, Gm, torch.where(lo, S2 - r * g, torch.where(hmod, Gm + sl, Gb)))
            psi1 = torch.where(mid, Pm, torch.where(lo, S2p - g, torch.where(hmod, Pm + si, Pb)))
            D = torch.where(mid, S2p, torch.where(lo, S2p - one / r, psi1 + g - one / r))
            lr = torch.where(r < FMT[dt]['minnorm'], torch.log(b) - torch.log(c), torch.log(r))
            H = torch.where(mid, S2, torch.where(lo, S2 + (zero - lr - g), G + rho * g - torch.log(torch.clamp_max(r, fmax))))
            u = G - t
            xx = torch.expm1(u)
            inseg = torch.isfinite(xx) & (xx >= SEG_LO) & (xx <= SEG_HI)
            xs = torch.where(inseg, xx, zero)
            phi = torch.where(inseg, _phiS(xs, xs / (torch.full_like(m, 2.0) + xs), K), torch.where(torch.isinf(xx), xx, xx - u))
            kl = torch.clamp_min(phi, 0) + torch.clamp_min(H, 0)
            kl = torch.where(torch.isinf(H), H, kl)
            ex = torch.exp(u)
            v = u - torch.log(c)
            ec = torch.exp(v)
            xinf = torch.isinf(xx) & (xx > 0)
            g_m = torch.where(xinf, zero - ec, (zero - xx) / c)
            g_n = torch.where(xinf, ec, xx / c)
            xp = xx * psi1
            ep = ex * psi1
            gir = g - one / r
            wA = xp.abs() + D.abs()
            wB = ep.abs() + (g + one / r)
            useB = (wB < wA) & (u < 0) & ~lo
            cb = torch.where(useB, ep + gir, xp + D)
            rD = torch.where(lo, r * S2p - one, r * D)
            Dc = torch.where(lo, rD / b, D / c)
            TL = _tl(dt)
            fA = t - r * psi1
            q = xx * fA
            lg = lambda z: torch.log(z.abs())
            lc = torch.log(c)
            dm = lambda S_, dlog, comb: torch.where(torch.isinf(S_) & (dlog > 6.0), S_, comb)
            Sbx = _mz(psi1, v, TL)
            Sbf = _xfc(xx, psi1, c)
            Scx = _mz(fA, v, TL)
            Scf = _xfc(xx, fA, c)
            dbx = lg(psi1) + v - torch.where(lo, lg(rD) - torch.log(b), lg(D) - lc)
            dcx = lg(fA) + v - (lg(rD) - lc)
            dbf = lg(xx) + lg(psi1) - lg(D)
            dcf = lg(xx) + lg(fA) - lg(rD)
            g_b = torch.where(useB, cb / c, torch.where(xinf, dm(Sbx, dbx, Sbx + Dc), torch.where(lo, (r * (xp + S2p) - one) / b, torch.where(torch.isfinite(xp), cb / c, dm(Sbf, dbf, Sbf + D / c)))))
            g_c = torch.where(useB, (t * xx - r * cb) / c, torch.where(xinf, dm(Scx, dcx, Scx - rD / c), torch.where(torch.isfinite(q), (q - rD) / c, dm(Scf, dcf, Scf - rD / c))))
            same = (m == n) & (b == c)
            kl = torch.where(same, zero, kl)
            g_m = torch.where(same, zero, g_m)
            g_n = torch.where(same, zero, g_n)
            g_b = torch.where(same, zero, g_b)
            g_c = torch.where(same, zero, g_c)
            ctx.save_for_backward(g_m, g_b, g_n, g_c)
            return kl

        @staticmethod
        def backward(ctx, gr):
            a, b2, c2, d = ctx.saved_tensors
            return (gr * a, gr * b2, gr * c2, gr * d)
NAME = 'gumbel'
PARAMS = ('loc', 'scale')
COMPS = ['KL', 'P.loc', 'P.scale', 'Q.loc', 'Q.scale']
EXPECTED_ROWS = {'fp16': 137249, 'fp32': 202000}
TRANSCENDENTAL_COMPS = ('KL', 'P.loc', 'P.scale', 'Q.loc', 'Q.scale')
TRANSCENDENTAL_FUNCS = ('log1p', 'log', 'expm1', 'exp')

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    leaves = [to_t(x, dt_in).to(TD[cdt]).requires_grad_(True) for x in (P['loc'], P['scale'], Q['loc'], Q['scale'])]
    kl = GumbelKLC.apply(*leaves)
    kl.sum().backward()
    vals = {'KL': kl.detach()}
    for name, leaf in zip(('P.loc', 'P.scale', 'Q.loc', 'Q.scale'), leaves):
        vals[name] = leaf.grad.detach()
    return {k: v.cpu().numpy().astype(NPF[cdt]) for k, v in vals.items()}

def exact_grads(pl, ps, ql, qs):
    return None

def identity(pl, ps, ql, qs):
    return pl == ql and ps == qs

def calibrate_exp_pos(seed=11, dts=('fp16', 'fp32', 'fp64')):
    from decimal import localcontext
    TS = {'fp16': 11.089866488461016, 'fp32': 88.72283905206835, 'fp64': 709.782712893384}
    res = {}
    with localcontext() as ctx:
        ctx.prec = 40
        for k_, dt in enumerate(('fp16', 'fp32', 'fp64')):
            if dt not in dts:
                continue
            rng = np.random.default_rng(seed + k_)
            T = FMT[dt]['np']
            T1 = TS[dt]
            UI = {'fp16': np.uint16, 'fp32': np.uint32, 'fp64': np.uint64}[dt]
            if dt == 'fp16':
                xs = np.arange(1, 31744, dtype=np.uint16).view(np.float16)
                xs = xs[xs.astype(float) <= T1]
            else:
                lo = -30 if dt == 'fp32' else -60
                xs = np.unique(np.concatenate([np.exp2(rng.uniform(lo, math.log2(T1), 20000)), rng.uniform(0, T1, 10000)]).astype(T))
                xs = xs[(xs > 0) & (xs.astype(float) <= T1)]
            res[dt] = 0
            for fn in ('exp', 'expm1'):
                if torch is not None:
                    yv = np.asarray(getattr(torch, fn)(torch.from_numpy(xs)).numpy()).astype(T)
                    src = 'torch'
                else:
                    with np.errstate(all='ignore'):
                        yv = getattr(np, fn)(xs).astype(T)
                    src = 'numpy'
                dmax, wx = (0, None)
                for xv, yy in zip(xs.tolist(), yv):
                    ex = Decimal(xv).exp() - (1 if fn == 'expm1' else 0)
                    cc = T(float(ex))
                    cands = [np.nextafter(cc, T(-np.inf)), cc, np.nextafter(cc, T(np.inf))]
                    cr = min(cands, key=lambda q_: (abs(Decimal(float(q_)) - ex), int(np.array(q_).view(UI)) & 1))
                    d = abs(int(np.array(yy).view(UI)) - int(np.array(cr).view(UI)))
                    if d > dmax:
                        dmax, wx = (d, xv)
                res[dt] = max(res[dt], dmax)
                print(f'calibrate_exp_pos {dt} {fn} ({src}) n={len(xs)} max ULP vs correctly rounded d={dmax} (lambda would be 2d+1={2 * dmax + 1}) at x={wx!r}', flush=True)
    return res
_LAM_POS = {}

def lam_pos(dt):
    if dt not in _LAM_POS:
        _LAM_POS[dt] = 2 * calibrate_exp_pos(dts=(dt,))[dt] + 1
        print(f'fix_gumbel.running_bounds: {dt} lambda_pos = {_LAM_POS[dt]}; the bound uses max(lambda from p2_accept, lambda_pos)', flush=True)
    return _LAM_POS[dt]

def _tl(dt):
    T = FMT[dt]['np']
    x = T(math.log(float(np.finfo(T).max)))
    while float(x) > math.log(float(np.finfo(T).max)):
        x = np.nextafter(x, T(0))
    return float(x)

class _Tr:

    def __init__(self, dt, lam):
        f = FMT[dt]
        self.dt = dt
        self.T = f['np']
        self.u = f['u']
        self.mn = f['minnorm']
        self.h = f['eta'] / 2 if f['eta'] / 2 > 0 else f['eta']
        self.lam = lam
        self.OT = float(np.finfo(self.T).max) * (1 + self.u)

    def A(self, x):
        return min(abs(float(x)), self.OT)

    def sb(self, x):
        return 1.0 if abs(float(x)) < self.mn else 0.0

    def R(self, z):
        return self.u * self.A(z) + self.h * self.sb(z)

    def inp(self, x):
        return (self.T(x), 0.0)

    def const(self, c):
        v = self.T(float(c))
        return (v, float(abs(Fr(float(v)) - c)))

    def add(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] + b[0])
        return (z, a[1] + b[1] + (self.u * self.A(z) if abs(float(z)) >= self.mn else 0.0))

    def sub(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] - b[0])
        return (z, a[1] + b[1] + (self.u * self.A(z) if abs(float(z)) >= self.mn else 0.0))

    def subx(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] - b[0])
        return (z, a[1] + b[1])

    def mul(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] * b[0])
        return (z, self.A(b[0]) * a[1] + self.A(a[0]) * b[1] + self.R(z))

    def scale2(self, a, c):
        with np.errstate(all='ignore'):
            z = self.T(a[0] * self.T(c))
        return (z, abs(c) * a[1] + self.h * self.sb(z) * (c < 1))

    def div(self, a, b):
        with np.errstate(all='ignore'):
            z = self.T(a[0] / b[0])
        bb = abs(float(b[0]))
        if bb == math.inf:
            return (z, a[1] / bb if a[1] < math.inf else math.inf)
        return (z, (a[1] + self.A(z) * b[1]) / bb + self.R(z) if bb > 0 else float('inf'))

    def fn(self, name, a):
        with np.errstate(all='ignore'):
            z = self.T(getattr(np, name)(np.array([a[0]], dtype=self.T))[0])
        xv = float(a[0])
        e = float(a[1])
        if name in ('exp', 'expm1'):
            lx = min(xv, math.log(self.OT))
            if e == 0:
                prop = 0.0
            elif e < 1:
                prop = math.exp(lx) * math.expm1(e)
            else:
                prop = math.exp(lx + e) if lx + e < 709.7 else math.inf
        else:
            y = abs(xv if name == 'log' else 1.0 + xv)
            prop = 0.0 if e == 0 else -math.log1p(-e / y) if e < y else math.inf
        return (z, prop + self.lam * self.R(z))

    def clamp0(self, a):
        return (max(a[0], self.T(0)), a[1])

    def extra(self, a, e):
        return (a[0], a[1] + e)

def _tr_phiS(tr, x, s, dt):
    K = KTERMS[dt]
    z = tr.mul(s, s)
    q = tr.const(Fr(1, 2 * K + 1))
    for k in range(K - 1, 0, -1):
        q = tr.add(tr.mul(q, z), tr.const(Fr(1, 2 * k + 1)))
    q = tr.extra(q, q_tail_bound(float(z[0]), K))
    return tr.sub(tr.mul(x, s), tr.mul(tr.scale2(tr.mul(s, z), 2.0), q))

def _tr_horner(tr, x, coefs):
    p = tr.const(coefs[-1])
    for c in coefs[-2::-1]:
        p = tr.add(tr.mul(p, x), tr.const(c))
    return p

def _trace(m, b, n, c, dt, lam):
    tr = _Tr(dt, lam)
    T = tr.T
    K2 = KS2[dt]
    KS = KST[dt]
    mv, bv, nv, cv = (float(m), float(b), float(n), float(c))
    m, b, n, c = (tr.inp(v) for v in (m, b, n, c))
    one = tr.inp(1.0)
    zero = tr.inp(0.0)
    half = tr.inp(0.5)
    if mv == nv and bv == cv:
        return ({k: zero for k in COMPS}, {'_branch': 'id'})
    g = tr.const(EULER_FR)
    dl = tr.sub(m, n)
    t = tr.sub(tr.div(m, c), tr.div(n, c)) if math.isinf(float(dl[0])) else tr.div(dl, c)
    rho = tr.div(tr.sub(b, c), c)
    r = tr.div(b, c)
    rv, rhov = (float(r[0]), float(rho[0]))
    reg = 'mid' if abs(rhov) <= 0.5 else 'lo' if rhov < -0.5 else 'big' if rv >= R_BIG else 'hmod'

    def S2pair(x):
        xv = float(x[0])
        t2, t2p = s2_tails(xv, K2)
        S2 = tr.extra(tr.mul(tr.mul(x, x), _tr_horner(tr, x, P_FR(K2))), t2)
        S2p = tr.extra(tr.mul(x, _tr_horner(tr, x, Q_FR(K2))), t2p)
        return (S2, S2p)

    def mid_GP(y0, S2, S2p):
        Gm = tr.add(tr.sub(tr.fn('log1p', y0), tr.mul(y0, g)), S2)
        Pm = tr.add(tr.sub(tr.div(one, tr.add(one, y0)), g), S2p)
        return (Gm, Pm)
    if reg == 'mid':
        S2, S2p = S2pair(rho)
        G, psi1 = mid_GP(rho, S2, S2p)
        D = S2p
        H = S2
    elif reg == 'lo':
        S2, S2p = S2pair(r)
        G = tr.sub(S2, tr.mul(r, g))
        psi1 = tr.sub(S2p, g)
        D = None
        lr = tr.sub(tr.fn('log', b), tr.fn('log', c)) if rv < FMT[dt]['minnorm'] else tr.fn('log', r)
        H = tr.add(S2, tr.sub(tr.sub(zero, lr), g))
    else:
        if reg == 'hmod':
            nsh = T(np.floor(T(r[0] - T(0.5))))
            rp = tr.subx(r, (nsh, 0.0))
            rhop = tr.subx(rp, one)
            S2, S2p = S2pair(rhop)
            Gm, Pm = mid_GP(rhop, S2, S2p)
            sl = zero
            si = zero
            for j in range(1, 9):
                if nsh >= j:
                    rj = tr.add(rp, tr.inp(float(j)))
                    sl = tr.add(sl, tr.fn('log', rj))
                    si = tr.add(si, tr.div(one, rj))
                else:
                    sl = tr.add(sl, zero)
                    si = tr.add(si, zero)
            G = tr.add(Gm, sl)
            psi1 = tr.add(Pm, si)
        else:
            rG = r if math.isfinite(float(r[0])) else (T(np.finfo(T).max), 0.0)
            y = tr.add(one, rG)
            ly = tr.fn('log', y)
            iy = tr.div(one, y)
            iy2 = tr.mul(iy, iy)
            yv = float(y[0])
            G = tr.add(tr.add(tr.sub(tr.mul(tr.sub(y, half), tr.sub(ly, one)), half), tr.const(HLN2PI_FR)), tr.mul(iy, _tr_horner(tr, iy2, ST_LG_FR[:KS])))
            G = tr.extra(G, abs(float(ST_LG_FR[KS])) * yv ** (-(2 * KS + 1)))
            psi1 = tr.sub(tr.sub(ly, tr.mul(iy, half)), tr.mul(iy2, _tr_horner(tr, iy2, ST_PS_FR[:KS])))
            psi1 = tr.extra(psi1, abs(float(ST_PS_FR[KS])) * yv ** (-(2 * KS + 2)))
        D = tr.sub(tr.add(psi1, g), tr.div(one, r))
        H = tr.sub(tr.add(G, tr.mul(rho, g)), tr.fn('log', r if math.isfinite(rv) else (T(np.finfo(T).max), 0.0)))
    u = tr.sub(G, t)
    xx = tr.fn('expm1', u)
    xv = float(xx[0])
    if math.isfinite(xv) and SEG_LO <= xv <= SEG_HI:
        phi = _tr_phiS(tr, xx, tr.div(xx, tr.add(tr.inp(2.0), xx)), dt)
        bp = 'S'
    elif math.isinf(xv):
        phi = xx
        bp = 'I'
    else:
        phi = tr.sub(xx, u)
        bp = 'X'
    KL = tr.add(tr.clamp0(phi), tr.clamp0(H))
    if math.isinf(float(H[0])):
        KL = H
    ex = tr.fn('exp', u)
    vv = tr.sub(u, tr.fn('log', c))
    ec = tr.fn('exp', vv)
    xinf = math.isinf(xv) and xv > 0
    drop = math.exp(-min(float(u[0]), 700.0)) if xinf else 0.0
    if xinf:
        ec = tr.extra(ec, tr.A(ec[0]) * drop)
    gm = tr.sub(zero, ec) if xinf else tr.div(tr.sub(zero, xx), c)
    gn = ec if xinf else tr.div(xx, c)
    xp = tr.mul(xx, psi1)
    ep = tr.mul(ex, psi1)
    lo = reg == 'lo'
    if not lo:
        ir = tr.div(one, r)
        gir = tr.sub(g, ir)
        wA = T(abs(xp[0]) + abs(D[0]))
        wB = T(abs(ep[0]) + T(g[0] + ir[0]))
    rD = tr.sub(tr.mul(r, S2p), one) if lo else tr.mul(r, D)
    Dc = tr.div(rD, b) if lo else tr.div(D, c)
    TL = _tl(dt)

    def mz(a_):
        if float(a_[0]) == 0:
            ev = math.exp(min(float(vv[0]), 709.0)) if float(vv[0]) < 709 else math.inf
            return (zero[0], 0.0 if a_[1] == 0 else a_[1] * ev)
        w = float(vv[0])
        if w <= TL:
            o = tr.mul(a_, tr.fn('exp', vv))
        elif w <= 2 * TL:
            h = tr.fn('exp', tr.mul(vv, tr.inp(0.5)))
            o = tr.mul(tr.mul(a_, h), h)
        else:
            h = tr.fn('exp', tr.mul(vv, tr.inp(0.25)))
            o = tr.mul(tr.mul(tr.mul(tr.mul(a_, h), h), h), h)
        return tr.extra(o, tr.A(o[0]) * drop)

    def xfc(x_, f_):
        if abs(float(f_[0])) <= 1:
            return tr.div(tr.mul(x_, f_), c)
        if float(c[0]) >= 1:
            return tr.mul(tr.div(x_, c), f_)
        if abs(float(x_[0])) >= 1:
            return tr.mul(x_, tr.div(f_, c))
        return tr.div(tr.mul(x_, f_), c)
    if not lo and wB < wA and (float(u[0]) < 0):
        cb = tr.add(ep, gir)
        fb = 'B'
        gc = tr.div(tr.sub(tr.mul(t, xx), tr.mul(r, cb)), c)
        gb = tr.div(cb, c)
    else:
        fb = 'A'
        fA = tr.sub(t, tr.mul(r, psi1))

        def lg(z):
            with np.errstate(all='ignore'):
                return T(np.log(np.array([abs(T(z[0]))], dtype=T))[0])
        lcv = lg(c)
        with np.errstate(all='ignore'):
            dm = lambda S_, dl, comb: S_ if math.isinf(float(S_[0])) and dl > T(6.0) else comb
            if xinf:
                Sb = mz(psi1)
                db = T(T(lg(psi1) + vv[0]) - (T(lg(rD) - lg(b)) if lo else T(lg(D) - lcv)))
                gb = dm(Sb, db, tr.add(Sb, Dc))
                Sc = mz(fA)
                dc = T(T(lg(fA) + vv[0]) - T(lg(rD) - lcv))
                gc = dm(Sc, dc, tr.sub(Sc, tr.div(rD, c)))
            else:
                if lo:
                    gb = tr.div(tr.sub(tr.mul(r, tr.add(xp, S2p)), one), b)
                elif math.isfinite(float(xp[0])):
                    gb = tr.div(tr.add(xp, D), c)
                else:
                    Sb = xfc(xx, psi1)
                    gb = dm(Sb, T(T(lg(xx) + lg(psi1)) - lg(D)), tr.add(Sb, tr.div(D, c)))
                    fb = 'a'
                q = tr.mul(xx, fA)
                if math.isfinite(float(q[0])):
                    gc = tr.div(tr.sub(q, rD), c)
                else:
                    Sc = xfc(xx, fA)
                    gc = dm(Sc, T(T(lg(xx) + lg(fA)) - lg(rD)), tr.sub(Sc, tr.div(rD, c)))
                    fb = 'a'
    return ({'KL': KL, 'P.loc': gm, 'P.scale': gb, 'Q.loc': gn, 'Q.scale': gc}, {'_branch': f'{reg}/{bp}{fb}'})

def replay(pl, ps, ql, qs, dt):
    out, flags = _trace(pl, ps, ql, qs, dt, 1)
    res = {k: float(v[0]) for k, v in out.items()}
    res.update(flags)
    res['_in'] = (float(pl), float(ps), float(ql), float(qs))
    return res

def running_bounds(v, dt, lam):
    out, _ = _trace(*v['_in'], dt, max(lam, lam_pos(dt)))
    return {k: math.inf if math.isnan(float(e)) else SAFETY * float(e) for k, (_, e) in out.items()}
