from __future__ import annotations
import inspect, itertools, math, sys, warnings
from fractions import Fraction as Fr
import numpy as np, torch, torch.nn.functional as F, mpmath as mp
from stablekl import interface_b as K
DEV = sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv else 'cpu'
mp.mp.prec = 3000
NPD = {torch.float16: np.float16, torch.float32: np.float32, torch.float64: np.float64}
INF = math.inf
N = {'checks': 0, 'fail': 0}
NOTES = []

def ok(cond, msg):
    N['checks'] += 1
    if not cond:
        N['fail'] += 1
        if N['fail'] <= 40:
            print('FAIL:', msg)

def rn(v, d):
    f = NPD[d]
    if isinstance(v, mp.mpf):
        if mp.isinf(v):
            return float(v)
        if v == 0:
            return 0.0
        mag = mp.mag(v)
        if mag > 2000:
            return math.copysign(INF, v)
        if mag < -2000:
            return math.copysign(0.0, v)
        man, exp = v.man_exp if hasattr(v, 'man_exp') else (v.man, v.exp)
        q = Fr(int(man)) * Fr(2) ** int(exp)
        if v < 0 and q > 0:
            q = -q
    else:
        q = Fr(v)
    if q == 0:
        return 0.0
    with np.errstate(all='ignore'):
        fi = np.finfo(f)
        big = Fr(float(fi.max)) + Fr(float(fi.max) - float(np.nextafter(fi.max, f(0)))) / 2
        if abs(q) >= big:
            return INF if q > 0 else -INF
        c = f(float(q)) if abs(q) < Fr(float(fi.max)) else f(math.copysign(float(fi.max), q))
        best = None
        for cand in (np.nextafter(c, f(-INF)), c, np.nextafter(c, f(INF))):
            if not np.isfinite(cand):
                continue
            dist = abs(Fr(float(cand)) - q)
            even = int(np.asarray(cand).view({2: np.uint16, 4: np.uint32, 8: np.uint64}[np.dtype(f).itemsize])) & 1 == 0
            if best is None or dist < best[0] or (dist == best[0] and even):
                best = (dist, float(cand))
    return best[1]

def M(v):
    return mp.mpf(v)

def truth(x, t, lt, G=None, den=1):
    x, t = (float(x), float(t))
    g = M(1) if G is None else M(float(G))
    g0 = g == 0
    if not lt:
        if t == 0:
            if x == -INF:
                return (M(0), M(0), M(0) if g0 else 'nan')
            return (M(0), M(0), M(0) if g0 else -mp.inf * mp.sign(g))
        if x == -INF:
            return (mp.inf, -g * M(t) / den, M(0) if g0 else mp.inf * mp.sign(g))
        return (M(t) * (mp.log(M(t)) - M(x)) / den, -g * M(t) / den, g * (mp.log(M(t)) + 1 - M(x)) / den)
    if t == -INF:
        if x == -INF:
            return (M(0), M(0), M(0) if g0 else 'nan')
        return (M(0), M(0), M(0))
    e = mp.exp(M(t))
    if x == -INF:
        return (mp.inf, -g * e / den, M(0) if g0 else mp.inf * mp.sign(g))
    return (e * (M(t) - M(x)) / den, -g * e / den, g * e * (M(t) - M(x) + 1) / den)

def T(vals, d):
    return torch.tensor(vals, dtype=d, device=DEV)

def same(a, b):
    return all((math.isnan(p) and math.isnan(q) or p == q for p, q in zip(a, b)))

def flat(v):
    return v.detach().double().cpu().reshape(-1).tolist()

def run(x, t, lt, red, m, G=None, **kw):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y = K.kl_div(x, t, reduction=red, log_target=lt, method=m, **kw)
    gx, gt = torch.autograd.grad(y, [x, t], grad_outputs=torch.ones_like(y) if G is None else G)
    return (y, gx, gt)

def expect(x, t, lt, red, d, G=None):
    xs, ts = (flat(x), flat(t))
    n = len(xs)
    B = x.shape[0] if x.dim() else 1
    den = {'none': 1, 'sum': 1, 'mean': n, 'batchmean': B}[red]
    Gs = flat(G.expand(x.shape)) if G is not None and red == 'none' else [1.0 if G is None else float(G)] * n
    el = [truth(a, b, lt, g, den) for a, b, g in zip(xs, ts, Gs)]
    Ls = [truth(a, b, lt)[0] for a, b in zip(xs, ts)]
    if red == 'none':
        yv = [rn(v, d) for v in Ls]
    else:
        yv = [INF if any((mp.isinf(v) for v in Ls)) else rn(mp.fsum(Ls) / den, d)]
    r = lambda v: math.nan if isinstance(v, str) else rn(v, d)
    return (yv, [r(e[1]) for e in el], [r(e[2]) for e in el])

def test_helpers():
    rng = np.random.default_rng(1)
    h = rng.integers(1, 31743, 20000).astype(np.uint16).view(np.float16).astype(np.float64)
    nx = np.nextafter(h.astype(np.float16), np.float16(INF)).astype(np.float64)
    mid = (h + nx) / 2
    c = np.concatenate([mid, mid * (1 + 2.0 ** (-40)), mid * (1 - 2.0 ** (-40)), mid * (1 + 2.0 ** (-52)), h * 1.37, -mid * (1 - 2.0 ** (-45)), [1e-09, -1e-09, 2.0 ** (-25), 2.0 ** (-25) * (1 + 1e-12), 65519.99, 65520.0, 1e+300, -1e+300, 0.0, INF, -INF]])
    a = K._round(torch.from_numpy(c).to(DEV), torch.float16).cpu().numpy()
    with np.errstate(all='ignore'):
        b = c.astype(np.float16)
    ok((a.view(np.uint16) == b.view(np.uint16)).all(), '_round float64 -> float16 is a single rounding')
    ok(K._round(torch.tensor([math.nan], dtype=torch.float64, device=DEV), torch.float16).isnan().all().item(), '_round keeps NaN')
    for d in (torch.float32, torch.float64):
        v = torch.from_numpy(rng.standard_normal((50, 257)) * 10.0 ** rng.integers(-8, 8, (50, 257))).to(d).to(DEV)
        v = torch.cat([v, -v[:, :200] * (1 + 0.001)], 1)
        s = K._csum(v).double().cpu().tolist()
        e = [math.fsum(r) for r in v.double().cpu().tolist()]
        u = 2.0 ** (-24) if d == torch.float32 else 2.0 ** (-53)
        ok(all((abs(p - q) <= 2 * u * abs(q) + 1e-300 for p, q in zip(s, e))), f'_csum {d}: faithful to the exact sum')
    for d in (torch.float16, torch.float32):
        f = NPD[d]
        n = 4000
        ex = (-20, 12) if d == torch.float16 else (-140, 100)
        fa = [(rng.uniform(0.5, 1, n) * 2.0 ** rng.integers(ex[0], ex[1], n) * rng.choice([-1, 1], n)).astype(f) for _ in range(3)]
        for den in (1, 3, 8192, 10 ** 7):
            out = K._prod_div([torch.from_numpy(v).to(DEV) for v in fa], den, d).double().cpu().tolist()
            bad = 0
            for i in range(n):
                q = Fr(float(fa[0][i])) * Fr(float(fa[1][i])) * Fr(float(fa[2][i])) / den
                e = rn(q, d)
                if out[i] != e:
                    lo, hi = (float(np.nextafter(f(e), f(-INF))), float(np.nextafter(f(e), f(INF))))
                    lo2, hi2 = (float(np.nextafter(f(lo), f(-INF))), float(np.nextafter(f(hi), f(INF))))
                    if out[i] not in (lo, hi, lo2, hi2):
                        bad += 1
            ok(bad == 0, f'_prod_div {d} den={den}: within 2 ULP of the exact quotient, no spurious overflow / underflow ({bad} bad)')
    for d in (torch.float16, torch.float32):
        ok(K._ext('L', True, [999.0, 1001.0], [1000.0, 1000.0], [1.0, 1.0], 1, d) == 0.0, '_ext: exact cancellation')
        ok(K._ext('L', True, [999.0, 1001.0, 5.0], [1000.0, 1000.0, 4.0], [1.0] * 3, 1, d) == rn(-mp.exp(4), d), '_ext: cancellation at the top, lower group survives')
        ok(K._ext('L', False, [0.0] * 16 + [0.0], [0.25] * 16 + [4.0], [1.0] * 17, 1, d) == 0.0, '_ext: 16*0.25*log(0.25) + 4*log(4) = 0')
        xv = 3e+38 if d == torch.float32 else 60000.0
        ok(K._ext('gt', True, [xv], [-5.0], [1.0], 1, d) == rn(mp.exp(-5) * (M(-5) - M(xv) + 1), d), '_ext: huge coefficient')
    ok(K._ext('L', True, [0.0], [3e+38], [1.0], 1, torch.float32) == INF, '_ext: exp(3e38) overflows')
    ok(K._ext('L', True, [0.0], [-3e+38], [1.0], 1, torch.float32) == 0.0, '_ext: exp(-3e38) rounds to zero')

def rand_inputs(d, lt, shape, rng, near=False):
    z = torch.from_numpy(rng.standard_normal(shape) * 3).to(d)
    x = torch.log_softmax(z.float(), -1).to(d)
    if near:
        zt = z.float() + torch.from_numpy(rng.standard_normal(shape)).float() * 0.001
    else:
        zt = torch.from_numpy(rng.standard_normal(shape) * 3).float()
    t = (torch.log_softmax(zt, -1) if lt else torch.softmax(zt, -1)).to(d)
    return (x.to(DEV), t.to(DEV))
U = {torch.float16: 2.0 ** (-11), torch.float32: 2.0 ** (-24), torch.float64: 2.0 ** (-53)}
TINY = {torch.float16: 2.0 ** (-24), torch.float32: 2.0 ** (-149)}

def within_design(m, d, lt, part, got, exp_, xs, ts, Gs, den):
    u = U[K._wide(m, d)]
    bad = 0
    for g, e, x, t, G in zip(got, exp_, xs, ts, Gs):
        if math.isnan(e) or math.isinf(e) or math.isinf(x) or math.isinf(t) or (not lt and t == 0):
            bad += not (math.isnan(g) and math.isnan(e) or g == e)
            continue
        if lt:
            scale = abs(G) / den * math.exp(min(t, 700)) * (abs(t) + abs(x) + 1 if part != 'gx' else 1)
        else:
            scale = abs(t) * (abs(math.log(t)) + abs(x)) / den if part == 'y' else abs(G) / den * (abs(t) if part == 'gx' else abs(math.log(t)) + abs(x) + 1)
        sub = abs(G) / den * ((abs(t - x + 1) if part == 'gt' else 1) if lt else 0) * 2 * TINY[d] if m == 'C' else 0
        bad += abs(g - e) > 16 * u * scale + 2 * TINY[d] + sub
    return bad

def ulps(a, b, d):
    f = NPD[d]
    it = {torch.float16: np.int16, torch.float32: np.int32}[d]
    o = lambda v: np.where(v < 0, -(v & (v.dtype.type(-1) ^ np.iinfo(v.dtype).min)), v).astype(np.int64)
    return np.abs(o(np.asarray(a, dtype=f).view(it)) - o(np.asarray(b, dtype=f).view(it)))

def test_random():
    rng = np.random.default_rng(2)
    stat = {}
    for d, lt, near in itertools.product((torch.float16, torch.float32), (False, True), (False, True)):
        for rep in range(6):
            x, t = rand_inputs(d, lt, (4, 16), rng, near)
            G = torch.from_numpy(rng.standard_normal((4, 16)) * 10.0 ** rng.integers(-6, 4, (4, 16))).to(d).to(DEV)
            G[0, :3] = 0
            G[1, 0] = -G[1, 0].abs()
            for red in ('none', 'sum', 'mean', 'batchmean'):
                for m in K.METHODS:
                    Gm = G if red == 'none' else None
                    y, gx, gt = run(x, t, lt, red, m, Gm)
                    ey, egx, egt = expect(x, t, lt, red, d, Gm)
                    n_ = x.numel()
                    den = {'none': 1, 'sum': 1, 'mean': n_, 'batchmean': x.shape[0]}[red]
                    Gl = flat(G) if red == 'none' else [1.0] * n_
                    for nm, got, exp_ in (('y', flat(y), ey), ('gx', flat(gx), egx), ('gt', flat(gt), egt)):
                        u = ulps(got, exp_, d)
                        k = (m, str(d)[-2:], lt, nm, red != 'none')
                        s = stat.setdefault(k, [0, 0, 0, 0])
                        s[0] += len(u)
                        s[1] += int((u > 0).sum())
                        s[2] = max(s[2], int(u.max()))
                        if nm != 'y':
                            s[3] += within_design(m, d, lt, nm, got, exp_, flat(x), flat(t), Gl, den)
    for k, (n, bad, mx, out) in sorted(stat.items()):
        m, dn, lt, nm, reduced = k
        if m == 'D64' and dn == '16':
            ok(bad == 0, f'random {k}: D64 on float16 is correctly rounded ({bad}/{n}, max {mx} ULP)')
        elif m == 'D64':
            ok(mx <= 1 and bad <= max(2, n // 50), f'random {k}: D64 on float32 within 1 ULP, rare ({bad}/{n}, max {mx})')
        elif nm != 'y':
            ok(out == 0, f'random {k}: gradient error inside the design scale ({out} outside; not CR {bad}/{n}, max {mx} ULP)')
        NOTES.append(f"random  {m:3s} fp{dn} lt={int(lt)} {nm:2s} {('reduced' if reduced else 'element')}: not CR {bad}/{n}, max {mx} ULP")

def bc_expect(x, t, lt, red, d, G=None):
    xb, tb = torch.broadcast_tensors(x, t)
    n = xb.numel()
    B = x.shape[0] if x.dim() else 1
    den = {'none': 1, 'sum': 1, 'mean': n, 'batchmean': B if x.dim() else 1}[red]
    xs, ts = (flat(xb), flat(tb))
    Ls = [truth(a, b, lt)[0] for a, b in zip(xs, ts)]
    yv = [rn(v, d) for v in Ls] if red == 'none' else [rn(mp.fsum(Ls) / den, d)]
    idx = torch.arange(n).reshape(xb.shape)
    out = []
    for side, shp in ((1, x.shape), (2, t.shape)):
        el = [truth(a, b, lt, None, den)[side] for a, b in zip(xs, ts)]
        nd = len(xb.shape)
        lead = nd - len(shp)
        dims = list(range(lead)) + [lead + i for i, (a, b) in enumerate(zip(shp, xb.shape[lead:])) if a == 1 and b != 1]
        keep = [i for i in range(nd) if i not in dims]
        grp = idx.permute(keep + dims).reshape(max(1, int(np.prod([xb.shape[i] for i in keep]))), -1).tolist()
        out.append([rn(mp.fsum([el[i] for i in g]), d) for g in grp])
    return (yv, out[0], out[1])

def test_shapes():
    rng = np.random.default_rng(3)
    cases = [((2, 3, 4), (2, 3, 4)), ((2, 3, 2, 4), (2, 3, 2, 4)), ((2, 3, 4), (4,)), ((4,), (2, 3, 4)), ((2, 1, 4), (1, 3, 4)), ((3, 1), (1, 5)), ((), ()), ((5,), ()), ((), (5,))]
    for d, lt in itertools.product((torch.float16, torch.float32), (False, True)):
        for sx, st in cases:
            mk = lambda s: torch.from_numpy(np.asarray(-np.abs(rng.standard_normal(s) * 2) - 0.01)).to(d).to(DEV)
            x = mk(sx)
            t = mk(st) if lt else torch.exp(mk(st).float()).to(d)
            for red in ('none', 'sum', 'mean', 'batchmean'):
                xo = x.clone().requires_grad_(True)
                to = t.clone().requires_grad_(True)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    yo = F.kl_div(xo, to, reduction=red, log_target=lt)
                gxo, gto = torch.autograd.grad(yo, [xo, to], grad_outputs=torch.ones_like(yo))
                ey, egx, egt = bc_expect(x, t, lt, red, d)
                for m in K.METHODS:
                    y, gx, gt = run(x, t, lt, red, m)
                    tag = f'shapes {str(d)[-7:]} lt={lt} {sx}x{st} {red} {m}'
                    ok(y.shape == yo.shape and gx.shape == gxo.shape and (gt.shape == gto.shape) and (y.dtype == yo.dtype), tag + ': shapes / dtype as official')
                    if m == 'D64':
                        u = max(int(ulps(flat(y), ey, d).max()), int(ulps(flat(gx), egx, d).max()), int(ulps(flat(gt), egt, d).max()))
                        ok(u <= (0 if d == torch.float16 else 1), tag + f': against the exact broadcast truth (max {u} ULP)')
                    else:
                        tol = 0.05 if K._wide(m, d) == torch.float16 else 0.0001
                        cl = lambda a, b: all((abs(p - q) <= tol * (abs(q) + 1.0) for p, q in zip(flat(a), b)))
                        ok(cl(y, ey) and cl(gx, egx) and cl(gt, egt), tag + ': close to the exact broadcast truth')
                    if m == 'C' and red in ('none', 'sum'):
                        ok(same(flat(y), flat(yo)), tag + ': C forward identical to official on regular inputs')

def test_boundary():
    for d in (torch.float16, torch.float32):
        other = (-1.5, -0.7)
        tab = {False: [(-1.0, 0.0), (-INF, 0.25), (-INF, 0.0), (-3.0, 0.5)], True: [(-1.0, -INF), (-INF, -1.0), (-INF, -INF), (-3.0, -0.5)]}
        for lt in (False, True):
            for xv, tv in tab[lt]:
                x = T([xv, other[0]], d)
                t = T([tv, other[1] if lt else 0.5], d)
                for m in K.METHODS:
                    for Gv in ([1.0, 1.0], [-2.0, 1.0], [0.0, 1.0], [0.0, 0.0]):
                        G = T(Gv, d)
                        y, gx, gt = run(x, t, lt, 'none', m, G)
                        ey, egx, egt = expect(x, t, lt, 'none', d, G)
                        tag = f'boundary {str(d)[-7:]} lt={lt} x={xv} t={tv} G={Gv} {m}'
                        regular = math.isfinite(xv) and (math.isfinite(tv) if lt else tv > 0)
                        if regular and m != 'D64':
                            continue
                        ok(same(flat(y)[:1], ey[:1]) and same(flat(gx)[:1], egx[:1]) and same(flat(gt)[:1], egt[:1]), tag + f': special element {flat(y)[:1]} {flat(gx)[:1]} {flat(gt)[:1]} expected {ey[:1]} {egx[:1]} {egt[:1]}')
                        ok(not any((math.isnan(v) for v in flat(y)[1:] + flat(gx)[1:] + flat(gt)[1:])), tag + ': the regular neighbour is not polluted')
                    for red in ('sum', 'mean', 'batchmean'):
                        y, gx, gt = run(x, t, lt, red, m)
                        ey, egx, egt = expect(x, t, lt, red, d)
                        if m == 'D64':
                            ok(same(flat(y), ey) and same(flat(gx)[:1], egx[:1]) and same(flat(gt)[:1], egt[:1]), f'boundary {str(d)[-7:]} lt={lt} x={xv} t={tv} {red} {m}: {flat(y)} {flat(gx)} {flat(gt)} expected {ey} {egx} {egt}')
                        else:
                            ok(same(flat(gt)[:1], egt[:1]) and math.isinf(ey[0]) == math.isinf(flat(y)[0]) and (not math.isnan(flat(y)[0])), f'boundary {str(d)[-7:]} lt={lt} x={xv} t={tv} {red} {m}')
    for d, big in ((torch.float16, 60000.0), (torch.float32, 3e+38)):
        x = T([[-big, -big / 2, -1.0, -2.0]] * 2, d)
        t = T([[0.9, 0.9, 0.5, 0.25]] * 2, d)
        for m in K.METHODS:
            for red in ('mean', 'batchmean', 'sum'):
                y, gx, gt = run(x, t, False, red, m)
                ey, egx, egt = expect(x, t, False, red, d)
                ok(same(flat(y), ey) if m != 'C' else math.isinf(flat(y)[0]) == math.isinf(ey[0]) and (not math.isnan(flat(y)[0])), f'overflow before scaling {str(d)[-7:]} {red} {m}: {flat(y)} expected {ey}')
        x = T([[0.0, -1.0, 2.0, 1.0]] * 2, d)
        t = T([[9.0 if d == torch.float16 else 85.0, -1.0, 2.0, 0.0]] * 2, d)
        for m in K.METHODS:
            for red in ('mean', 'batchmean', 'sum'):
                y, gx, gt = run(x, t, True, red, m)
                ey, egx, egt = expect(x, t, True, red, d)
                ok(same(flat(y), ey) if m != 'C' else math.isinf(flat(y)[0]) == math.isinf(ey[0]) and (not math.isnan(flat(y)[0])), f'overflow before scaling lt=True {str(d)[-7:]} {red} {m}: {flat(y)} expected {ey}')
    d = torch.float32
    sets = {'t = x': ([1000.0], [1000.0]), 't != x': ([999.0, 1001.5], [1000.0, 1000.0]), 'exact sum 0': ([999.0, 1001.0], [1000.0, 1000.0]), 'top cancels, rest finite': ([999.0, 1001.0, -1.0, 0.5], [1000.0, 1000.0, -2.0, 0.25]), 'top cancels, rest overflows': ([999.0, 1001.0, 0.0], [1000.0, 1000.0, 500.0]), 'partial cancellation': ([999.0, 1001.0, 999.5], [1000.0, 1000.0, 1000.0]), 'gradient factor 0': ([1001.0, 2.0], [1000.0, 1.0]), 'x = t + 1 + ulp': ([float(np.nextafter(np.float32(1001), np.float32(INF))), 2.0], [1000.0, 1.0]), 'x = t + 1 - ulp': ([float(np.nextafter(np.float32(1001), np.float32(0))), 2.0], [1000.0, 1.0]), 't < -745': ([-800.0, -1.0, 3e+38], [-750.0, -0.5, -900.0]), 'moderate t = x + 1': ([3.0, 0.5], [2.0, -0.5])}
    for name, (xv, tv) in sets.items():
        x, t = (T(xv, d), T(tv, d))
        for m in K.METHODS:
            for red in ('none', 'sum', 'mean', 'batchmean'):
                for Gv in (None,) if red != 'none' else (None, [1.0, 0.0, -1.0, 2.0][:len(xv)]):
                    G = None if Gv is None else T(Gv, d)
                    y, gx, gt = run(x, t, True, red, m, G)
                    ey, egx, egt = expect(x, t, True, red, d, G)
                    exact_path = m == 'D64' or name not in ('t < -745', 'moderate t = x + 1')
                    if exact_path:
                        ok(same(flat(y), ey) and same(flat(gx), egx) and same(flat(gt), egt), f"exp range '{name}' {red} {m} G={Gv}: {flat(y)} {flat(gx)} {flat(gt)} expected {ey} {egx} {egt}")
                    else:
                        ok(not any((math.isnan(v) for v in flat(y) + flat(gx) + flat(gt))), f"exp range '{name}' {red} {m}: no NaN")
    d = torch.float16
    for name, (xv, tv) in {'fp16 sum 0': ([11.0, 13.0], [12.0, 12.0]), 'fp16 exp(100) cancels': ([99.0, 101.0, 1.0], [100.0, 100.0, 2.0]), 'fp16 false inf': ([12.0 - 2 ** (-7), 0.0], [12.0, -1.0]), 'fp16 zero gradient': ([13.0, 101.0], [12.0, 100.0])}.items():
        x, t = (T(xv, d), T(tv, d))
        for m in K.METHODS:
            for red in ('none', 'sum', 'mean'):
                y, gx, gt = run(x, t, True, red, m)
                ey, egx, egt = expect(x, t, True, red, d)
                ok(same(flat(y), ey) and same(flat(gx), egx) and same(flat(gt), egt), f"'{name}' {red} {m}: {flat(y)} {flat(gx)} {flat(gt)} expected {ey} {egx} {egt}")
    x, t = (T([11.0, 0.5], d), T([60000.0, 0.25], d))
    for m in K.METHODS:
        for red in ('none', 'sum', 'mean'):
            y, gx, gt = run(x, t, False, red, m)
            ey, egx, egt = expect(x, t, False, red, d)
            ok(same(flat(y), ey) if m != 'C' or red == 'none' else not math.isnan(flat(y)[0]), f'fp16 t log t overflow {red} {m}: {flat(y)} expected {ey}')
    d = torch.float32
    t = T([1000.0], d)
    x = T([1000.0, 1002.0, 1001.0], d)
    for m in K.METHODS:
        y, gx, gt = run(x, t, True, 'sum', m)
        ok(flat(gt) == [0.0] and flat(gx) == [-INF] * 3, f'broadcast gradient cancellation {m}: {flat(gt)} {flat(gx)}')
    for d, Gv, n, xv in ((torch.float16, 2.0 ** (-24), 256, -512.0), (torch.float32, 2.0 ** (-140), 4096, -1000.0)):
        x = torch.full((1, n), xv, dtype=d, device=DEV)
        t = torch.full((1, n), 0.25, dtype=d, device=DEV)
        G = T(Gv, d)
        for m in K.METHODS:
            y, gx, gt = run(x, t, False, 'mean', m, G)
            e = rn(M(Gv) / n * (mp.log(M(0.25)) + 1 - M(xv)), d)
            ex_ = rn(-M(Gv) / n * M(0.25), d)
            ok(e != 0 and set(flat(gt)) == {e} and (set(flat(gx)) == {ex_}), f'tiny upstream {str(d)[-7:]} {m}: {set(flat(gt))} expected {e}; gx {set(flat(gx))} expected {ex_}')
        w = torch.full((1, n), Gv, dtype=d, device=DEV)
        w[0, 0] = 0
        for m in K.METHODS:
            y, gx, gt = run(x, t, False, 'none', m, w)
            ok(flat(gt)[0] == 0 and flat(gt)[1] != 0, f'tiny element weights {str(d)[-7:]} {m}')
    for d in (torch.float16, torch.float32):
        x = T([[-0.1, -0.2, -0.3]], d)
        t = T([[0.05, 0.1, 0.02]], d)
        for m in K.METHODS:
            for lt, tt in ((False, t), (True, T([[-3.0, -2.5, -4.0]], d))):
                y, _, _ = run(x, tt, lt, 'sum', m)
                yn, _, _ = run(x, tt, lt, 'none', m)
                ey, _, _ = expect(x, tt, lt, 'sum', d)
                ok(ey[0] < 0 and flat(y)[0] < 0 and all((v < 0 for v in flat(yn))), f'negative values kept {str(d)[-7:]} lt={lt} {m}: {flat(y)}')
    d = torch.float32
    h = 2.0 ** (-24)
    mids = {'sum 1 + 2^-24 + e^-1000': ([-1.0, -h, -1001.0], [0.0, 0.0, -1000.0]), 'tiny term negative': ([-1.0, -h, -999.0], [0.0, 0.0, -1000.0]), 'mean of four': ([-1.0, -h, -1001.0, 5.0], [0.0, 0.0, -1000.0, 5.0]), 'exact tie, no tiny term': ([-1.0, -h, 3.0], [0.0, 0.0, 3.0]), 'tiny term not underflowed (e^-300)': ([-1.0, -h, -301.0], [0.0, 0.0, -300.0]), 'two tiny terms cancel': ([-1.0, -h, -1001.0, -999.0], [0.0, 0.0, -1000.0, -1000.0])}
    for name, (xv, tv) in mids.items():
        x, t = (T(xv, d), T(tv, d))
        for red in ('sum', 'mean', 'batchmean'):
            y, gx, gt = run(x, t, True, red, 'D64')
            ey, egx, egt = expect(x, t, True, red, d)
            ok(same(flat(y), ey) and same(flat(gx), egx) and same(flat(gt), egt), f"midpoint '{name}' {red} D64: {flat(y)} expected {ey}")
    x, t = (T([5.0, 5.0, 5.0], d), T([0.0, 0.0, -1000.0], d))
    G = T([1.0, h, 1.0], d)
    xs_ = T([5.0], d)
    y, gx, gt = run(xs_, t, True, 'none', 'D64', G)
    e = rn(-(1 + M(h) + mp.exp(-1000)), d)
    ok(flat(gx) == [e] and e != -1.0, f'midpoint in a broadcast gradient sum D64: {flat(gx)} expected {e}')
    d = torch.float16
    h = 2.0 ** (-11)
    for name, (xv, tv) in {'fp16 1 + 2^-11 + e^-200': ([-1.0, -h, -201.0], [0.0, 0.0, -200.0]), 'fp16 negative': ([-1.0, -h, -199.0], [0.0, 0.0, -200.0]), 'fp16 e^-50': ([-1.0, -h, -51.0], [0.0, 0.0, -50.0])}.items():
        x, t = (T(xv, d), T(tv, d))
        for m in ('D32', 'D64'):
            if m == 'D32' and 'e^-50' in name:
                continue
            for red in ('sum', 'mean'):
                y, gx, gt = run(x, t, True, red, m)
                ey, egx, egt = expect(x, t, True, red, d)
                ok(same(flat(y), ey), f"midpoint '{name}' {red} {m}: {flat(y)} expected {ey}")
    for xv, tv, lt in (([-1.0 - 2.0 ** (-12), -3.0], [0.0, -2.0], True), ([-(1 + 2.0 ** (-11)) / 0.5 + math.log(0.5), -1.0], [0.5, 0.25], False)):
        x, t = (T(xv, d), T(tv, d))
        for m in ('D32', 'D64'):
            y, gx, gt = run(x, t, lt, 'none', m)
            ey, egx, egt = expect(x, t, lt, 'none', d)
            if m == 'D64':
                ok(same(flat(y), ey) and same(flat(gx), egx) and same(flat(gt), egt), f'element near a midpoint lt={lt} {m}: {flat(y)} expected {ey}')
    old = K.EXT_LADDER
    K.EXT_LADDER = (8,)
    try:
        try:
            K._ext('L', False, [0.0] * 17, [0.25] * 16 + [4.0], [1.0] * 17, 1, torch.float32)
            ok(False, 'UncertifiedRounding raised when the ladder is exhausted')
        except K.UncertifiedRounding:
            ok(True, 'UncertifiedRounding raised when the ladder is exhausted')
    finally:
        K.EXT_LADDER = old
    x, t = (T([-10.0], torch.float16), T([-18.0], torch.float16))
    for m in ('D32', 'D64'):
        y, gx, gt = run(x, t, True, 'none', m)
        ey, egx, egt = expect(x, t, True, 'none', torch.float16)
        ok(same(flat(y), ey) and ey[0] != 0 and same(flat(gx), egx) and same(flat(gt), egt), f'exp underflow {m}: {flat(y)} expected {ey}')
    y, _, _ = run(x, t, True, 'none', 'C')
    NOTES.append(f'R5: method C at t=-18, x=-10 (float16) returns {flat(y)[0]} (registered residual)')

def test_domain():
    eqb = lambda a, b: a.shape == b.shape and bool(((a == b) | a.isnan() & b.isnan()).all()) and bool((torch.signbit(a) == torch.signbit(b))[~a.isnan()].all())
    for d in (torch.float16, torch.float32):
        for lt, xv, tv in ((False, [-1.0, -2.0, -1.5], [-0.5, 0.5, 0.25]), (False, [INF, -2.0, -1.5], [0.5, 0.5, 0.25]), (False, [math.nan, -2.0, -1.5], [0.5, 0.5, 0.25]), (False, [-1.0, -2.0, -1.5], [math.nan, 0.5, 0.25]), (False, [-1.0, -2.0, -1.5], [INF, 0.5, 0.25]), (False, [-1.0, -2.0, -INF], [-0.5, 0.0, 0.0]), (True, [INF, -2.0, -1.5], [-0.5, -0.5, -3.0]), (True, [-1.0, -2.0, -1.5], [INF, -0.5, -3.0]), (True, [math.nan, -2.0, -1.5], [-0.5, -0.5, -3.0]), (True, [-1.0, -2.0, -1.5], [-0.5, math.nan, -INF])):
            x, t = (T(xv, d), T(tv, d))
            badm = K._outside(x, t, lt)
            for red in ('none', 'sum', 'mean', 'batchmean'):
                for Gv in (None, [2.0, 0.0, -1.0]) if red == 'none' else (None,):
                    G = None if Gv is None else T(Gv, d)
                    xo = x.clone().requires_grad_(True)
                    to = t.clone().requires_grad_(True)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        yo = F.kl_div(xo, to, reduction=red, log_target=lt)
                    gxo, gto = torch.autograd.grad(yo, [xo, to], grad_outputs=torch.ones_like(yo) if G is None else G)
                    for m in K.METHODS:
                        y, gx, gt = run(x, t, lt, red, m, G)
                        tag = f'domain {str(d)[-7:]} lt={lt} x={xv} t={tv} {red} G={Gv} {m}'
                        if red == 'none':
                            ok(eqb(y[badm], yo[badm]), tag + ': outside elements keep the official value')
                        else:
                            ok(eqb(y, yo), tag + ': a reduction with an outside element returns the official value')
                        ok(eqb(gx[badm], gxo[badm]) and eqb(gt[badm], gto[badm]), tag + ': outside elements keep the official gradients')
                        if m == 'D64':
                            good = (~badm).nonzero().flatten().tolist()
                            n_ = x.numel()
                            den = {'none': 1, 'sum': 1, 'mean': n_, 'batchmean': n_}[red]
                            for i in good:
                                e = truth(xv[i], tv[i], lt, 1.0 if Gv is None else Gv[i], den)
                                r = lambda v: math.nan if isinstance(v, str) else rn(v, d)
                                got = (flat(gx)[i], flat(gt)[i])
                                exp_ = (r(e[1]), r(e[2]))
                                ok(same(got, exp_) and (red != 'none' or same([flat(y)[i]], [rn(truth(xv[i], tv[i], lt)[0], d)])), tag + f': element {i} repaired {got} expected {exp_}')

def test_api():
    so, sf = (inspect.signature(F.kl_div), inspect.signature(K.kl_div))
    ok(list(sf.parameters)[:-1] == list(so.parameters) and all((sf.parameters[k].default == so.parameters[k].default for k in so.parameters)), 'signature = official + method')
    d = torch.float32
    rng = np.random.default_rng(5)
    for shp in ((3, 4), (2, 3, 4), ()):
        x = torch.from_numpy(np.asarray(-np.abs(rng.standard_normal(shp)) - 0.1)).to(d).to(DEV)
        t = torch.from_numpy(np.asarray(np.abs(rng.standard_normal(shp)) * 0.2 + 0.01)).to(d).to(DEV)
        for sa, rd, red in itertools.product((None, True, False), (None, True, False), ('none', 'sum', 'mean', 'batchmean')):

            def call(fn, **kw):
                xo = x.clone().requires_grad_(True)
                to = t.clone().requires_grad_(True)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    y = fn(xo, to, size_average=sa, reduce=rd, reduction=red, **kw)
                g = torch.autograd.grad(y, [xo, to], grad_outputs=torch.ones_like(y))
                return (y, g, sorted((str(i.message)[:60] for i in w)))
            yo, go, wo = call(F.kl_div)
            for m in K.METHODS:
                y, g, w = call(K.kl_div, method=m)
                tag = f'legacy {shp} size_average={sa} reduce={rd} reduction={red} {m}'
                ok(y.shape == yo.shape and w == wo, tag + f': shape and warnings as official ({w} / {wo})')
                cl = lambda a, b: torch.allclose(a.double(), b.double(), rtol=2e-05, atol=1e-07)
                ok(cl(y, yo) and cl(g[0], go[0]) and cl(g[1], go[1]), tag + ': forward and both gradients carry the official scaling')
    for bad in ('elementwise_mean', 'bogus'):
        for m in K.METHODS[-1:]:
            x = T([-1.0], d)
            t = T([0.5], d)

            def outcome(fn, **kw):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        return ('value', float(fn(x, t, reduction=bad, **kw)))
                except Exception as e:
                    return ('error', type(e).__name__, str(e))
            ok(outcome(F.kl_div) == outcome(K.kl_div, method=m), f'reduction={bad!r}: same outcome as official')
    for xs, ts in (((2, 3), (4,)), ((0, 3), (0, 3)), ((2, 0), (2, 0))):
        x = torch.zeros(xs, dtype=d, device=DEV) - 1
        t = torch.zeros(ts, dtype=d, device=DEV) + 0.5
        for red in ('none', 'sum', 'mean', 'batchmean'):

            def outcome(fn, **kw):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        y = fn(x, t, reduction=red, **kw)
                        return ('value', tuple(y.shape), [repr(v) for v in flat(y)])
                except Exception as e:
                    return ('error', type(e).__name__)
            ok(outcome(F.kl_div) == outcome(K.kl_div, method='D64'), f'shapes {xs} / {ts} {red}: same outcome as official (error or empty-tensor value)')
    x = T([-1.0], torch.float16)
    t = T([0.5], torch.float32)

    def mixed(fn, **kw):
        try:
            return ('value', str(fn(x, t, reduction='sum', **kw).dtype))
        except Exception as e:
            return ('error', type(e).__name__)
    ok(mixed(F.kl_div) == mixed(K.kl_div, method='D64'), 'mixed dtypes are passed to the official function')
    for d in (torch.float16, torch.float32, torch.float64):
        x = T([-1.0, -2.0], d)
        t = T([0.5, 0.25], d)
        for m in K.METHODS:
            y, gx, gt = run(x, t, False, 'sum', m)
            ok(y.dtype == d and gx.dtype == d and (gt.dtype == d) and (y.device == x.device), f'dtype / device preserved {d} {m}')
    try:
        K.kl_div(T([-1.0], torch.float32), T([0.5], torch.float32), method='X')
        ok(False, 'unknown method rejected')
    except ValueError:
        ok(True, 'unknown method rejected')

def test_second_order():
    rng = np.random.default_rng(7)
    d = torch.float64
    for lt in (False, True):
        for red in ('none', 'sum', 'mean', 'batchmean'):
            for sx, st in (((2, 3), (2, 3)), ((2, 3), (3,)), ((3,), (2, 3))):
                x = torch.from_numpy(-np.abs(rng.standard_normal(sx)) - 0.2).to(d).to(DEV).requires_grad_(True)
                t0 = torch.from_numpy(np.abs(rng.standard_normal(st)) * 0.3 + 0.2).to(d).to(DEV)
                t = (t0.log() if lt else t0).detach().requires_grad_(True)
                for m in K.METHODS:
                    fn = lambda a, b: K.kl_div(a, b, reduction=red, log_target=lt, method=m)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        ok(torch.autograd.gradcheck(fn, (x, t), eps=1e-06, atol=1e-05, rtol=0.0001, raise_exception=False), f'gradcheck lt={lt} {red} {sx}x{st} {m}')
                        ok(torch.autograd.gradgradcheck(fn, (x, t), eps=1e-06, atol=1e-05, rtol=0.0001, raise_exception=False), f'gradgradcheck lt={lt} {red} {sx}x{st} {m}')
    x = T([-1.2, -0.4], d).requires_grad_(True)
    t = T([0.3, 0.7], d).requires_grad_(True)
    y = K.kl_div(x, t, reduction='sum', method='D64')
    gx, gt = torch.autograd.grad(y, [x, t], create_graph=True)
    htt = torch.autograd.grad(gt.sum(), t, retain_graph=True)[0]
    htx = torch.autograd.grad(gt.sum(), x, retain_graph=True)[0]
    ok(torch.allclose(htt, 1 / t.detach()) and torch.allclose(htx, -torch.ones_like(x)), 'd2L/dt2 = 1/t, d2L/dtdx = -1 (log_target=False)')
    third = torch.autograd.grad(torch.autograd.grad(gt.sum(), t, create_graph=True)[0].sum(), t)[0]
    ok(torch.allclose(third, -1 / t.detach() ** 2), 'third derivative through the analytic node')
if __name__ == '__main__':
    for f in (test_helpers, test_random, test_shapes, test_boundary, test_domain, test_api, test_second_order):
        n0, f0 = (N['checks'], N['fail'])
        f()
        print(f"{f.__name__}: {N['checks'] - n0} checks, {N['fail'] - f0} failed")
    print('\n'.join(NOTES))
    print(f'device {DEV}, torch {torch.__version__}, mpmath {mp.__version__}; extended path: {K.STATS}')
    print(f"TOTAL {N['checks']} checks, {N['fail']} failed")
    sys.exit(1 if N['fail'] else 0)
