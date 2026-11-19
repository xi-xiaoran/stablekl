"""Repaired KL divergence between two Normal distributions: formula, analytic gradients, and the
entry point run_rewrite_torch used by the benchmark (dtype in, compute dtype, outputs per component).
The derivation is in the paper's appendix on interface A."""
from __future__ import annotations
import math
import numpy as np
NAME = 'normal'
PARAMS = ('loc', 'scale')
COMPS = ['KL', 'P.loc', 'P.scale', 'Q.loc', 'Q.scale']
EXPECTED_ROWS = {'fp16': 137499, 'fp32': 202000}
TRANSCENDENTAL_COMPS = ('KL',)
FMT = {'fp16': dict(np=np.float16, u=2.0 ** (-11), eta=2.0 ** (-24), minnorm=2.0 ** (-14), emax=15), 'fp32': dict(np=np.float32, u=2.0 ** (-24), eta=2.0 ** (-149), minnorm=2.0 ** (-126), emax=127), 'fp64': dict(np=np.float64, u=2.0 ** (-53), eta=2.0 ** (-1074), minnorm=2.0 ** (-1022), emax=1023)}
SAFETY = 1.02
KTERMS = {'fp16': 9, 'fp32': 17, 'fp64': 36}
SEG_LO, SEG_HI = (-0.75, 3.0)
TEXP = {'fp16': 6, 'fp32': 40, 'fp64': 300}
VELT = {'fp16': 65.0, 'fp32': 4097.0, 'fp64': 134217729.0}
PREC = {'fp16': 11, 'fp32': 24, 'fp64': 53}
EMIN = {'fp16': -14, 'fp32': -126, 'fp64': -1022}
LIMS = {'fp16': 15, 'fp32': 127, 'fp64': 1023}

def q_tail_bound(z, K):
    return z ** K / ((2 * K + 3) * (1 - z))
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    _MINN = {torch.float16: 2.0 ** (-14), torch.float32: 2.0 ** (-126), torch.float64: 2.0 ** (-1022)}
    _K = {torch.float16: KTERMS['fp16'], torch.float32: KTERMS['fp32'], torch.float64: KTERMS['fp64']}
    _DT = {torch.float16: 'fp16', torch.float32: 'fp32', torch.float64: 'fp64'}

    def _frexp(x):
        if x.dtype == torch.float16:
            m, e = torch.frexp(x.to(torch.float32))
            return (m.to(torch.float16), e.to(torch.int64))
        m, e = torch.frexp(x)
        return (m, e.to(torch.int64))

    def _exp(x):
        _, e = _frexp(x)
        return e

    def _pow2(kk, like):
        return torch.exp2(kk.to(torch.float64)).to(like.dtype)

    def _scale(x, k):
        L = LIMS[_DT[x.dtype]]
        k1 = k.clamp(-L, L)
        k2 = k - k1
        assert int(k2.abs().max()) <= L if k2.numel() else True, 'scale exponent out of range'
        assert bool((k2 >= 0).all()), 'down-scaling needs one factor'
        return x * _pow2(k1, x) * _pow2(k2, x)

    def _ldexp(x, E):
        dt = _DT[x.dtype]
        L = LIMS[dt]
        pos = E >= 0
        E1p = E.clamp(0, L)
        E2p = (E - E1p).clamp(0, L)
        E1n = torch.clamp(E, min=EMIN[dt] + 1).clamp(max=0)
        E2n = (E - E1n).clamp(-(PREC[dt] + 5), 0)
        E1 = torch.where(pos, E1p, E1n)
        E2 = torch.where(pos, E2p, E2n)
        return x * _pow2(E1, x) * _pow2(E2, x)

    def _two_sum(a, b):
        s = a + b
        bb = s - a
        return (s, a - (s - bb) + (b - bb))

    def _two_prod(a, b, C):
        p = a * b
        ga = C * a
        ah = ga - (ga - a)
        al = a - ah
        gb = C * b
        bh = gb - (gb - b)
        bl = b - bh
        err = ah * bh - p + ah * bl + al * bh + al * bl
        return (p, err)

    def _exact_sum(terms, zero):
        e = []
        for b in terms:
            Q = b
            new = []
            for c in e:
                Q, h = _two_sum(Q, c)
                new.append(h)
            new.append(Q)
            e = new
        S = e[0]
        for c in e[1:]:
            S = S + c
        return S + zero

    def _route_n(pl, ql, ps, qs, dm, fdm, zero):
        dt = _DT[ps.dtype]
        dh = torch.where(fdm, dm, zero)
        _, dl = _two_sum(pl, zero - ql)
        dl = torch.where(fdm, dl, zero)
        km = torch.maximum(_exp(torch.maximum(ps, qs)), torch.where(dh == 0, torch.full_like(fdm, -100000, dtype=torch.int64), _exp(dh.abs())))
        k = TEXP[dt] + 1 - km
        ps_, qs_, dh_, dl_ = (_scale(ps, k), _scale(qs, k), _scale(dh, k), _scale(dl, k))
        C = torch.full_like(ps, VELT[dt])
        P1 = _two_prod(qs_, qs_, C)
        P2 = _two_prod(ps_, ps_, C)
        P3 = _two_prod(dh_, dh_, C)
        P4 = _two_prod(dh_, 2 * dl_, C)
        P5 = _two_prod(dl_, dl_, C)
        terms = [P1[0], P1[1]] + [zero - t for P in (P2, P3, P4, P5) for t in P]
        Nh = _exact_sum(terms, zero)
        Nm, eN = _frexp(Nh)
        Qm, eQ = _frexp(qs_)
        v = Nm / Qm / Qm / Qm
        return _ldexp(v, eN - 3 * eQ + k)

    class NormalKLC(torch.autograd.Function):

        @staticmethod
        def forward(ctx, pl, ps, ql, qs):
            one = torch.ones_like(ps)
            zero = torch.zeros_like(ps)
            half = torch.full_like(ps, 0.5)
            two = torch.full_like(ps, 2)
            dm = pl - ql
            fdm = torch.isfinite(dm)
            dmh = pl * half - ql * half
            w = torch.where(fdm, dm / qs, 2 * (dmh / qs))
            gm = torch.where(fdm, w / qs, 2 * (dmh / qs / qs))
            ds = ps - qs
            e = ds / qs
            fe = torch.isfinite(e)
            ef = torch.where(fe, e, zero)
            inS = fe & (e >= SEG_LO) & (e <= SEG_HI)
            inA = fe & (e > SEG_HI)
            dss = torch.where(inS, ds, zero)
            es = torch.where(inS, e, zero)
            ss = ps + qs
            fss = torch.isfinite(ss)
            s = torch.where(fss, dss / torch.where(fss, ss, one), dss * half / (ps * half + qs * half))
            z = s * s
            q = torch.full_like(ps, 1.0 / (2 * _K[ps.dtype] + 1))
            for k in range(_K[ps.dtype] - 1, 0, -1):
                q = q * z + torch.full_like(ps, 1.0 / (2 * k + 1))
            phiS = es * s - 2 * (s * z) * q
            LA = torch.log1p(torch.where(inA, e, zero))
            r = ps / qs
            rn = r >= _MINN[ps.dtype]
            Lab = torch.log(ps) - torch.log(qs)
            LB = torch.where(rn, torch.log(torch.where(rn, r, one)), Lab)
            L = torch.where(inA, LA, LB)
            phi = torch.clamp_min(torch.where(inS, phiS, ef - L), 0)
            KLs = phi + ef * half * ef
            KLm = w * half * w
            kl = torch.where(fe, KLs + KLm, torch.full_like(ps, float('inf')))
            ep2 = e + two
            Pe = e * ep2
            fP = torch.isfinite(Pe)
            g_ps = torch.where(fP, Pe / ps, e / ps * ep2)
            A1 = torch.where(fP, Pe / qs, e / qs * ep2)
            A2 = w * gm
            A12 = A1 + A2
            fA = torch.isfinite(A12)
            g_old = torch.where(fA, zero - A12, zero - (Pe + w * w) / qs)
            useN = fdm & (qs >= ps * half) & (qs >= dm.abs() * half)
            g_qs = torch.where(useN, _route_n(pl, ql, ps, qs, dm, fdm, zero), g_old)
            g_pl = gm
            g_ql = zero - gm
            ctx.save_for_backward(g_pl, g_ps, g_ql, g_qs)
            return kl

        @staticmethod
        def backward(ctx, g):
            a, b, c, d = ctx.saved_tensors
            return (g * a, g * b, g * c, g * d)

def _np_route_n(pl, ql, ps, qs, dm, fdm, dt):
    T = FMT[dt]['np']
    L = LIMS[dt]
    with np.errstate(all='ignore'):
        A = lambda x: np.asarray(x, dtype=T)
        pl, ql, ps, qs, dm = (A(pl), A(ql), A(ps), A(qs), A(dm))
        z0 = T(0)

        def two_sum(a, b):
            s = (a + b).astype(T)
            bb = (s - a).astype(T)
            return (s, ((a - (s - bb).astype(T)).astype(T) + (b - bb).astype(T)).astype(T))
        Cv = T(VELT[dt])

        def split(a):
            g = (Cv * a).astype(T)
            hi = (g - (g - a).astype(T)).astype(T)
            return (hi, (a - hi).astype(T))

        def two_prod(a, b):
            p = (a * b).astype(T)
            ah, al = split(a)
            bh, bl = split(b)
            t1 = ((ah * bh).astype(T) - p).astype(T)
            t2 = (t1 + (ah * bl).astype(T)).astype(T)
            t3 = (t2 + (al * bh).astype(T)).astype(T)
            err = (t3 + (al * bl).astype(T)).astype(T)
            return (p, err)
        dh = np.where(fdm, dm, z0).astype(T)
        _, dl = two_sum(pl, (z0 - ql).astype(T))
        dl = np.where(fdm, dl, z0).astype(T)
        ex = lambda x: np.frexp(x.astype(np.float64))[1].astype(np.int64)
        km = np.maximum(ex(np.maximum(ps, qs)), np.where(dh == 0, -100000, ex(np.abs(dh))))
        k = TEXP[dt] + 1 - km

        def sc(x):
            k1 = np.clip(k, -L, L)
            k2 = k - k1
            assert (k2 >= 0).all() and (k2 <= L).all()
            return ((x * np.exp2(k1.astype(np.float64)).astype(T)).astype(T) * np.exp2(k2.astype(np.float64)).astype(T)).astype(T)
        ps_, qs_, dh_, dl_ = (sc(ps), sc(qs), sc(dh), sc(dl))
        prods = [(qs_, qs_), (ps_, ps_), (dh_, dh_), (dh_, (T(2) * dl_).astype(T)), (dl_, dl_)]
        PE = [two_prod(a, b) for a, b in prods]
        terms = [PE[0][0], PE[0][1]] + [(z0 - t).astype(T) for P in PE[1:] for t in P]
        e = []
        for b in terms:
            Q = b
            new = []
            for c in e:
                Q, h = two_sum(Q, c)
                new.append(h)
            new.append(Q)
            e = new
        S = e[0]
        partial = []
        for c in e[1:]:
            S = (S + c).astype(T)
            partial.append(S)
        Nh = (S + z0).astype(T)
        mN, eN = np.frexp(Nh.astype(np.float64))
        mQ, eQ = np.frexp(qs_.astype(np.float64))
        Nm, Qm = (mN.astype(T), mQ.astype(T))
        eN, eQ = (eN.astype(np.int64), eQ.astype(np.int64))
        v1 = (Nm / Qm).astype(T)
        v2 = (v1 / Qm).astype(T)
        v3 = (v2 / Qm).astype(T)
        E = eN - 3 * eQ + k
        pos = E >= 0
        E1 = np.where(pos, np.clip(E, 0, L), np.minimum(np.maximum(E, EMIN[dt] + 1), 0))
        E2 = np.where(pos, np.clip(E - E1, 0, L), np.clip(E - E1, -(PREC[dt] + 5), 0))
        g = ((v3 * np.exp2(E1.astype(np.float64)).astype(T)).astype(T) * np.exp2(E2.astype(np.float64)).astype(T)).astype(T)
    return dict(g=g, k=k, ps_=ps_, qs_=qs_, dh_=dh_, dl_=dl_, dh=dh, dl=dl, prods=prods, PE=PE, partial=partial, Nh=Nh, eN=eN, Qm=Qm, v1=v1, v2=v2, v3=v3, E=E)

def run_rewrite_torch(TD, to_t, dt_in, cdt, P, Q, NPF):
    leaves = [to_t(x, dt_in).to(TD[cdt]).requires_grad_(True) for x in (P['loc'], P['scale'], Q['loc'], Q['scale'])]
    kl = NormalKLC.apply(*leaves)
    kl.sum().backward()
    vals = {'KL': kl.detach()}
    for name, leaf in zip(('P.loc', 'P.scale', 'Q.loc', 'Q.scale'), leaves):
        vals[name] = leaf.grad.detach()
    return {k: v.cpu().numpy().astype(NPF[cdt]) for k, v in vals.items()}

def exact_grads(pl, ps, ql, qs):
    dm = pl - ql
    return {'P.loc': dm / (qs * qs), 'P.scale': (ps * ps - qs * qs) / (ps * qs * qs), 'Q.loc': -dm / (qs * qs), 'Q.scale': (qs * qs - ps * ps - dm * dm) / (qs * qs * qs)}

def identity(pl, ps, ql, qs):
    return pl == ql and ps == qs

def replay(pl, ps, ql, qs, dt):
    f = FMT[dt]
    T = f['np']
    pl, ps, ql, qs = (T(pl), T(ps), T(ql), T(qs))
    v = {}
    H = T(0.5)
    with np.errstate(all='ignore'):
        v['dm'] = T(pl - ql)
        fdm = bool(np.isfinite(v['dm']))
        if fdm:
            v['w'] = T(v['dm'] / qs)
            v['gm'] = T(v['w'] / qs)
        else:
            v['dmh'] = T(T(pl * H) - T(ql * H))
            v['w1'] = T(v['dmh'] / qs)
            v['w'] = T(T(2) * v['w1'])
            v['gm'] = T(T(2) * T(v['w1'] / qs))
        v['ds'] = T(ps - qs)
        v['e'] = T(v['ds'] / qs)
        fe = bool(np.isfinite(v['e']))
        ev = float(v['e'])
        inS = fe and SEG_LO <= ev <= SEG_HI
        inA = fe and ev > SEG_HI
        br = 'S' if inS else 'A' if inA else 'B' if fe else 'INF'
        qs_list = None
        if inS:
            K = KTERMS[dt]
            v['ss'] = T(ps + qs)
            fss = bool(np.isfinite(v['ss']))
            if fss:
                v['s'] = T(v['ds'] / v['ss'])
            else:
                v['ss'] = T(T(ps * H) + T(qs * H))
                v['s'] = T(T(v['ds'] * H) / v['ss'])
            v['_fss'] = fss
            v['z'] = T(v['s'] * v['s'])
            qs_list = [T(1.0 / (2 * K + 1))]
            for k in range(K - 1, 0, -1):
                qs_list.append(T(T(qs_list[-1] * v['z']) + T(1.0 / (2 * k + 1))))
            v['q'] = qs_list[-1]
            v['es'] = T(v['e'] * v['s'])
            v['sz'] = T(v['s'] * v['z'])
            v['t2'] = T(T(T(2) * v['sz']) * v['q'])
            v['phit'] = T(v['es'] - v['t2'])
        elif fe:
            v['r'] = T(ps / qs)
            rn = float(v['r']) >= f['minnorm']
            v['_rn'] = rn
            if inA:
                v['L'] = T(np.log1p(v['e']))
            elif rn:
                v['L'] = T(np.log(v['r']))
            else:
                v['L'] = T(T(np.log(ps)) - T(np.log(qs)))
            v['phit'] = T(v['e'] - v['L'])
        if fe:
            v['phi'] = max(v['phit'], T(0))
            v['eh'] = T(v['e'] * H)
            v['ee'] = T(v['eh'] * v['e'])
            v['KLs'] = T(v['phi'] + v['ee'])
            v['wh'] = T(v['w'] * H)
            v['KLm'] = T(v['wh'] * v['w'])
            v['KL'] = T(v['KLs'] + v['KLm'])
        else:
            v['KL'] = T(np.inf)
        v['ep2'] = T(v['e'] + T(2))
        v['Pe'] = T(v['e'] * v['ep2'])
        fP = bool(np.isfinite(v['Pe']))
        if fP:
            v['gps'] = T(v['Pe'] / ps)
        else:
            v['eps_'] = T(v['e'] / ps)
            v['gps'] = T(v['eps_'] * v['ep2'])
        useN = fdm and float(qs) >= float(T(ps * H)) and (float(qs) >= float(T(abs(v['dm']) * H)))
        if not useN:
            v['A1'] = T(v['Pe'] / qs) if fP else T(T(v['e'] / qs) * v['ep2'])
            v['A2'] = T(v['w'] * v['gm'])
            v['A12'] = T(v['A1'] + v['A2'])
            fA = bool(np.isfinite(v['A12']))
            if fA:
                v['gqs'] = T(T(0) - v['A12'])
            else:
                v['ww'] = T(v['w'] * v['w'])
                v['N'] = T(v['Pe'] + v['ww'])
                v['gqs'] = T(T(0) - T(v['N'] / qs))
        else:
            fA = True
        if useN:
            RN = _np_route_n([pl], [ql], [ps], [qs], [v['dm']], np.array([True]), dt)
            v['gqs'] = T(RN['g'][0])
            flatN = {key: val[0] if isinstance(val, np.ndarray) else val for key, val in RN.items() if key not in ('prods', 'PE', 'partial')}
            v['_RN'] = dict(flatN, prods=[(a[0], b[0]) for a, b in RN['prods']], PE=[(p[0], q[0]) for p, q in RN['PE']], partial=[x[0] for x in RN['partial']])
        v['gpl'] = v['gm']
        v['gql'] = T(T(0) - v['gm'])
    flags = {k: v.pop(k) for k in list(v) if k.startswith('_')}
    out = {k: float(x) for k, x in v.items()}
    out.update(flags)
    out.update({'_branch': br, '_fdm': fdm, '_fP': fP, '_useN': useN, '_fA': fA, '_in': (float(pl), float(ps), float(ql), float(qs)), '_qs': [float(t) for t in qs_list] if qs_list is not None else None, 'P.loc': out['gpl'], 'P.scale': out['gps'], 'Q.loc': out['gql'], 'Q.scale': out['gqs']})
    return out

def running_bounds(v, dt, lam):
    f = FMT[dt]
    u, h = (f['u'], f['eta'] / 2)
    mn = f['minnorm']
    sub = lambda x: 1.0 if abs(float(x)) < mn else 0.0
    OTf = float(np.finfo(f['np']).max) * (1 + u)
    A = lambda x: min(abs(float(x)), OTf)
    R = lambda x: u * A(x) + h * sub(x)
    pl, ps, ql, qs = v['_in']
    if v['_fdm']:
        e_dm = u * A(v['dm'])
        e_w = e_dm / qs + R(v['w'])
        e_gm = e_w / qs + R(v['gm'])
    else:
        e_dmh = u * A(v['dmh'])
        e_w1 = e_dmh / qs + R(v['w1'])
        e_w = 2 * e_w1
        e_gm = 2 * (e_w1 / qs + R(v['gm'] / 2))
    e_ds = u * A(v['ds'])
    e_e = e_ds / qs + R(v['e'])
    br = v['_branch']
    if br == 'INF':
        e_KL = 0.0
    else:
        if br == 'S':
            K = KTERMS[dt]
            sv, z, ss = (v['s'], v['z'], v['ss'])
            e_ss = u * A(ss)
            dd = v['ds'] if v['_fss'] else v['ds'] / 2
            e_dd = e_ds if v['_fss'] else e_ds / 2
            e_s = e_dd / ss + abs(sv) * e_ss / ss + R(sv)
            e_z = 2 * abs(sv) * e_s + R(z)
            ql_ = v['_qs']
            e_q = u * ql_[0]
            for i, k in enumerate(range(K - 1, 0, -1)):
                qprev = ql_[i]
                prod = qprev * z
                e_q = e_q * z + qprev * e_z + R(prod) + u / (2 * k + 1) + u * A(ql_[i + 1])
            e_q += q_tail_bound(z, K)
            e_es = e_e * abs(sv) + abs(v['e']) * e_s + R(v['es'])
            e_sz = e_s * z + abs(sv) * e_z + R(v['sz'])
            e_t2 = 2 * e_sz * v['q'] + 2 * abs(v['sz']) * e_q + R(v['t2'])
            e_phi = e_es + e_t2 + u * A(v['phit']) + h * sub(v['phit'])
        else:
            if br == 'A':
                e_L = e_e / (1 + v['e']) + lam * u * A(v['L'])
            elif v['_rn']:
                e_L = u * A(v['r']) / v['r'] + lam * u * A(v['L'])
            else:
                e_L = lam * u * (abs(math.log(ps)) + abs(math.log(qs))) + u * A(v['L'])
            e_phi = e_e + e_L + u * A(v['phit']) + h * sub(v['phit'])
        e_ee = abs(v['e']) * e_e + R(v['ee'])
        e_ee += abs(v['e']) * h * sub(v['eh'])
        e_KLs = e_phi + e_ee + u * A(v['KLs'])
        e_KLm = abs(v['w']) * e_w + R(v['KLm']) + abs(v['w']) * h * sub(v['wh'])
        e_KL = e_KLs + e_KLm + u * A(v['KL'])
    e_ep2 = e_e + u * A(v['ep2'])
    if v['_fP']:
        e_P = e_e * abs(v['ep2']) + abs(v['e']) * e_ep2 + R(v['Pe'])
        e_gps = e_P / ps + R(v['gps'])
    else:
        e_eps = e_e / ps + R(v['eps_'])
        e_gps = e_eps * abs(v['ep2']) + abs(v['eps_']) * e_ep2 + R(v['gps'])
    if not v['_useN']:
        if v['_fP']:
            e_A1 = e_P / qs + R(v['A1'])
        else:
            e_eqs = e_e / qs + u * A(v['e'] / qs) + h * sub(v['e'] / qs)
            e_A1 = e_eqs * abs(v['ep2']) + abs(v['e'] / qs) * e_ep2 + R(v['A1'])
        e_A2 = e_w * abs(v['gm']) + abs(v['w']) * e_gm + R(v['A2'])
        if v['_fA']:
            e_gqs = e_A1 + e_A2 + u * A(v['A12'])
        else:
            e_Pe = e_e * abs(v['ep2']) + abs(v['e']) * e_ep2 + R(v['Pe'])
            e_N2 = e_Pe + 2 * abs(v['w']) * e_w + R(v['ww']) + u * A(v['N'])
            e_gqs = e_N2 / qs + R(v['gqs'])
        S = SAFETY
        return {'KL': S * e_KL, 'P.loc': S * e_gm, 'Q.loc': S * e_gm, 'P.scale': S * e_gps, 'Q.scale': S * e_gqs}
    from fractions import Fraction as Fr
    RN = v['_RN']
    k = int(RN['k'])
    eta = f['eta']
    dn = (lambda x: h * sub(x)) if k < 0 else lambda x: 0.0
    e_ps_, e_qs_, e_dh_, e_dl_ = (dn(RN['ps_']), dn(RN['qs_']), dn(RN['dh_']), dn(RN['dl_']))
    e_N = 2 * abs(float(RN['qs_'])) * e_qs_ + 2 * abs(float(RN['ps_'])) * e_ps_ + 2 * (abs(float(RN['dh_'])) + abs(float(RN['dl_']))) * (e_dh_ + e_dl_)
    n_inexact = 0
    for (a, b), (p, er) in zip(RN['prods'], RN['PE']):
        if Fr(float(p)) + Fr(float(er)) != Fr(float(a)) * Fr(float(b)):
            n_inexact += 1
            e_N += 4 * eta
    e_N += u * sum((abs(float(x)) for x in RN['partial']))
    v['_n_inexact'] = n_inexact
    s = 2.0 ** (-int(RN['eN']))
    Qm = float(RN['Qm'])
    rq = e_qs_ / abs(float(RN['qs_']))
    e_v1 = e_N * s / Qm + abs(float(RN['v1'])) * rq + R(RN['v1'])
    e_v2 = e_v1 / Qm + abs(float(RN['v2'])) * rq + R(RN['v2'])
    e_v3 = e_v2 / Qm + abs(float(RN['v3'])) * rq + R(RN['v3'])
    e_gqs = e_v3 * 2.0 ** int(RN['E']) + h * sub(v['gqs'])
    S = SAFETY
    return {'KL': S * e_KL, 'P.loc': S * e_gm, 'Q.loc': S * e_gm, 'P.scale': S * e_gps, 'Q.scale': S * e_gqs}
