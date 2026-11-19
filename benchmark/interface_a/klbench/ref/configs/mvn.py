from __future__ import annotations
from fractions import Fraction
from typing import Any
import mpmath as mp
from flint import arb
from ..arbutil import arb_bounds, arb_precision, fraction_to_arb
Params = dict[str, list[Fraction]]
Mat = list[list[Fraction]]

def _zeros(n: int, m: int) -> Mat:
    return [[Fraction(0)] * m for _ in range(n)]

def _eye(n: int) -> Mat:
    out = _zeros(n, n)
    for i in range(n):
        out[i][i] = Fraction(1)
    return out

def _matmul(a: Mat, b: Mat) -> Mat:
    n, m, r = (len(a), len(b), len(b[0]))
    return [[sum((a[i][t] * b[t][j] for t in range(m)), Fraction(0)) for j in range(r)] for i in range(n)]

def _transpose(a: Mat) -> Mat:
    return [list(row) for row in zip(*a)]

def _lower_solve(lower: Mat, rhs: Mat) -> Mat:
    n, m = (len(lower), len(rhs[0]))
    x = _zeros(n, m)
    for j in range(m):
        for i in range(n):
            acc = rhs[i][j] - sum((lower[i][t] * x[t][j] for t in range(i)), Fraction(0))
            x[i][j] = acc / lower[i][i]
    return x

def _upper_solve_transposed(lower: Mat, rhs: Mat) -> Mat:
    n, m = (len(lower), len(rhs[0]))
    x = _zeros(n, m)
    for j in range(m):
        for i in range(n - 1, -1, -1):
            acc = rhs[i][j] - sum((lower[t][i] * x[t][j] for t in range(i + 1, n)), Fraction(0))
            x[i][j] = acc / lower[i][i]
    return x

def _inverse_det(a: Mat) -> tuple[Mat, Fraction]:
    n = len(a)
    aug = [list(row) + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    det = Fraction(1)
    for col in range(n):
        piv = next((r for r in range(col, n) if aug[r][col] != 0))
        if piv != col:
            aug[col], aug[piv] = (aug[piv], aug[col])
            det = -det
        pv = aug[col][col]
        det *= pv
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return ([row[n:] for row in aug], det)

def _mp(x: Fraction) -> Any:
    return mp.mpf(x.numerator) / mp.mpf(x.denominator)

def _arb_mat(a: Mat) -> list[list[arb]]:
    return [[fraction_to_arb(v) for v in row] for row in a]

def _arb_cholesky(sigma: list[list[arb]]) -> list[list[arb]]:
    n = len(sigma)
    low = [[arb(0)] * n for _ in range(n)]
    for j in range(n):
        s = sigma[j][j] - sum((low[j][t] * low[j][t] for t in range(j)), arb(0))
        if not s > 0:
            raise ValueError('Cholesky pivot not proven positive')
        low[j][j] = s.sqrt()
        for i in range(j + 1, n):
            low[i][j] = (sigma[i][j] - sum((low[i][t] * low[j][t] for t in range(j)), arb(0))) / low[j][j]
    return low

def _arb_lower_solve(low, rhs):
    n, m = (len(low), len(rhs[0]))
    x = [[arb(0)] * m for _ in range(n)]
    for j in range(m):
        for i in range(n):
            x[i][j] = (rhs[i][j] - sum((low[i][t] * x[t][j] for t in range(i)), arb(0))) / low[i][i]
    return x

def _arb_upper_solve_transposed(low, rhs):
    n, m = (len(low), len(rhs[0]))
    x = [[arb(0)] * m for _ in range(n)]
    for j in range(m):
        for i in range(n - 1, -1, -1):
            x[i][j] = (rhs[i][j] - sum((low[t][i] * x[t][j] for t in range(i + 1, n)), arb(0))) / low[i][i]
    return x

def _arb_matmul(a, b):
    n, m, r = (len(a), len(b), len(b[0]))
    return [[sum((a[i][t] * b[t][j] for t in range(m)), arb(0)) for j in range(r)] for i in range(n)]

def _arb_inverse_from_cholesky(low):
    n = len(low)
    ident = [[arb(int(i == j)) for j in range(n)] for i in range(n)]
    return _arb_upper_solve_transposed(low, _arb_lower_solve(low, ident))

def _arb_kl_and_parts(sig_p, sig_q, delta):
    d = len(sig_p)
    lp, lq = (_arb_cholesky(sig_p), _arb_cholesky(sig_q))
    inv_q = _arb_inverse_from_cholesky(lq)
    inv_p = _arb_inverse_from_cholesky(lp)
    trace = sum((inv_q[i][j] * sig_p[j][i] for i in range(d) for j in range(d)), arb(0))
    quad = sum((delta[i] * inv_q[i][j] * delta[j] for i in range(d) for j in range(d)), arb(0))
    logdet = 2 * sum((lq[i][i].log() - lp[i][i].log() for i in range(d)), arb(0))
    kl = (trace + quad - d + logdet) / 2
    return (kl, inv_q, inv_p, lq)

def _flat(m) -> list:
    return [v for row in m for v in row]

def make_mvn(d: int):

    class MVN:
        D = d
        PARAM_SPEC = {'loc': {'shape': (d,), 'differentiable': True, 'domain': 'finite real'}, 'scale_tril': {'shape': (d, d), 'differentiable': True, 'domain': 'lower triangular, positive diagonal (row-major)'}}

        @staticmethod
        def _unpack(params: Params):
            if set(params) != {'loc', 'scale_tril'}:
                raise ValueError('Expected loc and scale_tril')
            loc, flat = (list(params['loc']), list(params['scale_tril']))
            if len(loc) != d or len(flat) != d * d:
                raise ValueError(f'Expected loc ({d},) and scale_tril ({d},{d})')
            return (loc, [flat[i * d:(i + 1) * d] for i in range(d)])

        @staticmethod
        def domain_ok(p: Params, q: Params) -> bool | str:
            try:
                for side in (p, q):
                    _, low = MVN._unpack(side)
                    if any((low[i][j] != 0 for i in range(d) for j in range(i + 1, d))):
                        return 'scale_tril upper triangle must be exactly zero'
                    if any((low[i][i] <= 0 for i in range(d))):
                        return 'scale_tril diagonal must be positive'
            except (KeyError, ValueError) as exc:
                return str(exc)
            return True

        @staticmethod
        def semantic_equal(p: Params, q: Params) -> bool:
            return list(p['loc']) == list(q['loc']) and list(p['scale_tril']) == list(q['scale_tril'])

        @staticmethod
        def _exact(p: Params, q: Params):
            mp_, lp = MVN._unpack(p)
            mq, lq = MVN._unpack(q)
            delta = [[a - b] for a, b in zip(mp_, mq)]
            m = _lower_solve(lq, lp)
            v = _lower_solve(lq, delta)
            t = sum((x * x for row in m for x in row), Fraction(0)) + sum((row[0] * row[0] for row in v), Fraction(0)) - d
            prod_p = Fraction(1)
            prod_q = Fraction(1)
            for i in range(d):
                prod_p *= lp[i][i]
                prod_q *= lq[i][i]
            return (mp_, lp, mq, lq, m, v, t, prod_p, prod_q)

        @staticmethod
        def exact_kl(p: Params, q: Params) -> Fraction | None:
            *_, t, prod_p, prod_q = MVN._exact(p, q)
            return t / 2 if prod_p == prod_q else None

        @staticmethod
        def exact_grad(p: Params, q: Params):
            _, lp, _, lq, m, v, _, _, _ = MVN._exact(p, q)
            g_mu = _upper_solve_transposed(lq, v)
            gp_full = _upper_solve_transposed(lq, m)
            mmt = _matmul(m, _transpose(m))
            inner = [[int(i == j) - mmt[i][j] - v[i][0] * v[j][0] for j in range(d)] for i in range(d)]
            gq_full = _upper_solve_transposed(lq, inner)
            gp = _zeros(d, d)
            gq = _zeros(d, d)
            for i in range(d):
                for j in range(i + 1):
                    gp[i][j] = gp_full[i][j] - (1 / lp[i][i] if i == j else 0)
                    gq[i][j] = gq_full[i][j]
            nil = lambda g: [g[i][j] if j <= i else None for i in range(d) for j in range(d)]
            return {'P': {'loc': [row[0] for row in g_mu], 'scale_tril': nil(gp)}, 'Q': {'loc': [-row[0] for row in g_mu], 'scale_tril': nil(gq)}}

        @staticmethod
        def exact_zero_mask(p: Params, q: Params):
            same = MVN.semantic_equal(p, q)
            tri = [same and j <= i for i in range(d) for j in range(d)]
            return {'P': {'loc': [same] * d, 'scale_tril': list(tri)}, 'Q': {'loc': [same] * d, 'scale_tril': list(tri)}}

        @staticmethod
        def nondiff_mask(p: Params, q: Params):
            upper = [j > i for i in range(d) for j in range(d)]
            return {'P': {'loc': [False] * d, 'scale_tril': list(upper)}, 'Q': {'loc': [False] * d, 'scale_tril': list(upper)}}

        @staticmethod
        def _arb_sigmas(p: Params, q: Params):
            mp_, lp = MVN._unpack(p)
            mq, lq = MVN._unpack(q)
            alp, alq = (_arb_mat(lp), _arb_mat(lq))
            sig_p = _arb_matmul(alp, [list(r) for r in zip(*alp)])
            sig_q = _arb_matmul(alq, [list(r) for r in zip(*alq)])
            delta = [fraction_to_arb(a - b) for a, b in zip(mp_, mq)]
            return (sig_p, sig_q, delta, alp, alq)

        @staticmethod
        def kl_arb(p: Params, q: Params, prec: int) -> arb:
            with arb_precision(prec):
                sig_p, sig_q, delta, _, _ = MVN._arb_sigmas(p, q)
                return _arb_kl_and_parts(sig_p, sig_q, delta)[0]

        @staticmethod
        def grad_arb(p: Params, q: Params, prec: int):
            with arb_precision(prec):
                sig_p, sig_q, delta, alp, alq = MVN._arb_sigmas(p, q)
                _, inv_q, _, _ = _arb_kl_and_parts(sig_p, sig_q, delta)
                g_mu = [sum((inv_q[i][j] * delta[j] for j in range(d)), arb(0)) for i in range(d)]
                s = [[sig_p[i][j] + delta[i] * delta[j] for j in range(d)] for i in range(d)]
                asa = _arb_matmul(_arb_matmul(inv_q, s), inv_q)
                gp_full = _arb_matmul(inv_q, alp)
                gq_full = _arb_matmul(asa, alq)
                gp = [[arb(0)] * d for _ in range(d)]
                gq = [[arb(0)] * d for _ in range(d)]
                for i in range(d):
                    for j in range(i + 1):
                        gp[i][j] = gp_full[i][j] - (1 / alp[i][i] if i == j else 0)
                        gq[i][j] = -gq_full[i][j] + (1 / alq[i][i] if i == j else 0)
                return {'P': {'loc': g_mu, 'scale_tril': _flat(gp)}, 'Q': {'loc': [-g for g in g_mu], 'scale_tril': _flat(gq)}}

        @staticmethod
        def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
            with mp.workdps(dps):
                mp_, lp = MVN._unpack(p)
                mq, lq = MVN._unpack(q)
                Lp = mp.matrix([[_mp(v) for v in row] for row in lp])
                Lq = mp.matrix([[_mp(v) for v in row] for row in lq])
                return _mp_kl(Lp * Lp.T, Lq * Lq.T, mp.matrix([_mp(a - b) for a, b in zip(mp_, mq)]), d)

        @staticmethod
        def kl_definition_check(p: Params, q: Params, dps: int=60) -> Any:
            return MVN.kl_mpmath_alt(p, q, dps)
    MVN.__name__ = MVN.__qualname__ = f'MVNd{d}'
    return MVN

def _mp_kl(sig_p, sig_q, delta, d) -> Any:
    inv_q = mp.inverse(sig_q)
    trace = sum(((inv_q * sig_p)[i, i] for i in range(d)), mp.mpf(0))
    quad = (delta.T * inv_q * delta)[0, 0]
    return (trace + quad - d + mp.log(mp.det(sig_q)) - mp.log(mp.det(sig_p))) / 2

def make_lowrank(d: int, r: int):

    class LowRankMVN:
        D = d
        R = r
        PARAM_SPEC = {'loc': {'shape': (d,), 'differentiable': True, 'domain': 'finite real'}, 'cov_factor': {'shape': (d, r), 'differentiable': True, 'domain': 'finite real (row-major)'}, 'cov_diag': {'shape': (d,), 'differentiable': True, 'domain': 'positive'}}

        @staticmethod
        def _unpack(params: Params):
            if set(params) != {'loc', 'cov_factor', 'cov_diag'}:
                raise ValueError('Expected loc, cov_factor, cov_diag')
            loc, w, diag = (list(params['loc']), list(params['cov_factor']), list(params['cov_diag']))
            if len(loc) != d or len(w) != d * r or len(diag) != d:
                raise ValueError(f'Expected loc ({d},), cov_factor ({d},{r}), cov_diag ({d},)')
            return (loc, [w[i * r:(i + 1) * r] for i in range(d)], diag)

        @staticmethod
        def _sigma(w: Mat, diag: list[Fraction]) -> Mat:
            out = _matmul(w, _transpose(w))
            for i in range(d):
                out[i][i] += diag[i]
            return out

        @staticmethod
        def domain_ok(p: Params, q: Params) -> bool | str:
            try:
                for side in (p, q):
                    _, _, diag = LowRankMVN._unpack(side)
                    if any((v <= 0 for v in diag)):
                        return 'cov_diag must be positive'
            except (KeyError, ValueError) as exc:
                return str(exc)
            return True

        @staticmethod
        def semantic_equal(p: Params, q: Params) -> bool:
            mp_, wp, dp = LowRankMVN._unpack(p)
            mq, wq, dq = LowRankMVN._unpack(q)
            return mp_ == mq and LowRankMVN._sigma(wp, dp) == LowRankMVN._sigma(wq, dq)

        @staticmethod
        def _exact(p: Params, q: Params):
            mp_, wp, dp = LowRankMVN._unpack(p)
            mq, wq, dq = LowRankMVN._unpack(q)
            sp, sq = (LowRankMVN._sigma(wp, dp), LowRankMVN._sigma(wq, dq))
            inv_q, det_q = _inverse_det(sq)
            inv_p, det_p = _inverse_det(sp)
            delta = [a - b for a, b in zip(mp_, mq)]
            trace = sum((inv_q[i][j] * sp[j][i] for i in range(d) for j in range(d)), Fraction(0))
            quad = sum((delta[i] * inv_q[i][j] * delta[j] for i in range(d) for j in range(d)), Fraction(0))
            return (mp_, wp, mq, wq, sp, sq, inv_p, inv_q, det_p, det_q, delta, trace + quad - d)

        @staticmethod
        def exact_kl(p: Params, q: Params) -> Fraction | None:
            *_, det_p, det_q, _, t = LowRankMVN._exact(p, q)
            return t / 2 if det_p == det_q else None

        @staticmethod
        def exact_grad(p: Params, q: Params):
            _, wp, _, wq, sp, _, inv_p, inv_q, _, _, delta, _ = LowRankMVN._exact(p, q)
            g_mu = [sum((inv_q[i][j] * delta[j] for j in range(d)), Fraction(0)) for i in range(d)]
            s = [[sp[i][j] + delta[i] * delta[j] for j in range(d)] for i in range(d)]
            asa = _matmul(_matmul(inv_q, s), inv_q)
            g_sp = [[(inv_q[i][j] - inv_p[i][j]) / 2 for j in range(d)] for i in range(d)]
            g_sq = [[(inv_q[i][j] - asa[i][j]) / 2 for j in range(d)] for i in range(d)]
            gw_p = _matmul(g_sp, wp)
            gw_q = _matmul(g_sq, wq)
            return {'P': {'loc': g_mu, 'cov_factor': [2 * v for v in _flat(gw_p)], 'cov_diag': [g_sp[i][i] for i in range(d)]}, 'Q': {'loc': [-g for g in g_mu], 'cov_factor': [2 * v for v in _flat(gw_q)], 'cov_diag': [g_sq[i][i] for i in range(d)]}}

        @staticmethod
        def exact_zero_mask(p: Params, q: Params):
            same = LowRankMVN.semantic_equal(p, q)
            return {side: {'loc': [same] * d, 'cov_factor': [same] * (d * r), 'cov_diag': [same] * d} for side in ('P', 'Q')}

        @staticmethod
        def _arb_parts(p: Params, q: Params):
            mp_, wp, dp = LowRankMVN._unpack(p)
            mq, wq, dq = LowRankMVN._unpack(q)
            awp, awq = (_arb_mat(wp), _arb_mat(wq))

            def sig(aw, diag):
                out = _arb_matmul(aw, [list(c) for c in zip(*aw)])
                for i in range(d):
                    out[i][i] = out[i][i] + fraction_to_arb(diag[i])
                return out
            sig_p, sig_q = (sig(awp, dp), sig(awq, dq))
            delta = [fraction_to_arb(a - b) for a, b in zip(mp_, mq)]
            return (sig_p, sig_q, delta, awp, awq)

        @staticmethod
        def kl_arb(p: Params, q: Params, prec: int) -> arb:
            with arb_precision(prec):
                sig_p, sig_q, delta, _, _ = LowRankMVN._arb_parts(p, q)
                return _arb_kl_and_parts(sig_p, sig_q, delta)[0]

        @staticmethod
        def grad_arb(p: Params, q: Params, prec: int):
            with arb_precision(prec):
                sig_p, sig_q, delta, awp, awq = LowRankMVN._arb_parts(p, q)
                _, inv_q, inv_p, _ = _arb_kl_and_parts(sig_p, sig_q, delta)
                g_mu = [sum((inv_q[i][j] * delta[j] for j in range(d)), arb(0)) for i in range(d)]
                s = [[sig_p[i][j] + delta[i] * delta[j] for j in range(d)] for i in range(d)]
                asa = _arb_matmul(_arb_matmul(inv_q, s), inv_q)
                g_sp = [[(inv_q[i][j] - inv_p[i][j]) / 2 for j in range(d)] for i in range(d)]
                g_sq = [[(inv_q[i][j] - asa[i][j]) / 2 for j in range(d)] for i in range(d)]
                gw_p = _arb_matmul(g_sp, awp)
                gw_q = _arb_matmul(g_sq, awq)
                return {'P': {'loc': g_mu, 'cov_factor': [2 * v for v in _flat(gw_p)], 'cov_diag': [g_sp[i][i] for i in range(d)]}, 'Q': {'loc': [-g for g in g_mu], 'cov_factor': [2 * v for v in _flat(gw_q)], 'cov_diag': [g_sq[i][i] for i in range(d)]}}

        @staticmethod
        def kl_mpmath_alt(p: Params, q: Params, dps: int) -> Any:
            with mp.workdps(dps):
                mp_, wp, dp = LowRankMVN._unpack(p)
                mq, wq, dq = LowRankMVN._unpack(q)

                def sig(w, diag):
                    wm = mp.matrix([[_mp(v) for v in row] for row in w])
                    return wm * wm.T + mp.diag([_mp(v) for v in diag])
                return _mp_kl(sig(wp, dp), sig(wq, dq), mp.matrix([_mp(a - b) for a, b in zip(mp_, mq)]), d)

        @staticmethod
        def kl_definition_check(p: Params, q: Params, dps: int=60) -> Any:
            return LowRankMVN.kl_mpmath_alt(p, q, dps)
    LowRankMVN.__name__ = LowRankMVN.__qualname__ = f'LowRankMVNd{d}r{r}'
    return LowRankMVN
