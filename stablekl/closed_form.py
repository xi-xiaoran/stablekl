"""Repaired closed forms of ``torch.distributions.kl_divergence`` for the 18 same-family pairs.

    kl_divergence(p, q, mode="ours")

``p`` and ``q`` are ``torch.distributions`` objects of the same family.  The result has the batch shape of the
broadcast parameters and supports autograd with respect to the parameters of both distributions.

mode="ours"     the repaired formula evaluated in float64 and rounded once to the dtype of the parameters
                (float16 or float32); gradients are rounded once in the same way.  This is the method evaluated
                in the paper as OURS.
mode="rewrite"  the repaired formula evaluated in the dtype of the parameters, without a wider type.
"""
from __future__ import annotations
import importlib
import torch
import torch.distributions as D
from ._round import Narrow, Widen, SUPPORTED

_MODULES = {}


def _mod(name):
    if name not in _MODULES:
        _MODULES[name] = importlib.import_module(f"stablekl._closed.{name}")
    return _MODULES[name]


def _given(d, a, b):
    """Which of two alternative parameterisations (e.g. probs / logits) the distribution was constructed with."""
    if hasattr(d, "_param"):
        return a if d._param is d.__dict__.get(a) else b
    has_a, has_b = a in d.__dict__, b in d.__dict__
    return b if has_b and not has_a else a


_TARGET = {torch.float16: "fp16", torch.float32: "fp32"}

# family -> (module, parameter names, function factory, takes target dtype)
_SCALAR = {
    D.Exponential: ("exponential", ("rate",), lambda m: m.ExponentialKLC, False),
    D.Poisson: ("poisson", ("rate",), lambda m: m.PoissonKLC, False),
    D.Normal: ("normal", ("loc", "scale"), lambda m: m.NormalKLC, False),
    D.Laplace: ("laplace", ("loc", "scale"), lambda m: m.LaplaceKLC, False),
    D.Gumbel: ("gumbel", ("loc", "scale"), lambda m: m.GumbelKLC, False),
    D.Cauchy: ("cauchy", ("loc", "scale"), lambda m: m.CauchyKLC, False),
    D.Gamma: ("gamma", ("concentration", "rate"), lambda m: m.GammaKLC, False),
    D.Beta: ("beta", ("concentration1", "concentration0"), lambda m: m.BetaKLC, False),
    D.Pareto: ("pareto", ("scale", "alpha"), lambda m: m.ParetoKLC, False),
    D.Uniform: ("uniform", ("low", "high"), lambda m: m.UniformKLC, False),
}
_TWO_PARAM = {   # family -> {parameterisation: (module, function factory, takes target dtype)}
    D.Bernoulli: {"probs": ("bernoulli_probs", lambda m: m.BernoulliProbsKLC, False), "logits": ("bernoulli_logits", lambda m: m.BernoulliLogitsKLC, True)},
    D.Geometric: {"probs": ("geometric_probs", lambda m: m.GeometricProbsKLC, False), "logits": ("geometric_logits", lambda m: m.GeometricLogitsKLC, True)},
    D.ContinuousBernoulli: {"probs": ("contbern", lambda m: m.KLC["probs"], True), "logits": ("contbern", lambda m: m.KLC["logits"], True)},
}


def _check(p, q):
    if type(p) is not type(q):
        raise NotImplementedError(f"stablekl repairs same-family pairs only, got {type(p).__name__} and {type(q).__name__}")


def _prepare(tensors, mode):
    """Broadcast, check the dtype, and widen to float64 for mode 'ours'."""
    tensors = torch.broadcast_tensors(*tensors)
    dt = tensors[0].dtype
    if any(t.dtype != dt for t in tensors):
        raise TypeError("all parameters must have the same dtype")
    if dt not in SUPPORTED:
        raise TypeError(f"supported parameter dtypes are float16, float32 and float64, got {dt}")
    if mode == "ours" and dt != torch.float64:
        tensors = [Widen.apply(t) for t in tensors]
    elif mode not in ("ours", "rewrite"):
        raise ValueError("mode must be 'ours' or 'rewrite'")
    return list(tensors), dt


def _finish(v, dt, mode, shape):
    if mode == "ours" and dt != torch.float64:
        v = Narrow.apply(v, dt)
    return v.reshape(shape)


def kl_divergence(p, q, mode="ours"):
    """Repaired ``torch.distributions.kl_divergence(p, q)``; see the module docstring for ``mode``."""
    _check(p, q)
    fam = type(p)
    if fam in _SCALAR:
        name, params, fn, _ = _SCALAR[fam]
        ts, dt = _prepare([getattr(p, k) for k in params] + [getattr(q, k) for k in params], mode)
        shape = ts[0].shape
        out = fn(_mod(name)).apply(*[t.reshape(-1) for t in ts])
        return _finish(out, dt, mode, shape)
    if fam in _TWO_PARAM:
        kind = _given(p, "probs", "logits")
        name, fn, wants_target = _TWO_PARAM[fam][kind]
        ts, dt = _prepare([getattr(p, kind), getattr(q, kind)], mode)
        shape = ts[0].shape
        args = [t.reshape(-1) for t in ts]
        if wants_target:
            args.append(_TARGET.get(dt) if mode == "ours" else None)
        return _finish(fn(_mod(name)).apply(*args), dt, mode, shape)
    if fam is D.Binomial:
        if not torch.equal(*torch.broadcast_tensors(p.total_count, q.total_count)):
            raise NotImplementedError("Binomial pairs with different total_count are not repaired")
        kind = _given(p, "probs", "logits")
        m = _mod("binomial_" + kind)
        ts, dt = _prepare([getattr(p, kind), getattr(q, kind)], mode)
        shape = ts[0].shape
        n = torch.broadcast_to(p.total_count, shape).to(ts[0].dtype).reshape(-1)
        if kind == "probs":
            out = m.BinomialProbsKLC.apply(n, ts[0].reshape(-1), ts[1].reshape(-1))
        else:
            out = m.BinomialLogitsKLC.apply(n, ts[0].reshape(-1), ts[1].reshape(-1), _TARGET.get(dt) if mode == "ours" else None)
        return _finish(out, dt, mode, shape)
    if fam is D.Categorical:
        kind = _given(p, "probs", "logits")
        ts, dt = _prepare([getattr(p, kind), getattr(q, kind)], mode)
        shape, K = ts[0].shape[:-1], ts[0].shape[-1]
        out = _mod("categorical").KLC[kind].apply(ts[0].reshape(-1, K), ts[1].reshape(-1, K))
        return _finish(out, dt, mode, shape)
    if fam is D.Dirichlet:
        ts, dt = _prepare([p.concentration, q.concentration], mode)
        shape, K = ts[0].shape[:-1], ts[0].shape[-1]
        out = _mod("dirichlet").DirichletKLC.apply(ts[0].reshape(-1, K), ts[1].reshape(-1, K))
        return _finish(out, dt, mode, shape)
    if fam in (D.MultivariateNormal, D.LowRankMultivariateNormal):
        m = _mod("mvn")
        if fam is D.MultivariateNormal:
            d = p.loc.shape[-1]
            parts = [p.loc, p.scale_tril.flatten(-2), q.loc, q.scale_tril.flatten(-2)]
            kind, rank, sizes = "mvn", None, (d, d * d)
        else:
            d, rank = p.cov_factor.shape[-2], p.cov_factor.shape[-1]
            if q.cov_factor.shape[-1] != rank:
                raise NotImplementedError("LowRankMultivariateNormal pairs with different ranks are not repaired")
            parts = [p.loc, p.cov_factor.flatten(-2), p.cov_diag, q.loc, q.cov_factor.flatten(-2), q.cov_diag]
            kind, sizes = "lowrank", (d, d * rank, d)
        if parts[0].dtype == torch.float16:
            raise TypeError("float16 multivariate normals are not supported (PyTorch has no float16 linear algebra)")
        batch = torch.broadcast_shapes(*[t.shape[:-1] for t in parts])
        parts = [t.expand(*batch, t.shape[-1]) for t in parts]
        ts, dt = _prepare_mv(parts, mode)
        k = len(sizes)
        A = torch.cat([t.reshape(-1, s) for t, s in zip(ts[:k], sizes)], dim=1)
        C = torch.cat([t.reshape(-1, s) for t, s in zip(ts[k:], sizes)], dim=1)
        key = (kind, d, rank)
        if key not in m._FCACHE:
            m._FCACHE[key] = m._make(m._ALGOS[kind], d, rank)
        return _finish(m._FCACHE[key].apply(A, C), dt, mode, batch)
    raise NotImplementedError(f"no repaired closed form for {fam.__name__}")


def _prepare_mv(tensors, mode):
    dt = tensors[0].dtype
    if any(t.dtype != dt for t in tensors):
        raise TypeError("all parameters must have the same dtype")
    if mode == "ours" and dt != torch.float64:
        return [Widen.apply(t) for t in tensors], dt
    if mode not in ("ours", "rewrite"):
        raise ValueError("mode must be 'ours' or 'rewrite'")
    return list(tensors), dt
