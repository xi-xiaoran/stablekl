"""Quick self-test of the installed package (CPU, under a minute)."""
import math
import mpmath as mp
import pytest
import torch
import torch.distributions as D
import stablekl

mp.mp.dps = 60


def exp_kl(a, b):          # KL(Exp(a) || Exp(b)) = r - 1 - log r, r = b / a, in 60-digit arithmetic
    r = mp.mpf(float(b)) / mp.mpf(float(a))
    return r - 1 - mp.log(r)


@pytest.mark.parametrize("dt", [torch.float16, torch.float32])
def test_exponential_small_kl_is_correctly_rounded(dt):
    a = torch.full((1,), 1.0, dtype=dt)
    for k in range(1, 12):
        b = (torch.ones(1, dtype=torch.float64) * (1 + 2.0 ** -k)).to(dt)
        if b == a:
            continue
        got = stablekl.kl_divergence(D.Exponential(a), D.Exponential(b))
        want = torch.tensor(float(exp_kl(a, b)), dtype=torch.float64).to(torch.float32).to(dt) if dt == torch.float32 else None
        if want is not None:
            assert got.item() == want.item()
        assert got.item() > 0


@pytest.mark.parametrize("mode", ["ours", "rewrite"])
def test_identical_distributions_give_exact_zero(mode):
    loc = torch.randn(64, requires_grad=True)
    scale = torch.rand(64).add(0.5).requires_grad_(True)
    kl = stablekl.kl_divergence(D.Normal(loc, scale), D.Normal(loc, scale), mode=mode)
    assert torch.all(kl == 0)
    kl.sum().backward()
    assert torch.all(loc.grad == 0) and torch.all(scale.grad == 0)


@pytest.mark.parametrize("fam", ["normal", "gamma", "beta", "bernoulli", "categorical", "dirichlet", "mvn"])
def test_agrees_with_float64_official_for_large_kl(fam):
    g = torch.Generator().manual_seed(0)
    u = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    mk = {
        "normal": lambda: D.Normal(u(32) * 4 - 2, u(32) + 0.5),
        "gamma": lambda: D.Gamma(u(32) * 3 + 0.5, u(32) + 0.5),
        "beta": lambda: D.Beta(u(32) * 3 + 0.5, u(32) * 3 + 0.5),
        "bernoulli": lambda: D.Bernoulli(probs=u(32) * 0.9 + 0.05),
        "categorical": lambda: D.Categorical(probs=u(32, 5) + 0.1),
        "dirichlet": lambda: D.Dirichlet(u(32, 4) * 3 + 0.5),
        "mvn": lambda: D.MultivariateNormal(u(32, 3), scale_tril=torch.diag_embed(u(32, 3) + 0.5) + torch.tril(u(32, 3, 3), -1) * 0.3),
    }[fam]
    p, q = mk(), mk()
    ref = D.kl_divergence(p, q)
    got = stablekl.kl_divergence(p, q, mode="rewrite")
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_kl_div_matches_official_where_official_is_accurate():
    x = torch.log_softmax(torch.randn(8, 10, dtype=torch.float64), -1)
    t = torch.softmax(torch.randn(8, 10, dtype=torch.float64), -1)
    for red in ("none", "sum", "mean", "batchmean"):
        assert torch.allclose(stablekl.kl_div(x.float(), t.float(), reduction=red).double(),
                              torch.nn.functional.kl_div(x, t, reduction=red), rtol=1e-6, atol=1e-7)


def test_kl_div_is_exactly_zero_for_equal_log_inputs():
    x = torch.log_softmax(torch.randn(256, 64), -1)
    for red in ("none", "sum", "batchmean"):
        xi = x.clone().requires_grad_(True)
        v = stablekl.kl_div(xi, x.clone(), reduction=red, log_target=True)
        (v.sum() if red == "none" else v).backward()
        assert torch.all(v == 0)
        g_exact = -(x.double().exp()) / (x.shape[0] if red == "batchmean" else 1)        # d/dx of e^t (t - x) at x = t
        assert torch.equal(xi.grad, g_exact.float())


def test_logits_kl_is_zero_with_zero_gradient_at_identity():
    a = torch.randn(4, 1000, dtype=torch.bfloat16)
    c = a.clone().requires_grad_(True)
    v = stablekl.logits_kl(a, c)
    v.backward()
    assert v.item() == 0 and torch.all(c.grad == 0)
