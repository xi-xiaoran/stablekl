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


def _exact_logits_kl(a, c):
    mp.mp.dps = 80
    a, c = [mp.mpf(float(v)) for v in a], [mp.mpf(float(v)) for v in c]
    la, lc = mp.log(mp.fsum(mp.exp(v - max(a)) for v in a)) + max(a), mp.log(mp.fsum(mp.exp(v - max(c)) for v in c)) + max(c)
    p, q = [mp.exp(v - la) for v in a], [mp.exp(v - lc) for v in c]
    return mp.fsum(pi * ((x - la) - (y - lc)) for pi, x, y in zip(p, a, c)), [qi - pi for pi, qi in zip(p, q)]


@pytest.mark.parametrize("a,c", [
    ([0.0, 2.0 ** -55], [1.0, 1.0]),                 # the float64 difference a - c is inexact
    ([1000.3, 998.8, 1002.0], [0.3, -1.2, 2.0]),     # common offset +1000: expm1 overflows without centring
    ([-29.7, -31.2, -28.0], [0.3, -1.2, 2.0]),       # common offset -30: 1 + sum q expm1 cancels without centring
    ([0.0, 0.0], [0.0, -2000.0]),                    # distributions far apart
    ([0.0, -40.0], [0.0, -1000.0]),                  # a rare category with a large difference; KL is 4e-15
    ([0.0, -60.0, 1.0], [0.0, -760.0, 1.0]),         # the student probability underflows, the teacher's does not
])
def test_logits_kl_guarded_rows(a, c):
    at = torch.tensor([a], dtype=torch.float32)
    ct = torch.tensor([c], dtype=torch.float32, requires_grad=True)
    out = stablekl.logits_kl(at, ct)
    out.backward()
    kl, grad = _exact_logits_kl(at[0], ct[0].detach())
    assert out.item() == torch.tensor(float(kl), dtype=torch.float64).to(torch.float32).item()
    for g, t in zip(ct.grad[0].tolist(), grad):
        assert g == pytest.approx(float(t), rel=1e-6, abs=1e-44)


def test_logits_kl_is_exactly_zero_for_a_common_offset():
    c = torch.tensor([[0.5, -1.0, 2.0]], requires_grad=True)
    for offset in (-40.0, -0.75, 8.0, 1024.0):
        c.grad = None
        out = stablekl.logits_kl(c.detach() + offset, c)
        out.backward()
        assert out.item() == 0.0 and not c.grad.abs().any()


@pytest.mark.parametrize("mode", ["ours", "rewrite"])
@pytest.mark.parametrize("dt", [torch.float16, torch.float32])
def test_categorical_with_zero_probabilities_in_p(mode, dt):
    ln2 = torch.tensor(math.log(2.0), dtype=torch.float64).to(dt).item()
    p, q = D.Categorical(probs=torch.tensor([1.0, 0.0], dtype=dt)), D.Categorical(probs=torch.tensor([0.5, 0.5], dtype=dt))
    assert stablekl.kl_divergence(p, q, mode=mode).item() == ln2
    p, q = D.Categorical(logits=torch.tensor([0.0, -math.inf], dtype=dt)), D.Categorical(logits=torch.tensor([0.0, 0.0], dtype=dt))
    assert stablekl.kl_divergence(p, q, mode=mode).item() == ln2
    p = D.Categorical(probs=torch.tensor([[0.3, 0.7, 0.0], [0.2, 0.3, 0.5], [1.0, 0.0, 0.0]], dtype=dt))
    q = D.Categorical(probs=torch.tensor([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4], [0.0, 0.5, 0.5]], dtype=dt))
    out = stablekl.kl_divergence(p, q, mode=mode)
    ref = D.kl_divergence(D.Categorical(probs=p.probs.double()), D.Categorical(probs=q.probs.double()))
    assert out[0].item() == pytest.approx(ref[0].item(), rel=4e-3 if dt == torch.float16 else 1e-6)
    assert out[1].item() == pytest.approx(ref[1].item(), rel=4e-3 if dt == torch.float16 else 1e-6)
    assert math.isinf(out[2].item()) and out[2].item() > 0
