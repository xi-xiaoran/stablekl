"""Repaired ``torch.nn.functional.kl_div`` (interface B).

    kl_div(input, target, size_average=None, reduce=None, reduction="mean", log_target=False, mode="ours")

Same arguments and semantics as ``torch.nn.functional.kl_div``.

mode="ours"     inputs raised exactly to float64, merged formulas and analytic gradients, every output rounded
                once; a rounding guard falls back to exact interval evaluation (mpmath) where the fast path cannot
                decide the rounding, so every output is the correctly rounded exact value.  Raises
                ``UncertifiedRounding`` instead of guessing if even the interval evaluation cannot decide.
mode="ours32"   the same with float32 internals, intended for float16 inputs (cheaper; not correctly rounded in
                every case).
mode="rewrite"  no wider type: the official element formula with repaired backward, scaling and boundary handling.

Scope: float16 and float32 inputs of equal dtype.  Elements outside the domain (NaN, +inf, negative targets with
log_target=False) keep the official value and gradient.  Empty tensors and other dtypes are passed to the official
function unchanged."""
from . import interface_b as _impl

UncertifiedRounding = _impl.UncertifiedRounding
_METHOD = {"ours": "D64", "ours32": "D32", "rewrite": "C"}


def kl_div(input, target, size_average=None, reduce=None, reduction="mean", log_target=False, mode="ours"):
    if mode not in _METHOD:
        raise ValueError("mode must be 'ours', 'ours32' or 'rewrite'")
    return _impl.kl_div(input, target, size_average=size_average, reduce=reduce, reduction=reduction,
                        log_target=log_target, method=_METHOD[mode])
