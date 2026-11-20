"""stablekl: numerically repaired KL divergences for PyTorch.

    stablekl.kl_divergence(p, q, mode="ours")      # torch.distributions.kl_divergence, 18 same-family pairs
    stablekl.kl_div(input, target, reduction="mean", log_target=False, mode="ours")   # torch.nn.functional.kl_div
    stablekl.logits_kl(teacher_logits, student_logits)   # KL(softmax(teacher) || softmax(student)), mean over rows
"""
from .closed_form import kl_divergence
from .api_b import kl_div
from .logits import logits_kl

__version__ = "1.0.1"
__all__ = ["kl_divergence", "kl_div", "logits_kl", "__version__"]
