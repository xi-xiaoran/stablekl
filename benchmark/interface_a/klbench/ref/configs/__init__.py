from __future__ import annotations
from types import MappingProxyType
from . import bernoulli, binomial, categorical, contbern, dirichlet, elementary, geometric, location_scale, mvn, normal, special, support
_SCALAR = {'bernoulli_probs': bernoulli.Probs, 'bernoulli_logits': bernoulli.Logits, 'normal': normal.Normal, 'binomial_probs': binomial.Probs, 'binomial_logits': binomial.Logits, 'geometric_probs': geometric.Probs, 'geometric_logits': geometric.Logits, 'poisson': elementary.Poisson, 'exponential': elementary.Exponential, 'gamma': special.Gamma, 'beta': special.Beta, 'uniform': support.Uniform, 'pareto': support.Pareto, 'cauchy': location_scale.Cauchy, 'gumbel': location_scale.Gumbel, 'laplace': location_scale.Laplace}
_MV = {'contbern_probs': contbern.Probs, 'contbern_logits': contbern.Logits, **{f'categorical_probs_K{k}': categorical.make_probs(k) for k in (2, 4, 16)}, **{f'categorical_logits_K{k}': categorical.make_logits(k) for k in (2, 4, 16)}, **{f'dirichlet_K{k}': dirichlet.make(k) for k in (2, 4, 16)}, **{f'mvn_d{d}': mvn.make_mvn(d) for d in (2, 4)}, **{f'lowrank_mvn_d{d}_r{r}': mvn.make_lowrank(d, r) for d in (2, 4) for r in (1, 2)}}
CONFIGS = MappingProxyType({**_SCALAR, **_MV})
MV_CONFIGS = tuple(_MV)

def get_config(config_id: str):
    try:
        return CONFIGS[config_id]
    except KeyError as exc:
        raise ValueError(f'Unknown or not-yet-implemented config {config_id!r}') from exc
