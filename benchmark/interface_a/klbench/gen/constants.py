from __future__ import annotations
from fractions import Fraction as F
BASE_SEED = 'KL-superbug-phase1-v1.0-20261105'
ATTEMPT_MULTIPLIER = 50
INFEASIBLE_PROBE = 2000
ROOT_DPS = 55
ROOT_MAX_DOUBLINGS = 96
ROOT_BISECTION_STEPS = 190
REGIONS = ('R0', 'R1', 'R2')
REGION_PRIORITY = ('R0', 'R1', 'R2', 'R3')
ULP_RANGE = (1, 8)
NORMAL_QUOTA = 10000
DANGER_QUOTA = 1000
ULP_QUOTA = 10000
PROB_ORDINARY_MIN = F(1, 100)
PROB_ORDINARY_MAX = F(99, 100)
POS_R0 = (F(1, 100), F(100))
POS_R1 = (F(100), F(10000))
POS_R2 = (F(1, 10000), F(1, 100))
LOC_R0 = (F(-10), F(10))
LOC_RATIO_R1 = (F(100), F(10000))
LOC_SCALE_R2 = (F(1, 10000), F(1, 100))
UNIFORM_WIDTH_R0 = (F(1, 100), F(100))
UNIFORM_RATIO_R1 = (F(100), F(10000))
PARETO_SCALE_R0 = (F(1, 100), F(100))
PARETO_ALPHA_R0 = (F(1, 10), F(100))
PARETO_ALPHA_R1 = (F(100), F(10000))
LOGITS_R1 = (F(15), F(15))
LOGITS_R2 = (F(15), F(1000))
FAMILIES = {'bernoulli_probs': 'prob', 'binomial_probs': 'prob', 'geometric_probs': 'prob', 'bernoulli_logits': 'logits', 'binomial_logits': 'logits', 'geometric_logits': 'logits', 'poisson': 'positive', 'exponential': 'positive', 'gamma': 'positive', 'beta': 'positive', 'normal': 'location_scale', 'cauchy': 'location_scale', 'gumbel': 'location_scale', 'laplace': 'location_scale', 'uniform': 'uniform', 'pareto': 'pareto', 'contbern_probs': 'prob', 'contbern_logits': 'logits', 'categorical_probs': 'categorical_probs', 'categorical_logits': 'categorical_logits', 'dirichlet': 'positive', 'mvn': 'mvn', 'lowrank_mvn': 'lowrank_mvn'}
CONFIG_DISTRIBUTIONS = {'bernoulli_probs': 'Bernoulli', 'bernoulli_logits': 'Bernoulli', 'binomial_probs': 'Binomial', 'binomial_logits': 'Binomial', 'geometric_probs': 'Geometric', 'geometric_logits': 'Geometric', 'poisson': 'Poisson', 'exponential': 'Exponential', 'gamma': 'Gamma', 'beta': 'Beta', 'uniform': 'Uniform', 'pareto': 'Pareto', 'normal': 'Normal', 'cauchy': 'Cauchy', 'gumbel': 'Gumbel', 'laplace': 'Laplace', 'contbern_probs': 'ContinuousBernoulli', 'contbern_logits': 'ContinuousBernoulli', 'categorical_probs': 'Categorical', 'categorical_logits': 'Categorical', 'dirichlet': 'Dirichlet', 'mvn': 'MultivariateNormal', 'lowrank_mvn': 'LowRankMultivariateNormal'}
from ..mv_dims import BASE_CONFIG as _MV_BASE
CONFIG_DISTRIBUTIONS.update({sub: CONFIG_DISTRIBUTIONS[base] for sub, base in _MV_BASE.items()})
SCALAR_CONFIGS = ('bernoulli_probs', 'bernoulli_logits', 'normal', 'binomial_probs', 'binomial_logits', 'geometric_probs', 'geometric_logits', 'poisson', 'exponential', 'gamma', 'beta', 'uniform', 'pareto', 'cauchy', 'gumbel', 'laplace')
