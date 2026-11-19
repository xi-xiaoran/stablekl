from __future__ import annotations
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from klbench.gen.constants import CONFIG_DISTRIBUTIONS
from klbench.mv_dims import BASE_CONFIG
from klbench.torchrun.runner import _call_batch
SHAPES = {'contbern_probs': {'probs': (1,)}, 'contbern_logits': {'logits': (1,)}, **{f'categorical_probs_K{k}': {'probs': (k,)} for k in (2, 4, 16)}, **{f'categorical_logits_K{k}': {'logits': (k,)} for k in (2, 4, 16)}, **{f'dirichlet_K{k}': {'concentration': (k,)} for k in (2, 4, 16)}, **{f'mvn_d{d}': {'loc': (d,), 'scale_tril': (d, d)} for d in (2, 4)}, **{f'lowrank_mvn_d{d}_r{r}': {'loc': (d,), 'cov_factor': (d, r), 'cov_diag': (d,)} for d in (2, 4) for r in (1, 2)}}

def _values(config, name, shape, rng, shift):
    n = int(np.prod(shape))
    if name == 'probs' and config.startswith('contbern'):
        v = rng.uniform(0.2, 0.8, n)
    elif name in ('probs', 'concentration', 'cov_diag'):
        v = rng.uniform(0.5, 2.0, n)
    elif name == 'scale_tril':
        d = shape[0]
        m = np.tril(rng.uniform(-0.3, 0.3, (d, d)), -1) + np.diag(rng.uniform(0.5, 2, d))
        v = m.reshape(-1)
    else:
        v = rng.uniform(-1, 1, n)
    if shift and name != 'scale_tril':
        v = v * 1.01
    elif shift:
        v = (v.reshape(shape) * (1 + 0.01 * np.tri(*shape))).reshape(-1)
    return v

@pytest.mark.parametrize('config', sorted(SHAPES))
@pytest.mark.parametrize('dtype', ['fp16', 'fp32'])
def test_official_kl_runs_or_is_marked_unsupported(config, dtype):
    assert CONFIG_DISTRIBUTIONS[config] == CONFIG_DISTRIBUTIONS[BASE_CONFIG[config]]
    npdt, bt = (np.float16, np.uint16) if dtype == 'fp16' else (np.float32, np.uint32)
    rng = np.random.default_rng(1)
    frozen = {'_parameter_shapes': {n: {'shape': list(s), 'differentiable': True} for n, s in SHAPES[config].items()}}
    n_rows = 3
    for side, shift in (('p', False), ('q', True)):
        for name, shape in SHAPES[config].items():
            rows = [_values(config, name, shape, rng, shift).astype(npdt).view(bt) for _ in range(n_rows)]
            frozen[f'{side}__{name}'] = np.stack(rows)
    methods = ('A', 'B32', 'B64') if dtype == 'fp16' else ('A', 'B64')
    for method in methods:
        out = _call_batch(config, dtype, method, frozen, list(range(n_rows)), 'cpu')
        assert len(out) == n_rows
        for row in out:
            status, error = (row['status'], row.get('error', ''))
            if method != 'A':
                assert status == 'OK', (config, dtype, method, status, error)
            elif status != 'OK':
                cpu_half = dtype == 'fp16' and BASE_CONFIG[config] in ('mvn', 'lowrank_mvn') and (status in ('UNSUPPORTED_CPU', 'BACKWARD_UNSUPPORTED_CPU'))
                validation = status == 'CALL_ERROR' and 'constraint' in error
                assert cpu_half or validation, (config, dtype, method, status, error)
            if status == 'OK':
                for name, shape in SHAPES[config].items():
                    for side in ('P', 'Q'):
                        assert len(row['grad'][f'{side}__{name}']['bits']) == int(np.prod(shape))
