from __future__ import annotations
import os
import random
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
from klbench.fp import get_format
from klbench.gen import multivariate
from klbench.gen.runner import _decode, build_tasks, generate_cell
from klbench.mv_dims import MV_SUBCONFIGS, expand
from klbench.pipeline_io import load_unit
from klbench.ref.configs import CONFIGS, MV_CONFIGS

@pytest.mark.parametrize('config_id', MV_CONFIGS)
@pytest.mark.parametrize('dtype', ['fp16', 'fp32'])
def test_sampled_p_lands_in_its_region(config_id, dtype):
    fmt = get_format(dtype)
    rng = random.Random(f'{config_id}/{dtype}')
    for region in multivariate.regions_for(config_id):
        hits = Counter()
        for _ in range(16):
            try:
                bits = multivariate.sample_p(config_id, region, dtype, rng)
            except ValueError:
                hits['sampler_error'] += 1
                continue
            try:
                p = _decode(bits, dtype)
            except ValueError:
                hits['nonfinite'] += 1
                continue
            if CONFIGS[config_id].domain_ok(p, p) is not True:
                hits['domain'] += 1
                continue
            tags, outside = multivariate.side_tags(config_id, p, fmt)
            label = 'OUTSIDE' if outside else max(tags, key=('R0', 'R1', 'R2', 'R3').index)
            hits[label] += 1
        assert hits[region] >= 6, (config_id, dtype, region, dict(hits))

def test_task_list_splits_quotas_and_regions(tmp_path):
    tasks = list(build_tasks(['dirichlet', 'contbern_probs'], ['fp32'], None, 't', str(tmp_path)))
    cells = Counter(((t[0], t[2], t[3]) for t in tasks))
    assert set((c for c, _, _ in cells)) == set(expand(['dirichlet', 'contbern_probs']))
    quota = {(t[0], t[2], t[3]): t[4] for t in tasks}
    assert sum((quota[s, 'B_3', 'R0'] for s in MV_SUBCONFIGS['dirichlet'])) == 10000
    assert sum((quota[s, 'ULP', 'ALL'] for s in MV_SUBCONFIGS['dirichlet'])) == 10000
    assert quota['contbern_probs', 'B_3', 'R3'] == 1000
    assert ('dirichlet_K4', 'B_3', 'R3') not in quota

@pytest.mark.parametrize('config_id', MV_CONFIGS)
def test_tiny_cells_complete(tmp_path, config_id):
    for dtype in ('fp16', 'fp32'):
        for group, region in (('identity', 'R0'), ('B_3', 'R0'), ('ULP', 'ALL')):
            res = generate_cell((config_id, dtype, group, region, 2, 'mv_test', str(tmp_path)))
            assert res['status'] == 'COMPLETE', (config_id, dtype, group, res)
            arrays, meta = load_unit(Path(res['path']))
            for name, spec in CONFIGS[config_id].PARAM_SPEC.items():
                size = int(np.prod(spec['shape']))
                assert arrays[f'p__{name}'].shape == (2, size)
                assert arrays[f'grad__P__{name}__bits'].shape == (2, size)
            assert set(arrays['kl_status']) == {'CERTIFIED'}
FROZEN_V1 = os.environ.get('KL_FROZEN_V1')

@pytest.mark.skipif(not FROZEN_V1, reason='set KL_FROZEN_V1 to the frozen data directory of the release')
@pytest.mark.parametrize('config_id,dtype,group,region', [('gamma', 'fp32', 'B_3', 'R0'), ('beta', 'fp16', 'B_5', 'R2'), ('poisson', 'fp32', 'ULP', 'ALL'), ('pareto', 'fp16', 'identity', 'R1')])
def test_scalar_generation_unchanged(tmp_path, config_id, dtype, group, region):
    frozen, _ = load_unit(Path(FROZEN_V1) / config_id / dtype / f'{group}__{region}.npz')
    res = generate_cell((config_id, dtype, group, region, 3, 'regress', str(tmp_path)))
    new, _ = load_unit(Path(res['path']))
    for key in [k for k in new if k.startswith(('p__', 'q__'))]:
        assert np.array_equal(new[key], frozen[key][:3]), key
