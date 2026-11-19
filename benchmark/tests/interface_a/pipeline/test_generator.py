from __future__ import annotations
from fractions import Fraction as F
import random
import pytest
from klbench.fp import BINARY16, BINARY32, round_fraction
from klbench.gen.regions import label_sides
from klbench.gen.runner import _neighbor, derive_seed, build_tasks, _quantize, _draw_ulp_p_region

def test_sha_seeds_per_cell_independent_and_stable():
    a = derive_seed('normal', 'fp16', 'B_1', 'R0')
    assert a == derive_seed('normal', 'fp16', 'B_1', 'R0')
    assert a != derive_seed('normal', 'fp32', 'B_1', 'R0')
    assert a != derive_seed('normal', 'fp16', 'B_2', 'R0')

def test_quota_counts_and_registration(tmp_path):
    tasks = list(build_tasks(['normal'], ['fp16'], 2, 'smoke', tmp_path))
    assert len(tasks) == 16 * 3 + 1
    assert sum((task[4] for task in tasks)) == 2 * (16 * 3 + 1)
    assert {t[0] for t in build_tasks(['mvn'], ['fp16'], 2, 'smoke', tmp_path)} == {'mvn_d2', 'mvn_d4'}
    with pytest.raises(ValueError, match='Unknown configuration'):
        list(build_tasks(['no_such_config'], ['fp16'], 2, 'smoke', tmp_path))

def test_ulp_neighbor_moves_exactly_one_ieee_step():
    assert _neighbor(15360, BINARY16, +1) == 15361
    assert _neighbor(15360, BINARY16, -1) == 15359
    assert _neighbor(48128, BINARY16, +1) == 48127
    assert _neighbor(48128, BINARY16, -1) == 48129

def test_region_priority_q_danger_override():
    p = {'rate': [F(1)]}
    q = {'rate': [F(1, 1000)]}
    a, b, combined, _, _ = label_sides('exponential', p, q, BINARY16)
    assert (a, b, combined) == ('R0', 'R2', 'R2')

def test_provided_probabilities_are_frozen_before_regions():
    fmt = BINARY16
    bits = _quantize({'probs': [F(1, 100)]}, 'fp16')
    from klbench.fp.bits import exact_value
    p = {'probs': [exact_value(bits['probs'][0], fmt)]}
    a, _, _, _, _ = label_sides('bernoulli_probs', p, p, fmt)
    assert a in ('R0', 'R1')

def test_f1_ulp_draws_all_three_real_regions():
    rng = random.Random(123)
    draws = [_draw_ulp_p_region(rng) for _ in range(100000)]
    frequencies = {r: draws.count(r) / len(draws) for r in ('R0', 'R1', 'R2')}
    assert set(draws) == {'R0', 'R1', 'R2'}
    assert abs(frequencies['R0'] - 0.7) < 0.01
    assert abs(frequencies['R1'] - 0.15) < 0.01
    assert abs(frequencies['R2'] - 0.15) < 0.01

def test_f2_identity_support_coincidence_does_not_force_r2():
    uniform = {'low': [F(1)], 'high': [F(2)]}
    assert label_sides('uniform', uniform, uniform, BINARY16)[2] == 'R2'
    assert label_sides('uniform', uniform, uniform, BINARY16, identity_group=True)[2] == 'R0'
    far_uniform = {'low': [F(1000)], 'high': [F(1001)]}
    assert label_sides('uniform', far_uniform, far_uniform, BINARY16, identity_group=True)[2] == 'R1'
    pareto = {'scale': [F(1)], 'alpha': [F(2)]}
    assert label_sides('pareto', pareto, pareto, BINARY16)[2] == 'R2'
    assert label_sides('pareto', pareto, pareto, BINARY16, identity_group=True)[2] == 'R0'
    high_alpha = {'scale': [F(1)], 'alpha': [F(1000)]}
    assert label_sides('pareto', high_alpha, high_alpha, BINARY16, identity_group=True)[2] == 'R1'
