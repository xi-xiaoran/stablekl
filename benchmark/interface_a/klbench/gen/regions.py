from __future__ import annotations
from fractions import Fraction as F
import mpmath as mp
from ..fp import FloatFormat
from .constants import FAMILIES, REGION_PRIORITY
from ..mv_dims import is_multivariate
from . import multivariate

def _priority(labels: set[str]) -> str:
    return max(labels, key=REGION_PRIORITY.index) if labels else 'OUTSIDE'

def _positive(value: F) -> str:
    if F(1, 100) <= value <= F(100):
        return 'R0'
    if F(100) < value <= F(10000):
        return 'R1'
    if F(1, 10000) <= value < F(1, 100):
        return 'R2'
    return 'OUTSIDE'

def label_sides(config_id: str, p: dict[str, list[F]], q: dict[str, list[F]], fmt: FloatFormat, *, identity_group: bool=False) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]]:
    if is_multivariate(config_id):
        return multivariate.label_sides(config_id, p, q, fmt, identity_group=identity_group)
    family = FAMILIES[config_id]
    result = []
    outside = [False, False]
    for side_index, side in enumerate((p, q)):
        tags: set[str] = set()
        if family == 'prob':
            for v in side['probs']:
                if fmt.minimum_subnormal <= v < fmt.eps or F(1) - fmt.eps < v < 1:
                    tags.add('R2')
                elif fmt.eps <= v < F(1, 100) or F(99, 100) < v <= 1 - fmt.eps:
                    tags.add('R1')
                elif F(1, 100) <= v <= F(99, 100):
                    tags.add('R0')
        elif family == 'logits':
            with mp.workdps(90):
                limit = mp.log(99)
                for z in side['logits']:
                    v = abs(mp.mpf(z.numerator) / z.denominator)
                    if v <= limit:
                        tags.add('R0')
                    elif v < 15:
                        tags.add('R1')
                    elif v <= 1000:
                        tags.add('R2')
                    else:
                        outside[side_index] = True
        elif family == 'positive':
            labels = [_positive(v) for name, vals in side.items() if name != 'total_count' for v in vals]
            tags.update(labels)
            if 'OUTSIDE' in labels:
                outside[side_index] = True
        elif family == 'location_scale':
            scale = side['scale'][0]
            loc = side['loc'][0]
            if F(1, 10000) <= scale < F(1, 100):
                tags.add('R2')
            if F(100) <= abs(loc) / scale <= F(10000):
                tags.add('R1')
            if F(-10) <= loc <= F(10) and F(1, 100) <= scale <= F(100):
                tags.add('R0')
        elif family == 'uniform':
            low = side['low'][0]
            width = side['high'][0] - low
            if F(100) <= abs(low) / width <= F(10000):
                tags.add('R1')
            if F(-10) <= low <= F(10) and F(1, 100) <= width <= F(100):
                tags.add('R0')
        elif family == 'pareto':
            scale = side['scale'][0]
            alpha = side['alpha'][0]
            if F(100) <= alpha <= F(10000):
                tags.add('R1')
            if F(1, 100) <= scale <= F(100) and F(1, 10) <= alpha <= F(100):
                tags.add('R0')
        else:
            raise NotImplementedError(f'no region adapter for {family}')
        tags.discard('OUTSIDE')
        if not tags:
            outside[side_index] = True
        result.append(tags)
    if not identity_group and family == 'uniform' and (p['low'][0] == q['low'][0] or p['high'][0] == q['high'][0]):
        result[0].add('R2')
        result[1].add('R2')
    if not identity_group and family == 'pareto' and (p['scale'][0] == q['scale'][0]):
        result[0].add('R2')
        result[1].add('R2')
    pl, ql = (_priority(result[0]), _priority(result[1]))
    if outside[0]:
        pl = 'OUTSIDE'
    if outside[1]:
        ql = 'OUTSIDE'
    combined = _priority(result[0] | result[1])
    return (pl, ql, combined, tuple(sorted(result[0])), tuple(sorted(result[1])))
