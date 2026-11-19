from __future__ import annotations
MV_SUBCONFIGS: dict[str, tuple[str, ...]] = {'contbern_probs': ('contbern_probs',), 'contbern_logits': ('contbern_logits',), 'categorical_probs': tuple((f'categorical_probs_K{k}' for k in (2, 4, 16))), 'categorical_logits': tuple((f'categorical_logits_K{k}' for k in (2, 4, 16))), 'dirichlet': tuple((f'dirichlet_K{k}' for k in (2, 4, 16))), 'mvn': tuple((f'mvn_d{d}' for d in (2, 4))), 'lowrank_mvn': tuple((f'lowrank_mvn_d{d}_r{r}' for d in (2, 4) for r in (1, 2)))}
BASE_CONFIG: dict[str, str] = {sub: base for base, subs in MV_SUBCONFIGS.items() for sub in subs}

def base_config(config_id: str) -> str:
    return BASE_CONFIG.get(config_id, config_id)

def is_multivariate(config_id: str) -> bool:
    return config_id in BASE_CONFIG or config_id in MV_SUBCONFIGS

def expand(config_ids) -> list[str]:
    out: list[str] = []
    for config in config_ids:
        out.extend(MV_SUBCONFIGS.get(config, (config,)))
    return out

def split_quota(config_id: str, quota: int) -> int:
    base = base_config(config_id)
    subs = MV_SUBCONFIGS.get(base)
    if not subs or len(subs) == 1:
        return quota
    n = len(subs)
    index = subs.index(config_id)
    return quota // n + (1 if index < quota % n else 0)
