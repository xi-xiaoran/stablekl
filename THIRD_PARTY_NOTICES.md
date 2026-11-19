# Third-party code

`experiments/common/trl_gkd_loss.py` reproduces the function `GKDTrainer.generalized_jsd_loss` of the TRL
library (https://github.com/huggingface/trl, `trl/trainer/gkd_trainer.py`), so that the distillation
experiments call exactly the loss used by TRL's GKD trainer. `SOURCE_MD5` and `SOURCE_LINES` identify the TRL
source file and the line range it was taken from. TRL is distributed under the Apache License 2.0
(https://www.apache.org/licenses/LICENSE-2.0); that license applies to this function.

All other third-party software (PyTorch, NumPy, mpmath, python-flint, transformers, datasets, auto-circuit,
TransformerLens, rsl_rl, matplotlib, pandas, scipy) is used as an installed dependency and is not
redistributed here.
