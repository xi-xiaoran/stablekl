# Benchmark data

## Download

The data are hosted on Hugging Face: <https://huggingface.co/datasets/xi-xiaoran/stablekl-benchmark> (10.8 GB, 44,000 files).

```bash
pip install -U huggingface_hub
huggingface-cli download --repo-type dataset xi-xiaoran/stablekl-benchmark --local-dir data      # everything
huggingface-cli download --repo-type dataset xi-xiaoran/stablekl-benchmark --local-dir data \
        --include "interface_a/*/normal/*"                                                         # one configuration
cd data && sha256sum -c SHA256SUMS --ignore-missing
```

License: CC BY 4.0 ([../LICENSE-DATA](../LICENSE-DATA)).

## What is in it

| Part | Content | Samples |
|---|---|---|
| `interface_a/frozen/` | inputs (P, Q) and certified truth of `kl_divergence` and its gradients, 33 configurations x {fp16, fp32} | 7.5 million |
| `interface_a/official_cpu/` | outputs of the official implementation on CPU on exactly these inputs: `A` (official), `B32`/`B64` (official formula on exactly promoted parameters, rounded once) | |
| `interface_b/<config>/inputs/` | inputs of `F.kl_div` (`x` = log-probabilities, `t` = target), with the logits they were derived from | 942,200 batches |
| `interface_b/<config>/truth/` | certified truth of every output: element-wise loss, the four reductions, gradients w.r.t. both arguments | 5.69e8 values |
| `interface_b/<config>/out/` | official outputs on CPU for batch sizes 1 and 8: `A` and the official formula on exactly promoted inputs (`B64`) | |

Platform of the official outputs: PyTorch 2.15.0+cpu, Python 3.11, single thread, batch size 4,096 (interface A).

## Organisation

**Interface A.** One directory per configuration (`normal`, `categorical_logits_K4`, `mvn_d2`,
`lowrank_mvn_d4_r2`, ...; the full list is `scripts/configs_a.txt`) and dtype. Each file is one *unit*: one
parameter region and one KL group.

* Regions: `R0` ordinary parameters; `R1`, `R2` (and `R3` for some families) stress regions with very large or
  very small parameters, or probabilities close to 0 or 1.
* KL groups: `B_c` control interval [1e-1, 1); `B_k` the interval [10^-(k+1), 10^-k) of the true KL, k = 1..14;
  `ULP` Q obtained by moving parameters of P by 1–8 ULP; `identity` Q = P.
* File names: `<group>__<region>.npz`, e.g. `B_9__R0.npz`; official outputs `<group>__<region>__<method>.npz`.

**Interface B.** One directory per configuration `lt{F,T}_{fp16,fp32}_K{2,4,16,256,1024}` (`log_target`, dtype,
number of classes). Units are named `<layer>__<group>__<region>`: layer `S1` samples from logits, `S1t` the same
with the probabilities computed by torch, `S2`/`S2b` boundary strata (zeros, -inf, extreme magnitudes), `S3`–`S7`
further strata of the sampling plan; `group` is the KL interval as above. Truth and outputs carry a suffix
`__B1` or `__B8` for the batch size.

## File format

Every unit is a NumPy `.npz` (no pickled objects; load with `np.load(path, allow_pickle=False)`) with a
`.npz.manifest.json` beside it. The manifest's `sha256` is the SHA-256 of the `.npz`; the loaders
(`klbench.pipeline_io.load_unit`, `kldivbench.store.load`) refuse a unit whose hash does not match. Manifests
also record the configuration, dtype, unit, quotas, parameter shapes, the hashes of the generating source files
and, for derived files, the hash of the inputs they were computed from.

**Floating-point values are stored as bit patterns**: `uint16` for fp16, `uint32` for fp32. To get the numbers:

```python
import numpy as np
z = np.load("data/interface_a/frozen/normal/fp32/B_9__R0.npz", allow_pickle=False)
loc_p = z["p__loc"].view(np.float32)            # shape (n, 1)
kl_truth = z["kl_truth_bits"].view(np.float32)  # correctly rounded true KL
```

**Interface A, frozen unit** (n samples):

| Array | Meaning |
|---|---|
| `sample_id` | stable identifier |
| `p__<param>`, `q__<param>` | parameters of P and Q, bit patterns, shape (n, size of the parameter) |
| `kl_truth_bits` | true KL rounded once to the dtype |
| `kl_lower`, `kl_upper` | certified enclosure of the true KL as exact rationals (strings `num/den`) |
| `kl_status`, `kl_group`, `kl_precision`, `kl_overflow` | certification status (`CERTIFIED`), KL group, working precision in bits, overflow on rounding |
| `grad__{P,Q}__<param>__bits` | true gradient components rounded once |
| `grad__{P,Q}__<param>__{lower,upper,status,exact_zero,nondiff,overflow,precision}` | enclosure and status per component; `nondiff` components are not scored |
| `region`, `region_p`, `region_q`, `group`, `direction`, `ulp_k`, `semantic_identity`, `exact_bitwise_identity`, `sample_tag`, `issues_json`, `xcheck` | sampling metadata and the result of an independent cross-check of the truth |

**Interface A, official output** (`__A`, `__B32`, `__B64`): `kl_bits`, `grad__{P,Q}__<param>__bits` (rounded to the
input dtype), `status` and `error` per sample (`OK`, or the exception raised by PyTorch), and the unrounded native
values for `B32`/`B64` (`*_native_bits`, `native_dtype`).

**Interface B**: `inputs` hold `x`, `t` (bit patterns, shape (n, K)), the source logits `zp`, `zq` and a JSON
`meta` string per row (sampling details and the end-to-end truth interval). `truth` holds per row the element
losses `L_bits`, reductions `S_bits` (sum, mean, batchmean), gradients `gx_bits_<reduction>`, `gt_bits_<reduction>`,
their status flags (`*_st`, `*_q`) and the end-to-end truth `E2E_bits`. `out` holds per method
(`A__...`, `B64__...`) the element losses (`none`), reductions and gradients, plus call and gradient status codes.

## Integrity guarantees

The released `.npz` files are byte-identical to the files the paper's numbers were computed from. Manifests were
rewritten only to remove host-specific fields (platform string, interpreter build) and to normalise schema names;
the hashes they contain are unchanged.

Scoring the released official outputs and the repair is deterministic. Methods that you recompute yourself in
float16/float32 (the official implementation and the same-precision rewrite) can differ by a few samples from
the published counts on other CPU or GPU types, because PyTorch's transcendental functions are not bit-identical
across hardware.
