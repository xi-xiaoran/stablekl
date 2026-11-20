# Reproducing the paper

All commands run from the repository root after

```bash
pip install -e .                                  # the stablekl package
pip install -r requirements-benchmark.txt -r requirements-experiments.txt
source scripts/env.sh                             # puts the benchmark harnesses on PYTHONPATH
```

Figures are written to `figures/out/`. Model checkpoints and datasets are downloaded through Hugging Face
(`transformers`, `datasets`) or passed as paths; nothing of them is included here.

## Level 0: figures from the included results (minutes)

| Paper item | Command | Input |
|---|---|---|
| Figure 1 (a): relative error of the Exponential KL vs. the true KL | `python figures/fig1_overview.py` | recomputed on 3,000 random pairs per point against 50-digit arithmetic |
| Figure 1 (b): identical-start distillation | (same script) | `results/distillation/runs.jsonl` |
| Figure 2: error rate per KL interval, 16 scalar-family configurations pooled (region R0) | `python figures/fig2_errkl.py` | `results/interface_a/unit_metrics.csv.gz` |
| Experiment 1 figure: Geometric fit, true KL along training | `python figures/fig_exp1.py` | recomputed (12 starts, Adam, 3,000 steps; CPU) |
| Experiment 3 figure: drift vs. learning rate, vocabulary, precision | `python figures/fig_grid_lr.py` | `results/distillation/runs.jsonl` |
| GPU appendix heatmaps (official CPU / official GPU / ours) | `python figures/heat_paper.py` | `results/heatmaps/*.csv` |
| All device x interface x method heatmaps | `python figures/heat_all.py` | `results/heatmaps/*.csv` |

`results/heatmaps/` holds, per configuration, KL interval, method and device, the number of samples and the
number with a relative KL error of at least 1% (parameter region R0); the heatmaps are these ratios.

## Level 1: the measurement and the repair on the frozen data

Download the data first ([DATA.md](DATA.md)).

**Interface A** (`torch.distributions.kl_divergence`, 33 configurations listed in `scripts/configs_a.txt`):

```bash
DATA=data OUT=runs/interface_a bash scripts/run_interface_a.sh <config> <cpu|cuda> <workers>
```

This writes

* `runs/interface_a/score/<config>.json`: per dtype and method the number of samples with a significant KL error
  (`kl_sig`), correctly rounded values (`kl_cr`), negative values (`KL_NEG`), and the same for gradient components
  (`g_sig`, `g_cr`). These are the counts behind the main table (official, official one precision higher,
  rewrite, ours) and the GPU table. Methods: `A` official; `B32`/`B64` official formula in float32/float64;
  `C` rewrite in the working precision; `D32`/`D64` repaired formula in float32/float64 with one rounding
  (`D64` is `mode="ours"` of the package). A suffix `_cpu` marks the released official CPU outputs, scored in the
  same pass; without suffix the method was recomputed on `<device>`.
* `runs/interface_a/bins/<config>.json`: the same per unit (KL interval x parameter region), every KL metric.

Counts of methods recomputed on your machine (`A`, `B32`, `B64`, `C`) can differ slightly from the paper on other
CPU or GPU types, since PyTorch's float16/float32 math functions are not bit-identical across hardware.

A significant KL error is any of: at least 8 ULP away from the correctly rounded truth, relative error of at
least 1%, wrong sign, NaN or infinity where the truth is finite, a false zero, or a non-zero value at identity.
For gradients: at least 8 ULP, NaN/inf, wrong sign, vanishing or spurious components.

**Interface B** (`torch.nn.functional.kl_div`, 20 configurations: `log_target` x dtype x K, listed in
`scripts/configs_b.txt`):

```bash
cp -r data/interface_b work/interface_b                      # the official scoring writes score/ next to the data
DATA=work/interface_b OUT=runs/interface_b bash scripts/run_interface_b.sh <0|1> <fp16|fp32> <K> <cpu|cuda>
```

This scores the official outputs (`runs/interface_b/<config>/report_official/`), runs the repaired `kl_div`
(methods `C`, `D32`, `D64`) and writes the acceptance verdict of every method to `runs/interface_b/verdict/`.
The verdict compares each method with the official outputs on the same inputs: `H1a` counts new significant
errors relative to the official code, `H1b` outputs that are correctly rounded with the official code but not
with the method, and `H2` hard failures (NaN where the truth is defined, false infinities, gradient errors
against the certified truth). The repair `D64` is accepted when all three are zero (paper, protocol appendix).

**Official `kl_div` on a GPU:**

```bash
python benchmark/gpu/run_official_b.py run   --root work/interface_b --root3 runs/interface_b_gpu --lt 0 --dt fp32 --K 16 --device cuda
python benchmark/gpu/run_official_b.py score --root work/interface_b --root3 runs/interface_b_gpu --lt 0 --dt fp32 --K 16
```

**Cluster.** `scripts/slurm/array_template.sbatch` runs one configuration per array task; fill in the partition,
account and environment lines.

**Regenerating the data instead of downloading it.** `klbench.gen.runner` (interface A) and the `gen`, `fill`,
`certify`, `run` stages of `kldivbench.pipeline` (interface B) regenerate inputs with the fixed seed string in
`klbench/gen/constants.py`, certify the truth with python-flint and run the official implementation. This takes
thousands of CPU hours and produces bit-identical inputs only with the same library versions; the released data
are the reference.

## Level 2: downstream experiments 1–3

**Experiment 1, fitting a fixed Geometric distribution** (CPU):

```bash
python experiments/exp1_geometric_fit/geo_fit.py            # all parameterisations, directions, targets, methods
python experiments/exp1_geometric_fit/geo_fit2.py A         # 12 starts per setting (figure data)
python experiments/exp1_geometric_fit/geo_fit2.py B         # gradient and trajectory figures
python experiments/exp1_geometric_fit/geo_fit_robust.py     # other optimisers and learning rates
```

**Experiment 2, sequential Bayesian experimental design with Beta posteriors** (CPU):

```bash
python experiments/exp2_beta_design/beta_ig.py --setting A --out runs/beta     # 10 arms, 100,000 rounds, 5 seeds
python experiments/exp2_beta_design/beta_ig.py --setting B --out runs/beta     # warm start, 20,000 rounds
python experiments/exp2_beta_design/beta_ig.py batch runs/beta_batch            # batch design, m = 1, 100, 1000, 10000
python experiments/exp2_beta_design/summarize_beta_ig.py runs/beta
python experiments/exp2_beta_design/summarize_batch.py runs/beta_batch
python experiments/exp2_beta_design/test_beta_ig.py                             # unit tests
```

**Experiment 3, distillation from an identical start** (one GPU; 2,880 runs: 3 vocabularies x 3 dtypes x 8 optimiser settings x 4 losses x 10 seeds):

```bash
python experiments/exp3_identical_distillation/distill_grid.py --out runs/grid --device cuda
python experiments/exp3_identical_distillation/summarize_grid.py runs/grid
```

`runs/grid/runs.jsonl` has the format of `results/distillation/runs.jsonl`; pass `--compare
results/distillation/runs.jsonl` to report whether the new runs are bit-identical (they are on the same GPU type
and software versions; expect small differences elsewhere). `distill_gpu.py` is the per-run module and also runs
the smaller first grid (`--out`, `--device`).

## Level 3: experiment 4 and the settings without an effect (GPU)

**Experiment 4, Qwen2.5-1.5B-Instruct** (WikiText-2 test; checkpoints from the Hugging Face hub):

```bash
# Part A: KL(base || quantised) for RTN, GPTQ and AWQ variants (experiments/exp4a_quantized_eval/config.json)
python experiments/exp4a_quantized_eval/run_eval.py --config experiments/exp4a_quantized_eval/config.json \
       --out runs/exp4a --scratch work/exp4a --device cuda --hub "$HF_HUB_CACHE"
# Part B: layer-wise sensitivity ranking and mixed-precision choice
python experiments/exp4b_layer_ranking/run_rank.py --config experiments/exp4b_layer_ranking/config.json \
       --out runs/exp4b --scratch work/exp4b --device cuda --hub "$HF_HUB_CACHE"
# Part C: reported KL under different batch sizes / reductions (R) and execution-mismatch KL (M)
M=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct'))")
python experiments/exp4c_repeated_execution/instrument.py --part R --fwd fp16 --model $M --out runs/exp4c/R_fp16.jsonl
python experiments/exp4c_repeated_execution/instrument.py --part R --fwd bf16 --model $M --out runs/exp4c/R_bf16.jsonl
python experiments/exp4c_repeated_execution/instrument.py --part M --dtypes fp32,fp16,bf16 --model $M --seq 257 --out runs/exp4c/M.jsonl
python experiments/exp4c_repeated_execution/summarize_instrument.py runs/exp4c
```

Part D, circuit analysis with `auto-circuit` (its repository provides the prompt datasets under `datasets/`):

```bash
DS=/path/to/auto-circuit/datasets
python experiments/exp4d_circuit_edges/model_files.py online models gpt2-small gpt2-medium attn-only-4l   # local copies + digests
for DT in fp32 fp16; do
  python experiments/exp4d_circuit_edges/edge_scores.py --model gpt2-small --dtype $DT --data $DS/ioi/ioi_vanilla_template_prompts.json --out runs/exp4d_ioi
  python experiments/exp4d_circuit_edges/edge_scores2.py --tag gt  --dtype $DT --out runs/exp4d --model-dir models --model gpt2-small \
         --data $DS/greaterthan_gpt2-small_prompts.json --true-edges greaterthan_official.greaterthan_true_edges
  python experiments/exp4d_circuit_edges/edge_scores2.py --tag doc --dtype $DT --out runs/exp4d --model-dir models --model attn-only-4l \
         --data $DS/docstring_prompts.json --batch 128 --n-train 256 --true-edges docstring_official.docstring_true_edges
  python experiments/exp4d_circuit_edges/edge_scores2.py --tag med --dtype $DT --out runs/exp4d --model-dir models --model gpt2-medium \
         --data $DS/ioi/ioi_vanilla_template_prompts.json --sample 30000 --seed 0
done
python experiments/exp4d_circuit_edges/summarize_edge.py runs/exp4d_ioi
python experiments/exp4d_circuit_edges/summarize_edge2.py runs/exp4d
```

**Settings without a measurable effect** (paper appendix): `experiments/no_effect/`

| Setting | Script | Grid used |
|---|---|---|
| GKD distillation of Qwen2.5-1.5B on TRL's loss | `llm_distillation/gkd.py --model M --out O --start {S0,S1,S2} --loss {bf16,fp32,logits} --seeds 0-9` | 9 runs x 10 seeds; `summarize_gkd.py O` |
| LoRA with a KL term | `lora/lora_screen.py --scenario {S0,S1,S2,ANCHOR} --lr {1e-4,3e-4}`; `lora/lora_strong.py` with `--b-init-std` or `--anchor-lambda` | `summarize_lora.py`, `summarize_strong.py` |
| KL in RL / calibration on real logits | `rl_and_calibration/probe.py --model M --out O` | single run |
| PPO with a Beta policy (rsl_rl) | `ppo_beta_policy/run_beta.py --mode {official,fp64} --seed S --out F` | |
| Sequential ACDC | `exp4d_circuit_edges/acdc_run.py --dtype {fp32,fp16} --data ... --out O`; `sequential_acdc/summarize_acdc.py O` | |

**Cost of the implementations** (paper appendix): `python experiments/cost/cost_bench.py` (CPU, closed forms) and
`python experiments/cost/cost_bench_gpu.py --out runs/cost --devices cuda,cpu`; `make_cost_tables.py` formats them.

**Call-site survey** (paper appendix): `python experiments/callsite_survey/survey_list.py <unpacked library
sources...> out.json` lists the call sites of both interfaces; `results/callsite_survey/survey_calls.json` is the
result for the libraries in the paper. One use that the pattern does not match was added by hand: `allennlp`
(`allennlp/fairness/fairness_metrics.py` stores `kl_divergence` and calls it through an attribute).
