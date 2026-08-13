# The Self-Pruning Neural Network

A feed-forward CIFAR-10 classifier whose linear layers each carry a learnable
per-weight sigmoid gate (`PrunableLinear`), trained with an L1 sparsity
penalty on the gate values so the network prunes its own connections during
ordinary gradient-descent training — no post-training pruning step.

```
gate_scores  -- learnable, same shape as weight
gates        = sigmoid(gate_scores)
effective_w  = weight * gates
output       = effective_w @ input + bias

SparsityLoss = sum(gates) over every PrunableLinear layer
TotalLoss    = CrossEntropy(logits, labels) + lambda * SparsityLoss
```

See `report.md` for the full methodology, mathematical formulation, results,
and analysis. This README covers setup, running, and reproducing the
experiments.

## Setup

Tested with **Python 3.14.3** on macOS (Apple Silicon, MPS backend). Any
Python >= 3.10 with the packages below should work; CUDA and CPU are both
auto-detected as fallbacks (see "Device" below).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins floor versions (`>=`); this project was developed
and verified against:

| Package | Tested version |
|---|---|
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| numpy | 2.5.2 |
| matplotlib | 3.11.1 |

**All commands below must be run from the repository root** (`self_pruning_nn/`)
— the script uses relative paths (`./data`, `./results`) for the dataset
cache and all outputs.

CIFAR-10 downloads automatically to `./data` on first run (~170MB via
`torchvision.datasets.CIFAR10`); it is cached there for every subsequent run.

## Run

```bash
# 1. Sanity tests for the PrunableLinear layer (run this first, seconds)
python test_prunable_linear.py

# 2. Short pilot run -- sanity-checks that training works and that lambda
#    is in a useful range before committing to a full experiment.
python self_pruning_nn.py --mode pilot

# 3. Full experiment (only after reviewing pilot results)
python self_pruning_nn.py --mode full --epochs 50 --lambdas 0 1e-5 1e-4 1e-3 \
    --output-prefix primary --checkpoint-subdir checkpoints_primary --diagnostic-plots
```

**Note:** command 3 above **retrains all four models from scratch** (about
18 minutes on the reference machine — see "Expected runtime" below); it
does not read from or depend on any file already in `results/`. The CSVs,
logs, checkpoints, and plots already committed under `results/` (used
throughout `report.md`) are the **completed experiment's saved output**,
produced by this exact command with this exact seed. Re-running it will
independently regenerate results that should match those files up to the
normal run-to-run variance discussed in `report.md` §11 (not bit-for-bit
identical, since dataloader/model-init RNG state can differ slightly by
platform/library version even with a fixed seed); it will not overwrite
them unless you reuse the same `--output-prefix`/`--checkpoint-subdir`.

`--mode` only accepts `pilot` or `full` — there is no separate `test` mode;
the `PrunableLinear` sanity tests are a plain `unittest` script, run
directly as in step 1.

### CLI flags (`--mode full`)

| Flag | Default | Purpose |
|---|---|---|
| `--epochs N` | 25 | Epochs per lambda. The reported results use `--epochs 50` for the primary experiment. |
| `--lambdas ...` | `0 1e-7 1e-6 1e-5 1e-4` | Space-separated lambda grid. The reported primary experiment used `0 1e-5 1e-4 1e-3`. |
| `--seed N` | 42 | Re-applied before building each lambda's model, so every run in a sweep starts from an identical initialization. |
| `--output-prefix NAME` | `full` | Prefix for this run's output filenames (`results/<prefix>_epoch_logs.csv`, etc.) — lets separate experiments coexist without overwriting each other. |
| `--checkpoint-subdir NAME` | `checkpoints` | Subdirectory under `results/` for this run's checkpoints. |
| `--diagnostic-plots` | off | After training, save sparsity/accuracy/mean-gate-vs-epoch line plots (one line per lambda) to `results/<prefix>_diagnostic_*.png`. |

`--mode pilot` accepts `--epochs`, `--lambdas`, and `--seed` the same way
(defaults: 4 lambdas, 5 epochs); it always writes to
`results/pilot_epoch_logs.csv` / `results/pilot_summary.csv`.

### Expected runtime

On the reference machine (Apple Silicon, MPS backend), one epoch takes
roughly 5 seconds. The pilot (4 lambdas x 5 epochs) takes about 2 minutes;
the reported primary experiment (4 lambdas x 50 epochs = 200 epochs) took
about 18 minutes. CPU-only machines will be significantly slower. The
one-time CIFAR-10 download adds a few minutes depending on connection speed.

## Output

Every `--mode full` run writes, under `results/`, prefixed with
`--output-prefix` (default `full`):

- `<prefix>_epoch_logs.csv` — every epoch, every lambda: losses, test
  accuracy, sparsity %, mean/median/min/max gate, gate quantiles
  (Q01/Q05/Q25/Q75/Q95/Q99), gate-gradient norm.
- `<prefix>_results_table.csv` — the assignment's exact 3-column table
  (Lambda, Test Accuracy, Sparsity Level %).
- `<prefix>_gate_summary.csv` — richer final-epoch gate diagnostics per
  lambda (full quantile breakdown).
- `<prefix>_diagnostic_*.png` — only with `--diagnostic-plots`: sparsity /
  test accuracy / mean gate vs. epoch, one line per lambda.
- `<checkpoint-subdir>/lambda_<lam>_final.pt` — model state after the last
  epoch of that lambda's run.
- `<checkpoint-subdir>/lambda_<lam>_best.pt` — model state at the epoch
  with the highest test accuracy seen *during that run* (an automatic,
  predefined rule applied during training, never a post-hoc manual pick).

The final gate-value distribution plot for the recommended model is at
`results/gate_distribution_lambda_1e-4.png` (a two-panel plot: all gates
on a log-scaled axis with the pruning threshold marked, plus a zoomed
panel on the active-gate population only).

## Results summary

Primary experiment — `λ ∈ {0, 1e-5, 1e-4, 1e-3}`, 50 epochs each, identical
architecture/optimizer/seed/preprocessing:

| Lambda | Test Accuracy | Sparsity Level (%) |
|---:|---:|---:|
| 0    | 53.12% | 0.00%  |
| 1e-5 | 55.35% | 50.05% |
| 1e-4 | 55.88% | 88.45% |
| 1e-3 | 55.39% | 99.72% |

**Recommended model: λ = 1e-4** — it provides the strongest overall balance
of accuracy, sparsity, stability, and effective-weight cleanliness. It has
the highest final accuracy of any condition tested at 88.45% sparsity; its
1.29-point peak-to-final erosion is lower than the dense baseline's 1.83
points (though higher than λ=1e-5's 0.83 points); no meaningful bulk
weight-gate compensation was observed, with the maximum effective weight
among threshold-pruned connections remaining small; and its gate
distribution clearly shows the expected spike-at-zero-plus-active-cluster
shape. See `report.md` §§8-12 for the full results, the accuracy/sparsity
trade-off analysis, and limitations (including why λ=1e-3's higher
sparsity number comes with real trade-offs that make it the less
defensible choice despite scoring higher on that one metric).

## Reproducibility

- All RNGs (`random`, `numpy`, `torch`, `torch.cuda`) are seeded via
  `set_seed()`, called fresh before building each lambda's model.
- Device is auto-detected (CUDA > MPS > CPU) and logged (along with Python/
  torch/torchvision versions and the seed) at the start of every run.
- No data augmentation is used, so preprocessing is deterministic
  (`ToTensor` + fixed per-channel CIFAR-10 mean/std normalization).
