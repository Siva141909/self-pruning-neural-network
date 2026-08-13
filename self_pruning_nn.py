"""
The Self-Pruning Neural Network
================================

A feed-forward network for CIFAR-10 classification in which every linear
layer carries a learnable per-weight "gate". Gates are pushed toward zero
by an L1 sparsity penalty, so the network learns to prune its own
connections during ordinary gradient-descent training (no post-training
pruning step).

Mechanism (exactly as specified by the assignment -- not substituted with
any alternative pruning scheme):

    gate_scores  -- learnable, same shape as weight
    gates        = sigmoid(gate_scores)                 in (0, 1)
    eff_weight   = weight * gates
    output       = eff_weight @ input + bias

    ClassificationLoss = CrossEntropy(logits, labels)
    SparsityLoss       = sum(gates) over every PrunableLinear layer
    TotalLoss          = ClassificationLoss + lambda * SparsityLoss

IMPORTANT CAVEAT (see report.md for the full discussion): sigmoid(s) is
strictly > 0 for any finite s. Gates never reach mathematical zero during
training; "pruned" is an *operational* definition (gate < threshold, e.g.
1e-2), not a literal zero. The dense weight/gate tensors are never resized
during training -- pruning here is functional (near-zero multiplicative
gates), not structural.

Usage:
    python test_prunable_linear.py             # sanity tests (run first)
    python self_pruning_nn.py --mode pilot      # short pilot run  (Stage G)
    python self_pruning_nn.py --mode full       # full experiment  (after approval)
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class Config:
    """All tunable knobs for a single training run, in one place."""

    seed: int = 42
    data_dir: str = "./data"
    results_dir: str = "./results"

    batch_size: int = 128
    lr: float = 1e-3

    input_dim: int = 3072  # 3 x 32 x 32, flattened
    hidden_dims: Sequence[int] = (1024, 512, 256)
    num_classes: int = 10

    # sigmoid(3.0) ~= 0.953 -- layers start "mostly open" (close to, but not
    # identical to, an un-gated nn.Linear). See report.md for why this is an
    # approximation, not an equivalence.
    gate_init: float = 3.0

    sparsity_threshold: float = 1e-2

    device: str = field(default_factory=lambda: _detect_device())


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Seed every RNG that affects this experiment's reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_environment(cfg: Config) -> Dict[str, str]:
    """Capture the environment info needed to reproduce a run later."""
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": cfg.device,
        "seed": str(cfg.seed),
    }


# -----------------------------------------------------------------------------
# Stage B: PrunableLinear
# -----------------------------------------------------------------------------


class PrunableLinear(nn.Module):
    """
    A linear layer where every weight has its own learnable multiplicative
    gate in (0, 1). Implemented with plain differentiable tensor ops only
    (sigmoid, elementwise multiply, F.linear) so PyTorch autograd derives
    correct gradients for `weight` and `gate_scores` without any custom
    backward pass, `.detach()`, or in-place tensor surgery.
    """

    def __init__(self, in_features: int, out_features: int, gate_init: float = 3.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Same shape as weight, as required by the assignment.
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), float(gate_init)))

        self._reset_weight_and_bias()

    def _reset_weight_and_bias(self) -> None:
        # Mirrors nn.Linear.reset_parameters exactly, so at gate ~ 1 this
        # layer's weight/bias distribution matches a standard nn.Linear.
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = self.in_features
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        effective_weight = self.weight * gates
        return F.linear(x, effective_weight, self.bias)

    @torch.no_grad()
    def gate_values(self) -> torch.Tensor:
        """Detached gate tensor, for logging/analysis only -- never used in the loss."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# -----------------------------------------------------------------------------
# Stage D: Network
# -----------------------------------------------------------------------------


class SelfPruningMLP(nn.Module):
    """
    Feed-forward classifier for CIFAR-10 built entirely from PrunableLinear
    layers (including the output layer). No CNN layers, BatchNorm, or
    Dropout, per the agreed design -- the assignment specifies a standard
    feed-forward network, and we keep the architecture minimal so that any
    accuracy/sparsity effect we observe is attributable to lambda, not to
    extra architectural machinery.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dims: Sequence[int] = (1024, 512, 256),
        num_classes: int = 10,
        gate_init: float = 3.0,
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims, num_classes]
        self.layers = nn.ModuleList(
            [PrunableLinear(dims[i], dims[i + 1], gate_init=gate_init) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)  # raw logits

    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.layers if isinstance(m, PrunableLinear)]


# -----------------------------------------------------------------------------
# Stage E: Sparsity loss
# -----------------------------------------------------------------------------


def sparsity_loss(model: SelfPruningMLP) -> torch.Tensor:
    """
    SUM (not mean) of every gate value across every PrunableLinear layer.
    Because gates = sigmoid(.) > 0 everywhere, this sum equals the L1 norm
    of the gates, exactly as the assignment specifies.
    """
    total: Optional[torch.Tensor] = None
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        layer_sum = gates.sum()
        total = layer_sum if total is None else total + layer_sum
    if total is None:
        raise ValueError("Model has no PrunableLinear layers -- nothing to regularize.")
    return total


# -----------------------------------------------------------------------------
# Stage F: Evaluation / gate diagnostics
# -----------------------------------------------------------------------------


@dataclass
class GateStats:
    total_gates: int
    pruned_gates: int
    sparsity_pct: float
    min_gate: float
    max_gate: float
    mean_gate: float
    median_gate: float
    q01: float
    q05: float
    q25: float
    q75: float
    q95: float
    q99: float


# Quantile levels requested for diagnosing whether a small population of
# gates is moving toward zero while the majority remain active, vs. a
# uniform shift of the whole distribution.
_QUANTILE_LEVELS = torch.tensor([0.01, 0.05, 0.25, 0.75, 0.95, 0.99])


@torch.no_grad()
def compute_gate_stats(model: SelfPruningMLP, threshold: float) -> GateStats:
    """
    Sparsity is computed over gated WEIGHT elements only (biases have no
    gate, so they are excluded), aggregated across all PrunableLinear
    layers. "Pruned" means gate < threshold, per the assignment's
    operational definition -- not literal mathematical zero.
    """
    all_gates = torch.cat([layer.gate_values().flatten() for layer in model.prunable_layers()])
    total = int(all_gates.numel())
    pruned = int((all_gates < threshold).sum().item())
    # torch.quantile is not implemented for the MPS backend in this torch
    # version, and CPU is plenty fast for a ~3.8M-element tensor -- always
    # compute quantiles on CPU regardless of the training device.
    cpu_gates = all_gates.float().cpu()
    q01, q05, q25, q75, q95, q99 = [float(v) for v in torch.quantile(cpu_gates, _QUANTILE_LEVELS)]
    return GateStats(
        total_gates=total,
        pruned_gates=pruned,
        sparsity_pct=100.0 * pruned / total,
        min_gate=float(all_gates.min().item()),
        max_gate=float(all_gates.max().item()),
        mean_gate=float(all_gates.mean().item()),
        median_gate=float(all_gates.median().item()),
        q01=q01, q05=q05, q25=q25, q75=q75, q95=q95, q99=q99,
    )


@torch.no_grad()
def gate_grad_norm(model: SelfPruningMLP) -> Optional[float]:
    """L2 norm of gate_scores.grad across all layers, whatever grad is currently held."""
    grads = [layer.gate_scores.grad.flatten() for layer in model.prunable_layers() if layer.gate_scores.grad is not None]
    if not grads:
        return None
    return float(torch.cat(grads).norm().item())


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_set = torchvision.datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# Training / evaluation loops
# -----------------------------------------------------------------------------


def train_one_epoch(model: SelfPruningMLP, loader: DataLoader, optimizer: torch.optim.Optimizer, lam: float, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = total_ce = total_sparse = 0.0
    n_batches = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        ce_loss = F.cross_entropy(logits, labels)
        sparse_loss = sparsity_loss(model)
        loss = ce_loss + lam * sparse_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_sparse += sparse_loss.item()
        n_batches += 1

    # gate_scores.grad still holds the *last batch's* gradient here (optimizer.step
    # does not clear it) -- a cheap, indicative (not epoch-averaged) probe of
    # gradient scale for the pilot diagnostics.
    grad_norm = gate_grad_norm(model)

    return {
        "train_loss": total_loss / n_batches,
        "classification_loss": total_ce / n_batches,
        "sparsity_loss": total_sparse / n_batches,
        "gate_grad_norm_last_batch": grad_norm,
    }


@torch.no_grad()
def evaluate(model: SelfPruningMLP, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    lam: float
    epoch_logs: List[dict]
    final_test_accuracy: float
    final_gate_stats: GateStats
    model: SelfPruningMLP


def build_model(cfg: Config) -> SelfPruningMLP:
    return SelfPruningMLP(
        input_dim=cfg.input_dim,
        hidden_dims=cfg.hidden_dims,
        num_classes=cfg.num_classes,
        gate_init=cfg.gate_init,
    )


def run_experiment(
    lam: float,
    cfg: Config,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    checkpoint_dir: Optional[Path] = None,
) -> ExperimentResult:
    """
    Train one model from scratch at a given lambda. Re-seeds before building
    the model so every lambda in a sweep starts from an identical
    initialization -- isolating lambda as the only varying factor.

    If checkpoint_dir is given, saves two checkpoints per run:
      - lambda_<lam>_final.pt: model state after the last epoch.
      - lambda_<lam>_best.pt : model state at the epoch with the highest
        test accuracy seen so far *within this run*. This is an automatic,
        predefined rule applied during training -- not a post-hoc manual
        pick across lambda values after seeing final results.
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None

    epoch_logs: List[dict] = []
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, lam, device)
        test_acc = evaluate(model, test_loader, device)
        gate_stats = compute_gate_stats(model, cfg.sparsity_threshold)

        log = {
            "lambda": lam,
            "epoch": epoch,
            **train_stats,
            "test_accuracy": test_acc,
            "sparsity_pct": gate_stats.sparsity_pct,
            "mean_gate": gate_stats.mean_gate,
            "median_gate": gate_stats.median_gate,
            "min_gate": gate_stats.min_gate,
            "max_gate": gate_stats.max_gate,
            "gate_q01": gate_stats.q01,
            "gate_q05": gate_stats.q05,
            "gate_q25": gate_stats.q25,
            "gate_q75": gate_stats.q75,
            "gate_q95": gate_stats.q95,
            "gate_q99": gate_stats.q99,
        }
        epoch_logs.append(log)
        print(_format_log_line(log))

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"lambda_{lam:g}_final.pt")
        if best_state is not None:
            torch.save(best_state, checkpoint_dir / f"lambda_{lam:g}_best.pt")

    final_gate_stats = compute_gate_stats(model, cfg.sparsity_threshold)
    return ExperimentResult(
        lam=lam,
        epoch_logs=epoch_logs,
        final_test_accuracy=epoch_logs[-1]["test_accuracy"],
        final_gate_stats=final_gate_stats,
        model=model,
    )


def _format_log_line(log: dict) -> str:
    return (
        f"lambda={log['lambda']:.0e} epoch={log['epoch']:>3} "
        f"train_loss={log['train_loss']:.4f} ce={log['classification_loss']:.4f} "
        f"sparse={log['sparsity_loss']:.2f} test_acc={log['test_accuracy']:.2f}% "
        f"sparsity={log['sparsity_pct']:.2f}% mean_gate={log['mean_gate']:.4f} "
        f"median_gate={log['median_gate']:.4f} q01={log['gate_q01']:.4f} "
        f"q99={log['gate_q99']:.4f} grad_norm={log['gate_grad_norm_last_batch']}"
    )


# -----------------------------------------------------------------------------
# Gate-distribution visualization
# -----------------------------------------------------------------------------


def plot_gate_distribution(model: SelfPruningMLP, save_path: Path, threshold: float, title_suffix: str = "") -> None:
    """
    Histogram of every gate value in the model (all PrunableLinear layers
    concatenated), log-scaled y-axis so a large near-zero spike and a
    smaller "active gate" cluster are both visible at once, with a dashed
    line marking the assignment's pruning threshold.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_gates = torch.cat([layer.gate_values().flatten() for layer in model.prunable_layers()]).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(all_gates, bins=60, range=(0.0, 1.0), color="#4C72B0", edgecolor="none")
    ax.axvline(threshold, color="crimson", linestyle="--", linewidth=1.5, label=f"threshold = {threshold:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Gate value = sigmoid(gate_score)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"Final gate value distribution{title_suffix}")
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_gate_distribution_two_panel(model: SelfPruningMLP, save_path: Path, threshold: float, title_suffix: str = "") -> None:
    """
    Two-panel version of the final gate-distribution plot for the report.

    Left panel: the full [0, 1] histogram, log-scaled y-axis, threshold
    marked -- shows the near-zero "pruned" spike at its true scale.
    Right panel: zoomed on gates >= threshold only, x-axis restricted to
    that range (also log-y, since the active population itself spans
    orders of magnitude in count) -- the pruned spike is excluded entirely
    so the "active gate" population's own shape is visible at full
    resolution instead of being squeezed into a corner of the full-range
    plot. This makes it possible to see, at high sparsity, whether the
    remaining active gates form a real cluster near 1.0 or are just thinly
    scattered noise.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_gates = torch.cat([layer.gate_values().flatten() for layer in model.prunable_layers()]).cpu().numpy()
    active_gates = all_gates[all_gates >= threshold]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(all_gates, bins=60, range=(0.0, 1.0), color="#4C72B0", edgecolor="none")
    ax1.axvline(threshold, color="crimson", linestyle="--", linewidth=1.5, label=f"threshold = {threshold:g}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Gate value = sigmoid(gate_score)")
    ax1.set_ylabel("Count (log scale)")
    ax1.set_title("All gates")
    ax1.legend()

    if active_gates.size > 0:
        ax2.hist(active_gates, bins=60, range=(threshold, 1.0), color="#55A868", edgecolor="none")
        ax2.set_yscale("log")
    ax2.set_xlabel("Gate value = sigmoid(gate_score)")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title(f"Active gates only (gate >= {threshold:g}), n={active_gates.size:,}")

    fig.suptitle(f"Final gate value distribution{title_suffix}")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(epoch_logs_csv: Path, metric: str, ylabel: str, title: str, save_path: Path, logy: bool = False) -> None:
    """
    Diagnostic line plot of a per-epoch metric, one line per lambda, read
    directly from a saved epoch-logs CSV (so this can be re-run on any past
    experiment without retraining).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with epoch_logs_csv.open() as f:
        rows = list(csv.DictReader(f))

    by_lambda: Dict[str, List[tuple]] = {}
    for r in rows:
        by_lambda.setdefault(r["lambda"], []).append((int(r["epoch"]), float(r[metric])))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lam_str, points in sorted(by_lambda.items(), key=lambda kv: float(kv[0])):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=3, label=f"lambda={float(lam_str):g}")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------


def save_epoch_logs_csv(all_logs: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_logs[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_logs)


def save_results_table_csv(results: List[ExperimentResult], path: Path) -> None:
    """The exact 3-column table the assignment asks for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Lambda", "Test Accuracy", "Sparsity Level (%)"])
        for r in results:
            writer.writerow([r.lam, f"{r.final_test_accuracy:.2f}", f"{r.final_gate_stats.sparsity_pct:.2f}"])


def save_final_gate_summary_csv(results: List[ExperimentResult], path: Path) -> None:
    """Richer per-lambda diagnostics: full gate distribution summary, not just the headline sparsity %."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Lambda", "Test Accuracy", "Sparsity (%)", "Total Gates", "Pruned Gates",
            "Min", "Q01", "Q05", "Q25", "Median", "Mean", "Q75", "Q95", "Q99", "Max",
        ])
        for r in results:
            gs = r.final_gate_stats
            writer.writerow([
                r.lam, f"{r.final_test_accuracy:.2f}", f"{gs.sparsity_pct:.4f}", gs.total_gates, gs.pruned_gates,
                f"{gs.min_gate:.6f}", f"{gs.q01:.6f}", f"{gs.q05:.6f}", f"{gs.q25:.6f}", f"{gs.median_gate:.6f}",
                f"{gs.mean_gate:.6f}", f"{gs.q75:.6f}", f"{gs.q95:.6f}", f"{gs.q99:.6f}", f"{gs.max_gate:.6f}",
            ])


# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------


def run_pilot(cfg: Config, lambdas: Sequence[float], epochs: int) -> List[ExperimentResult]:
    print(f"=== PILOT RUN === device={cfg.device} epochs={epochs} lambdas={lambdas}")
    print(json.dumps(log_environment(cfg), indent=2))
    train_loader, test_loader = get_dataloaders(cfg)

    results: List[ExperimentResult] = []
    all_logs: List[dict] = []
    for lam in lambdas:
        print(f"\n--- lambda = {lam:g} ---")
        result = run_experiment(lam, cfg, epochs, train_loader, test_loader)
        results.append(result)
        all_logs.extend(result.epoch_logs)

    results_dir = Path(cfg.results_dir)
    save_epoch_logs_csv(all_logs, results_dir / "pilot_epoch_logs.csv")
    save_results_table_csv(results, results_dir / "pilot_summary.csv")

    print("\n=== PILOT SUMMARY (final epoch of each run) ===")
    for r in results:
        gs = r.final_gate_stats
        print(
            f"lambda={r.lam:g}: test_acc={r.final_test_accuracy:.2f}% "
            f"sparsity={gs.sparsity_pct:.2f}% mean_gate={gs.mean_gate:.4f} "
            f"median_gate={gs.median_gate:.4f} min={gs.min_gate:.6f} max={gs.max_gate:.4f}"
        )
    return results


def run_full_experiment(
    cfg: Config,
    lambdas: Sequence[float],
    epochs: int,
    output_prefix: str = "full",
    checkpoint_subdir: str = "checkpoints",
) -> List[ExperimentResult]:
    """
    Runs every lambda with identical architecture/optimizer/LR/batch
    size/epochs/seed/eval procedure, logs everything, and checkpoints every
    run. Deliberately does NOT select a "best model" or generate the final
    gate-distribution plot -- that selection is a separate, explicit step
    (see select_best_model / plot_gate_distribution) taken only after the
    full results table has been reviewed.

    output_prefix/checkpoint_subdir let a follow-up experiment (e.g. an
    extended-epoch run on a subset of lambdas) write to its own files
    instead of overwriting a previous experiment's results.
    """
    print(f"=== FULL EXPERIMENT [{output_prefix}] === device={cfg.device} epochs={epochs} lambdas={lambdas}")
    print(json.dumps(log_environment(cfg), indent=2))
    train_loader, test_loader = get_dataloaders(cfg)

    checkpoint_dir = Path(cfg.results_dir) / checkpoint_subdir
    results: List[ExperimentResult] = []
    all_logs: List[dict] = []
    for lam in lambdas:
        print(f"\n--- lambda = {lam:g} ---")
        result = run_experiment(lam, cfg, epochs, train_loader, test_loader, checkpoint_dir=checkpoint_dir)
        results.append(result)
        all_logs.extend(result.epoch_logs)

    results_dir = Path(cfg.results_dir)
    save_epoch_logs_csv(all_logs, results_dir / f"{output_prefix}_epoch_logs.csv")
    save_results_table_csv(results, results_dir / f"{output_prefix}_results_table.csv")
    save_final_gate_summary_csv(results, results_dir / f"{output_prefix}_gate_summary.csv")

    print("\n=== FULL EXPERIMENT SUMMARY (final epoch of each run) ===")
    for r in results:
        gs = r.final_gate_stats
        print(
            f"lambda={r.lam:g}: test_acc={r.final_test_accuracy:.2f}% sparsity={gs.sparsity_pct:.4f}% "
            f"mean={gs.mean_gate:.4f} median={gs.median_gate:.4f} "
            f"q01={gs.q01:.4f} q05={gs.q05:.4f} q25={gs.q25:.4f} q75={gs.q75:.4f} q95={gs.q95:.4f} q99={gs.q99:.4f} "
            f"min={gs.min_gate:.6f} max={gs.max_gate:.4f} pruned={gs.pruned_gates}/{gs.total_gates}"
        )
    print("\nBest-model selection and the final gate-distribution plot are deferred pending review.")
    return results


def select_best_model(results: List[ExperimentResult], accuracy_drop_budget: float = 5.0) -> tuple[Optional[ExperimentResult], List[ExperimentResult]]:
    """
    Predefined rule: highest sparsity among runs within `accuracy_drop_budget`
    points of the lambda=0 baseline. Returns (best_or_None, candidates) so the
    caller can inspect which runs qualified before accepting the pick --
    returns None for best if no run qualifies, rather than forcing a choice.
    """
    baseline = next((r for r in results if r.lam == 0), None)
    if baseline is None:
        return None, []
    candidates = [r for r in results if baseline.final_test_accuracy - r.final_test_accuracy <= accuracy_drop_budget]
    if not candidates:
        return None, []
    best = max(candidates, key=lambda r: r.final_gate_stats.sparsity_pct)
    return best, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network on CIFAR-10")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epoch count for the chosen mode.")
    parser.add_argument("--lambdas", type=float, nargs="+", default=None, help="Override default lambda grid.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", type=str, default="full", help="Filename prefix for full-mode results (keeps separate experiments from overwriting each other).")
    parser.add_argument("--checkpoint-subdir", type=str, default="checkpoints", help="Subdirectory under results/ for full-mode checkpoints.")
    parser.add_argument("--diagnostic-plots", action="store_true", help="After a full-mode run, save sparsity/accuracy/mean-gate-vs-epoch line plots.")
    args = parser.parse_args()

    cfg = Config(seed=args.seed)

    if args.mode == "pilot":
        lambdas = args.lambdas if args.lambdas is not None else [0.0, 1e-7, 1e-6, 1e-5]
        epochs = args.epochs if args.epochs is not None else 5
        run_pilot(cfg, lambdas, epochs)
    else:
        lambdas = args.lambdas if args.lambdas is not None else [0.0, 1e-7, 1e-6, 1e-5, 1e-4]
        epochs = args.epochs if args.epochs is not None else 25
        run_full_experiment(cfg, lambdas, epochs, output_prefix=args.output_prefix, checkpoint_subdir=args.checkpoint_subdir)

        if args.diagnostic_plots:
            results_dir = Path(cfg.results_dir)
            logs_csv = results_dir / f"{args.output_prefix}_epoch_logs.csv"
            plot_training_curves(logs_csv, "sparsity_pct", "Sparsity (%)", f"Sparsity vs Epoch [{args.output_prefix}]", results_dir / f"{args.output_prefix}_diagnostic_sparsity_vs_epoch.png")
            plot_training_curves(logs_csv, "test_accuracy", "Test Accuracy (%)", f"Test Accuracy vs Epoch [{args.output_prefix}]", results_dir / f"{args.output_prefix}_diagnostic_accuracy_vs_epoch.png")
            plot_training_curves(logs_csv, "mean_gate", "Mean Gate", f"Mean Gate vs Epoch [{args.output_prefix}]", results_dir / f"{args.output_prefix}_diagnostic_mean_gate_vs_epoch.png")
            print(f"\nDiagnostic plots saved to {results_dir} with prefix '{args.output_prefix}_diagnostic_*.png'")


if __name__ == "__main__":
    main()
