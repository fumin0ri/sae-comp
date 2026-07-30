from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .config import ExperimentConfig

plt.switch_backend("Agg")


COLORS = {
    "standard": "#4C78A8",
    "temporal": "#F58518",
    "proposal_w016": "#54A24B",
    "proposal_w032": "#E45756",
    "proposal_w064": "#B279A2",
}


def display_name(label: str) -> str:
    if label == "standard":
        return "Standard Top-K"
    if label == "temporal":
        return "Temporal SAE"
    if label.startswith("proposal_w"):
        return f"Proposal W={int(label.removeprefix('proposal_w'))}"
    return label


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _bar_grid(
    methods: dict[str, dict[str, Any]],
    metrics: list[tuple[str, str]],
    destination: Path,
    title: str,
) -> None:
    labels = list(methods)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    figure.suptitle(title, fontsize=16, fontweight="bold")
    for axis, (key, axis_title) in zip(axes.flat, metrics, strict=True):
        values = [methods[label][key] for label in labels]
        bars = axis.bar(
            range(len(labels)),
            values,
            color=[COLORS.get(label, "#777777") for label in labels],
        )
        axis.set_title(axis_title)
        axis.set_xticks(
            range(len(labels)),
            [display_name(label) for label in labels],
            rotation=25,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    _save_figure(figure, destination)


def _plot_common_metrics(metrics: dict[str, Any], output_dir: Path) -> Path:
    destination = output_dir / "common_metrics.png"
    _bar_grid(
        metrics["methods"],
        [
            ("fve", "Fraction of variance explained (higher is better)"),
            ("cosine_similarity", "Reconstruction cosine (higher is better)"),
            ("fraction_alive", "Fraction of alive features (higher is better)"),
            ("l0", "Observed L0"),
        ],
        destination,
        "Controlled SAE comparison: common metrics",
    )
    return destination


def _plot_smoothness(metrics: dict[str, Any], output_dir: Path) -> Path:
    destination = output_dir / "temporal_smoothness.png"
    smoothness = {
        label: values["smoothness"]["full"]
        for label, values in metrics["methods"].items()
    }
    _bar_grid(
        smoothness,
        [
            ("lipschitz", "Lipschitz score (lower is smoother)"),
            ("fourier", "High/low Fourier energy (lower is smoother)"),
            ("wavelet", "Wavelet detail ratio (lower is smoother)"),
            ("multiscale", "Multiscale variation ratio (lower is smoother)"),
        ],
        destination,
        "Controlled SAE comparison: temporal smoothness",
    )
    return destination


def _plot_probes(
    probes: list[dict[str, Any]], cfg: ExperimentConfig, output_dir: Path
) -> Path:
    destination = output_dir / "mmlu_probes.png"
    methods = list(dict.fromkeys(item["method"] for item in probes))
    tasks = ("semantics", "context", "syntax")
    sparsities: list[int | str] = list(cfg.evaluation.probe_sparsities)
    if cfg.evaluation.probe_include_dense:
        sparsities.append("dense")
    lookup = {
        (item["method"], item["task"], str(item["sparsity"])): item["accuracy"]
        for item in probes
        if item["split"] == "full"
    }
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    figure.suptitle(
        "MMLU sparse linear probes (higher is better)",
        fontsize=16,
        fontweight="bold",
    )
    x = list(range(len(sparsities)))
    for axis, task in zip(axes, tasks, strict=True):
        for method in methods:
            values = [
                lookup.get((method, task, str(sparsity)), float("nan"))
                for sparsity in sparsities
            ]
            axis.plot(
                x,
                values,
                marker="o",
                linewidth=2,
                color=COLORS.get(method, "#777777"),
                label=display_name(method),
            )
        axis.set_title(task.capitalize())
        axis.set_xticks(x, [str(value) for value in sparsities])
        axis.set_xlabel("Features per class")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    axes[-1].legend(loc="lower right", fontsize=8)
    _save_figure(figure, destination)
    return destination


def _plot_forecasts(metrics: dict[str, Any], output_dir: Path) -> Path:
    destination = output_dir / "forecast_diagnostics.png"
    forecasts = metrics["proposal_forecasts"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    figure.suptitle(
        "Fixed-endpoint forecast diagnostics by horizon",
        fontsize=16,
        fontweight="bold",
    )
    for label, values in forecasts.items():
        offsets = [item["offset"] for item in values["offsets"]]
        axes[0].plot(
            offsets,
            [item["code_cosine"] for item in values["offsets"]],
            color=COLORS.get(label, "#777777"),
            label=display_name(label),
        )
        axes[1].plot(
            offsets,
            [item["true_minus_shuffled"] for item in values["offsets"]],
            color=COLORS.get(label, "#777777"),
            label=display_name(label),
        )
    common_horizon = min(
        values["common_horizon_max_offset"] for values in forecasts.values()
    )
    for axis in axes:
        axis.axvline(
            common_horizon,
            color="#333333",
            linestyle="--",
            linewidth=1,
            label="Common comparison horizon",
        )
        axis.set_xlabel("Forecast horizon")
        axis.grid(alpha=0.25)
    axes[0].set_title("Target-code cosine")
    axes[0].set_ylabel("Cosine similarity")
    axes[1].set_title("True-context minus shuffled-context cosine")
    axes[1].set_ylabel("Cosine difference")
    axes[1].legend(loc="best", fontsize=8)
    _save_figure(figure, destination)
    return destination


def _plot_training(cfg: ExperimentConfig, output_dir: Path) -> Path:
    run_dir = Path(cfg.run_dir)
    controls = json.loads(
        (run_dir / "controlled_training_history.json").read_text(encoding="utf-8")
    )
    proposals = json.loads(
        (run_dir / "window_sweep_training_history.json").read_text(encoding="utf-8")
    )["histories"]
    series = {
        "standard": (
            controls["standard"],
            "fvu",
        ),
        "temporal": (
            controls["temporal"],
            "full_fvu",
        ),
    }
    series.update(
        {
            label: (history, "online_reconstruction_fvu")
            for label, history in proposals.items()
        }
    )
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for label, (history, metric) in series.items():
        axis.plot(
            [item["step"] for item in history],
            [item[metric] for item in history],
            color=COLORS.get(label, "#777777"),
            linewidth=2,
            label=display_name(label),
        )
    axis.set_title(
        "Branch training reconstruction FVU", fontsize=16, fontweight="bold"
    )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("FVU (lower is better)")
    axis.grid(alpha=0.25)
    axis.legend()
    destination = output_dir / "training_reconstruction.png"
    _save_figure(figure, destination)
    return destination


def build_controlled_plots(cfg: ExperimentConfig) -> list[Path]:
    evaluation_dir = Path(cfg.run_dir) / "evaluation"
    output_dir = evaluation_dir / "plots"
    metrics = json.loads(
        (evaluation_dir / "controlled_metrics.json").read_text(encoding="utf-8")
    )
    probes = json.loads(
        (evaluation_dir / "controlled_probes.json").read_text(encoding="utf-8")
    )
    paths = [
        _plot_common_metrics(metrics, output_dir),
        _plot_smoothness(metrics, output_dir),
        _plot_probes(probes, cfg, output_dir),
        _plot_forecasts(metrics, output_dir),
        _plot_training(cfg, output_dir),
    ]
    (output_dir / "manifest.json").write_text(
        json.dumps(
            [str(path.relative_to(Path(cfg.run_dir))) for path in paths], indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
