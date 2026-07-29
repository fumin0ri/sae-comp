from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ExperimentConfig
from .evaluation import controlled_checkpoint_paths
from .plots import COLORS, display_name
from .saebench import CUSTOM_SAE_ID


def _load_eval_result(root: Path, eval_type: str, label: str) -> dict[str, Any]:
    path = root / eval_type / f"{label}_{CUSTOM_SAE_ID}_eval_results.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing SAEBench result for {eval_type}/{label}: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def collect_saebench_summary(cfg: ExperimentConfig) -> dict[str, Any]:
    root = Path(cfg.run_dir) / "saebench_results"
    labels = list(controlled_checkpoint_paths(cfg))
    summary: dict[str, Any] = {
        "saebench_version": cfg.sae_bench.version,
        "model_name": cfg.sae_bench.model_name,
        "eval_types": list(cfg.sae_bench.eval_types),
        "excluded_eval_types": list(cfg.sae_bench.excluded_eval_types),
        "conditions": {},
    }
    for label in labels:
        condition: dict[str, Any] = {}
        if "core" in cfg.sae_bench.eval_types:
            result = _load_eval_result(root, "core", label)
            metrics = result["eval_result_metrics"]
            condition["core"] = {
                "explained_variance": metrics["reconstruction_quality"][
                    "explained_variance"
                ],
                "mse": metrics["reconstruction_quality"]["mse"],
                "ce_loss_score": metrics["model_performance_preservation"][
                    "ce_loss_score"
                ],
                "l0": metrics["sparsity"]["l0"],
            }
        if "sparse_probing" in cfg.sae_bench.eval_types:
            result = _load_eval_result(root, "sparse_probing", label)
            metrics = result["eval_result_metrics"]["sae"]
            condition["sparse_probing"] = {
                f"top_{k}_accuracy": metrics[f"sae_top_{k}_test_accuracy"]
                for k in (1, 2, 5)
            }
        if "sparse_probing_sae_probes" in cfg.sae_bench.eval_types:
            result = _load_eval_result(
                root, "sparse_probing_sae_probes", label
            )
            metrics = result["eval_result_metrics"]["sae"]
            condition["sparse_probing_sae_probes"] = {
                f"top_{k}_accuracy": metrics[f"sae_top_{k}_test_accuracy"]
                for k in (1, 2, 5)
            }
        if "ravel" in cfg.sae_bench.eval_types:
            result = _load_eval_result(root, "ravel", label)
            metrics = result["eval_result_metrics"]["ravel"]
            condition["ravel"] = {
                "disentanglement_score": metrics["disentanglement_score"],
                "cause_score": metrics["cause_score"],
                "isolation_score": metrics["isolation_score"],
            }
        summary["conditions"][label] = condition

    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_summary_csv(root / "summary.csv", summary)
    return summary


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    metric_names = [
        "core.explained_variance",
        "core.mse",
        "core.ce_loss_score",
        "core.l0",
        "sparse_probing.top_1_accuracy",
        "sparse_probing.top_2_accuracy",
        "sparse_probing.top_5_accuracy",
        "sparse_probing_sae_probes.top_1_accuracy",
        "sparse_probing_sae_probes.top_2_accuracy",
        "sparse_probing_sae_probes.top_5_accuracy",
        "ravel.disentanglement_score",
        "ravel.cause_score",
        "ravel.isolation_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", *metric_names])
        writer.writeheader()
        for label, condition in summary["conditions"].items():
            row: dict[str, Any] = {"method": label}
            for metric_name in metric_names:
                section, metric = metric_name.split(".", maxsplit=1)
                row[metric_name] = condition.get(section, {}).get(metric)
            writer.writerow(row)


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_core(summary: dict[str, Any], output_dir: Path) -> Path:
    labels = list(summary["conditions"])
    metrics = (
        ("explained_variance", "Explained variance", "higher is better"),
        ("ce_loss_score", "CE loss score", "higher is better"),
        ("l0", "Observed L0", "matched sparsity"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    figure.suptitle("SAEBench core evaluation", fontsize=16, fontweight="bold")
    for axis, (key, title, direction) in zip(axes, metrics, strict=True):
        values = [summary["conditions"][label]["core"][key] for label in labels]
        bars = axis.bar(
            range(len(labels)),
            values,
            color=[COLORS.get(label, "#777777") for label in labels],
        )
        axis.set_title(f"{title}\n({direction})")
        axis.set_xticks(
            range(len(labels)),
            [display_name(label) for label in labels],
            rotation=28,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    return _save_figure(figure, output_dir / "core.png")


def _plot_probing(summary: dict[str, Any], output_dir: Path) -> Path:
    labels = list(summary["conditions"])
    sections = ("sparse_probing", "sparse_probing_sae_probes")
    titles = ("SAEBench sparse probing", "SAEBench SAE-Probes")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    figure.suptitle(
        "Sparse feature probing accuracy", fontsize=16, fontweight="bold"
    )
    for axis, section, title in zip(axes, sections, titles, strict=True):
        for label in labels:
            values = [
                summary["conditions"][label][section][f"top_{k}_accuracy"]
                for k in (1, 2, 5)
            ]
            axis.plot(
                (1, 2, 5),
                values,
                marker="o",
                linewidth=2,
                color=COLORS.get(label, "#777777"),
                label=display_name(label),
            )
        axis.set_title(title)
        axis.set_xlabel("Selected SAE features")
        axis.set_ylabel("Accuracy")
        axis.set_xticks((1, 2, 5))
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    axes[-1].legend(loc="lower right", fontsize=8)
    return _save_figure(figure, output_dir / "probing.png")


def _plot_ravel(summary: dict[str, Any], output_dir: Path) -> Path:
    labels = list(summary["conditions"])
    metrics = (
        ("disentanglement_score", "Disentanglement"),
        ("cause_score", "Cause"),
        ("isolation_score", "Isolation"),
    )
    width = 0.8 / len(labels)
    x_values = list(range(len(metrics)))
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for index, label in enumerate(labels):
        positions = [
            x - 0.4 + width / 2 + index * width for x in x_values
        ]
        values = [
            summary["conditions"][label]["ravel"][key] for key, _ in metrics
        ]
        axis.bar(
            positions,
            values,
            width=width,
            color=COLORS.get(label, "#777777"),
            label=display_name(label),
        )
    axis.set_title("SAEBench RAVEL (higher is better)", fontsize=16, fontweight="bold")
    axis.set_xticks(x_values, [title for _, title in metrics])
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.02)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    return _save_figure(figure, output_dir / "ravel.png")


def build_saebench_report(cfg: ExperimentConfig) -> Path:
    summary = collect_saebench_summary(cfg)
    run_dir = Path(cfg.run_dir)
    output_dir = run_dir / "saebench_results" / "plots"
    plots: list[Path] = []
    if "core" in cfg.sae_bench.eval_types:
        plots.append(_plot_core(summary, output_dir))
    if {
        "sparse_probing",
        "sparse_probing_sae_probes",
    }.issubset(cfg.sae_bench.eval_types):
        plots.append(_plot_probing(summary, output_dir))
    if "ravel" in cfg.sae_bench.eval_types:
        plots.append(_plot_ravel(summary, output_dir))

    lines = [
        "# SAEBench controlled SAE comparison",
        "",
        "## Setup",
        "",
        f"- SAEBench: `{cfg.sae_bench.version}`",
        (
            f"- Frozen model: `{cfg.sae_bench.model_name}`, "
            f"`blocks.{cfg.model.layer}.hook_resid_post`"
        ),
        f"- Dictionary: {cfg.sae.dictionary_size:,} features; target k={cfg.sae.k}",
        f"- Shared pretraining: {cfg.train.standard_steps:,} steps",
        f"- Branch training: {cfg.train.branch_steps:,} steps for every condition",
        (
            "- Compared conditions: Standard Top-K SAE, Temporal SAE, and proposal "
            "W=16, W=32, W=64"
        ),
        "- Explicitly excluded SAEBench evaluations: SCR and TPP",
        "",
        (
            "Every proposal width uses the same reconstruction-token and forecast-pair "
            "budget. All five conditions are evaluated with the same SAEBench settings."
        ),
        "",
    ]
    conditions = summary["conditions"]
    labels = list(conditions)
    if "core" in cfg.sae_bench.eval_types:
        lines.extend(
            [
                "## Core",
                "",
                "| Method | Explained variance | MSE | CE loss score | L0 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label in labels:
            values = conditions[label]["core"]
            lines.append(
                f"| {display_name(label)} | "
                f"{_number(values['explained_variance'])} | "
                f"{_number(values['mse'])} | "
                f"{_number(values['ce_loss_score'])} | "
                f"{_number(values['l0'])} |"
            )
        lines.append("")
    for section, title in (
        ("sparse_probing", "Sparse probing"),
        ("sparse_probing_sae_probes", "Sparse probing with SAE-Probes"),
    ):
        if section not in cfg.sae_bench.eval_types:
            continue
        lines.extend(
            [
                f"## {title}",
                "",
                "| Method | Top-1 | Top-2 | Top-5 |",
                "|---|---:|---:|---:|",
            ]
        )
        for label in labels:
            values = conditions[label][section]
            lines.append(
                f"| {display_name(label)} | "
                f"{_number(values['top_1_accuracy'])} | "
                f"{_number(values['top_2_accuracy'])} | "
                f"{_number(values['top_5_accuracy'])} |"
            )
        lines.append("")
    if "ravel" in cfg.sae_bench.eval_types:
        lines.extend(
            [
                "## RAVEL",
                "",
                "| Method | Disentanglement | Cause | Isolation |",
                "|---|---:|---:|---:|",
            ]
        )
        for label in labels:
            values = conditions[label]["ravel"]
            lines.append(
                f"| {display_name(label)} | "
                f"{_number(values['disentanglement_score'])} | "
                f"{_number(values['cause_score'])} | "
                f"{_number(values['isolation_score'])} |"
            )
        lines.append("")
    lines.extend(["## Figures", ""])
    for plot in plots:
        relative = plot.relative_to(run_dir).as_posix()
        lines.extend(
            [
                f"### {plot.stem.replace('_', ' ').title()}",
                "",
                f"![{plot.stem}]({relative})",
                "",
            ]
        )
    lines.extend(
        [
            (
                "Raw official result JSON files, the run manifest, and machine-readable "
                "summary files are under `saebench_results/`."
            ),
            "",
        ]
    )
    destination = run_dir / "SAEBENCH_REPORT.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
