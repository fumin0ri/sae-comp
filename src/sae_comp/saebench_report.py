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


def _result_path(root: Path, eval_type: str, label: str) -> Path:
    return root / eval_type / f"{label}_{CUSTOM_SAE_ID}_eval_results.json"


def _load_eval_result(
    root: Path, eval_type: str, label: str
) -> dict[str, Any] | None:
    path = _result_path(root, eval_type, label)
    if not path.is_file():
        return None
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
        "availability": {
            eval_type: {label: False for label in labels}
            for eval_type in cfg.sae_bench.eval_types
        },
    }
    for label in labels:
        condition: dict[str, Any] = {}
        if "core" in cfg.sae_bench.eval_types:
            result = _load_eval_result(root, "core", label)
            if result is not None:
                summary["availability"]["core"][label] = True
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
            if result is not None:
                summary["availability"]["sparse_probing"][label] = True
                metrics = result["eval_result_metrics"]["sae"]
                condition["sparse_probing"] = {
                    f"top_{k}_accuracy": metrics[f"sae_top_{k}_test_accuracy"]
                    for k in (1, 2, 5)
                }
        if "sparse_probing_sae_probes" in cfg.sae_bench.eval_types:
            result = _load_eval_result(
                root, "sparse_probing_sae_probes", label
            )
            if result is not None:
                summary["availability"]["sparse_probing_sae_probes"][label] = True
                metrics = result["eval_result_metrics"]["sae"]
                condition["sparse_probing_sae_probes"] = {
                    f"top_{k}_accuracy": metrics[f"sae_top_{k}_test_accuracy"]
                    for k in (1, 2, 5)
                }
        if "ravel" in cfg.sae_bench.eval_types:
            result = _load_eval_result(root, "ravel", label)
            if result is not None:
                summary["availability"]["ravel"][label] = True
                metrics = result["eval_result_metrics"]["ravel"]
                condition["ravel"] = {
                    "disentanglement_score": metrics["disentanglement_score"],
                    "cause_score": metrics["cause_score"],
                    "isolation_score": metrics["isolation_score"],
                }
        summary["conditions"][label] = condition

    summary["completed_eval_types"] = [
        eval_type
        for eval_type, availability in summary["availability"].items()
        if all(availability.values())
    ]
    summary["available_eval_types"] = [
        eval_type
        for eval_type, availability in summary["availability"].items()
        if any(availability.values())
    ]
    summary["missing_results"] = [
        str(_result_path(root, eval_type, label))
        for eval_type, availability in summary["availability"].items()
        for label, available in availability.items()
        if not available
    ]
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


def _grouped_bars(
    axis: plt.Axes,
    summary: dict[str, Any],
    metrics: tuple[tuple[str, str, str], ...],
) -> None:
    labels = list(summary["conditions"])
    width = 0.8 / len(labels)
    x_values = list(range(len(metrics)))
    for index, label in enumerate(labels):
        positions = [x - 0.4 + width / 2 + index * width for x in x_values]
        values = [
            summary["conditions"][label][section][metric]
            for section, metric, _ in metrics
        ]
        axis.bar(
            positions,
            values,
            width=width,
            color=COLORS.get(label, "#777777"),
            label=display_name(label),
        )
    axis.set_xticks(x_values, [title for _, _, title in metrics])
    axis.grid(axis="y", alpha=0.25)


def _plot_overview(summary: dict[str, Any], output_dir: Path, target_l0: int) -> Path:
    labels = list(summary["conditions"])
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    figure.suptitle(
        "SAEBench controlled comparison overview",
        fontsize=18,
        fontweight="bold",
    )

    _grouped_bars(
        axes[0, 0],
        summary,
        (
            ("core", "explained_variance", "Explained\nvariance"),
            ("core", "ce_loss_score", "CE loss\nscore"),
        ),
    )
    axes[0, 0].set_title("Reconstruction and model preservation")
    axes[0, 0].set_ylabel("Score (higher is better)")
    axes[0, 0].set_ylim(0, 1.02)

    _grouped_bars(
        axes[0, 1],
        summary,
        (
            ("sparse_probing", "top_5_accuracy", "Sparse probing"),
            (
                "sparse_probing_sae_probes",
                "top_5_accuracy",
                "SAE-Probes",
            ),
        ),
    )
    axes[0, 1].set_title("Top-5 sparse feature probing")
    axes[0, 1].set_ylabel("Accuracy (higher is better)")
    axes[0, 1].set_ylim(0, 1.02)

    _grouped_bars(
        axes[1, 0],
        summary,
        (
            ("ravel", "disentanglement_score", "Disentangle"),
            ("ravel", "cause_score", "Cause"),
            ("ravel", "isolation_score", "Isolation"),
        ),
    )
    axes[1, 0].set_title("RAVEL intervention metrics")
    axes[1, 0].set_ylabel("Score (higher is better)")
    axes[1, 0].set_ylim(0, 1.02)

    l0_values = [summary["conditions"][label]["core"]["l0"] for label in labels]
    bars = axes[1, 1].bar(
        range(len(labels)),
        l0_values,
        color=[COLORS.get(label, "#777777") for label in labels],
    )
    axes[1, 1].axhline(
        target_l0,
        color="#222222",
        linestyle="--",
        linewidth=1.5,
        label=f"Training target k={target_l0}",
    )
    axes[1, 1].set_title("Observed sparsity sanity check")
    axes[1, 1].set_ylabel("Mean active features (L0)")
    axes[1, 1].set_xticks(
        range(len(labels)),
        [display_name(label) for label in labels],
        rotation=25,
        ha="right",
    )
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axes[1, 1].legend(fontsize=8)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=len(labels),
        fontsize=9,
    )
    return _save_figure(figure, output_dir / "overview.png")


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
    metrics = (
        ("ravel", "disentanglement_score", "Disentanglement"),
        ("ravel", "cause_score", "Cause"),
        ("ravel", "isolation_score", "Isolation"),
    )
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    _grouped_bars(axis, summary, metrics)
    axis.set_title("SAEBench RAVEL (higher is better)", fontsize=16, fontweight="bold")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.02)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    return _save_figure(figure, output_dir / "ravel.png")


def _point_estimate_rows(summary: dict[str, Any]) -> list[tuple[str, str, float]]:
    metrics = (
        ("Explained variance", "core", "explained_variance", True),
        ("Reconstruction MSE", "core", "mse", False),
        ("CE loss score", "core", "ce_loss_score", True),
        ("Sparse probing Top-5", "sparse_probing", "top_5_accuracy", True),
        (
            "SAE-Probes Top-5",
            "sparse_probing_sae_probes",
            "top_5_accuracy",
            True,
        ),
        ("RAVEL disentanglement", "ravel", "disentanglement_score", True),
        ("RAVEL cause", "ravel", "cause_score", True),
        ("RAVEL isolation", "ravel", "isolation_score", True),
    )
    rows: list[tuple[str, str, float]] = []
    for title, section, metric, higher_is_better in metrics:
        values = {
            label: condition[section][metric]
            for label, condition in summary["conditions"].items()
        }
        best_value = (
            max(values.values()) if higher_is_better else min(values.values())
        )
        best_labels = [
            display_name(label)
            for label, value in values.items()
            if abs(value - best_value) <= 1e-12
        ]
        rows.append((title, ", ".join(best_labels), best_value))
    return rows


def build_saebench_report(cfg: ExperimentConfig) -> Path:
    summary = collect_saebench_summary(cfg)
    run_dir = Path(cfg.run_dir)
    output_dir = run_dir / "saebench_results" / "plots"
    plots: list[Path] = []
    completed_eval_types = set(summary["completed_eval_types"])
    available_eval_types = set(summary["available_eval_types"])
    complete_overview = {
        "core",
        "sparse_probing",
        "sparse_probing_sae_probes",
        "ravel",
    }.issubset(completed_eval_types)
    overview_path: Path | None = None
    if complete_overview:
        overview_path = _plot_overview(summary, output_dir, cfg.sae.k)
        plots.append(overview_path)
    if "core" in completed_eval_types:
        plots.append(_plot_core(summary, output_dir))
    if {
        "sparse_probing",
        "sparse_probing_sae_probes",
    }.issubset(completed_eval_types):
        plots.append(_plot_probing(summary, output_dir))
    if "ravel" in completed_eval_types:
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
    lines.extend(
        [
            "## Evaluation completion",
            "",
            "| Evaluation | Completed conditions | Missing conditions |",
            "|---|---:|---|",
        ]
    )
    for eval_type in cfg.sae_bench.eval_types:
        availability = summary["availability"][eval_type]
        completed = sum(availability.values())
        missing = [
            display_name(label)
            for label, available in availability.items()
            if not available
        ]
        lines.append(
            f"| `{eval_type}` | {completed}/{len(availability)} | "
            f"{', '.join(missing) if missing else '-'} |"
        )
    lines.append("")
    if summary["missing_results"]:
        lines.extend(
            [
                (
                    "> **Partial report:** some SAEBench evaluations have not completed. "
                    "The tables and figures below use only available official result "
                    "files. Resume with "
                    "`sae-comp saebench --config configs/controlled_rtx4090.toml`, "
                    "then regenerate this report."
                ),
                "",
            ]
        )
    if overview_path is not None:
        relative = overview_path.relative_to(run_dir).as_posix()
        lines.extend(
            [
                "## Overview",
                "",
                f"![SAEBench comparison overview]({relative})",
                "",
                "### Best point estimates",
                "",
                "| Metric | Best condition | Value |",
                "|---|---|---:|",
            ]
        )
        for metric, label, value in _point_estimate_rows(summary):
            lines.append(f"| {metric} | {label} | {_number(value)} |")
        lines.extend(
            [
                "",
                (
                    "These are descriptive point estimates, not "
                    "statistical-significance claims. L0 is shown as a sparsity "
                    "sanity check and is not ranked."
                ),
                "",
            ]
        )
    conditions = summary["conditions"]
    labels = list(conditions)
    if "core" in available_eval_types:
        lines.extend(
            [
                "## Core",
                "",
                "| Method | Explained variance | MSE | CE loss score | L0 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label in labels:
            values = conditions[label].get("core", {})
            lines.append(
                f"| {display_name(label)} | "
                f"{_number(values.get('explained_variance'))} | "
                f"{_number(values.get('mse'))} | "
                f"{_number(values.get('ce_loss_score'))} | "
                f"{_number(values.get('l0'))} |"
            )
        lines.append("")
    for section, title in (
        ("sparse_probing", "Sparse probing"),
        ("sparse_probing_sae_probes", "Sparse probing with SAE-Probes"),
    ):
        if section not in available_eval_types:
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
            values = conditions[label].get(section, {})
            lines.append(
                f"| {display_name(label)} | "
                f"{_number(values.get('top_1_accuracy'))} | "
                f"{_number(values.get('top_2_accuracy'))} | "
                f"{_number(values.get('top_5_accuracy'))} |"
            )
        lines.append("")
    if "ravel" in available_eval_types:
        lines.extend(
            [
                "## RAVEL",
                "",
                "| Method | Disentanglement | Cause | Isolation |",
                "|---|---:|---:|---:|",
            ]
        )
        for label in labels:
            values = conditions[label].get("ravel", {})
            lines.append(
                f"| {display_name(label)} | "
                f"{_number(values.get('disentanglement_score'))} | "
                f"{_number(values.get('cause_score'))} | "
                f"{_number(values.get('isolation_score'))} |"
            )
        lines.append("")
    detail_plots = [plot for plot in plots if plot != overview_path]
    if detail_plots:
        lines.extend(["## Detailed figures", ""])
        for plot in detail_plots:
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
