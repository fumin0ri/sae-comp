from __future__ import annotations

import json
from pathlib import Path

from .config import ExperimentConfig


def _number(value: float) -> str:
    return f"{value:.4f}"


def build_report(cfg: ExperimentConfig) -> Path:
    run_dir = Path(cfg.run_dir)
    evaluation_dir = run_dir / "evaluation"
    metrics = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    probes = json.loads((evaluation_dir / "probes.json").read_text(encoding="utf-8"))
    lines = [
        "# SAE comparison report",
        "",
        "## Common SAE metrics",
        "",
        "| Method | FVE | Cosine | Alive | L0 | Lipschitz | Fourier | Wavelet | Multiscale |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("standard", "temporal", "proposal"):
        values = metrics["common"][method]
        smooth = values["smoothness"]["full"]
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    _number(values["fve"]),
                    _number(values["cosine_similarity"]),
                    _number(values["fraction_alive"]),
                    _number(values["l0"]),
                    _number(smooth["lipschitz"]),
                    _number(smooth["fourier"]),
                    _number(smooth["wavelet"]),
                    _number(smooth["multiscale"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Lower values indicate smoother features for all four smoothness metrics.",
            "",
            "## MMLU sparse probes",
            "",
            "| Method | Split | Task | Sparsity | Accuracy |",
            "|---|---|---|---:|---:|",
        ]
    )
    for result in probes:
        if result["split"] == "full":
            lines.append(
                f"| {result['method']} | {result['split']} | "
                f"{result['task']} | {result['sparsity']} | "
                f"{_number(result['accuracy'])} |"
            )
    forecast = metrics["proposal_forecast"]
    lines.extend(
        [
            "",
            "## Proposal-specific forecast diagnostic",
            "",
            f"- Mean target-code cosine: {_number(forecast['mean_code_cosine'])}",
            "- Mean true-context minus shuffled-context cosine: "
            f"{_number(forecast['mean_true_minus_shuffled'])}",
            "",
            "This forecast diagnostic is not a three-way common metric. "
            "The common reconstruction, smoothness, and probe tables are the "
            "confirmatory cross-method comparison.",
            "",
        ]
    )
    destination = run_dir / "REPORT.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def build_window_sweep_report(cfg: ExperimentConfig) -> Path:
    run_dir = Path(cfg.run_dir)
    evaluation_dir = run_dir / "evaluation"
    metrics = json.loads(
        (evaluation_dir / "window_sweep.json").read_text(encoding="utf-8")
    )
    probes = json.loads(
        (evaluation_dir / "window_sweep_probes.json").read_text(encoding="utf-8")
    )
    preferred_sparsity: int | str = (
        5 if 5 in cfg.evaluation.probe_sparsities else cfg.evaluation.probe_sparsities[0]
    )
    probe_lookup = {
        (item["method"], item["task"]): item["accuracy"]
        for item in probes
        if item["split"] == "full" and item["sparsity"] == preferred_sparsity
    }
    lines = [
        "# Proposal window-width sweep",
        "",
        "Every condition uses the same shared SAE initialization, optimizer-step "
        "count, reconstruction-token count, forecast-pair count, and pool of "
        f"sequences with at least {max(cfg.proposal.window_sizes)} valid tokens.",
        "",
        "| W | Batch windows | Forecast offsets/window | Total reconstruction tokens | "
        "Total forecast pairs | FVE | Cosine | Alive | L0 | Forecast cosine (1..7) | "
        "True-shuffled (1..7) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window_size in cfg.proposal.window_sizes:
        label = f"proposal_w{window_size:03d}"
        values = metrics[label]
        budget = values["training_budget"]
        common = values["common"]
        forecast = values["forecast"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(window_size),
                    str(budget["batch_windows"]),
                    str(budget["forecast_offsets_per_window"]),
                    str(budget["total_reconstruction_tokens"]),
                    str(budget["total_forecast_pairs"]),
                    _number(common["fve"]),
                    _number(common["cosine_similarity"]),
                    _number(common["fraction_alive"]),
                    _number(common["l0"]),
                    _number(forecast["common_horizon_mean_code_cosine"]),
                    _number(
                        forecast["common_horizon_mean_true_minus_shuffled"]
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"## MMLU sparse probes (k={preferred_sparsity})",
            "",
            "| W | Semantics | Context | Syntax |",
            "|---:|---:|---:|---:|",
        ]
    )
    for window_size in cfg.proposal.window_sizes:
        label = f"proposal_w{window_size:03d}"
        lines.append(
            f"| {window_size} | "
            f"{_number(probe_lookup[(label, 'semantics')])} | "
            f"{_number(probe_lookup[(label, 'context')])} | "
            f"{_number(probe_lookup[(label, 'syntax')])} |"
        )
    lines.extend(
        [
            "",
            "The table compares the common forecast horizon (offsets 1..7). "
            "All-offset means and offset-wise results are retained in "
            "`evaluation/window_sweep.json`.",
            "",
        ]
    )
    destination = run_dir / "WINDOW_SWEEP_REPORT.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
