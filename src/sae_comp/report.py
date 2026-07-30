from __future__ import annotations

import json
from pathlib import Path

from .config import ExperimentConfig
from .plots import build_controlled_plots, display_name


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
        5
        if 5 in cfg.evaluation.probe_sparsities
        else cfg.evaluation.probe_sparsities[0]
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
        "count, residual-position count, and pool of sequences with at least "
        f"{max(cfg.proposal.window_sizes)} valid tokens. The new architecture "
        "uses every context position, so context-target pair counts are reported "
        "rather than artificially subsampled to equality.",
        "",
        "| W | Batch windows | Contexts/window | Total residual positions | "
        "Total endpoint reconstructions | Total context-target pairs | FVE | "
        "Cosine | Alive | L0 | Forecast cosine (h=1..7) | "
        "True-shuffled (1..7) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                    str(budget["context_positions_per_window"]),
                    str(budget["total_residual_positions"]),
                    str(budget["total_endpoint_reconstructions"]),
                    str(budget["total_context_target_pairs"]),
                    _number(common["fve"]),
                    _number(common["cosine_similarity"]),
                    _number(common["fraction_alive"]),
                    _number(common["l0"]),
                    _number(forecast["common_horizon_mean_code_cosine"]),
                    _number(forecast["common_horizon_mean_true_minus_shuffled"]),
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
            "The table compares the common forecast horizons 1..7, where "
            "`h = (W - 1) - context_position`. All-horizon means and "
            "position-wise results are retained in "
            "`evaluation/window_sweep.json`.",
            "",
        ]
    )
    destination = run_dir / "WINDOW_SWEEP_REPORT.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def build_controlled_report(cfg: ExperimentConfig) -> Path:
    run_dir = Path(cfg.run_dir)
    evaluation_dir = run_dir / "evaluation"
    metrics = json.loads(
        (evaluation_dir / "controlled_metrics.json").read_text(encoding="utf-8")
    )
    probes = json.loads(
        (evaluation_dir / "controlled_probes.json").read_text(encoding="utf-8")
    )
    plot_paths = build_controlled_plots(cfg)
    experiment = metrics["experiment"]
    preferred_sparsity: int | str = (
        5
        if 5 in cfg.evaluation.probe_sparsities
        else cfg.evaluation.probe_sparsities[0]
    )
    probe_lookup = {
        (item["method"], item["task"]): item["accuracy"]
        for item in probes
        if item["split"] == "full" and item["sparsity"] == preferred_sparsity
    }
    labels = list(metrics["methods"])
    lines = [
        "# Controlled SAE comparison",
        "",
        "## Experimental setup",
        "",
        f"- Frozen model: `{experiment['model']}` revision "
        f"`{experiment['revision']}`, layer {experiment['layer']}",
        f"- SAE: {experiment['dictionary_size']:,} features, target k={experiment['k']}",
        f"- Shared standard pretraining: {cfg.train.standard_steps:,} steps",
        f"- Controlled branch training: {experiment['branch_optimizer_steps']:,} "
        "steps per method",
        f"- Data pool: sequences with at least "
        f"{experiment['minimum_sequence_length']} valid tokens",
        f"- Seed: {experiment['seed']}",
        "",
        "The Pythia-160m layer-8, 16k-feature, k=20 configuration follows the "
        "Temporal SAE paper's principal Pythia setup. Standard and proposal "
        "conditions use token-wise Top-K; Temporal SAE uses BatchTopK training.",
        "",
        "## Controlled branch-training budget",
        "",
        "| Method | Windows/step | Residual positions/step | Reconstruction "
        "targets/step | Predictive pairs/step | Total residual positions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    total_tokens = experiment["total_reconstruction_tokens_per_method"]
    lines.extend(
        [
            f"| Standard Top-K | - | {cfg.train.token_batch_size} | "
            f"{cfg.train.token_batch_size} | - | {total_tokens:,} |",
            f"| Temporal SAE | - | {cfg.train.token_batch_size} | "
            f"{cfg.train.token_batch_size} | {cfg.train.temporal_pairs_per_step} | "
            f"{total_tokens:,} |",
        ]
    )
    for window_size in cfg.proposal.window_sizes:
            budget = cfg.proposal.sweep_budget(window_size)
            lines.append(
                f"| Proposal W={window_size} | {budget['batch_windows']} | "
                f"{budget['residual_positions_per_step']} | "
                f"{budget['endpoint_reconstructions_per_step']} | "
                f"{budget['context_target_pairs_per_step']} | {total_tokens:,} |"
            )
    lines.extend(
        [
            "",
            "Method-specific objectives and optimizers are preserved; forcing them "
            "to be identical would change the methods being compared.",
            "",
            "## Common metrics",
            "",
            "| Method | FVE | Reconstruction cosine | Alive fraction | L0 | "
            "Lipschitz | Fourier | Wavelet | Multiscale |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in labels:
        values = metrics["methods"][label]
        smooth = values["smoothness"]["full"]
        lines.append(
            f"| {display_name(label)} | {_number(values['fve'])} | "
            f"{_number(values['cosine_similarity'])} | "
            f"{_number(values['fraction_alive'])} | {_number(values['l0'])} | "
            f"{_number(smooth['lipschitz'])} | {_number(smooth['fourier'])} | "
            f"{_number(smooth['wavelet'])} | "
            f"{_number(smooth['multiscale'])} |"
        )
    lines.extend(
        [
            "",
            f"## MMLU sparse probes (k={preferred_sparsity})",
            "",
            "| Method | Semantics | Context | Syntax |",
            "|---|---:|---:|---:|",
        ]
    )
    for label in labels:
        lines.append(
            f"| {display_name(label)} | "
            f"{_number(probe_lookup[(label, 'semantics')])} | "
            f"{_number(probe_lookup[(label, 'context')])} | "
            f"{_number(probe_lookup[(label, 'syntax')])} |"
        )
    lines.extend(
        [
            "",
            "## Proposal forecast diagnostic",
            "",
            f"All proposal means below use the common forecast-horizon range 1.."
            f"{min(cfg.proposal.window_sizes) - 1}.",
            "",
            "| Method | Target-code cosine | True minus shuffled cosine |",
            "|---|---:|---:|",
        ]
    )
    for label, forecast in metrics["proposal_forecasts"].items():
        lines.append(
            f"| {display_name(label)} | "
            f"{_number(forecast['common_horizon_mean_code_cosine'])} | "
            f"{_number(forecast['common_horizon_mean_true_minus_shuffled'])} |"
        )
    lines.extend(["", "## Figures", ""])
    for path in plot_paths:
        relative = path.relative_to(run_dir).as_posix()
        lines.extend(
            [
                f"### {path.stem.replace('_', ' ').title()}",
                "",
                f"![{path.stem}]({relative})",
                "",
            ]
        )
    lines.extend(
        [
            "Raw machine-readable results are in "
            "`evaluation/controlled_metrics.json` and "
            "`evaluation/controlled_probes.json`.",
            "",
        ]
    )
    destination = run_dir / "CONTROLLED_REPORT.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
