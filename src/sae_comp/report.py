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
    views = metrics["proposal_views"]
    lines.extend(
        [
            "",
            "## Proposal shared-view diagnostic",
            "",
            f"- Mean same-span high cosine: {_number(views['mean_high_cosine'])}",
            ("- Mean same-span minus shuffled high cosine: "
            f"{_number(views['mean_high_margin'])}"),
            f"- Same-span high swap FVU: {_number(views['swap_fvu'])}",
            f"- Shuffled high swap FVU: {_number(views['shuffled_swap_fvu'])}",
            "",
            ("This shared-view diagnostic is not a three-way common metric. "
            "The common reconstruction, smoothness, and probe tables are the "
            "confirmatory cross-method comparison."),
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
        "# Proposal maximum-span sweep",
        "",
        ("Every condition uses the same random initialization, optimizer-step "
        "count, exchangeable-view pair batch, reconstruction count, and long-sequence "
        "pool. W is the maximum sampled span length, not a stored window boundary. "
        "Span length is uniform in 2..W and two distinct positions are sampled."),
        "",
        ("| Max span W | Pair batch | Distance support | Total sampled pairs | "
        "Total reconstructions | FVE | Cosine | Alive | L0 | "
        "High cosine (common d) | Same-shuffled margin (common d) |"),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window_size in cfg.proposal.window_sizes:
        label = f"proposal_w{window_size:03d}"
        values = metrics[label]
        budget = values["training_budget"]
        common = values["common"]
        views = values["views"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(window_size),
                    str(budget["pair_batch_size"]),
                    f"1..{budget['maximum_distance']}",
                    str(budget["total_sampled_pairs"]),
                    str(budget["total_reconstructions"]),
                    _number(common["fve"]),
                    _number(common["cosine_similarity"]),
                    _number(common["fraction_alive"]),
                    _number(common["l0"]),
                    _number(views["common_mean_high_cosine"]),
                    _number(views["common_mean_high_margin"]),
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
            (f"The table compares the common token distances 1.."
            f"{min(cfg.proposal.window_sizes) - 1}. All-distance means and "
            "distance-wise results are retained in "
            "`evaluation/window_sweep.json`."),
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
        (f"- Frozen model: `{experiment['model']}` revision "
        f"`{experiment['revision']}`, layer {experiment['layer']}"),
        f"- SAE: {experiment['dictionary_size']:,} features, target k={experiment['k']}",
        f"- Shared standard pretraining: {cfg.train.standard_steps:,} steps",
        (f"- Controlled branch training: {experiment['branch_optimizer_steps']:,} "
        "steps per method"),
        (f"- Data pool: sequences with at least "
        f"{experiment['minimum_sequence_length']} valid tokens"),
        f"- Seed: {experiment['seed']}",
        "",
        ("The Pythia-160m layer-8, 16k-feature, k=20 configuration follows the "
        "Temporal SAE paper's principal Pythia setup. Standard uses global "
        "token-wise Top-K; Rectified LpJEPA uses shifted-ReLU high features and "
        "Top-K only for low features; Temporal SAE uses BatchTopK training."),
        "",
        "## Controlled branch-training budget",
        "",
        ("| Method | Pair batch | Residual values/step | Reconstruction "
        "targets/step | View pairs/step | Total reconstruction targets |"),
        "|---|---:|---:|---:|---:|---:|",
    ]
    total_tokens = experiment["total_reconstruction_tokens_per_method"]
    lines.extend(
        [
            (f"| Standard Top-K | - | {cfg.train.token_batch_size} | "
            f"{cfg.train.token_batch_size} | - | {total_tokens:,} |"),
            (f"| Temporal SAE | - | {cfg.train.token_batch_size} | "
            f"{cfg.train.token_batch_size} | {cfg.train.temporal_pairs_per_step} | "
            f"{total_tokens:,} |"),
        ]
    )
    for window_size in cfg.proposal.window_sizes:
            budget = cfg.proposal.sweep_budget(window_size)
            lines.append(
                f"| Proposal W={window_size} | {budget['pair_batch_size']} | "
                f"{budget['residual_values_per_step']} | "
                f"{budget['reconstructions_per_step']} | "
                f"{budget['sampled_pairs_per_step']} | {total_tokens:,} |"
            )
    lines.extend(
        [
            "",
            ("Method-specific objectives and optimizers are preserved; forcing them "
            "to be identical would change the methods being compared."),
            "",
            "## Common metrics",
            "",
            ("| Method | FVE | Reconstruction cosine | Alive fraction | L0 | "
            "Lipschitz | Fourier | Wavelet | Multiscale |"),
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
            "## Proposal shared-view diagnostic",
            "",
            (f"All proposal means below use the common token-distance range 1.."
            f"{min(cfg.proposal.window_sizes) - 1}."),
            "",
            "| Method | Same-span high cosine | Same minus shuffled cosine |",
            "|---|---:|---:|",
        ]
    )
    for label, views in metrics["proposal_views"].items():
        lines.append(
            f"| {display_name(label)} | "
            f"{_number(views['common_mean_high_cosine'])} | "
            f"{_number(views['common_mean_high_margin'])} |"
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
            ("Raw machine-readable results are in "
            "`evaluation/controlled_metrics.json` and "
            "`evaluation/controlled_probes.json`."),
            "",
        ]
    )
    destination = run_dir / "CONTROLLED_REPORT.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
