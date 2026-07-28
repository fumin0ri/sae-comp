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
