import json
from dataclasses import replace
from pathlib import Path

from sae_comp.config import ExperimentConfig
from sae_comp.report import build_controlled_report


def test_controlled_report_renders_all_plots(tmp_path: Path) -> None:
    base = ExperimentConfig()
    cfg = replace(
        base,
        run_dir=str(tmp_path),
        proposal=replace(base.proposal, window_sizes=(16, 32, 64)),
    )
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    labels = ("standard", "temporal", "proposal_w016", "proposal_w032", "proposal_w064")
    methods = {
        label: {
            "fve": 0.8,
            "cosine_similarity": 0.9,
            "fraction_alive": 0.7,
            "l0": 20.0,
            "smoothness": {
                "full": {
                    "lipschitz": 0.4,
                    "fourier": 0.3,
                    "wavelet": 0.2,
                    "multiscale": 0.1,
                }
            },
        }
        for label in labels
    }
    forecasts = {
        label: {
            "common_horizon_max_offset": 15,
            "common_horizon_mean_code_cosine": 0.7,
            "common_horizon_mean_true_minus_shuffled": 0.2,
            "offsets": [
                {
                    "offset": offset,
                    "code_cosine": 0.7,
                    "true_minus_shuffled": 0.2,
                }
                for offset in range(1, window_size)
            ],
        }
        for label, window_size in (
            ("proposal_w016", 16),
            ("proposal_w032", 32),
            ("proposal_w064", 64),
        )
    }
    metrics = {
        "experiment": {
            "model": cfg.model.name,
            "revision": cfg.model.revision,
            "layer": cfg.model.layer,
            "dictionary_size": cfg.sae.dictionary_size,
            "k": cfg.sae.k,
            "seed": cfg.train.seed,
            "minimum_sequence_length": 64,
            "branch_optimizer_steps": cfg.train.branch_steps,
            "reconstruction_tokens_per_step": cfg.train.token_batch_size,
            "temporal_pairs_per_step": cfg.train.temporal_pairs_per_step,
            "total_reconstruction_tokens_per_method": 3_072_000,
            "total_temporal_pairs_per_temporal_method": 2_688_000,
        },
        "methods": methods,
        "proposal_forecasts": forecasts,
    }
    (evaluation_dir / "controlled_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    probes = [
        {
            "method": label,
            "split": "full",
            "task": task,
            "sparsity": sparsity,
            "accuracy": 0.75,
        }
        for label in labels
        for task in ("semantics", "context", "syntax")
        for sparsity in (*cfg.evaluation.probe_sparsities, "dense")
    ]
    (evaluation_dir / "controlled_probes.json").write_text(
        json.dumps(probes), encoding="utf-8"
    )
    (tmp_path / "controlled_training_history.json").write_text(
        json.dumps(
            {
                "standard": [{"step": 1, "fvu": 0.9}, {"step": 2, "fvu": 0.8}],
                "temporal": [
                    {"step": 1, "full_fvu": 0.9},
                    {"step": 2, "full_fvu": 0.8},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "window_sweep_training_history.json").write_text(
        json.dumps(
            {
                "histories": {
                        label: [
                            {"step": 1, "online_reconstruction_fvu": 0.9},
                            {"step": 2, "online_reconstruction_fvu": 0.8},
                    ]
                    for label in labels
                    if label.startswith("proposal")
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_controlled_report(cfg)

    assert report.exists()
    assert "Standard Top-K" in report.read_text(encoding="utf-8")
    plot_paths = list((evaluation_dir / "plots").glob("*.png"))
    assert len(plot_paths) == 5
    assert all(path.stat().st_size > 0 for path in plot_paths)
