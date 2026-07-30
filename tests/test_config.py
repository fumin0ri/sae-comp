from pathlib import Path

import pytest

from sae_comp.cli import build_parser
from sae_comp.config import apply_training_overrides, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_loads() -> None:
    cfg = load_config(ROOT / "configs" / "smoke.toml")
    assert cfg.model.layer == 3
    assert cfg.proposal.window_size == 10
    assert cfg.proposal.window_sizes == [8, 16, 32]
    assert cfg.proposal.high_fraction == 0.2
    assert cfg.proposal.high_reconstruction_weight == 0.2
    assert cfg.proposal.sweep_budget(8) == {
        "window_size": 8,
        "batch_windows": 16,
        "residual_positions_per_step": 128,
        "endpoint_reconstructions_per_step": 16,
        "context_positions_per_window": 7,
        "context_target_pairs_per_step": 112,
    }
    assert cfg.sae.dictionary_size == 2048
    assert len(cfg.fingerprint()) == 64


def test_config_rejects_window_larger_than_sequence(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "smoke.toml").read_text(encoding="utf-8")
    source = source.replace("sequence_length = 32", "sequence_length = 8")
    path = tmp_path / "invalid.toml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="sequence_length"):
        load_config(path)


def test_controlled_sweep_has_equal_training_volume() -> None:
    cfg = load_config(ROOT / "configs" / "controlled_rtx4090.toml")
    budgets = [
        cfg.proposal.sweep_budget(window_size)
        for window_size in cfg.proposal.window_sizes
    ]
    assert cfg.proposal.window_sizes == [16, 32, 64]
    assert [item["batch_windows"] for item in budgets] == [32, 16, 8]
    assert [item["context_positions_per_window"] for item in budgets] == [
        15,
        31,
        63,
    ]
    assert {item["residual_positions_per_step"] for item in budgets} == {512}
    assert [item["endpoint_reconstructions_per_step"] for item in budgets] == [
        32,
        16,
        8,
    ]
    assert [item["context_target_pairs_per_step"] for item in budgets] == [
        480,
        496,
        504,
    ]
    assert cfg.train.temporal_pairs_per_step == 448


def test_training_scale_updates_steps_schedules_and_run_directory() -> None:
    cfg = load_config(ROOT / "configs" / "controlled_rtx4090.toml")
    scaled = apply_training_overrides(cfg, training_scale=2)
    assert scaled.train.standard_steps == 24_000
    assert scaled.train.branch_steps == 12_000
    assert scaled.train.warmup_steps == 1_000
    assert scaled.proposal.predictor_warmup_steps == 1_600
    assert scaled.proposal.prediction_ramp_steps == 1_600
    assert scaled.train.token_batch_size == cfg.train.token_batch_size
    assert scaled.proposal.sweep_residual_positions_per_step == 512
    assert scaled.run_dir == f"{cfg.run_dir}-trainx2"


def test_explicit_training_steps_override_scale_and_can_set_run_directory() -> None:
    cfg = load_config(ROOT / "configs" / "controlled_rtx4090.toml")
    resolved = apply_training_overrides(
        cfg,
        training_scale=2,
        standard_steps=30_000,
        branch_steps=15_000,
        predictor_warmup_steps=2_000,
        run_dir="runs/custom-budget",
    )
    assert resolved.train.standard_steps == 30_000
    assert resolved.train.branch_steps == 15_000
    assert resolved.train.warmup_steps == 1_000
    assert resolved.proposal.predictor_warmup_steps == 2_000
    assert resolved.proposal.prediction_ramp_steps == 1_600
    assert resolved.run_dir == "runs/custom-budget"


def test_custom_budget_gets_deterministic_suffix() -> None:
    cfg = load_config(ROOT / "configs" / "controlled_rtx4090.toml")
    resolved = apply_training_overrides(
        cfg,
        standard_steps=18_000,
        branch_steps=9_000,
    )
    assert resolved.run_dir.endswith(
        "-budget-s18000-b9000-w500-pw800-pr800"
    )


@pytest.mark.parametrize("scale", [0, -1, float("inf")])
def test_training_scale_must_be_finite_and_positive(scale: float) -> None:
    cfg = load_config(ROOT / "configs" / "controlled_rtx4090.toml")
    with pytest.raises(ValueError, match="training_scale"):
        apply_training_overrides(cfg, training_scale=scale)


def test_cli_accepts_training_budget_overrides() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--config",
            "configs/controlled_rtx4090.toml",
            "--training-scale",
            "2",
            "--branch-steps",
            "15000",
            "--run-dir",
            "runs/custom",
        ]
    )
    assert args.training_scale == 2
    assert args.branch_steps == 15_000
    assert args.run_dir == "runs/custom"
