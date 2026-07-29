from pathlib import Path

import pytest

from sae_comp.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_loads() -> None:
    cfg = load_config(ROOT / "configs" / "smoke.toml")
    assert cfg.model.layer == 3
    assert cfg.proposal.window_size == 10
    assert cfg.proposal.window_sizes == [8, 16, 32]
    assert cfg.proposal.sweep_budget(8) == {
        "window_size": 8,
        "batch_windows": 16,
        "reconstruction_tokens_per_step": 128,
        "forecast_offsets_per_window": 7,
        "forecast_pairs_per_step": 112,
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
    assert [item["forecast_offsets_per_window"] for item in budgets] == [
        14,
        28,
        56,
    ]
    assert {item["reconstruction_tokens_per_step"] for item in budgets} == {512}
    assert {item["forecast_pairs_per_step"] for item in budgets} == {448}
    assert cfg.train.temporal_pairs_per_step == 448
