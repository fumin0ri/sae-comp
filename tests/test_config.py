from pathlib import Path

import pytest

from sae_comp.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_loads() -> None:
    cfg = load_config(ROOT / "configs" / "smoke.toml")
    assert cfg.model.layer == 3
    assert cfg.proposal.window_size == 10
    assert cfg.sae.dictionary_size == 2048
    assert len(cfg.fingerprint()) == 64


def test_config_rejects_window_larger_than_sequence(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "smoke.toml").read_text(encoding="utf-8")
    source = source.replace("sequence_length = 32", "sequence_length = 8")
    path = tmp_path / "invalid.toml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="sequence_length"):
        load_config(path)
