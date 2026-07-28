import importlib

import pytest

from sae_comp.probes import _load_spacy_model


def test_spacy_dependency_error_names_missing_module(monkeypatch) -> None:
    def missing_click(name: str):
        assert name == "spacy"
        raise ModuleNotFoundError("No module named 'click'", name="click")

    monkeypatch.setattr(importlib, "import_module", missing_click)
    with pytest.raises(RuntimeError, match=r"`click` is missing"):
        _load_spacy_model()


def test_missing_spacy_recommends_probe_extra(monkeypatch) -> None:
    def missing_spacy(name: str):
        assert name == "spacy"
        raise ModuleNotFoundError("No module named 'spacy'", name="spacy")

    monkeypatch.setattr(importlib, "import_module", missing_spacy)
    with pytest.raises(RuntimeError, match=r"\.\[probe\]"):
        _load_spacy_model()
