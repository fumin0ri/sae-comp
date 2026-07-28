import importlib

import numpy as np
import pytest

from sae_comp.probes import (
    _feature_rankings,
    _fit_probe,
    _load_spacy_model,
    _select_ranked_features,
)


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


def test_feature_rankings_are_reused_across_sparsities() -> None:
    features = np.array(
        [
            [5.0, 0.0, 1.0, 0.0],
            [4.0, 0.0, 1.0, 0.0],
            [0.0, 5.0, 0.0, 1.0],
            [0.0, 4.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1])
    rankings = _feature_rankings(features, labels, 2)
    selected_one = _select_ranked_features(rankings, 1)
    selected_two = _select_ranked_features(rankings, 2)
    assert set(selected_one).issubset(set(selected_two))
    assert set(selected_one) == {0, 1}


def test_sparse_sgd_probe_finishes() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), 30)
    features = np.zeros((90, 200), dtype=np.float32)
    features[np.arange(90), labels] = 5
    features += rng.normal(0, 0.05, features.shape).astype(np.float32)
    train = np.concatenate(
        [np.arange(label * 30, label * 30 + 20) for label in range(3)]
    )
    test = np.setdiff1d(np.arange(90), train)
    accuracy, selected, iterations = _fit_probe(
        features,
        labels,
        train,
        test,
        np.array([0, 1, 2]),
        max_iter=50,
        tolerance=1e-3,
    )
    assert accuracy > 0.9
    assert selected == 3
    assert iterations <= 50
