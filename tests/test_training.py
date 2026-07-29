import torch

from sae_comp.config import ExperimentConfig
from sae_comp.models import (
    SparseAutoencoder,
    SparseAutoencoderConfig,
    TransitionJEPA,
    TransitionJEPAConfig,
)
from sae_comp.training import (
    _proposal_loss,
    _save_checkpoint,
    _symmetric_contrastive,
    _temporal_loss,
    load_checkpoint,
)


def test_contrastive_prefers_aligned_pairs() -> None:
    features = torch.eye(8)
    aligned = _symmetric_contrastive(features, features, 0.2)
    shuffled = _symmetric_contrastive(features, features.roll(1, 0), 0.2)
    assert aligned < shuffled


def test_temporal_loss_is_finite() -> None:
    cfg = ExperimentConfig()
    sae = SparseAutoencoder(SparseAutoencoderConfig(d_in=12, d_sae=40, k=4))
    loss, metrics, threshold, active = _temporal_loss(
        sae, torch.randn(16, 12), torch.randn(16, 12), cfg
    )
    assert torch.isfinite(loss)
    assert threshold >= 0
    assert metrics["l0"] == 4
    assert active.shape == (40,)


def test_proposal_loss_supports_budgeted_offsets() -> None:
    cfg = ExperimentConfig()
    sae = SparseAutoencoder(SparseAutoencoderConfig(d_in=12, d_sae=40, k=4))
    proposal = TransitionJEPA(
        TransitionJEPAConfig(
            d_in=12,
            d_sae=40,
            k=4,
            window_size=8,
            predictor_width=16,
        ),
        sae,
    )
    loss, metrics = _proposal_loss(
        proposal,
        torch.randn(4, 8, 12),
        prediction_weight=1.0,
        cfg=cfg,
        offsets=torch.tensor([1, 3, 7]),
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_checkpoint_round_trip(tmp_path) -> None:
    cfg = ExperimentConfig()
    sae = SparseAutoencoder(SparseAutoencoderConfig(d_in=6, d_sae=12, k=2))
    path = tmp_path / "checkpoint.pt"
    _save_checkpoint(
        path,
        "standard",
        sae,
        sae.checkpoint_config(),
        cfg,
        {"config_fingerprint": "activation-test"},
    )
    loaded = load_checkpoint(path)
    assert loaded["method"] == "standard"
    assert loaded["model_config"]["d_sae"] == 12
