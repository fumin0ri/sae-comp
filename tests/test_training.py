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


def test_temporal_loss_can_match_pair_budget() -> None:
    cfg = ExperimentConfig()
    sae = SparseAutoencoder(SparseAutoencoderConfig(d_in=12, d_sae=40, k=4))
    loss, metrics, _, _ = _temporal_loss(
        sae,
        torch.randn(16, 12),
        torch.randn(16, 12),
        cfg,
        contrastive_rows=8,
    )
    assert torch.isfinite(loss)
    assert metrics["l0"] == 4


def test_proposal_loss_uses_fixed_endpoint_and_all_contexts() -> None:
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
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert "online_reconstruction_fvu" in metrics
    assert "online_high_reconstruction_fvu" in metrics
    assert "weighted_reconstruction_fvu" in metrics
    assert "ema_reconstruction_fvu" in metrics
    assert "ema_high_reconstruction_fvu" in metrics
    assert metrics["high_l0"] <= proposal.cfg.k_high
    assert metrics["low_l0"] <= proposal.cfg.k_low
    assert "variance_loss" not in metrics
    assert all(
        f"context_{position}_horizon_{7 - position}_cosine" in metrics
        for position in range(7)
    )


def test_proposal_loss_does_not_backpropagate_into_ema_teacher() -> None:
    cfg = ExperimentConfig()
    proposal = TransitionJEPA(
        TransitionJEPAConfig(
            d_in=12,
            d_sae=40,
            k=4,
            window_size=8,
            predictor_width=16,
        ),
        SparseAutoencoder(SparseAutoencoderConfig(d_in=12, d_sae=40, k=4)),
    )
    loss, _ = _proposal_loss(
        proposal, torch.randn(4, 8, 12), prediction_weight=1.0, cfg=cfg
    )
    loss.backward()
    assert proposal.sae.decoder.grad is not None
    assert proposal.predictor.output.weight.grad is not None
    assert proposal.ema_decoder.grad is None
    assert all(p.grad is None for p in proposal.ema_encoder.parameters())


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
