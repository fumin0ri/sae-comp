import json
from dataclasses import replace

import pytest
import torch

from sae_comp.activations import FORMAT, ActivationStore
from sae_comp.config import ExperimentConfig
from sae_comp.models import (
    RectifiedLpJEPAConfig,
    RectifiedLpJEPASAE,
    SparseAutoencoder,
    SparseAutoencoderConfig,
)
from sae_comp.training import (
    _proposal_loss,
    _save_checkpoint,
    _symmetric_contrastive,
    _temporal_loss,
    _train_proposal,
    axis_aligned_distribution_matching_loss,
    load_checkpoint,
    rectified_distribution_matching_loss,
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


def make_proposal() -> RectifiedLpJEPASAE:
    model = RectifiedLpJEPASAE(
        RectifiedLpJEPAConfig(
            d_in=12,
            d_sae=40,
            low_k=4,
            max_span_length=8,
            target_active_fraction=0.1,
        )
    )
    model.initialize_normalization(torch.zeros(12), 1.0)
    return model


def small_cfg() -> ExperimentConfig:
    base = ExperimentConfig()
    return replace(
        base,
        proposal=replace(
            base.proposal,
            low_k=4,
            rdm_projections=8,
            rdm_projection_chunk_size=4,
            axis_rdm_features=3,
        ),
    )


def test_proposal_loss_uses_invariance_and_both_rdm_terms() -> None:
    cfg = small_cfg()
    proposal = make_proposal()
    loss, metrics = _proposal_loss(
        proposal,
        torch.randn(6, 12),
        torch.randn(6, 12),
        invariance_weight=1.0,
        rdm_weight=2.0,
        cfg=cfg,
        distance=torch.tensor([1, 2, 3, 4, 5, 7]),
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert "full_reconstruction_fvu" in metrics
    assert "high_reconstruction_fvu" in metrics
    assert "invariance_loss" in metrics
    assert "random_projection_rdm_loss" in metrics
    assert "axis_aligned_rdm_loss" in metrics
    assert metrics["axis_sampled_features"] == 3
    assert metrics["low_l0"] <= proposal.cfg.low_k
    assert "prediction_loss" not in metrics


def test_identical_views_have_zero_invariance_error() -> None:
    cfg = small_cfg()
    proposal = make_proposal()
    values = torch.randn(6, 12)
    _, metrics = _proposal_loss(
        proposal,
        values,
        values,
        invariance_weight=1.0,
        rdm_weight=1.0,
        cfg=cfg,
    )
    assert metrics["invariance_raw_mse"] == pytest.approx(0.0, abs=1e-8)


def test_proposal_loss_updates_the_single_sae() -> None:
    proposal = make_proposal()
    loss, _ = _proposal_loss(
        proposal,
        torch.randn(6, 12),
        torch.randn(6, 12),
        invariance_weight=1.0,
        rdm_weight=2.0,
        cfg=small_cfg(),
    )
    loss.backward()
    assert proposal.encoder.weight.grad is not None
    assert proposal.decoder.grad is not None
    assert not hasattr(proposal, "ema_decoder")


def test_rdm_loss_is_finite_and_differentiable() -> None:
    model = make_proposal()
    first = torch.rand(12, model.cfg.d_high, requires_grad=True)
    second = torch.rand(12, model.cfg.d_high, requires_grad=True)
    loss, metrics = rectified_distribution_matching_loss(
        (first, second),
        model.cfg,
        projections=12,
        projection_chunk_size=5,
        axis_features=3,
        axis_weight=1.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["random_projection"])
    assert torch.isfinite(metrics["axis_aligned"])
    assert int(metrics["axis_sampled_features"]) == 3
    assert first.grad is not None and second.grad is not None


def test_axis_rdm_zero_features_is_exact_ablation() -> None:
    target = torch.rand(4, 3)
    view = torch.rand(4, 3, requires_grad=True)
    loss, sampled = axis_aligned_distribution_matching_loss((view,), target, 0)
    loss.backward()
    assert float(loss) == 0.0
    assert int(sampled) == 0
    assert view.grad is not None


def test_proposal_training_runs_distribution_warmup_then_joint(tmp_path) -> None:
    shard_path = tmp_path / "train.pt"
    torch.save(
        {
            "activations": torch.randn(8, 12, 6),
            "attention_mask": torch.ones(8, 12, dtype=torch.bool),
            "valid_lengths": torch.full((8,), 12, dtype=torch.int32),
        },
        shard_path,
    )
    manifest = {
        "format": FORMAT,
        "sequence_length": 12,
        "min_span_length": 2,
        "max_span_length": 4,
        "max_horizon": 3,
        "burn_in_tokens": 2,
        "minimum_valid_length": 6,
        "train": {"shards": [{"path": shard_path.name}]},
        "validation": {"shards": [{"path": shard_path.name}]},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    base = ExperimentConfig()
    cfg = replace(
        base,
        train=replace(
            base.train, branch_steps=3, warmup_steps=0, log_every=1, device="cpu"
        ),
        proposal=replace(
            base.proposal,
            window_size=4,
            window_sizes=(2, 4),
            min_span_length=2,
            sweep_pairs_per_step=4,
            low_k=2,
            sae_warmup_steps=1,
            regularization_ramp_steps=1,
            rdm_projections=4,
            rdm_projection_chunk_size=2,
            axis_rdm_features=2,
        ),
    )
    model = RectifiedLpJEPASAE(
        RectifiedLpJEPAConfig(
            d_in=6, d_sae=12, low_k=2, max_span_length=4
        )
    )
    before = model.decoder.clone()
    history = _train_proposal(
        model,
        ActivationStore(manifest_path, seed=0),
        cfg,
        pair_batch_size=4,
        boundary_max_distance=3,
    )
    assert [record["phase"] for record in history] == [
        "distribution_warmup",
        "joint",
        "joint",
    ]
    assert history[0]["active_invariance_weight"] == 0
    assert history[0]["active_rdm_weight"] == cfg.proposal.rdm_weight
    assert history[-1]["active_invariance_weight"] == 1
    assert not torch.allclose(before, model.decoder)


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
