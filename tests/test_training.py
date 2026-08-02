import json
from dataclasses import replace

import torch

from sae_comp.activations import FORMAT, ActivationStore
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
    _train_proposal,
    horizon_loss_weight_table,
    horizon_sampling_probabilities,
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


def test_proposal_loss_uses_random_pairs_and_balanced_horizons() -> None:
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
        torch.randn(4, 12),
        torch.randn(4, 12),
        torch.tensor([1, 2, 4, 7]),
        prediction_weight=1.0,
        cfg=cfg,
        span_length=torch.tensor([2, 3, 5, 8]),
        horizon_weight_table=horizon_loss_weight_table(2, 8, "inverse_probability"),
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
    assert "horizon_7_cosine" in metrics
    assert "prediction_loss_unweighted" in metrics
    assert "mean_horizon_loss_weight" in metrics


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
        proposal,
        torch.randn(4, 12),
        torch.randn(4, 12),
        torch.tensor([1, 2, 4, 7]),
        prediction_weight=1.0,
        cfg=cfg,
    )
    loss.backward()
    assert proposal.sae.decoder.grad is not None
    assert proposal.predictor.output.weight.grad is not None
    assert proposal.ema_decoder.grad is None
    assert all(p.grad is None for p in proposal.ema_encoder.parameters())


def test_inverse_probability_weights_equalize_horizon_mass() -> None:
    probabilities = horizon_sampling_probabilities(2, 8)
    weights = horizon_loss_weight_table(2, 8, "inverse_probability")
    expected_mass = probabilities[1:] * weights[1:]
    torch.testing.assert_close(
        expected_mass,
        torch.full_like(expected_mass, 1 / 7),
    )
    torch.testing.assert_close(
        (probabilities * weights).sum(), torch.tensor(1.0, dtype=torch.float64)
    )


def test_proposal_training_runs_sae_warmup_then_joint(tmp_path) -> None:
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
            base.train,
            branch_steps=3,
            warmup_steps=0,
            log_every=1,
            device="cpu",
        ),
        proposal=replace(
            base.proposal,
            window_size=4,
            window_sizes=(2, 4),
            sweep_pairs_per_step=4,
            sae_warmup_steps=1,
            prediction_ramp_steps=1,
        ),
    )
    model = TransitionJEPA(
        TransitionJEPAConfig(d_in=6, d_sae=12, k=2, window_size=4),
        SparseAutoencoder(SparseAutoencoderConfig(d_in=6, d_sae=12, k=2)),
    )
    before_ema = model.ema_decoder.clone()
    history = _train_proposal(
        model,
        ActivationStore(manifest_path, seed=0),
        cfg,
        pair_batch_size=4,
        boundary_max_horizon=3,
    )
    assert [record["phase"] for record in history] == [
        "sae_warmup",
        "joint",
        "joint",
    ]
    assert history[0]["prediction_weight"] == 0
    assert history[-1]["prediction_weight"] == 1
    assert not torch.allclose(before_ema, model.ema_decoder)


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
