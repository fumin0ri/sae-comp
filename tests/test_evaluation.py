import torch

from sae_comp.config import ExperimentConfig
from sae_comp.evaluation import (
    fourier_smoothness,
    lipschitz_smoothness,
    load_method,
    multiscale_smoothness,
    wavelet_smoothness,
)
from sae_comp.models import (
    SparseAutoencoder,
    SparseAutoencoderConfig,
    TransitionJEPA,
    TransitionJEPAConfig,
)
from sae_comp.training import _save_checkpoint


def test_smooth_signal_scores_below_alternating_signal() -> None:
    time = 32
    x = torch.arange(time, dtype=torch.float32)[:, None].repeat(1, 4)
    smooth = torch.linspace(0, 1, time)[:, None].repeat(1, 3)
    alternating = (torch.arange(time) % 2).float()[:, None].repeat(1, 3)
    assert lipschitz_smoothness(x, smooth) < lipschitz_smoothness(x, alternating)
    assert fourier_smoothness(smooth) < fourier_smoothness(alternating)
    assert wavelet_smoothness(smooth) < wavelet_smoothness(alternating)
    assert multiscale_smoothness(smooth) < multiscale_smoothness(alternating)


def test_metrics_handle_all_zero_features() -> None:
    x = torch.randn(8, 4)
    features = torch.zeros(8, 10)
    assert lipschitz_smoothness(x, features) == 0
    assert fourier_smoothness(features) == 0
    assert wavelet_smoothness(features) == 0
    assert multiscale_smoothness(features) == 0


def test_proposal_checkpoint_exports_full_ema_sae(tmp_path) -> None:
    cfg = ExperimentConfig()
    sae = SparseAutoencoder(SparseAutoencoderConfig(d_in=6, d_sae=12, k=2))
    proposal_cfg = TransitionJEPAConfig(
        d_in=6, d_sae=12, k=2, window_size=4
    )
    proposal = TransitionJEPA(proposal_cfg, sae)
    with torch.no_grad():
        proposal.sae.pre_bias.add_(3)
        proposal.sae.encoder.weight.add_(2)
        proposal.sae.decoder.add_(1)
    path = tmp_path / "proposal.pt"
    _save_checkpoint(
        path,
        "proposal",
        proposal,
        vars(proposal_cfg),
        cfg,
        {"config_fingerprint": "activation-test"},
    )
    final_sae, loaded_proposal, method = load_method(path, torch.device("cpu"))
    assert method == "proposal"
    assert loaded_proposal is not None
    torch.testing.assert_close(final_sae.pre_bias, proposal.ema_pre_bias)
    torch.testing.assert_close(final_sae.encoder.weight, proposal.ema_encoder.weight)
    torch.testing.assert_close(final_sae.decoder, proposal.ema_decoder)
    assert not torch.allclose(final_sae.pre_bias, proposal.sae.pre_bias)
    assert final_sae.cfg.group_topk


def test_obsolete_proposal_checkpoint_is_rejected(tmp_path) -> None:
    cfg = ExperimentConfig()
    sae = SparseAutoencoder(SparseAutoencoderConfig(d_in=6, d_sae=12, k=2))
    proposal_cfg = TransitionJEPAConfig(
        d_in=6, d_sae=12, k=2, window_size=4
    )
    proposal = TransitionJEPA(proposal_cfg, sae)
    path = tmp_path / "old-proposal.pt"
    _save_checkpoint(
        path,
        "proposal",
        proposal,
        vars(proposal_cfg),
        cfg,
        {"config_fingerprint": "activation-test"},
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint.pop("architecture_id")
    torch.save(checkpoint, path)
    try:
        load_method(path, torch.device("cpu"))
    except ValueError as exc:
        assert "obsolete architecture" in str(exc)
    else:
        raise AssertionError("obsolete proposal checkpoint was accepted")
