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
    RectifiedLpJEPAConfig,
    RectifiedLpJEPASAE,
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


def test_proposal_checkpoint_loads_the_single_rectified_sae(tmp_path) -> None:
    cfg = ExperimentConfig()
    proposal_cfg = RectifiedLpJEPAConfig(
        d_in=6, d_sae=12, low_k=2, max_span_length=4
    )
    proposal = RectifiedLpJEPASAE(proposal_cfg)
    with torch.no_grad():
        proposal.pre_bias.add_(3)
        proposal.encoder.weight.add_(2)
        proposal.decoder.add_(1)
    path = tmp_path / "proposal.pt"
    _save_checkpoint(
        path,
        "proposal",
        proposal,
        vars(proposal_cfg),
        cfg,
        {"config_fingerprint": "activation-test"},
    )
    loaded_sae, loaded_proposal, method = load_method(path, torch.device("cpu"))
    assert method == "proposal"
    assert loaded_proposal is not None
    assert loaded_sae is loaded_proposal
    torch.testing.assert_close(loaded_sae.pre_bias, proposal.pre_bias)
    torch.testing.assert_close(loaded_sae.encoder.weight, proposal.encoder.weight)
    torch.testing.assert_close(loaded_sae.decoder, proposal.decoder)
    assert loaded_sae.cfg.low_k == 2


def test_obsolete_proposal_checkpoint_is_rejected(tmp_path) -> None:
    cfg = ExperimentConfig()
    proposal_cfg = RectifiedLpJEPAConfig(
        d_in=6, d_sae=12, low_k=2, max_span_length=4
    )
    proposal = RectifiedLpJEPASAE(proposal_cfg)
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
