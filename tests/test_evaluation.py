import torch

from sae_comp.evaluation import (
    fourier_smoothness,
    lipschitz_smoothness,
    multiscale_smoothness,
    wavelet_smoothness,
)


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
