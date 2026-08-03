import pytest
import torch

from sae_comp.models import (
    RectifiedLpJEPAConfig,
    RectifiedLpJEPASAE,
    SparseAutoencoder,
    SparseAutoencoderConfig,
    batch_topk,
    rgg_mean_for_active_fraction,
    sample_rectified_generalized_gaussian,
    token_topk,
    unit_variance_generalized_gaussian_sigma,
)


def test_token_topk_is_per_example() -> None:
    values = torch.randn(7, 19)
    encoded = token_topk(values, 3)
    assert encoded.shape == values.shape
    assert bool(((encoded > 0).sum(dim=-1) <= 3).all())
    assert bool((encoded >= 0).all())


def test_batch_topk_has_global_budget() -> None:
    values = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    encoded, threshold = batch_topk(values, 2)
    assert int((encoded > 0).sum()) == 10
    assert threshold > 0


def test_sparse_autoencoder_shapes_and_unit_decoder() -> None:
    cfg = SparseAutoencoderConfig(d_in=12, d_sae=32, k=4)
    model = SparseAutoencoder(cfg)
    x = torch.randn(6, 12)
    code = model.encode_token_topk(x)
    assert code.shape == (6, 32)
    assert model.decode(code).shape == x.shape
    torch.testing.assert_close(
        model.decoder.norm(dim=1), torch.ones(32), atol=1e-5, rtol=1e-5
    )


def make_proposal() -> RectifiedLpJEPASAE:
    model = RectifiedLpJEPASAE(
        RectifiedLpJEPAConfig(
            d_in=8,
            d_sae=20,
            low_k=4,
            max_span_length=6,
            high_fraction=0.2,
            target_active_fraction=0.1,
        )
    )
    model.initialize_normalization(torch.zeros(8), 1.0)
    return model


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_rgg_parameterization_controls_active_fraction(p: float) -> None:
    sigma = unit_variance_generalized_gaussian_sigma(p)
    mu = rgg_mean_for_active_fraction(p, 0.1, sigma)
    torch.manual_seed(1)
    samples = sample_rectified_generalized_gaussian(
        (200_000,), p=p, mu=mu, sigma=sigma, device=torch.device("cpu")
    )
    assert abs(float((samples > 0).float().mean()) - 0.1) < 0.005


def test_proposal_high_is_shifted_relu_and_only_low_is_topk() -> None:
    model = make_proposal()
    assert model.cfg.d_high == 4
    assert model.cfg.d_low == 16
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.fill_(1.0)
    high, low = model.split_code(model.encode(torch.randn(3, 8)))
    assert torch.all(high > 0)
    assert torch.all((low > 0).sum(dim=-1) == model.cfg.low_k)


def test_proposal_has_two_exchangeable_views_and_no_predictor() -> None:
    model = make_proposal()
    outputs = model(torch.randn(2, 8), torch.randn(2, 8))
    assert "predicted_codes" not in outputs
    assert not hasattr(model, "predictor")
    assert not hasattr(model, "ema_encoder")
    assert outputs["high_a"].shape == (2, model.cfg.d_high)
    expected = outputs["high_reconstruction_a"] + model.decode_low(
        outputs["low_a"]
    )
    torch.testing.assert_close(outputs["full_reconstruction_a"], expected)


def test_proposal_initialization_covers_single_full_sae() -> None:
    model = make_proposal()
    torch.testing.assert_close(
        model.encoder.bias[: model.cfg.d_high],
        torch.full((model.cfg.d_high,), model.cfg.target_mu),
    )
    torch.testing.assert_close(
        model.decoder.norm(dim=1), torch.ones(model.cfg.d_sae), atol=1e-5, rtol=0
    )
