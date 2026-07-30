import torch

from sae_comp.models import (
    SparseAutoencoder,
    SparseAutoencoderConfig,
    TransitionJEPA,
    TransitionJEPAConfig,
    batch_topk,
    token_topk,
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


def test_transition_jepa_shapes() -> None:
    sae_cfg = SparseAutoencoderConfig(d_in=8, d_sae=24, k=3)
    sae = SparseAutoencoder(sae_cfg)
    cfg = TransitionJEPAConfig(
        d_in=8,
        d_sae=24,
        k=3,
        window_size=5,
        predictor_width=10,
    )
    model = TransitionJEPA(cfg, sae)
    output = model(torch.randn(4, 5, 8))
    assert output["predicted_codes"].shape == (4, 4, 24)
    assert output["target_codes"].shape == (4, 4, 24)
    assert output["online_target_reconstruction"].shape == (4, 8)
    assert output["target_reconstruction"].shape == (4, 8)
    torch.testing.assert_close(
        output["target_codes"],
        output["target_code"][:, None].expand(-1, 4, -1),
    )
    assert not output["target_codes"].requires_grad


def test_transition_jepa_uses_every_context_for_one_endpoint() -> None:
    sae_cfg = SparseAutoencoderConfig(d_in=8, d_sae=24, k=3)
    model = TransitionJEPA(
        TransitionJEPAConfig(
            d_in=8,
            d_sae=24,
            k=3,
            window_size=8,
            predictor_width=10,
        ),
        SparseAutoencoder(sae_cfg),
    )
    output = model(torch.randn(4, 8, 8))
    assert output["context_codes"].shape == (4, 7, 24)
    assert output["predicted_codes"].shape == (4, 7, 24)
    assert output["target_codes"].shape == (4, 7, 24)
    assert output["target_residual"].shape == (4, 7, 8)


def test_ema_update_tracks_full_sae_and_normalizes_decoder() -> None:
    sae = SparseAutoencoder(SparseAutoencoderConfig(d_in=8, d_sae=24, k=3))
    model = TransitionJEPA(
        TransitionJEPAConfig(d_in=8, d_sae=24, k=3, window_size=5),
        sae,
    )
    before_bias = model.ema_pre_bias.clone()
    before_decoder = model.ema_decoder.clone()
    with torch.no_grad():
        model.sae.pre_bias.add_(2)
        model.sae.decoder.add_(0.5)
    model.update_ema_sae(decay=0.5)
    torch.testing.assert_close(model.ema_pre_bias, before_bias + 1)
    assert not torch.allclose(model.ema_decoder, before_decoder)
    torch.testing.assert_close(
        model.ema_decoder.norm(dim=1),
        torch.ones(model.cfg.d_sae),
        atol=1e-6,
        rtol=0,
    )


def test_final_sae_is_the_full_ema_teacher() -> None:
    sae = SparseAutoencoder(SparseAutoencoderConfig(d_in=8, d_sae=24, k=3))
    model = TransitionJEPA(
        TransitionJEPAConfig(d_in=8, d_sae=24, k=3, window_size=5),
        sae,
    )
    x = torch.randn(6, 8)
    final = model.final_ema_sae()
    expected = model.encode_ema(x)
    torch.testing.assert_close(final.encode_token_topk(x), expected)
    torch.testing.assert_close(final.decode(expected), model.decode_ema(expected))
