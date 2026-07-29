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
    assert output["prediction"].shape == (4, 4, 24)
    assert output["targets"].shape == (4, 4, 24)
    assert output["reconstruction"].shape == (4, 5, 8)
    assert not output["targets"].requires_grad


def test_transition_jepa_accepts_sampled_offsets() -> None:
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
    output = model(torch.randn(4, 8, 8), torch.tensor([1, 3, 7]))
    assert output["prediction"].shape == (4, 3, 24)
    assert output["targets"].shape == (4, 3, 24)
    assert output["target_residual"].shape == (4, 3, 8)
    torch.testing.assert_close(output["offsets"], torch.tensor([1, 3, 7]))
