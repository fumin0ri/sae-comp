import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sae_comp.config import load_config
from sae_comp.models import SparseAutoencoder, SparseAutoencoderConfig
from sae_comp.saebench import (
    AdapterConfig,
    SAEBenchAdapter,
    _ravel_protocol_mismatch,
)
from sae_comp.saebench_report import build_saebench_report, collect_saebench_summary

ROOT = Path(__file__).resolve().parents[1]


def _adapter(sae: SparseAutoencoder, *, temporal: bool) -> SAEBenchAdapter:
    scale = sae.pre_scale.detach()
    return SAEBenchAdapter(
        W_enc=sae.encoder.weight.detach().T / scale,
        W_dec=sae.decoder.detach(),
        b_enc=sae.encoder.bias.detach(),
        b_dec=sae.pre_bias.detach(),
        feature_scale=scale,
        threshold=sae.threshold.detach(),
        use_threshold=temporal,
        use_group_topk=sae.cfg.group_topk,
        group_high_size=sae.cfg.group_high_size,
        group_high_k=sae.cfg.group_high_k,
        k=sae.cfg.k,
        cfg=AdapterConfig(
            model_name="pythia-160m-deduped",
            d_in=sae.cfg.d_in,
            d_sae=sae.cfg.d_sae,
            hook_layer=8,
            hook_name="blocks.8.hook_resid_post",
            context_size=128,
            architecture="test",
            activation_fn_str="threshold" if temporal else "topk",
        ),
    )


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("shape", [(7, 12), (2, 7, 12)])
def test_saebench_adapter_preserves_local_sae(
    temporal: bool, shape: tuple[int, ...]
) -> None:
    torch.manual_seed(1)
    sae = SparseAutoencoder(SparseAutoencoderConfig(d_in=12, d_sae=32, k=4))
    sae.initialize_normalization(torch.randn(12), 2.3)
    sae.threshold.fill_(0.15)
    adapter = _adapter(sae, temporal=temporal)
    values = torch.randn(shape)
    method = "temporal" if temporal else "standard"
    torch.testing.assert_close(adapter.encode(values), sae.encode(values, method) * 2.3)
    torch.testing.assert_close(adapter(values), sae.decode(sae.encode(values, method)))
    assert adapter.check_decoder_norms()
    assert adapter.W_enc.shape == (12, 32)
    assert adapter.W_dec.shape == (32, 12)


@pytest.mark.parametrize("shape", [(7, 12), (2, 7, 12)])
def test_saebench_adapter_preserves_grouped_topk(shape: tuple[int, ...]) -> None:
    torch.manual_seed(2)
    sae = SparseAutoencoder(
        SparseAutoencoderConfig(
            d_in=12,
            d_sae=20,
            k=5,
            high_fraction=0.2,
            group_topk=True,
        )
    )
    sae.initialize_normalization(torch.randn(12), 1.7)
    adapter = _adapter(sae, temporal=False)
    values = torch.randn(shape)
    code = sae.encode_token_topk(values)
    torch.testing.assert_close(adapter.encode(values), code * 1.7)
    torch.testing.assert_close(adapter(values), sae.decode(code))
    assert adapter.use_group_topk


def test_controlled_config_enables_only_allowlisted_saebench_evals() -> None:
    cfg = load_config(ROOT / "configs" / "controlled_rtx4090.toml")
    assert cfg.sae_bench.enabled
    assert cfg.sae_bench.eval_types == [
        "core",
        "sparse_probing",
        "sparse_probing_sae_probes",
        "ravel",
    ]
    assert cfg.sae_bench.excluded_eval_types == ["scr", "tpp"]
    assert cfg.sae_bench.ravel_entity_attribute_selection == {
        "city": ["Country", "Continent", "Language"]
    }
    assert cfg.proposal.window_sizes == [16, 32, 64]


def test_controlled_config_rejects_scr_or_tpp(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "controlled_rtx4090.toml").read_text(
        encoding="utf-8"
    )
    source = source.replace('"ravel",\n]', '"ravel",\n  "tpp",\n]')
    path = tmp_path / "invalid.toml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly excluded"):
        load_config(path)


def test_ravel_protocol_mismatch_detects_stale_results(tmp_path: Path) -> None:
    expected = {"city": ["Country", "Continent", "Language"]}
    assert not _ravel_protocol_mismatch(tmp_path, ["standard"], expected)
    path = tmp_path / "standard_custom_sae_eval_results.json"
    path.write_text(
        json.dumps(
            {
                "eval_config": {
                    "entity_attribute_selection": expected,
                }
            }
        ),
        encoding="utf-8",
    )
    assert not _ravel_protocol_mismatch(tmp_path, ["standard"], expected)
    path.write_text(
        json.dumps(
            {
                "eval_config": {
                    "entity_attribute_selection": {
                        "nobel_prize_winner": ["Field", "Gender"]
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    assert _ravel_protocol_mismatch(tmp_path, ["standard"], expected)


def test_saebench_summary_reads_official_result_shape(tmp_path: Path) -> None:
    cfg = replace(
        load_config(ROOT / "configs" / "controlled_rtx4090.toml"),
        run_dir=str(tmp_path),
    )
    root = tmp_path / "saebench_results"
    labels = ["standard", "temporal", "proposal_w016", "proposal_w032", "proposal_w064"]
    fixtures = {
        "core": {
            "reconstruction_quality": {"explained_variance": 0.8, "mse": 0.2},
            "model_performance_preservation": {"ce_loss_score": 0.9},
            "sparsity": {"l0": 20.0},
        },
        "sparse_probing": {
            "sae": {
                "sae_top_1_test_accuracy": 0.6,
                "sae_top_2_test_accuracy": 0.7,
                "sae_top_5_test_accuracy": 0.8,
            }
        },
        "sparse_probing_sae_probes": {
            "sae": {
                "sae_top_1_test_accuracy": 0.61,
                "sae_top_2_test_accuracy": 0.71,
                "sae_top_5_test_accuracy": 0.81,
            }
        },
        "ravel": {
            "ravel": {
                "disentanglement_score": 0.5,
                "cause_score": 0.6,
                "isolation_score": 0.4,
            }
        },
    }
    for eval_type, metrics in fixtures.items():
        output_dir = root / eval_type
        output_dir.mkdir(parents=True)
        for label in labels:
            (output_dir / f"{label}_custom_sae_eval_results.json").write_text(
                json.dumps({"eval_result_metrics": metrics}), encoding="utf-8"
            )

    summary = collect_saebench_summary(cfg)
    assert list(summary["conditions"]) == labels
    assert summary["conditions"]["standard"]["core"]["l0"] == 20.0
    assert (
        summary["conditions"]["proposal_w064"]["ravel"]["disentanglement_score"]
        == 0.5
    )
    assert (root / "summary.json").is_file()
    assert (root / "summary.csv").is_file()
    report = build_saebench_report(cfg)
    assert report.is_file()
    assert (root / "plots" / "overview.png").is_file()
    assert (root / "plots" / "core.png").is_file()
    assert (root / "plots" / "probing.png").is_file()
    assert (root / "plots" / "ravel.png").is_file()


def test_partial_saebench_report_uses_available_results(tmp_path: Path) -> None:
    cfg = replace(
        load_config(ROOT / "configs" / "controlled_rtx4090.toml"),
        run_dir=str(tmp_path),
    )
    root = tmp_path / "saebench_results"
    output_dir = root / "core"
    output_dir.mkdir(parents=True)
    core_metrics = {
        "reconstruction_quality": {"explained_variance": 0.8, "mse": 0.2},
        "model_performance_preservation": {"ce_loss_score": 0.9},
        "sparsity": {"l0": 20.0},
    }
    labels = ["standard", "temporal", "proposal_w016", "proposal_w032", "proposal_w064"]
    for label in labels:
        (output_dir / f"{label}_custom_sae_eval_results.json").write_text(
            json.dumps({"eval_result_metrics": core_metrics}), encoding="utf-8"
        )

    report = build_saebench_report(cfg)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    report_text = report.read_text(encoding="utf-8")
    assert summary["completed_eval_types"] == ["core"]
    assert len(summary["missing_results"]) == 15
    assert "**Partial report:**" in report_text
    assert "| `core` | 5/5 | - |" in report_text
    assert "| `sparse_probing` | 0/5 |" in report_text
    assert (root / "plots" / "core.png").is_file()
    assert not (root / "plots" / "overview.png").exists()


def test_saebench_report_without_results_explains_how_to_resume(
    tmp_path: Path,
) -> None:
    cfg = replace(
        load_config(ROOT / "configs" / "controlled_rtx4090.toml"),
        run_dir=str(tmp_path),
    )
    report = build_saebench_report(cfg)
    report_text = report.read_text(encoding="utf-8")
    assert "**Partial report:**" in report_text
    assert "sae-comp saebench --config" in report_text
    assert "| `core` | 0/5 |" in report_text
    assert not (tmp_path / "saebench_results" / "plots").exists()
