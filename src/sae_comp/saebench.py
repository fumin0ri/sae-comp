from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import ExperimentConfig
from .evaluation import controlled_checkpoint_paths, load_method

CUSTOM_SAE_ID = "custom_sae"


@dataclass
class AdapterConfig:
    """The configuration surface expected by SAEBench v0.6.0 custom SAEs."""

    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    context_size: int
    architecture: str
    activation_fn_str: str
    activation_fn_kwargs: dict[str, Any] = field(default_factory=dict)
    hook_head_index: int | None = None
    apply_b_dec_to_input: bool = True
    finetuning_scaling_factor: bool = False
    prepend_bos: bool = True
    normalize_activations: str = "none"
    dtype: str = "float32"
    device: str = "cpu"
    model_from_pretrained_kwargs: dict[str, Any] = field(default_factory=dict)
    dataset_path: str = ""
    dataset_trust_remote_code: bool = True
    seqpos_slice: tuple[None] = (None,)
    training_tokens: int = -1
    sae_lens_training_version: str | None = None
    neuronpedia_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SAEBenchAdapter(nn.Module):
    """Lossless reparameterization of a local SAE for SAEBench.

    The local decoder is unit-normalized but multiplies decoded values by a
    scalar activation normalization factor. SAEBench requires unit decoder
    vectors, so encode emits scalar-rescaled features and decode uses the unit
    decoder directly. The resulting reconstruction is exactly unchanged.
    """

    def __init__(
        self,
        *,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        feature_scale: torch.Tensor,
        threshold: torch.Tensor,
        use_threshold: bool,
        k: int,
        cfg: AdapterConfig,
    ):
        super().__init__()
        self.W_enc = nn.Parameter(W_enc.detach().clone(), requires_grad=False)
        self.W_dec = nn.Parameter(W_dec.detach().clone(), requires_grad=False)
        self.b_enc = nn.Parameter(b_enc.detach().clone(), requires_grad=False)
        self.b_dec = nn.Parameter(b_dec.detach().clone(), requires_grad=False)
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        self.register_buffer("threshold", threshold.detach().clone())
        self.use_threshold = use_threshold
        self.k = k
        self.cfg = cfg
        self.device = self.W_enc.device
        self.dtype = self.W_enc.dtype

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        preactivations = (x - self.b_dec) @ self.W_enc + self.b_enc
        positive = F.relu(preactivations)
        if self.use_threshold:
            code = positive * (positive > self.threshold)
        else:
            selected = positive.topk(self.k, dim=-1, sorted=False)
            code = torch.zeros_like(positive).scatter_(
                -1, selected.indices, selected.values
            )
        return code * self.feature_scale

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def to(self, *args: Any, **kwargs: Any) -> SAEBenchAdapter:
        super().to(*args, **kwargs)
        self.device = self.W_enc.device
        self.dtype = self.W_enc.dtype
        self.cfg.device = str(self.device)
        self.cfg.dtype = str(self.dtype).removeprefix("torch.")
        return self

    @torch.no_grad()
    def check_decoder_norms(self) -> bool:
        tolerance = 1e-2 if self.W_dec.dtype in {torch.float16, torch.bfloat16} else 1e-5
        norms = self.W_dec.norm(dim=1)
        return bool(
            torch.allclose(norms, torch.ones_like(norms), atol=tolerance, rtol=0)
        )


def checkpoint_to_saebench(
    checkpoint_path: str | Path,
    label: str,
    cfg: ExperimentConfig,
) -> SAEBenchAdapter:
    sae, _, method = load_method(checkpoint_path, torch.device("cpu"))
    scale = sae.pre_scale.detach().float()
    architecture = {
        "standard": "standard-topk",
        "temporal": "temporal-batchtopk",
        "proposal": f"transition-jepa-{label.removeprefix('proposal_')}",
    }[method]
    activation_fn = "threshold" if method == "temporal" else "topk"
    training_tokens = (
        (cfg.train.standard_steps + cfg.train.branch_steps)
        * cfg.train.token_batch_size
        * cfg.train.gradient_accumulation_steps
    )
    adapter_cfg = AdapterConfig(
        model_name=cfg.sae_bench.model_name,
        d_in=sae.cfg.d_in,
        d_sae=sae.cfg.d_sae,
        hook_layer=cfg.model.layer,
        hook_name=f"blocks.{cfg.model.layer}.hook_resid_post",
        context_size=cfg.sae_bench.context_size,
        architecture=architecture,
        activation_fn_str=activation_fn,
        activation_fn_kwargs={"k": sae.cfg.k},
        dtype="float32",
        device="cpu",
        dataset_path=cfg.data.dataset,
        training_tokens=training_tokens,
    )
    adapter = SAEBenchAdapter(
        W_enc=sae.encoder.weight.detach().T.float() / scale,
        W_dec=sae.decoder.detach().float(),
        b_enc=sae.encoder.bias.detach().float(),
        b_dec=sae.pre_bias.detach().float(),
        feature_scale=scale,
        threshold=sae.threshold.detach().float(),
        use_threshold=method == "temporal",
        k=sae.cfg.k,
        cfg=adapter_cfg,
    )
    if not adapter.check_decoder_norms():
        raise ValueError(f"checkpoint has non-unit decoder vectors: {checkpoint_path}")
    return adapter


def _check_saebench_install(cfg: ExperimentConfig) -> str:
    if not cfg.sae_bench.enabled:
        raise ValueError("SAEBench is disabled in this configuration")
    try:
        installed = version("sae-bench")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "SAEBench is not installed; run `python -m pip install -e '.[saebench]'`"
        ) from exc
    if installed != cfg.sae_bench.version:
        raise RuntimeError(
            f"SAEBench {cfg.sae_bench.version} is required, found {installed}"
        )
    return installed


def _result_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"{label}_{CUSTOM_SAE_ID}_eval_results.json"


def _verify_results(output_dir: Path, labels: list[str], eval_type: str) -> None:
    missing = [
        str(_result_path(output_dir, label))
        for label in labels
        if not _result_path(output_dir, label).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"SAEBench {eval_type} did not produce every expected result: {missing}"
        )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_saebench(cfg: ExperimentConfig) -> Path:
    """Run only the explicitly allowlisted SAEBench evaluations."""

    installed_version = _check_saebench_install(cfg)
    checkpoint_paths = controlled_checkpoint_paths(cfg)
    missing_checkpoints = [
        str(path) for path in checkpoint_paths.values() if not path.is_file()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            f"train all controlled conditions before SAEBench: {missing_checkpoints}"
        )

    root = Path(cfg.run_dir) / "saebench_results"
    artifacts = root / "artifacts"
    manifest_path = root / "manifest.json"
    labels = list(checkpoint_paths)
    selected_saes = [
        (label, checkpoint_to_saebench(path, label, cfg))
        for label, path in checkpoint_paths.items()
    ]
    manifest: dict[str, Any] = {
        "status": "running",
        "saebench_version": installed_version,
        "config_fingerprint": cfg.fingerprint(),
        "model_name": cfg.sae_bench.model_name,
        "hook_name": f"blocks.{cfg.model.layer}.hook_resid_post",
        "eval_types": list(cfg.sae_bench.eval_types),
        "excluded_eval_types": list(cfg.sae_bench.excluded_eval_types),
        "conditions": {
            label: str(path) for label, path in checkpoint_paths.items()
        },
        "stages": {},
    }
    _write_manifest(manifest_path, manifest)

    try:
        for eval_type in cfg.sae_bench.eval_types:
            output_dir = root / eval_type
            manifest["stages"][eval_type] = "running"
            _write_manifest(manifest_path, manifest)

            if eval_type == "core":
                from sae_bench.evals.core import main as core

                core.multiple_evals(
                    selected_saes=selected_saes,
                    n_eval_reconstruction_batches=(
                        cfg.sae_bench.core_reconstruction_batches
                    ),
                    n_eval_sparsity_variance_batches=(
                        cfg.sae_bench.core_sparsity_variance_batches
                    ),
                    eval_batch_size_prompts=cfg.sae_bench.core_prompt_batch_size,
                    compute_featurewise_density_statistics=True,
                    compute_featurewise_weight_based_metrics=True,
                    exclude_special_tokens_from_reconstruction=True,
                    dataset=cfg.sae_bench.core_dataset,
                    context_size=cfg.sae_bench.context_size,
                    output_folder=str(output_dir),
                    verbose=True,
                    dtype=cfg.sae_bench.llm_dtype,
                    device=cfg.train.device,
                    force_rerun=cfg.sae_bench.force_rerun,
                )
            elif eval_type == "sparse_probing":
                from sae_bench.evals.sparse_probing import main as sparse_probing
                from sae_bench.evals.sparse_probing.eval_config import (
                    SparseProbingEvalConfig,
                )

                sparse_probing.run_eval(
                    SparseProbingEvalConfig(
                        model_name=cfg.sae_bench.model_name,
                        random_seed=cfg.train.seed,
                        context_length=cfg.sae_bench.context_size,
                        llm_batch_size=cfg.sae_bench.llm_batch_size,
                        llm_dtype=cfg.sae_bench.llm_dtype,
                        k_values=[1, 2, 5],
                    ),
                    selected_saes,
                    cfg.train.device,
                    str(output_dir),
                    force_rerun=cfg.sae_bench.force_rerun,
                    clean_up_activations=True,
                    save_activations=False,
                    artifacts_path=str(artifacts),
                )
            elif eval_type == "sparse_probing_sae_probes":
                from sae_bench.evals.sparse_probing_sae_probes import (
                    main as sparse_probing_sae_probes,
                )
                from sae_bench.evals.sparse_probing_sae_probes.eval_config import (
                    SparseProbingSaeProbesEvalConfig,
                )

                sae_probe_artifacts = artifacts / "sparse_probing_sae_probes"
                sparse_probing_sae_probes.run_eval(
                    SparseProbingSaeProbesEvalConfig(
                        model_name=cfg.sae_bench.model_name,
                        random_seed=cfg.train.seed,
                        ks=[1, 2, 5],
                        results_path=str(sae_probe_artifacts / "results"),
                        model_cache_path=str(sae_probe_artifacts / "model_cache"),
                    ),
                    selected_saes,
                    cfg.train.device,
                    str(output_dir),
                    force_rerun=cfg.sae_bench.force_rerun,
                )
            elif eval_type == "ravel":
                from sae_bench.evals.ravel import main as ravel
                from sae_bench.evals.ravel.eval_config import RAVELEvalConfig

                ravel.run_eval(
                    RAVELEvalConfig(
                        model_name=cfg.sae_bench.model_name,
                        random_seed=cfg.train.seed,
                        llm_batch_size=max(1, cfg.sae_bench.llm_batch_size // 4),
                        llm_dtype=cfg.sae_bench.llm_dtype,
                        artifact_dir=str(artifacts / "ravel"),
                    ),
                    selected_saes,
                    cfg.train.device,
                    str(output_dir),
                    force_rerun=cfg.sae_bench.force_rerun,
                    artifacts_path=str(artifacts),
                )
            else:  # Protected by ExperimentConfig.validate.
                raise AssertionError(f"unreachable SAEBench eval type: {eval_type}")

            _verify_results(output_dir, labels, eval_type)
            manifest["stages"][eval_type] = "completed"
            _write_manifest(manifest_path, manifest)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as exc:
        for eval_type, status in manifest["stages"].items():
            if status == "running":
                manifest["stages"][eval_type] = "failed"
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "completed"
    _write_manifest(manifest_path, manifest)
    return manifest_path
