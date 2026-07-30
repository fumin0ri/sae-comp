from __future__ import annotations

import copy
import json
import math
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import trange

from .activations import ActivationStore, load_manifest
from .config import ExperimentConfig
from .models import (
    PROPOSAL_ARCHITECTURE_ID,
    SparseAutoencoder,
    SparseAutoencoderConfig,
    TransitionJEPA,
    TransitionJEPAConfig,
)

PROPOSAL_SOURCE_COMMIT = "bdc0b4183741df4e0ecb62708c95bfc78cf79194"


def _autocast(device: torch.device, dtype: str):
    if device.type != "cuda" or dtype == "none":
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
    )


def _configure_accelerator(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def _learning_rate(step: int, total: int, maximum: float, warmup: int) -> float:
    if step <= warmup:
        return maximum * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return maximum * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def _fvu(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denominator = target.float().var(unbiased=False).clamp_min(1e-8)
    return (reconstruction - target).float().square().mean() / denominator


@torch.no_grad()
def _project_decoder_gradient(sae: SparseAutoencoder) -> None:
    if sae.decoder.grad is None:
        return
    parallel = (sae.decoder.grad * sae.decoder).sum(dim=1, keepdim=True) * sae.decoder
    sae.decoder.grad.sub_(parallel)


def _state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _save_checkpoint(
    path: Path,
    method: str,
    module: torch.nn.Module,
    model_config: dict[str, Any],
    cfg: ExperimentConfig,
    manifest: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "sae-comp-checkpoint-v2",
            "method": method,
            "architecture_id": (
                PROPOSAL_ARCHITECTURE_ID if method == "proposal" else None
            ),
            "state_dict": _state_dict_cpu(module),
            "model_config": model_config,
            "experiment_config": cfg.as_dict(),
            "config_fingerprint": cfg.fingerprint(),
            "activation_config_fingerprint": manifest["config_fingerprint"],
            "metadata": metadata or {},
            "source": {
                "proposal": {
                    "repository": "https://github.com/fumin0ri/my-sae",
                    "commit": PROPOSAL_SOURCE_COMMIT,
                },
                "temporal": ("https://github.com/AI4LIFE-GROUP/temporal-saes"),
            },
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("format") not in {
        "sae-comp-checkpoint-v1",
        "sae-comp-checkpoint-v2",
    }:
        raise ValueError(f"unsupported checkpoint: {path}")
    return value


def _standard_loss(
    sae: SparseAutoencoder, batch: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    code = sae.encode_token_topk(batch)
    reconstruction = sae.decode(code)
    loss = _fvu(reconstruction, batch)
    return loss, {
        "loss": float(loss.detach()),
        "fvu": float(loss.detach()),
        "l0": float((code > 0).sum(dim=-1).float().mean().detach()),
    }


def _train_standard_phase(
    sae: SparseAutoencoder,
    store: ActivationStore,
    cfg: ExperimentConfig,
    steps: int,
    description: str,
    minimum_sequence_length: int | None = None,
) -> list[dict[str, float | int]]:
    device = next(sae.parameters()).device
    iterator = store.token_batches(
        cfg.train.token_batch_size,
        minimum_sequence_length=minimum_sequence_length,
    )
    optimizer = torch.optim.AdamW(
        sae.parameters(), lr=cfg.train.standard_lr, fused=device.type == "cuda"
    )
    history: list[dict[str, float | int]] = []
    for step in trange(1, steps + 1, desc=description):
        lr = _learning_rate(step, steps, cfg.train.standard_lr, cfg.train.warmup_steps)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, float] = {}
        for _ in range(cfg.train.gradient_accumulation_steps):
            batch = next(iterator).to(device, non_blocking=True)
            with _autocast(device, cfg.train.amp_dtype):
                loss, metrics = _standard_loss(sae, batch)
                loss = loss / cfg.train.gradient_accumulation_steps
            loss.backward()
            for name, value in metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + value
        _project_decoder_gradient(sae)
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.train.gradient_clip)
        optimizer.step()
        sae.normalize_decoder()
        if step == 1 or step % cfg.train.log_every == 0 or step == steps:
            history.append(
                {
                    "step": step,
                    "lr": lr,
                    "reconstruction_tokens": (
                        cfg.train.token_batch_size
                        * cfg.train.gradient_accumulation_steps
                    ),
                    **{
                        key: value / cfg.train.gradient_accumulation_steps
                        for key, value in metric_sums.items()
                    },
                }
            )
    return history


def _symmetric_contrastive(
    current: torch.Tensor, previous: torch.Tensor, temperature: float
) -> torch.Tensor:
    current = F.normalize(current.float(), dim=-1, eps=1e-8)
    previous = F.normalize(previous.float(), dim=-1, eps=1e-8)
    logits = current @ previous.T / temperature
    labels = torch.arange(len(current), device=current.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _temporal_loss(
    sae: SparseAutoencoder,
    current: torch.Tensor,
    previous: torch.Tensor,
    cfg: ExperimentConfig,
    dead_features: torch.Tensor | None = None,
    contrastive_rows: int | None = None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    code, selected_minimum, post_relu = sae.encode_batch_topk(current)
    pair_rows = contrastive_rows or len(current)
    if not 1 <= pair_rows <= len(current):
        raise ValueError("contrastive_rows must lie in [1, batch size]")
    previous_code, _, _ = sae.encode_batch_topk(previous[:pair_rows])
    high = sae.cfg.high_size
    high_reconstruction = sae.pre_bias + sae.pre_scale * (
        code[:, :high] @ sae.decoder[:high]
    )
    full_reconstruction = sae.decode(code)
    high_loss = _fvu(high_reconstruction, current)
    full_loss = _fvu(full_reconstruction, current)
    contrastive = _symmetric_contrastive(
        code[:pair_rows, :high],
        previous_code[:, :high],
        cfg.sae.contrastive_temperature,
    )
    auxiliary = torch.zeros((), device=current.device)
    if dead_features is not None and bool(dead_features.any()):
        residual = (current - full_reconstruction).detach()
        k_aux = min(sae.cfg.d_in // 2, int(dead_features.sum()))
        candidates = torch.where(dead_features[None], post_relu, -torch.inf)
        selected = candidates.topk(k_aux, dim=-1, sorted=False)
        auxiliary_code = torch.zeros_like(post_relu).scatter_(
            -1, selected.indices, selected.values
        )
        auxiliary_reconstruction = sae.pre_scale * (auxiliary_code @ sae.decoder)
        numerator = (
            (residual.float() - auxiliary_reconstruction.float())
            .square()
            .sum(dim=-1)
            .mean()
        )
        denominator = (
            (residual.float() - residual.float().mean(dim=0))
            .square()
            .sum(dim=-1)
            .mean()
            .clamp_min(1e-8)
        )
        auxiliary = numerator / denominator
    loss = (
        cfg.sae.high_reconstruction_weight * high_loss
        + cfg.sae.full_reconstruction_weight * full_loss
        + cfg.sae.temporal_alpha * contrastive
        + cfg.sae.auxiliary_weight * auxiliary
    )
    return (
        loss,
        {
            "loss": float(loss.detach()),
            "high_fvu": float(high_loss.detach()),
            "full_fvu": float(full_loss.detach()),
            "contrastive": float(contrastive.detach()),
            "auxiliary": float(auxiliary.detach()),
            "l0": float((code > 0).sum(dim=-1).float().mean().detach()),
        },
        selected_minimum,
        (code > 0).any(dim=0),
    )


def _train_temporal(
    sae: SparseAutoencoder,
    store: ActivationStore,
    cfg: ExperimentConfig,
    *,
    minimum_sequence_length: int | None = None,
) -> list[dict[str, float | int]]:
    device = next(sae.parameters()).device
    iterator = store.temporal_pair_batches(
        cfg.train.token_batch_size,
        minimum_sequence_length=minimum_sequence_length,
    )
    optimizer = torch.optim.Adam(
        sae.parameters(), lr=cfg.train.temporal_lr, betas=(0.9, 0.999)
    )
    history: list[dict[str, float | int]] = []
    tokens_since_fired = torch.zeros(sae.cfg.d_sae, dtype=torch.long, device=device)
    for step in trange(1, cfg.train.branch_steps + 1, desc="temporal SAE"):
        lr = _learning_rate(
            step,
            cfg.train.branch_steps,
            cfg.train.temporal_lr,
            cfg.train.warmup_steps,
        )
        optimizer.param_groups[0]["lr"] = lr
        current, previous = next(iterator)
        current = current.to(device, non_blocking=True)
        previous = previous.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        dead_features = tokens_since_fired >= cfg.sae.dead_token_threshold
        with _autocast(device, cfg.train.amp_dtype):
            loss, metrics, selected_minimum, active_features = _temporal_loss(
                sae,
                current,
                previous,
                cfg,
                dead_features,
                contrastive_rows=cfg.train.temporal_pairs_per_step,
            )
        loss.backward()
        _project_decoder_gradient(sae)
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.train.gradient_clip)
        optimizer.step()
        sae.normalize_decoder()
        with torch.no_grad():
            tokens_since_fired.add_(len(current))
            tokens_since_fired[active_features] = 0
            if sae.threshold < 0:
                sae.threshold.copy_(selected_minimum.float())
            else:
                sae.threshold.mul_(cfg.sae.threshold_beta).add_(
                    selected_minimum.float(),
                    alpha=1 - cfg.sae.threshold_beta,
                )
        if (
            step == 1
            or step % cfg.train.log_every == 0
            or step == cfg.train.branch_steps
        ):
            history.append(
                {
                    "step": step,
                    "lr": lr,
                    "threshold": float(sae.threshold),
                    "reconstruction_tokens": len(current),
                    "temporal_pairs": cfg.train.temporal_pairs_per_step,
                    **metrics,
                }
            )
    return history


def _proposal_loss(
    model: TransitionJEPA,
    windows: torch.Tensor,
    prediction_weight: float,
    cfg: ExperimentConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(windows)
    endpoint = windows[:, -1]
    residual_scale = (
        endpoint - model.sae.pre_bias.detach()
    ).float().square().mean().clamp_min(1e-8)
    reconstruction = (
        output["online_target_reconstruction"] - endpoint
    ).float().square().mean() / residual_scale
    ema_residual_scale = (
        endpoint - model.ema_pre_bias.detach()
    ).float().square().mean().clamp_min(1e-8)
    ema_reconstruction = (
        output["target_reconstruction"] - endpoint
    ).float().square().mean() / ema_residual_scale
    prediction = output["predicted_codes"]
    targets = output["target_codes"].detach()
    cosine = F.cosine_similarity(prediction, targets, dim=-1)
    target_energy = targets.float().square().mean(dim=-1).clamp_min(1e-8)
    normalized_mse = (prediction - targets).float().square().mean(
        dim=-1
    ) / target_energy
    prediction_loss = (1 - cosine + 0.25 * normalized_mse).mean()
    residual_prediction = (
        output["predictable_residual"] - output["target_residual"]
    ).float().square().mean() / ema_residual_scale
    loss = (
        reconstruction
        + prediction_weight
        * (
            prediction_loss
            + cfg.proposal.residual_prediction_weight * residual_prediction
        )
    )
    predicted_active = output["sparse_predicted_codes"] > 0
    target_active = targets > 0
    intersection = (predicted_active & target_active).sum(dim=-1).float()
    precision = intersection / predicted_active.sum(dim=-1).float().clamp_min(1)
    recall = intersection / target_active.sum(dim=-1).float().clamp_min(1)
    union = (predicted_active | target_active).sum(dim=-1).float().clamp_min(1)
    metrics = {
        "loss": float(loss.detach()),
        "online_reconstruction_fvu": float(reconstruction.detach()),
        "ema_reconstruction_fvu": float(ema_reconstruction.detach()),
        "prediction_loss": float(prediction_loss.detach()),
        "code_cosine": float(cosine.mean().detach()),
        "code_nrmse": float(normalized_mse.mean().detach()),
        "support_precision": float(precision.mean().detach()),
        "support_recall": float(recall.mean().detach()),
        "support_jaccard": float((intersection / union).mean().detach()),
        "residual_prediction_fvu": float(residual_prediction.detach()),
        "sae_l0": float(
            (output["codes"] > 0).sum(dim=-1).float().mean().detach()
        ),
    }
    target_position = model.cfg.window_size - 1
    for context_position in range(target_position):
        horizon = target_position - context_position
        prefix = f"context_{context_position}_horizon_{horizon}"
        metrics[f"{prefix}_cosine"] = float(
            cosine[:, context_position].mean().detach()
        )
        metrics[f"{prefix}_nrmse"] = float(
            normalized_mse[:, context_position].mean().detach()
        )
        metrics[f"{prefix}_support_recall"] = float(
            recall[:, context_position].mean().detach()
        )
    return loss, metrics


def _train_proposal(
    model: TransitionJEPA,
    store: ActivationStore,
    cfg: ExperimentConfig,
    *,
    window_batch_size: int | None = None,
    minimum_sequence_length: int | None = None,
    description: str = "transition JEPA",
) -> list[dict[str, float | int | str]]:
    device = next(model.parameters()).device
    batch_size = window_batch_size or cfg.train.window_batch_size
    iterator = store.window_batches(
        batch_size,
        model.cfg.window_size,
        minimum_sequence_length=minimum_sequence_length,
    )
    model.set_sae_trainable(False)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.predictor.parameters(),
                "lr": cfg.train.proposal_predictor_lr,
                "base_lr": cfg.train.proposal_predictor_lr,
            }
        ],
        fused=device.type == "cuda",
    )
    history: list[dict[str, float | int | str]] = []
    for step in trange(1, cfg.train.branch_steps + 1, desc=description):
        phase = (
            "predictor_warmup"
            if step <= cfg.proposal.predictor_warmup_steps
            else "joint"
        )
        if step == cfg.proposal.predictor_warmup_steps + 1:
            model.set_sae_trainable(True)
            optimizer.add_param_group(
                {
                    "params": [
                        parameter
                        for parameter in model.sae.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": cfg.train.proposal_sae_lr,
                    "base_lr": cfg.train.proposal_sae_lr,
                }
            )
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(
                step,
                cfg.train.branch_steps,
                float(group["base_lr"]),
                min(cfg.train.warmup_steps, cfg.train.branch_steps // 10),
            )
        joint_step = max(0, step - cfg.proposal.predictor_warmup_steps)
        prediction_weight = cfg.proposal.prediction_weight
        if phase == "joint":
            prediction_weight *= min(
                1.0,
                joint_step / max(cfg.proposal.prediction_ramp_steps, 1),
            )
        windows = next(iterator).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, cfg.train.amp_dtype):
            loss, metrics = _proposal_loss(model, windows, prediction_weight, cfg)
        loss.backward()
        if phase == "joint":
            _project_decoder_gradient(model.sae)
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            cfg.train.gradient_clip,
        )
        optimizer.step()
        if phase == "joint":
            model.sae.normalize_decoder()
            model.update_ema_sae()
        if (
            step == 1
            or step % cfg.train.log_every == 0
            or step in {cfg.proposal.predictor_warmup_steps, cfg.train.branch_steps}
        ):
            history.append(
                {
                    "step": step,
                    "phase": phase,
                    "prediction_weight": prediction_weight,
                    "window_size": model.cfg.window_size,
                    "batch_windows": batch_size,
                    "residual_positions": batch_size * model.cfg.window_size,
                    "endpoint_reconstructions": batch_size,
                    "context_target_pairs": (
                        batch_size * (model.cfg.window_size - 1)
                    ),
                    **metrics,
                }
            )
    return history


def train_controls(cfg: ExperimentConfig) -> dict[str, Path]:
    """Train the shared initialization and the two non-proposal controls."""
    device = torch.device(cfg.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _configure_accelerator(device)
    torch.manual_seed(cfg.train.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.train.seed)

    manifest_path = Path(cfg.activation_dir) / "manifest.json"
    _, manifest = load_manifest(manifest_path)
    minimum_sequence_length = max(cfg.proposal.window_sizes)
    sae_cfg = SparseAutoencoderConfig(
        d_in=int(manifest["d_in"]),
        d_sae=cfg.sae.dictionary_size,
        k=cfg.sae.k,
        high_fraction=cfg.sae.high_fraction,
    )
    base = SparseAutoencoder(sae_cfg).to(device)
    base.initialize_normalization(
        torch.tensor(manifest["normalization"]["mean"]),
        float(manifest["normalization"]["scalar_rms"]),
    )
    run_dir = Path(cfg.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    history: dict[str, Any] = {
        "comparison_budget": {
            "minimum_sequence_length": minimum_sequence_length,
            "branch_optimizer_steps": cfg.train.branch_steps,
            "reconstruction_tokens_per_step": cfg.train.token_batch_size,
            "temporal_pairs_per_step": cfg.train.temporal_pairs_per_step,
            "total_reconstruction_tokens_per_method": (
                cfg.train.token_batch_size * cfg.train.branch_steps
            ),
            "total_temporal_pairs_per_temporal_method": (
                cfg.train.temporal_pairs_per_step * cfg.train.branch_steps
            ),
        }
    }

    history["shared_standard_pretraining"] = _train_standard_phase(
        base,
        ActivationStore(manifest_path, cfg.train.seed),
        cfg,
        cfg.train.standard_steps,
        "shared standard SAE",
        minimum_sequence_length=minimum_sequence_length,
    )
    shared_path = checkpoint_dir / "shared_initialization.pt"
    _save_checkpoint(
        shared_path,
        "shared_initialization",
        base,
        base.checkpoint_config(),
        cfg,
        manifest,
        metadata={"minimum_sequence_length": minimum_sequence_length},
    )

    standard = copy.deepcopy(base)
    history["standard"] = _train_standard_phase(
        standard,
        ActivationStore(manifest_path, cfg.train.seed + 100),
        cfg,
        cfg.train.branch_steps,
        "standard Top-K SAE control",
        minimum_sequence_length=minimum_sequence_length,
    )
    standard_path = checkpoint_dir / "standard.pt"
    _save_checkpoint(
        standard_path,
        "standard",
        standard,
        standard.checkpoint_config(),
        cfg,
        manifest,
        metadata={
            "comparison_budget": {
                "optimizer_steps": cfg.train.branch_steps,
                "reconstruction_tokens_per_step": cfg.train.token_batch_size,
                "temporal_pairs_per_step": 0,
            }
        },
    )
    del standard

    temporal = copy.deepcopy(base)
    del base
    history["temporal"] = _train_temporal(
        temporal,
        ActivationStore(manifest_path, cfg.train.seed + 100),
        cfg,
        minimum_sequence_length=minimum_sequence_length,
    )
    temporal_path = checkpoint_dir / "temporal.pt"
    _save_checkpoint(
        temporal_path,
        "temporal",
        temporal,
        temporal.checkpoint_config(),
        cfg,
        manifest,
        metadata={
            "comparison_budget": {
                "optimizer_steps": cfg.train.branch_steps,
                "reconstruction_tokens_per_step": cfg.train.token_batch_size,
                "temporal_pairs_per_step": cfg.train.temporal_pairs_per_step,
            }
        },
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "controlled_training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "shared": shared_path,
        "standard": standard_path,
        "temporal": temporal_path,
    }


def train_all(cfg: ExperimentConfig) -> dict[str, Path]:
    device = torch.device(cfg.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _configure_accelerator(device)
    torch.manual_seed(cfg.train.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.train.seed)

    manifest_path = Path(cfg.activation_dir) / "manifest.json"
    _, manifest = load_manifest(manifest_path)
    store = ActivationStore(manifest_path, cfg.train.seed)
    sae_cfg = SparseAutoencoderConfig(
        d_in=int(manifest["d_in"]),
        d_sae=cfg.sae.dictionary_size,
        k=cfg.sae.k,
        high_fraction=cfg.sae.high_fraction,
    )
    base = SparseAutoencoder(sae_cfg).to(device)
    base.initialize_normalization(
        torch.tensor(manifest["normalization"]["mean"]),
        float(manifest["normalization"]["scalar_rms"]),
    )
    run_dir = Path(cfg.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    history: dict[str, Any] = {}

    history["shared_standard_pretraining"] = _train_standard_phase(
        base,
        store,
        cfg,
        cfg.train.standard_steps,
        "shared standard SAE",
    )
    base_path = checkpoint_dir / "shared_initialization.pt"
    _save_checkpoint(
        base_path,
        "shared_initialization",
        base,
        base.checkpoint_config(),
        cfg,
        manifest,
    )

    normal = copy.deepcopy(base)
    history["standard"] = _train_standard_phase(
        normal,
        ActivationStore(manifest_path, cfg.train.seed + 100),
        cfg,
        cfg.train.branch_steps,
        "standard SAE control",
    )
    normal_path = checkpoint_dir / "standard.pt"
    _save_checkpoint(
        normal_path,
        "standard",
        normal,
        normal.checkpoint_config(),
        cfg,
        manifest,
    )
    del normal

    temporal = copy.deepcopy(base)
    history["temporal"] = _train_temporal(
        temporal,
        ActivationStore(manifest_path, cfg.train.seed + 100),
        cfg,
    )
    temporal_path = checkpoint_dir / "temporal.pt"
    _save_checkpoint(
        temporal_path,
        "temporal",
        temporal,
        temporal.checkpoint_config(),
        cfg,
        manifest,
    )
    del temporal

    proposal_cfg = TransitionJEPAConfig(
        d_in=sae_cfg.d_in,
        d_sae=sae_cfg.d_sae,
        k=sae_cfg.k,
        window_size=cfg.proposal.window_size,
        high_fraction=sae_cfg.high_fraction,
        predictor_width=cfg.proposal.predictor_width,
        predictor_expansion=cfg.proposal.predictor_expansion,
        ema_decay=cfg.proposal.ema_decay,
    )
    proposal = TransitionJEPA(proposal_cfg, base).to(device)
    del base
    history["proposal"] = _train_proposal(
        proposal,
        ActivationStore(manifest_path, cfg.train.seed + 100),
        cfg,
    )
    proposal_path = checkpoint_dir / "proposal.pt"
    _save_checkpoint(
        proposal_path,
        "proposal",
        proposal,
        asdict(proposal_cfg),
        cfg,
        manifest,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "shared": base_path,
        "standard": normal_path,
        "temporal": temporal_path,
        "proposal": proposal_path,
    }


def train_proposal_window_sweep(cfg: ExperimentConfig) -> dict[str, Path]:
    """Train only proposal variants, all from the saved shared SAE initialization."""
    device = torch.device(cfg.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _configure_accelerator(device)

    manifest_path = Path(cfg.activation_dir) / "manifest.json"
    _, manifest = load_manifest(manifest_path)
    checkpoint_dir = Path(cfg.run_dir) / "checkpoints"
    shared_path = checkpoint_dir / "shared_initialization.pt"
    shared = load_checkpoint(shared_path)
    if shared["activation_config_fingerprint"] != manifest["config_fingerprint"]:
        raise ValueError(
            "shared initialization and activation cache use different configurations"
        )
    sae_cfg = SparseAutoencoderConfig(**shared["model_config"])
    base = SparseAutoencoder(sae_cfg)
    base.load_state_dict(shared["state_dict"])

    maximum_window = max(cfg.proposal.window_sizes)
    torch.manual_seed(cfg.train.seed)
    template_cfg = TransitionJEPAConfig(
        d_in=sae_cfg.d_in,
        d_sae=sae_cfg.d_sae,
        k=sae_cfg.k,
        window_size=maximum_window,
        high_fraction=sae_cfg.high_fraction,
        predictor_width=cfg.proposal.predictor_width,
        predictor_expansion=cfg.proposal.predictor_expansion,
        ema_decay=cfg.proposal.ema_decay,
    )
    template_state = TransitionJEPA(template_cfg, copy.deepcopy(base)).state_dict()
    paths: dict[str, Path] = {}
    histories: dict[str, Any] = {}
    budgets: dict[str, Any] = {}
    for window_size in cfg.proposal.window_sizes:
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.train.seed)
        budget = cfg.proposal.sweep_budget(window_size)
        proposal_cfg = TransitionJEPAConfig(
            d_in=sae_cfg.d_in,
            d_sae=sae_cfg.d_sae,
            k=sae_cfg.k,
            window_size=window_size,
            high_fraction=sae_cfg.high_fraction,
            predictor_width=cfg.proposal.predictor_width,
            predictor_expansion=cfg.proposal.predictor_expansion,
            ema_decay=cfg.proposal.ema_decay,
        )
        proposal = TransitionJEPA(proposal_cfg, copy.deepcopy(base))
        proposal_state = proposal.state_dict()
        for name, target in proposal_state.items():
            source = template_state[name]
            if source.shape == target.shape:
                target.copy_(source)
            elif name == "predictor.position_embedding.weight":
                target.copy_(source[:window_size])
            else:
                raise ValueError(f"cannot share sweep initialization for {name}")
        proposal.load_state_dict(proposal_state)
        proposal.to(device)
        label = f"proposal_w{window_size:03d}"
        histories[label] = _train_proposal(
            proposal,
            ActivationStore(manifest_path, cfg.train.seed + 100),
            cfg,
            window_batch_size=budget["batch_windows"],
            minimum_sequence_length=maximum_window,
            description=f"transition JEPA W={window_size}",
        )
        budget_record = {
            **budget,
            "optimizer_steps": cfg.train.branch_steps,
            "total_residual_positions": (
                budget["residual_positions_per_step"] * cfg.train.branch_steps
            ),
            "total_endpoint_reconstructions": (
                budget["endpoint_reconstructions_per_step"]
                * cfg.train.branch_steps
            ),
            "total_context_target_pairs": (
                budget["context_target_pairs_per_step"] * cfg.train.branch_steps
            ),
            "minimum_sequence_length": maximum_window,
        }
        budgets[label] = budget_record
        path = checkpoint_dir / f"{label}.pt"
        _save_checkpoint(
            path,
            "proposal",
            proposal,
            asdict(proposal_cfg),
            cfg,
            manifest,
            metadata={
                "window_sweep_budget": budget_record,
            },
        )
        paths[label] = path
        del proposal

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "window_sweep_training_history.json").write_text(
        json.dumps({"budgets": budgets, "histories": histories}, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths
