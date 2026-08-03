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
    RectifiedLpJEPAConfig,
    RectifiedLpJEPASAE,
    SparseAutoencoder,
    SparseAutoencoderConfig,
    sample_rectified_generalized_gaussian,
)

PROPOSAL_SOURCE_COMMIT = "66d8a6f87929a9a415929043863acaa0f14d4207"


def axis_aligned_distribution_matching_loss(
    views: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match sampled high-code coordinate marginals with axis-aligned 1D OT."""
    if not views or any(view.ndim != 2 for view in views):
        raise ValueError("views must be non-empty matrices")
    if target.ndim != 2 or any(view.shape != target.shape for view in views):
        raise ValueError("axis RDM views and target must have the same shape")
    if features < 0:
        raise ValueError("axis RDM feature count must be non-negative")
    zero = views[0].float().sum() * 0.0
    if features == 0:
        return zero, torch.zeros((), device=target.device, dtype=torch.long)
    count = min(features, target.shape[-1])
    indices = torch.randperm(target.shape[-1], device=target.device)[:count]
    target_axes = torch.sort(target.index_select(-1, indices), dim=0).values
    raw = sum(
        (
            torch.sort(view.float().index_select(-1, indices), dim=0).values
            - target_axes
        )
        .square()
        .mean()
        for view in views
    ) / len(views)
    target_energy = target_axes.square().mean().clamp_min(1e-8)
    return raw / target_energy, torch.as_tensor(count, device=target.device)


def rectified_distribution_matching_loss(
    views: tuple[torch.Tensor, ...],
    model_cfg: RectifiedLpJEPAConfig,
    projections: int,
    projection_chunk_size: int,
    axis_features: int = 0,
    axis_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Random-projection and axis-aligned 2-Wasserstein RGG matching."""
    if not views or any(view.ndim != 2 for view in views):
        raise ValueError("views must be non-empty [batch, d_high] matrices")
    if any(view.shape != views[0].shape for view in views):
        raise ValueError("all RDM views must have the same shape")
    if views[0].shape[-1] != model_cfg.d_high:
        raise ValueError("RDM views must contain only the high code")
    if projections < 1 or projection_chunk_size < 1:
        raise ValueError("projection counts must be positive")
    if axis_features < 0 or axis_weight < 0:
        raise ValueError("axis RDM feature count and weight must be non-negative")
    batch, dimension = views[0].shape
    target = sample_rectified_generalized_gaussian(
        (batch, dimension),
        p=model_cfg.rgg_p,
        mu=model_cfg.target_mu,
        sigma=model_cfg.resolved_target_sigma,
        device=views[0].device,
    )
    total = views[0].float().sum() * 0.0
    raw_total = total
    completed = 0
    while completed < projections:
        width = min(projection_chunk_size, projections - completed)
        directions = F.normalize(
            torch.randn(dimension, width, device=views[0].device), dim=0
        )
        target_projection = torch.sort(target @ directions, dim=0).values
        target_energy = target_projection.square().mean().clamp_min(1e-8)
        chunk_raw = sum(
            (
                torch.sort(view.float() @ directions, dim=0).values
                - target_projection
            )
            .square()
            .mean()
            for view in views
        ) / len(views)
        total = total + width * chunk_raw / target_energy
        raw_total = raw_total + width * chunk_raw
        completed += width
    random_projection = total / projections
    axis_aligned, sampled_axes = axis_aligned_distribution_matching_loss(
        views, target, axis_features
    )
    combined = random_projection + axis_weight * axis_aligned
    return combined, {
        "random_projection": random_projection,
        "random_projection_raw": raw_total / projections,
        "axis_aligned": axis_aligned,
        "axis_sampled_features": sampled_axes,
        "target_active_fraction": (target > 0).float().mean(),
        "target_l0": (target > 0).sum(dim=-1).float().mean(),
        "target_second_moment": target.square().mean(),
    }


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
    model: RectifiedLpJEPASAE,
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    invariance_weight: float,
    rdm_weight: float,
    cfg: ExperimentConfig,
    *,
    distance: torch.Tensor | None = None,
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(view_a, view_b)
    residual_scale = (
        torch.cat(
            (view_a - model.pre_bias, view_b - model.pre_bias), dim=0
        )
        .float()
        .square()
        .mean()
        .clamp_min(1e-8)
    )
    high_reconstruction = 0.5 * (
        (output["high_reconstruction_a"] - view_a).float().square().mean()
        + (output["high_reconstruction_b"] - view_b).float().square().mean()
    ) / residual_scale
    full_reconstruction = 0.5 * (
        (output["full_reconstruction_a"] - view_a).float().square().mean()
        + (output["full_reconstruction_b"] - view_b).float().square().mean()
    ) / residual_scale
    reconstruction = (
        model.cfg.high_reconstruction_weight * high_reconstruction
        + (1 - model.cfg.high_reconstruction_weight) * full_reconstruction
    )
    invariance_raw = (
        output["high_a"].float() - output["high_b"].float()
    ).square().mean()
    if rdm_weight > 0 or collect_metrics:
        rdm, rdm_metrics = rectified_distribution_matching_loss(
            (output["high_a"], output["high_b"]),
            model.cfg,
            cfg.proposal.rdm_projections,
            cfg.proposal.rdm_projection_chunk_size,
            cfg.proposal.axis_rdm_features,
            cfg.proposal.axis_rdm_weight,
        )
    else:
        target = sample_rectified_generalized_gaussian(
            output["high_a"].shape,
            p=model.cfg.rgg_p,
            mu=model.cfg.target_mu,
            sigma=model.cfg.resolved_target_sigma,
            device=output["high_a"].device,
        )
        rdm = output["high_a"].float().sum() * 0.0
        rdm_metrics = {
            "random_projection": rdm,
            "random_projection_raw": rdm,
            "axis_aligned": rdm,
            "axis_sampled_features": torch.zeros(
                (), device=rdm.device, dtype=torch.long
            ),
            "target_active_fraction": (target > 0).float().mean(),
            "target_l0": (target > 0).sum(dim=-1).float().mean(),
            "target_second_moment": target.square().mean(),
        }
    invariance = invariance_raw / rdm_metrics["target_second_moment"].clamp_min(
        1e-8
    )
    loss = reconstruction + invariance_weight * invariance + rdm_weight * rdm
    if not collect_metrics:
        return loss, {}

    with torch.no_grad():
        permutation = torch.roll(
            torch.arange(len(view_a), device=view_a.device), shifts=1
        )
        high_positive = F.cosine_similarity(
            output["high_a"].float(), output["high_b"].float(), dim=-1
        )
        high_shuffled = F.cosine_similarity(
            output["high_a"].float(),
            output["high_b"].index_select(0, permutation).float(),
            dim=-1,
        )
        low_positive = F.cosine_similarity(
            output["low_a"].float(), output["low_b"].float(), dim=-1
        )
        swap_a = model.decode_high(output["high_b"]) + model.decode_low(
            output["low_a"]
        )
        swap_b = model.decode_high(output["high_a"]) + model.decode_low(
            output["low_b"]
        )
        swap_fvu = 0.5 * (
            (swap_a - view_a).float().square().mean()
            + (swap_b - view_b).float().square().mean()
        ) / residual_scale
    return loss, {
        "loss": float(loss.detach()),
        "reconstruction_loss": float(reconstruction.detach()),
        "full_reconstruction_fvu": float(full_reconstruction.detach()),
        "high_reconstruction_fvu": float(high_reconstruction.detach()),
        "invariance_loss": float(invariance.detach()),
        "invariance_raw_mse": float(invariance_raw.detach()),
        "rdm_loss": float(rdm.detach()),
        "random_projection_rdm_loss": float(
            rdm_metrics["random_projection"].detach()
        ),
        "random_projection_rdm_raw": float(
            rdm_metrics["random_projection_raw"].detach()
        ),
        "axis_aligned_rdm_loss": float(rdm_metrics["axis_aligned"].detach()),
        "axis_sampled_features": float(rdm_metrics["axis_sampled_features"]),
        "high_positive_cosine": float(high_positive.mean()),
        "high_shuffled_cosine": float(high_shuffled.mean()),
        "high_positive_margin": float((high_positive - high_shuffled).mean()),
        "low_positive_cosine": float(low_positive.mean()),
        "swap_reconstruction_fvu": float(swap_fvu),
        "high_l0": float(
            0.5
            * (
                (output["high_a"] > 0).sum(dim=-1).float().mean()
                + (output["high_b"] > 0).sum(dim=-1).float().mean()
            )
        ),
        "low_l0": float(
            0.5
            * (
                (output["low_a"] > 0).sum(dim=-1).float().mean()
                + (output["low_b"] > 0).sum(dim=-1).float().mean()
            )
        ),
        "high_active_fraction": float(
            0.5
            * (
                (output["high_a"] > 0).float().mean()
                + (output["high_b"] > 0).float().mean()
            )
        ),
        "sampled_target_active_fraction": float(
            rdm_metrics["target_active_fraction"]
        ),
        "sampled_target_l0": float(rdm_metrics["target_l0"]),
        "mean_distance": (
            float(distance.float().mean()) if distance is not None else 0.0
        ),
    }


def _train_proposal(
    model: RectifiedLpJEPASAE,
    store: ActivationStore,
    cfg: ExperimentConfig,
    *,
    pair_batch_size: int | None = None,
    boundary_max_distance: int | None = None,
    description: str = "Rectified LpJEPA-SAE",
) -> list[dict[str, float | int | str]]:
    device = next(model.parameters()).device
    batch_size = pair_batch_size or cfg.proposal.sweep_pairs_per_step
    iterator = store.random_view_pair_batches(
        batch_size,
        max_span_length=model.cfg.max_span_length,
        min_span_length=cfg.proposal.min_span_length,
        boundary_max_horizon=boundary_max_distance,
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "lr": cfg.train.proposal_sae_lr,
                "base_lr": cfg.train.proposal_sae_lr,
            },
        ],
        weight_decay=0.0,
        fused=device.type == "cuda",
    )
    history: list[dict[str, float | int | str]] = []
    for step in trange(1, cfg.train.branch_steps + 1, desc=description):
        joint_step = max(0, step - cfg.proposal.sae_warmup_steps)
        phase = "distribution_warmup" if joint_step == 0 else "joint"
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(
                step,
                cfg.train.branch_steps,
                float(group["base_lr"]),
                min(cfg.train.warmup_steps, cfg.train.branch_steps // 10),
            )
        rdm_ramp = min(
            1.0,
            step / max(cfg.proposal.regularization_ramp_steps, 1),
        )
        invariance_ramp = min(
            1.0,
            joint_step / max(cfg.proposal.regularization_ramp_steps, 1),
        )
        active_rdm_weight = cfg.proposal.rdm_weight * rdm_ramp
        active_invariance_weight = (
            cfg.proposal.invariance_weight * invariance_ramp
        )
        should_log = (
            step == 1
            or step % cfg.train.log_every == 0
            or step in {cfg.proposal.sae_warmup_steps, cfg.train.branch_steps}
        )
        optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, float] = {}
        for _ in range(cfg.train.gradient_accumulation_steps):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in next(iterator).items()
            }
            with _autocast(device, cfg.train.amp_dtype):
                loss, metrics = _proposal_loss(
                    model,
                    batch["view_a"],
                    batch["view_b"],
                    active_invariance_weight,
                    active_rdm_weight,
                    cfg,
                    distance=batch["distance"],
                    collect_metrics=should_log,
                )
                scaled_loss = loss / cfg.train.gradient_accumulation_steps
            scaled_loss.backward()
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value
        metrics = {
            key: value / cfg.train.gradient_accumulation_steps
            for key, value in metric_sums.items()
        }
        _project_decoder_gradient(model)
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            cfg.train.gradient_clip,
        )
        optimizer.step()
        model.normalize_decoder()
        if should_log:
            history.append(
                {
                    "step": step,
                    "phase": phase,
                    "active_invariance_weight": active_invariance_weight,
                    "active_rdm_weight": active_rdm_weight,
                    "window_size": model.cfg.max_span_length,
                    "pair_batch_size": batch_size,
                    "residual_values": (
                        2 * batch_size * cfg.train.gradient_accumulation_steps
                    ),
                    "view_reconstructions": (
                        2 * batch_size * cfg.train.gradient_accumulation_steps
                    ),
                    "exchangeable_view_pairs": (
                        batch_size * cfg.train.gradient_accumulation_steps
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
    minimum_sequence_length = (
        cfg.data.burn_in_tokens + max(cfg.proposal.window_sizes)
    )
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

    torch.manual_seed(cfg.train.seed)
    proposal_cfg = RectifiedLpJEPAConfig(
        d_in=sae_cfg.d_in,
        d_sae=sae_cfg.d_sae,
        low_k=cfg.proposal.low_k,
        max_span_length=cfg.proposal.window_size,
        high_fraction=cfg.proposal.high_fraction,
        high_reconstruction_weight=cfg.proposal.high_reconstruction_weight,
        rgg_p=cfg.proposal.rgg_p,
        target_active_fraction=cfg.proposal.target_active_fraction,
        target_sigma=cfg.proposal.target_sigma,
    )
    proposal = RectifiedLpJEPASAE(proposal_cfg).to(device)
    proposal.initialize_normalization(
        torch.tensor(manifest["normalization"]["mean"]),
        float(manifest["normalization"]["scalar_rms"]),
    )
    del base
    history["proposal"] = _train_proposal(
        proposal,
        ActivationStore(manifest_path, cfg.train.seed + 100),
        cfg,
        boundary_max_distance=max(cfg.proposal.window_sizes) - 1,
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
    """Train Rectified LpJEPA variants from one identical random initialization."""
    device = torch.device(cfg.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _configure_accelerator(device)

    manifest_path = Path(cfg.activation_dir) / "manifest.json"
    _, manifest = load_manifest(manifest_path)
    checkpoint_dir = Path(cfg.run_dir) / "checkpoints"
    d_in = int(manifest["d_in"])

    maximum_window = max(cfg.proposal.window_sizes)
    torch.manual_seed(cfg.train.seed)
    template_cfg = RectifiedLpJEPAConfig(
        d_in=d_in,
        d_sae=cfg.sae.dictionary_size,
        low_k=cfg.proposal.low_k,
        max_span_length=maximum_window,
        high_fraction=cfg.proposal.high_fraction,
        high_reconstruction_weight=cfg.proposal.high_reconstruction_weight,
        rgg_p=cfg.proposal.rgg_p,
        target_active_fraction=cfg.proposal.target_active_fraction,
        target_sigma=cfg.proposal.target_sigma,
    )
    template = RectifiedLpJEPASAE(template_cfg)
    template.initialize_normalization(
        torch.tensor(manifest["normalization"]["mean"]),
        float(manifest["normalization"]["scalar_rms"]),
    )
    template_state = _state_dict_cpu(template)
    paths: dict[str, Path] = {}
    histories: dict[str, Any] = {}
    budgets: dict[str, Any] = {}
    for window_size in cfg.proposal.window_sizes:
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.train.seed)
        budget = cfg.proposal.sweep_budget(window_size)
        proposal_cfg = RectifiedLpJEPAConfig(
            d_in=d_in,
            d_sae=cfg.sae.dictionary_size,
            low_k=cfg.proposal.low_k,
            max_span_length=window_size,
            high_fraction=cfg.proposal.high_fraction,
            high_reconstruction_weight=cfg.proposal.high_reconstruction_weight,
            rgg_p=cfg.proposal.rgg_p,
            target_active_fraction=cfg.proposal.target_active_fraction,
            target_sigma=cfg.proposal.target_sigma,
        )
        proposal = RectifiedLpJEPASAE(proposal_cfg)
        proposal.load_state_dict(template_state)
        proposal.to(device)
        label = f"proposal_w{window_size:03d}"
        histories[label] = _train_proposal(
            proposal,
            ActivationStore(manifest_path, cfg.train.seed + 100),
            cfg,
            pair_batch_size=budget["pair_batch_size"],
            boundary_max_distance=maximum_window - 1,
            description=f"Rectified LpJEPA-SAE max-span={window_size}",
        )
        budget_record = {
            **budget,
            "optimizer_steps": cfg.train.branch_steps,
            "total_residual_values": (
                budget["residual_values_per_step"] * cfg.train.branch_steps
            ),
            "total_reconstructions": (
                budget["reconstructions_per_step"]
                * cfg.train.branch_steps
            ),
            "total_sampled_pairs": (
                budget["sampled_pairs_per_step"] * cfg.train.branch_steps
            ),
            "minimum_sequence_length": (
                cfg.data.burn_in_tokens + maximum_window
            ),
            "burn_in_tokens": cfg.data.burn_in_tokens,
            "boundary_max_distance": maximum_window - 1,
            "axis_rdm_features": min(
                cfg.proposal.axis_rdm_features, proposal_cfg.d_high
            ),
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
