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
    SparseAutoencoder,
    SparseAutoencoderConfig,
    TransitionJEPA,
    TransitionJEPAConfig,
)


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "sae-comp-checkpoint-v1",
            "method": method,
            "state_dict": _state_dict_cpu(module),
            "model_config": model_config,
            "experiment_config": cfg.as_dict(),
            "config_fingerprint": cfg.fingerprint(),
            "activation_config_fingerprint": manifest["config_fingerprint"],
            "source": {
                "proposal": "https://github.com/fumin0ri/my-sae",
                "temporal": ("https://github.com/AI4LIFE-GROUP/temporal-saes"),
            },
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("format") != "sae-comp-checkpoint-v1":
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
) -> list[dict[str, float | int]]:
    device = next(sae.parameters()).device
    iterator = store.token_batches(cfg.train.token_batch_size)
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
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    code, selected_minimum, post_relu = sae.encode_batch_topk(current)
    previous_code, _, _ = sae.encode_batch_topk(previous)
    high = sae.cfg.high_size
    high_reconstruction = sae.pre_bias + sae.pre_scale * (
        code[:, :high] @ sae.decoder[:high]
    )
    full_reconstruction = sae.decode(code)
    high_loss = _fvu(high_reconstruction, current)
    full_loss = _fvu(full_reconstruction, current)
    contrastive = _symmetric_contrastive(
        code[:, :high],
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
) -> list[dict[str, float | int]]:
    device = next(sae.parameters()).device
    iterator = store.temporal_pair_batches(cfg.train.token_batch_size)
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
                sae, current, previous, cfg, dead_features
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
    reconstruction = _fvu(output["reconstruction"], windows)
    prediction = output["prediction"]
    targets = output["targets"].detach()
    cosine = F.cosine_similarity(prediction, targets, dim=-1)
    target_energy = targets.float().square().mean(dim=-1).clamp_min(1e-8)
    normalized_mse = (prediction - targets).float().square().mean(
        dim=-1
    ) / target_energy
    prediction_loss = (1 - cosine + 0.25 * normalized_mse).mean()
    residual_prediction = _fvu(output["predicted_residual"], windows[:, 1:])
    state_std = output["state"].float().std(dim=0, unbiased=False)
    variance = F.relu(cfg.proposal.variance_target - state_std).mean()
    loss = (
        reconstruction
        + prediction_weight
        * (
            prediction_loss
            + cfg.proposal.residual_prediction_weight * residual_prediction
        )
        + cfg.proposal.variance_weight * variance
    )
    return loss, {
        "loss": float(loss.detach()),
        "reconstruction_fvu": float(reconstruction.detach()),
        "prediction_loss": float(prediction_loss.detach()),
        "code_cosine": float(cosine.mean().detach()),
        "code_normalized_mse": float(normalized_mse.mean().detach()),
        "residual_prediction_fvu": float(residual_prediction.detach()),
        "variance_loss": float(variance.detach()),
    }


def _train_proposal(
    model: TransitionJEPA,
    store: ActivationStore,
    cfg: ExperimentConfig,
) -> list[dict[str, float | int | str]]:
    device = next(model.parameters()).device
    iterator = store.window_batches(
        cfg.train.window_batch_size, cfg.proposal.window_size
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
    for step in trange(1, cfg.train.branch_steps + 1, desc="transition JEPA"):
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
            model.update_target()
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
                    **metrics,
                }
            )
    return history


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
