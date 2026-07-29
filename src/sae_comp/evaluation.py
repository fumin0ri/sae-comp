from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .activations import ActivationStore
from .config import ExperimentConfig
from .models import (
    SparseAutoencoder,
    SparseAutoencoderConfig,
    TransitionJEPA,
    TransitionJEPAConfig,
)
from .training import load_checkpoint


def load_method(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[SparseAutoencoder, TransitionJEPA | None, str]:
    checkpoint = load_checkpoint(checkpoint_path)
    method = checkpoint["method"]
    raw = checkpoint["model_config"]
    if method == "proposal":
        proposal_cfg = TransitionJEPAConfig(**raw)
        sae_cfg = SparseAutoencoderConfig(
            d_in=proposal_cfg.d_in,
            d_sae=proposal_cfg.d_sae,
            k=proposal_cfg.k,
            high_fraction=proposal_cfg.high_fraction,
        )
        initialized = SparseAutoencoder(sae_cfg)
        proposal = TransitionJEPA(proposal_cfg, initialized).to(device)
        proposal.load_state_dict(checkpoint["state_dict"])
        proposal.eval()
        return proposal.sae, proposal, method
    sae_cfg = SparseAutoencoderConfig(**raw)
    sae = SparseAutoencoder(sae_cfg).to(device)
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    return sae, None, method


def _active(values: torch.Tensor) -> torch.Tensor:
    return values.sum(dim=0) != 0


def lipschitz_smoothness(x: torch.Tensor, features: torch.Tensor) -> float:
    active = _active(features)
    if not bool(active.any()) or len(features) < 2:
        return 0.0
    values = features[:, active].float()
    numerator = (values[1:] - values[:-1]).abs()
    denominator = (x[1:].float() - x[:-1].float()).norm(dim=-1).clamp_min(1e-8)[:, None]
    return float((numerator / denominator).max(dim=0).values.mean())


def fourier_smoothness(features: torch.Tensor) -> float:
    active = _active(features)
    if not bool(active.any()) or len(features) < 2:
        return 0.0
    values = features[:, active].float().cpu()
    power = torch.fft.rfft(values, dim=0).abs().square()
    cutoff = max(1, int(0.5 * len(power)))
    low = power[:cutoff].sum(dim=0)
    high = power[cutoff:].sum(dim=0)
    return float((high / low.clamp_min(1e-10)).mean())


def wavelet_smoothness(features: torch.Tensor, levels: int = 3) -> float:
    active = _active(features)
    if not bool(active.any()) or len(features) < 2:
        return 0.0
    signal = features[:, active].float().cpu()
    detail_energy = 0.0
    for _ in range(levels):
        if len(signal) < 2:
            break
        if len(signal) % 2:
            signal = signal[:-1]
        even, odd = signal[::2], signal[1::2]
        detail_energy += float(((even - odd) / 2).square().sum())
        signal = (even + odd) / 2
    approximation = float(signal.square().sum())
    return detail_energy / max(approximation, 1e-10)


def multiscale_smoothness(
    features: torch.Tensor, scales: tuple[int, ...] = (1, 2, 4, 8)
) -> float:
    active = _active(features)
    if not bool(active.any()) or len(features) < 2:
        return 0.0
    values = features[:, active].float().cpu()
    valid_scales = [scale for scale in scales if scale < len(values)]
    if not valid_scales:
        return 0.0
    variations = {
        scale: float(
            (values[scale:] - values[:-scale]).var(dim=0, unbiased=False).mean()
        )
        for scale in valid_scales
    }
    return variations[min(valid_scales)] / max(variations[max(valid_scales)], 1e-10)


def sequence_metrics(x: torch.Tensor, features: torch.Tensor) -> dict[str, float]:
    return {
        "lipschitz": lipschitz_smoothness(x, features),
        "fourier": fourier_smoothness(features),
        "wavelet": wavelet_smoothness(features),
        "multiscale": multiscale_smoothness(features),
    }


@torch.inference_mode()
def evaluate_method(
    checkpoint_path: str | Path,
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    device = torch.device(cfg.train.device)
    sae, _, method = load_method(checkpoint_path, device)
    store = ActivationStore(Path(cfg.activation_dir) / "manifest.json", cfg.train.seed)
    total_squared_error = 0.0
    total_centered_energy = 0.0
    cosine_sum = 0.0
    l0_sum = 0.0
    token_count = 0
    alive = torch.zeros(sae.cfg.d_sae, dtype=torch.bool)
    smooth_sums = {
        "full": {
            name: 0.0 for name in ("lipschitz", "fourier", "wavelet", "multiscale")
        },
        "high": {
            name: 0.0 for name in ("lipschitz", "fourier", "wavelet", "multiscale")
        },
        "low": {
            name: 0.0 for name in ("lipschitz", "fourier", "wavelet", "multiscale")
        },
    }
    sequences = 0
    high = sae.cfg.high_size
    mean = torch.tensor(store.manifest["normalization"]["mean"], device=device)
    progress = tqdm(
        total=cfg.evaluation.max_sequences,
        desc=f"evaluate {method}",
    )
    for shard in store.validation_shards():
        for row in range(len(shard["activations"])):
            if sequences >= cfg.evaluation.max_sequences:
                break
            mask = shard["attention_mask"][row]
            x = shard["activations"][row][mask].to(device)
            if len(x) < 2:
                continue
            features = sae.encode(x, method)
            reconstruction = sae.decode(features)
            total_squared_error += float(
                (reconstruction.float() - x.float()).square().sum()
            )
            total_centered_energy += float((x.float() - mean.float()).square().sum())
            cosine_sum += float(
                F.cosine_similarity(reconstruction.float(), x.float(), dim=-1).sum()
            )
            l0_sum += float((features > 0).sum())
            token_count += len(x)
            alive |= (features > 0).any(dim=0).cpu()
            values = sequence_metrics(x, features)
            for name, value in values.items():
                smooth_sums["full"][name] += value
            if method == "temporal":
                for split, selected in (
                    ("high", features[:, :high]),
                    ("low", features[:, high:]),
                ):
                    values = sequence_metrics(x, selected)
                    for name, value in values.items():
                        smooth_sums[split][name] += value
            sequences += 1
            progress.update(1)
        if sequences >= cfg.evaluation.max_sequences:
            break
    progress.close()
    smoothness = {
        split: {name: value / max(sequences, 1) for name, value in metrics.items()}
        for split, metrics in smooth_sums.items()
        if split == "full" or method == "temporal"
    }
    return {
        "method": method,
        "sequences": sequences,
        "tokens": token_count,
        "fve": 1 - total_squared_error / max(total_centered_energy, 1e-10),
        "cosine_similarity": cosine_sum / max(token_count, 1),
        "fraction_alive": float(alive.float().mean()),
        "l0": l0_sum / max(token_count, 1),
        "smoothness": smoothness,
    }


@torch.inference_mode()
def evaluate_proposal_forecast(
    checkpoint_path: str | Path,
    cfg: ExperimentConfig,
    minimum_sequence_length: int | None = None,
) -> dict[str, Any]:
    device = torch.device(cfg.train.device)
    _, proposal, method = load_method(checkpoint_path, device)
    if proposal is None:
        raise ValueError("proposal forecast requires a proposal checkpoint")
    store = ActivationStore(Path(cfg.activation_dir) / "manifest.json", cfg.train.seed)
    offsets = proposal.cfg.window_size - 1
    cosine_sum = torch.zeros(offsets, device=device)
    shuffled_sum = torch.zeros(offsets, device=device)
    normalized_mse_sum = torch.zeros(offsets, device=device)
    count = 0
    for shard in store.validation_shards():
        activations = shard["activations"]
        lengths = shard["attention_mask"].sum(dim=1).tolist()
        required_length = minimum_sequence_length or proposal.cfg.window_size
        windows = [
            activations[row, start : start + proposal.cfg.window_size]
            for row, length in enumerate(lengths)
            if int(length) >= required_length
            for start in range(
                0,
                int(length) - required_length + 1,
                required_length,
            )
        ]
        for start in range(0, len(windows), cfg.train.window_batch_size):
            batch = torch.stack(
                windows[start : start + cfg.train.window_batch_size]
            ).to(device)
            output = proposal(batch)
            targets = output["targets"]
            prediction = output["prediction"]
            context = output["codes"][:, 0]
            permutation = torch.roll(
                torch.arange(len(context), device=device), shifts=1
            )
            offset_ids = torch.arange(1, proposal.cfg.window_size, device=device)
            shuffled, _ = proposal.predictor(context[permutation], offset_ids)
            cosine_sum += F.cosine_similarity(prediction, targets, dim=-1).sum(dim=0)
            shuffled_sum += F.cosine_similarity(shuffled, targets, dim=-1).sum(dim=0)
            energy = targets.float().square().mean(dim=-1).clamp_min(1e-8)
            normalized_mse_sum += (
                (prediction - targets).float().square().mean(dim=-1) / energy
            ).sum(dim=0)
            count += len(batch)
            if count >= cfg.evaluation.max_sequences:
                break
        if count >= cfg.evaluation.max_sequences:
            break
    cosine = cosine_sum / max(count, 1)
    shuffled = shuffled_sum / max(count, 1)
    normalized_mse = normalized_mse_sum / max(count, 1)
    return {
        "method": method,
        "windows": count,
        "offsets": [
            {
                "offset": index + 1,
                "code_cosine": float(cosine[index]),
                "shuffled_code_cosine": float(shuffled[index]),
                "true_minus_shuffled": float(cosine[index] - shuffled[index]),
                "normalized_mse": float(normalized_mse[index]),
            }
            for index in range(offsets)
        ],
        "mean_code_cosine": float(cosine.mean()),
        "mean_true_minus_shuffled": float((cosine - shuffled).mean()),
    }


def evaluate_all(cfg: ExperimentConfig) -> dict[str, Any]:
    run_dir = Path(cfg.run_dir)
    checkpoints = {
        name: run_dir / "checkpoints" / f"{name}.pt"
        for name in ("standard", "temporal", "proposal")
    }
    common = {name: evaluate_method(path, cfg) for name, path in checkpoints.items()}
    forecast = evaluate_proposal_forecast(checkpoints["proposal"], cfg)
    results = {"common": common, "proposal_forecast": forecast}
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "fve",
                "cosine_similarity",
                "fraction_alive",
                "l0",
                "lipschitz",
                "fourier",
                "wavelet",
                "multiscale",
            ],
        )
        writer.writeheader()
        for method, values in common.items():
            writer.writerow(
                {
                    "method": method,
                    "fve": values["fve"],
                    "cosine_similarity": values["cosine_similarity"],
                    "fraction_alive": values["fraction_alive"],
                    "l0": values["l0"],
                    **values["smoothness"]["full"],
                }
            )
    return results


def evaluate_window_sweep(cfg: ExperimentConfig) -> dict[str, Any]:
    run_dir = Path(cfg.run_dir)
    maximum_window = max(cfg.proposal.window_sizes)
    common_horizon = min(cfg.proposal.window_sizes) - 1
    results: dict[str, Any] = {}
    for window_size in cfg.proposal.window_sizes:
        label = f"proposal_w{window_size:03d}"
        path = run_dir / "checkpoints" / f"{label}.pt"
        budget = cfg.proposal.sweep_budget(window_size)
        forecast = evaluate_proposal_forecast(
            path,
            cfg,
            minimum_sequence_length=maximum_window,
        )
        common_offsets = forecast["offsets"][:common_horizon]
        forecast["common_horizon_max_offset"] = common_horizon
        forecast["common_horizon_mean_code_cosine"] = sum(
            item["code_cosine"] for item in common_offsets
        ) / len(common_offsets)
        forecast["common_horizon_mean_true_minus_shuffled"] = sum(
            item["true_minus_shuffled"] for item in common_offsets
        ) / len(common_offsets)
        results[label] = {
            "window_size": window_size,
            "training_budget": {
                **budget,
                "optimizer_steps": cfg.train.branch_steps,
                "total_reconstruction_tokens": (
                    budget["reconstruction_tokens_per_step"]
                    * cfg.train.branch_steps
                ),
                "total_forecast_pairs": (
                    budget["forecast_pairs_per_step"] * cfg.train.branch_steps
                ),
            },
            "common": evaluate_method(path, cfg),
            "forecast": forecast,
        }

    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "window_sweep.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "window_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "method",
            "window_size",
            "batch_windows",
            "optimizer_steps",
            "total_reconstruction_tokens",
            "total_forecast_pairs",
            "fve",
            "cosine_similarity",
            "fraction_alive",
            "l0",
            "lipschitz",
            "fourier",
            "wavelet",
            "multiscale",
            "common_horizon_forecast_code_cosine",
            "common_horizon_forecast_true_minus_shuffled",
            "all_offsets_forecast_code_cosine",
            "all_offsets_forecast_true_minus_shuffled",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for label, values in results.items():
            common = values["common"]
            budget = values["training_budget"]
            writer.writerow(
                {
                    "method": label,
                    "window_size": values["window_size"],
                    "batch_windows": budget["batch_windows"],
                    "optimizer_steps": budget["optimizer_steps"],
                    "total_reconstruction_tokens": budget[
                        "total_reconstruction_tokens"
                    ],
                    "total_forecast_pairs": budget["total_forecast_pairs"],
                    "fve": common["fve"],
                    "cosine_similarity": common["cosine_similarity"],
                    "fraction_alive": common["fraction_alive"],
                    "l0": common["l0"],
                    **common["smoothness"]["full"],
                    "common_horizon_forecast_code_cosine": values["forecast"][
                        "common_horizon_mean_code_cosine"
                    ],
                    "common_horizon_forecast_true_minus_shuffled": values[
                        "forecast"
                    ]["common_horizon_mean_true_minus_shuffled"],
                    "all_offsets_forecast_code_cosine": values["forecast"][
                        "mean_code_cosine"
                    ],
                    "all_offsets_forecast_true_minus_shuffled": values[
                        "forecast"
                    ]["mean_true_minus_shuffled"],
                }
            )
    return results
