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
    PROPOSAL_ARCHITECTURE_ID,
    RectifiedLpJEPAConfig,
    RectifiedLpJEPASAE,
    SparseAutoencoder,
    SparseAutoencoderConfig,
)
from .training import load_checkpoint


def load_method(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[
    SparseAutoencoder | RectifiedLpJEPASAE,
    RectifiedLpJEPASAE | None,
    str,
]:
    checkpoint = load_checkpoint(checkpoint_path)
    method = checkpoint["method"]
    raw = checkpoint["model_config"]
    if method == "proposal":
        architecture_id = checkpoint.get("architecture_id")
        if architecture_id != PROPOSAL_ARCHITECTURE_ID:
            raise ValueError(
                "proposal checkpoint uses an obsolete architecture; rerun "
                "`sae-comp train-window-sweep` with the current code"
            )
        proposal_cfg = RectifiedLpJEPAConfig(**raw)
        proposal = RectifiedLpJEPASAE(proposal_cfg).to(device)
        proposal.load_state_dict(checkpoint["state_dict"])
        proposal.eval()
        return proposal, proposal, method
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
    minimum_sequence_length: int | None = None,
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
            if len(x) < (minimum_sequence_length or 2):
                continue
            x = x[cfg.data.burn_in_tokens :]
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
            if method in {"temporal", "proposal"}:
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
        if split == "full" or method in {"temporal", "proposal"}
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
def evaluate_proposal_views(
    checkpoint_path: str | Path,
    cfg: ExperimentConfig,
    minimum_sequence_length: int | None = None,
) -> dict[str, Any]:
    _ = minimum_sequence_length
    device = torch.device(cfg.train.device)
    _, proposal, method = load_method(checkpoint_path, device)
    if proposal is None:
        raise ValueError("shared-view evaluation requires a proposal checkpoint")
    store = ActivationStore(Path(cfg.activation_dir) / "manifest.json", cfg.train.seed)
    max_distance = proposal.cfg.max_span_length - 1
    cosine_sum = torch.zeros(max_distance + 1, device=device)
    shuffled_sum = torch.zeros(max_distance + 1, device=device)
    low_cosine_sum = torch.zeros(max_distance + 1, device=device)
    swap_error = torch.zeros(max_distance + 1, device=device)
    shuffled_swap_error = torch.zeros(max_distance + 1, device=device)
    centered_energy = torch.zeros(max_distance + 1, device=device)
    distance_counts = torch.zeros(max_distance + 1, device=device)
    count = 0
    active_high = 0
    high_values = 0
    iterator = store.random_view_pair_batches(
        cfg.train.window_batch_size,
        max_span_length=proposal.cfg.max_span_length,
        min_span_length=cfg.proposal.min_span_length,
        boundary_max_horizon=max(cfg.proposal.window_sizes) - 1,
        split="validation",
    )
    while count < cfg.evaluation.max_sequences:
        batch = {
            key: value.to(device)
            for key, value in next(iterator).items()
        }
        output = proposal(batch["view_a"], batch["view_b"])
        active_high += int((output["high_a"] > 0).sum())
        active_high += int((output["high_b"] > 0).sum())
        high_values += output["high_a"].numel() + output["high_b"].numel()
        permutation = torch.roll(
            torch.arange(len(batch["view_a"]), device=device), 1
        )
        high_cosine = F.cosine_similarity(
            output["high_a"].float(), output["high_b"].float(), dim=-1
        )
        shuffled_cosine = F.cosine_similarity(
            output["high_a"].float(),
            output["high_b"].index_select(0, permutation).float(),
            dim=-1,
        )
        low_cosine = F.cosine_similarity(
            output["low_a"].float(), output["low_b"].float(), dim=-1
        )
        swap_a = proposal.decode_high(output["high_b"]) + proposal.decode_low(
            output["low_a"]
        )
        swap_b = proposal.decode_high(output["high_a"]) + proposal.decode_low(
            output["low_b"]
        )
        shuffled_swap_a = proposal.decode_high(
            output["high_b"].index_select(0, permutation)
        ) + proposal.decode_low(output["low_a"])
        shuffled_swap_b = proposal.decode_high(
            output["high_a"].index_select(0, permutation)
        ) + proposal.decode_low(output["low_b"])
        per_pair_energy = (
            (batch["view_a"] - proposal.pre_bias).float().square().sum(dim=-1)
            + (batch["view_b"] - proposal.pre_bias).float().square().sum(dim=-1)
        )
        per_pair_swap_error = (
            (swap_a - batch["view_a"]).float().square().sum(dim=-1)
            + (swap_b - batch["view_b"]).float().square().sum(dim=-1)
        )
        per_pair_shuffled_swap_error = (
            (shuffled_swap_a - batch["view_a"]).float().square().sum(dim=-1)
            + (shuffled_swap_b - batch["view_b"]).float().square().sum(dim=-1)
        )
        distance = batch["distance"]
        ones = torch.ones_like(distance, dtype=torch.float32)
        distance_counts.index_add_(0, distance, ones)
        cosine_sum.index_add_(
            0, distance, high_cosine
        )
        shuffled_sum.index_add_(0, distance, shuffled_cosine)
        low_cosine_sum.index_add_(0, distance, low_cosine)
        swap_error.index_add_(0, distance, per_pair_swap_error)
        shuffled_swap_error.index_add_(0, distance, per_pair_shuffled_swap_error)
        centered_energy.index_add_(0, distance, per_pair_energy)
        count += len(distance)
    denominator = distance_counts.clamp_min(1)
    cosine = cosine_sum / denominator
    shuffled = shuffled_sum / denominator
    low_cosine = low_cosine_sum / denominator
    swap_fvu = swap_error / centered_energy.clamp_min(1e-8)
    shuffled_swap_fvu = shuffled_swap_error / centered_energy.clamp_min(1e-8)
    valid = distance_counts[1:] > 0
    return {
        "method": method,
        "pairs": count,
        "distances": [
            {
                "distance": distance,
                "high_cosine": float(cosine[distance]),
                "shuffled_high_cosine": float(shuffled[distance]),
                "high_margin": float(cosine[distance] - shuffled[distance]),
                "low_cosine": float(low_cosine[distance]),
                "swap_fvu": float(swap_fvu[distance]),
                "shuffled_swap_fvu": float(shuffled_swap_fvu[distance]),
                "pairs": int(distance_counts[distance]),
            }
            for distance in range(1, max_distance + 1)
        ],
        "mean_high_cosine": float(cosine[1:][valid].mean()),
        "mean_high_margin": float((cosine[1:] - shuffled[1:])[valid].mean()),
        "swap_fvu": float(swap_error.sum() / centered_energy.sum().clamp_min(1e-8)),
        "shuffled_swap_fvu": float(
            shuffled_swap_error.sum() / centered_energy.sum().clamp_min(1e-8)
        ),
        "high_active_fraction": active_high / max(high_values, 1),
    }


def evaluate_all(cfg: ExperimentConfig) -> dict[str, Any]:
    run_dir = Path(cfg.run_dir)
    checkpoints = {
        name: run_dir / "checkpoints" / f"{name}.pt"
        for name in ("standard", "temporal", "proposal")
    }
    common = {name: evaluate_method(path, cfg) for name, path in checkpoints.items()}
    views = evaluate_proposal_views(checkpoints["proposal"], cfg)
    results = {"common": common, "proposal_views": views}
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


def _summarize_common_distance(
    views: dict[str, Any], common_distance: int
) -> dict[str, Any]:
    common_rows = views["distances"][:common_distance]
    views["common_max_distance"] = common_distance
    views["common_mean_high_cosine"] = sum(
        item["high_cosine"] for item in common_rows
    ) / len(common_rows)
    views["common_mean_high_margin"] = sum(
        item["high_margin"] for item in common_rows
    ) / len(common_rows)
    return views


def evaluate_window_sweep(cfg: ExperimentConfig) -> dict[str, Any]:
    run_dir = Path(cfg.run_dir)
    maximum_window = max(cfg.proposal.window_sizes)
    common_distance = min(cfg.proposal.window_sizes) - 1
    results: dict[str, Any] = {}
    for window_size in cfg.proposal.window_sizes:
        label = f"proposal_w{window_size:03d}"
        path = run_dir / "checkpoints" / f"{label}.pt"
        budget = cfg.proposal.sweep_budget(window_size)
        views = evaluate_proposal_views(
            path,
            cfg,
            minimum_sequence_length=maximum_window,
        )
        views = _summarize_common_distance(views, common_distance)
        results[label] = {
            "window_size": window_size,
            "training_budget": {
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
                    budget["sampled_pairs_per_step"]
                    * cfg.train.branch_steps
                ),
            },
            "common": evaluate_method(path, cfg),
            "views": views,
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
            "pair_batch_size",
            "optimizer_steps",
            "total_residual_values",
            "total_reconstructions",
            "total_sampled_pairs",
            "fve",
            "cosine_similarity",
            "fraction_alive",
            "l0",
            "lipschitz",
            "fourier",
            "wavelet",
            "multiscale",
            "common_distance_high_cosine",
            "common_distance_high_margin",
            "all_distances_high_cosine",
            "all_distances_high_margin",
            "swap_fvu",
            "shuffled_swap_fvu",
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
                    "pair_batch_size": budget["pair_batch_size"],
                    "optimizer_steps": budget["optimizer_steps"],
                    "total_residual_values": budget[
                        "total_residual_values"
                    ],
                    "total_reconstructions": budget[
                        "total_reconstructions"
                    ],
                    "total_sampled_pairs": budget[
                        "total_sampled_pairs"
                    ],
                    "fve": common["fve"],
                    "cosine_similarity": common["cosine_similarity"],
                    "fraction_alive": common["fraction_alive"],
                    "l0": common["l0"],
                    **common["smoothness"]["full"],
                    "common_distance_high_cosine": values["views"][
                        "common_mean_high_cosine"
                    ],
                    "common_distance_high_margin": values["views"][
                        "common_mean_high_margin"
                    ],
                    "all_distances_high_cosine": values["views"][
                        "mean_high_cosine"
                    ],
                    "all_distances_high_margin": values["views"][
                        "mean_high_margin"
                    ],
                    "swap_fvu": values["views"]["swap_fvu"],
                    "shuffled_swap_fvu": values["views"]["shuffled_swap_fvu"],
                }
            )
    return results


def controlled_checkpoint_paths(cfg: ExperimentConfig) -> dict[str, Path]:
    checkpoint_dir = Path(cfg.run_dir) / "checkpoints"
    paths = {
        "standard": checkpoint_dir / "standard.pt",
        "temporal": checkpoint_dir / "temporal.pt",
    }
    paths.update(
        {
            f"proposal_w{window_size:03d}": (
                checkpoint_dir / f"proposal_w{window_size:03d}.pt"
            )
            for window_size in cfg.proposal.window_sizes
        }
    )
    return paths


def evaluate_controlled_comparison(cfg: ExperimentConfig) -> dict[str, Any]:
    minimum_sequence_length = (
        cfg.data.burn_in_tokens + max(cfg.proposal.window_sizes)
    )
    common_distance = min(cfg.proposal.window_sizes) - 1
    checkpoints = controlled_checkpoint_paths(cfg)
    methods = {
        label: evaluate_method(
            path,
            cfg,
            minimum_sequence_length=minimum_sequence_length,
        )
        for label, path in checkpoints.items()
    }
    views = {}
    for window_size in cfg.proposal.window_sizes:
        label = f"proposal_w{window_size:03d}"
        view_metrics = evaluate_proposal_views(
            checkpoints[label],
            cfg,
            minimum_sequence_length=minimum_sequence_length,
        )
        views[label] = _summarize_common_distance(view_metrics, common_distance)
    results = {
        "experiment": {
            "model": cfg.model.name,
            "revision": cfg.model.revision,
            "layer": cfg.model.layer,
            "dictionary_size": cfg.sae.dictionary_size,
            "k": cfg.sae.k,
            "seed": cfg.train.seed,
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
        },
        "methods": methods,
        "proposal_views": views,
    }
    output_dir = Path(cfg.run_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "controlled_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "controlled_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "method",
            "fve",
            "cosine_similarity",
            "fraction_alive",
            "l0",
            "lipschitz",
            "fourier",
            "wavelet",
            "multiscale",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for label, values in methods.items():
            writer.writerow(
                {
                    "method": label,
                    "fve": values["fve"],
                    "cosine_similarity": values["cosine_similarity"],
                    "fraction_alive": values["fraction_alive"],
                    "l0": values["l0"],
                    **values["smoothness"]["full"],
                }
            )
    return results
