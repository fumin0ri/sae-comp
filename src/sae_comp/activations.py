from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import ExperimentConfig

FORMAT = "sae-comp-exchangeable-view-sequences-v3"


def activation_config_fingerprint(cfg: ExperimentConfig) -> str:
    identity = {
        "format": FORMAT,
        "model": cfg.as_dict()["model"],
        "data": cfg.as_dict()["data"],
        "min_span_length": cfg.proposal.min_span_length,
        "max_span_length": max(cfg.proposal.window_sizes),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _split_for_document(text: str, validation_fraction: float) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if bucket < validation_fraction else "train"


def _model_dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"unsupported model dtype: {name}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@torch.inference_mode()
def extract_activations(cfg: ExperimentConfig, overwrite: bool = False) -> Path:
    """Extract one shared residual cache used by every method."""

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(cfg.activation_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("format") == FORMAT
            and existing.get("config_fingerprint")
            == activation_config_fingerprint(cfg)
        ):
            return manifest_path
        raise ValueError(
            f"{manifest_path} is an incompatible activation cache; use the new "
            "activation_dir or rerun extract with --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = _model_dtype(cfg.model.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, revision=cfg.model.revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    dataset_args: list[str] = [cfg.data.dataset]
    if cfg.data.dataset_config and cfg.data.dataset_config != "default":
        dataset_args.append(cfg.data.dataset_config)
    dataset = load_dataset(
        *dataset_args,
        split=cfg.data.split,
        streaming=True,
        revision=cfg.data.revision or None,
    )
    if cfg.data.shuffle_buffer > 1:
        dataset = dataset.shuffle(
            seed=cfg.train.seed, buffer_size=cfg.data.shuffle_buffer
        )

    targets = {
        "train": cfg.data.train_sequences,
        "validation": cfg.data.validation_sequences,
    }
    queued = {"train": 0, "validation": 0}
    pending: dict[str, list[list[int]]] = {"train": [], "validation": []}
    shard_rows: dict[str, list[dict[str, torch.Tensor]]] = {
        "train": [],
        "validation": [],
    }
    shard_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
    }
    sums: torch.Tensor | None = None
    square_sum = torch.zeros((), dtype=torch.float64)
    normalization_tokens = 0

    def save_shard(split: str) -> None:
        nonlocal shard_rows
        rows = shard_rows[split]
        if not rows:
            return
        index = len(shard_records[split])
        relative = Path(split) / f"shard-{index:05d}.pt"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        activations = torch.stack([row["activations"] for row in rows])
        input_ids = torch.stack([row["input_ids"] for row in rows])
        attention_mask = torch.stack([row["attention_mask"] for row in rows])
        torch.save(
            {
                "activations": activations,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "valid_lengths": attention_mask.sum(dim=1).to(torch.int32),
            },
            destination,
        )
        shard_records[split].append(
            {
                "path": relative.as_posix(),
                "sequences": len(rows),
                "valid_tokens": int(attention_mask.sum().item()),
            }
        )
        shard_rows[split] = []

    def flush_pending(split: str) -> None:
        nonlocal sums, square_sum, normalization_tokens
        rows = pending[split]
        if not rows:
            return
        length = cfg.data.sequence_length
        input_ids = torch.full(
            (len(rows), length),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros_like(input_ids)
        for index, tokens in enumerate(rows):
            count = min(length, len(tokens))
            input_ids[index, :count] = torch.tensor(tokens[:count], device=device)
            mask[index, :count] = 1
        output_values = model(
            input_ids=input_ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = output_values.hidden_states[cfg.model.layer + 1]
        hidden_cpu = hidden.to(dtype=torch.bfloat16, device="cpu")
        ids_cpu = input_ids.to(dtype=torch.int32, device="cpu")
        mask_cpu = mask.to(dtype=torch.bool, device="cpu")
        for index in range(len(rows)):
            shard_rows[split].append(
                {
                    "activations": hidden_cpu[index],
                    "input_ids": ids_cpu[index],
                    "attention_mask": mask_cpu[index],
                }
            )
        if split == "train":
            valid = hidden.float()[mask.bool()].to("cpu", torch.float64)
            batch_sum = valid.sum(dim=0)
            sums = batch_sum if sums is None else sums + batch_sum
            square_sum += valid.square().sum()
            normalization_tokens += len(valid)
        pending[split] = []
        if len(shard_rows[split]) >= cfg.data.shard_sequences:
            save_shard(split)

    progress = tqdm(total=sum(targets.values()), desc="activation sequences")
    random.seed(cfg.train.seed)
    for example in dataset:
        if all(queued[key] >= targets[key] for key in targets):
            break
        text = str(example.get(cfg.data.text_field, ""))
        if not text.strip():
            continue
        split = _split_for_document(text, cfg.data.validation_fraction)
        if queued[split] >= targets[split]:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, len(tokens), cfg.data.sequence_length):
            if queued[split] >= targets[split]:
                break
            chunk = tokens[start : start + cfg.data.sequence_length]
            if len(chunk) < cfg.data.min_valid_tokens:
                continue
            pending[split].append(chunk)
            queued[split] += 1
            progress.update(1)
            if len(pending[split]) >= cfg.data.extraction_batch_size:
                flush_pending(split)
    progress.close()
    for split in ("train", "validation"):
        flush_pending(split)
        save_shard(split)
        if queued[split] != targets[split]:
            raise RuntimeError(
                f"dataset ended with {queued[split]} {split} sequences; "
                f"expected {targets[split]}"
            )
    assert sums is not None and normalization_tokens > 0
    mean = sums / normalization_tokens
    mean_square = square_sum / normalization_tokens
    d_in = int(mean.numel())
    scalar_rms = torch.sqrt(
        (mean_square - mean.square().sum()).clamp_min(1e-12) / d_in
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    manifest = {
        "format": FORMAT,
        "config_fingerprint": activation_config_fingerprint(cfg),
        "dataset": {
            "name": cfg.data.dataset,
            "config": cfg.data.dataset_config,
            "revision": cfg.data.revision,
            "split": cfg.data.split,
            "document_split": "sha256",
            "validation_fraction": cfg.data.validation_fraction,
        },
        "model": {
            "name": cfg.model.name,
            "requested_revision": cfg.model.revision,
            "resolved_revision": resolved_revision,
            "layer": cfg.model.layer,
            "hook": "hidden_states[layer + 1]",
        },
        "sequence_length": cfg.data.sequence_length,
        "min_span_length": cfg.proposal.min_span_length,
        "max_span_length": max(cfg.proposal.window_sizes),
        "max_horizon": max(cfg.proposal.window_sizes) - 1,
        "burn_in_tokens": cfg.data.burn_in_tokens,
        "minimum_valid_length": cfg.data.min_valid_tokens,
        "pair_sampling": {
            "span_length": "uniform integer in [min_span_length, max_span_length]",
            "views": (
                "two distinct ordered positions sampled uniformly without "
                "replacement from the span"
            ),
            "view_marginals": "exchangeable; neither view is a prediction target",
            "span_end_support": "shared across span lengths and sweep conditions",
            "performed_online_during_training": True,
        },
        "d_in": d_in,
        "normalization": {
            "mean": mean.tolist(),
            "scalar_rms": float(scalar_rms.item()),
            "tokens": normalization_tokens,
        },
        "train": {
            "sequences": queued["train"],
            "valid_tokens": sum(
                item["valid_tokens"] for item in shard_records["train"]
            ),
            "shards": shard_records["train"],
        },
        "validation": {
            "sequences": queued["validation"],
            "valid_tokens": sum(
                item["valid_tokens"] for item in shard_records["validation"]
            ),
            "shards": shard_records["validation"],
        },
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(
            f"unsupported activation format: {manifest_path}; re-extract long "
            "residual sequences with the current extract stage"
        )
    max_span = int(manifest["max_span_length"])
    min_span = int(manifest["min_span_length"])
    burn_in = int(manifest["burn_in_tokens"])
    if not 2 <= min_span <= max_span:
        raise ValueError("span bounds must satisfy 2 <= min <= max")
    if int(manifest["max_horizon"]) != max_span - 1:
        raise ValueError("max_horizon must equal max_span_length - 1")
    if int(manifest["minimum_valid_length"]) < burn_in + max_span:
        raise ValueError("minimum_valid_length exposes a sequence boundary")
    return manifest_path.parent, manifest


def load_shard(root: Path, record: dict[str, Any]) -> dict[str, torch.Tensor]:
    return torch.load(root / record["path"], map_location="cpu", weights_only=True)


class ActivationStore:
    def __init__(self, manifest_path: str | Path, seed: int):
        self.root, self.manifest = load_manifest(manifest_path)
        self.seed = seed

    def _records(self, split: str, epoch: int) -> list[dict[str, Any]]:
        records = list(self.manifest[split]["shards"])
        random.Random(self.seed + epoch).shuffle(records)
        return records

    def token_batches(
        self,
        batch_size: int,
        split: str = "train",
        minimum_sequence_length: int | None = None,
    ) -> Iterator[torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed)
        epoch = 0
        while True:
            for record in self._records(split, epoch):
                shard = load_shard(self.root, record)
                source_mask = shard["attention_mask"]
                mask = source_mask.clone()
                burn_in = int(self.manifest["burn_in_tokens"])
                mask[:, :burn_in] = False
                if minimum_sequence_length is not None:
                    eligible = source_mask.sum(dim=1) >= minimum_sequence_length
                    mask = mask & eligible[:, None]
                values = shard["activations"][mask]
                order = torch.randperm(len(values), generator=generator)
                for start in range(0, len(order) - batch_size + 1, batch_size):
                    yield values.index_select(0, order[start : start + batch_size])
            epoch += 1

    def temporal_pair_batches(
        self,
        batch_size: int,
        mode: str = "previous",
        maximum_lookback: int = 24,
        minimum_sequence_length: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        generator = torch.Generator().manual_seed(self.seed + 1)
        epoch = 0
        while True:
            for record in self._records("train", epoch):
                shard = load_shard(self.root, record)
                activations = shard["activations"]
                mask = shard["attention_mask"]
                pair_mask = mask.clone()
                if minimum_sequence_length is not None:
                    eligible = mask.sum(dim=1) >= minimum_sequence_length
                    pair_mask &= eligible[:, None]
                burn_in = int(self.manifest["burn_in_tokens"])
                pair_mask[:, : burn_in + 1] = False
                rows, times = pair_mask.nonzero(as_tuple=True)
                order = torch.randperm(len(rows), generator=generator)
                for start in range(0, len(order) - batch_size + 1, batch_size):
                    selected = order[start : start + batch_size]
                    batch_rows = rows[selected]
                    current_times = times[selected]
                    if mode == "previous":
                        previous_times = current_times - 1
                    elif mode == "random":
                        lookback = torch.randint(
                            1,
                            maximum_lookback + 1,
                            (batch_size,),
                            generator=generator,
                        )
                        previous_times = torch.maximum(
                            torch.zeros_like(current_times),
                            current_times - lookback,
                        )
                    else:
                        raise ValueError(f"unknown temporal pair mode: {mode}")
                    yield (
                        activations[batch_rows, current_times],
                        activations[batch_rows, previous_times],
                    )
            epoch += 1

    def random_view_pair_batches(
        self,
        batch_size: int,
        max_span_length: int,
        min_span_length: int = 2,
        boundary_max_horizon: int | None = None,
        split: str = "train",
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield deterministic exchangeable view pairs from random spans."""
        stored_max_span = int(self.manifest["max_span_length"])
        if not 2 <= min_span_length <= max_span_length <= stored_max_span:
            raise ValueError(
                "span bounds must satisfy 2 <= min <= max <= stored maximum"
            )
        boundary_horizon = (
            stored_max_span - 1
            if boundary_max_horizon is None
            else boundary_max_horizon
        )
        if boundary_horizon < max_span_length - 1:
            raise ValueError("boundary horizon must cover the sampled horizon")
        burn_in = int(self.manifest["burn_in_tokens"])
        generator = torch.Generator().manual_seed(self.seed + 2)
        epoch = 0
        pending: dict[str, list[torch.Tensor]] = {}
        pending_count = 0
        while True:
            for record in self._records(split, epoch):
                shard = load_shard(self.root, record)
                sequences = shard["activations"]
                valid_lengths = shard.get("valid_lengths")
                if valid_lengths is None:
                    valid_lengths = shard["attention_mask"].sum(dim=1)
                eligible = valid_lengths > burn_in + boundary_horizon
                rows = eligible.nonzero(as_tuple=True)[0]
                rows = rows.index_select(
                    0, torch.randperm(len(rows), generator=generator)
                )
                if not len(rows):
                    continue
                lengths = valid_lengths.index_select(0, rows).long()
                span_lengths = torch.randint(
                    min_span_length,
                    max_span_length + 1,
                    (len(rows),),
                    generator=generator,
                )
                sampled = _sample_random_view_pairs(
                    sequences,
                    lengths,
                    rows,
                    span_lengths,
                    burn_in,
                    boundary_horizon,
                    generator,
                )
                for key, value in sampled.items():
                    pending.setdefault(key, []).append(value)
                pending_count += len(rows)
                while pending_count >= batch_size:
                    joined = {key: torch.cat(values) for key, values in pending.items()}
                    yield {key: value[:batch_size] for key, value in joined.items()}
                    pending = {
                        key: [value[batch_size:]]
                        for key, value in joined.items()
                        if len(value) > batch_size
                    }
                    pending_count -= batch_size
            epoch += 1

    def window_batches(
        self,
        batch_size: int,
        window_size: int,
        minimum_sequence_length: int | None = None,
    ) -> Iterator[torch.Tensor]:
        minimum_length = minimum_sequence_length or window_size
        if minimum_length < window_size:
            raise ValueError("minimum_sequence_length must be at least window_size")
        generator = torch.Generator().manual_seed(self.seed + 2)
        epoch = 0
        while True:
            for record in self._records("train", epoch):
                shard = load_shard(self.root, record)
                activations = shard["activations"]
                lengths = shard["attention_mask"].sum(dim=1).tolist()
                starts = [
                    (row, start)
                    for row, length in enumerate(lengths)
                    if int(length) >= minimum_length
                    for start in range(int(length) - window_size + 1)
                ]
                order = torch.randperm(len(starts), generator=generator)
                for offset in range(0, len(order) - batch_size + 1, batch_size):
                    selected = order[offset : offset + batch_size].tolist()
                    yield torch.stack(
                        [
                            activations[row, start : start + window_size]
                            for row, start in (starts[index] for index in selected)
                        ]
                    )
            epoch += 1

    def validation_shards(self) -> Iterator[dict[str, torch.Tensor]]:
        for record in self.manifest["validation"]["shards"]:
            yield load_shard(self.root, record)


def _sample_random_view_pairs(
    sequences: torch.Tensor,
    valid_lengths: torch.Tensor,
    rows: torch.Tensor,
    span_lengths: torch.Tensor,
    burn_in_tokens: int,
    boundary_max_horizon: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    minimum_endpoints = torch.full_like(
        span_lengths, burn_in_tokens + boundary_max_horizon
    )
    if torch.any(minimum_endpoints >= valid_lengths):
        raise ValueError("sequence is too short for burn-in and sampled horizon")
    endpoint_support = valid_lengths - minimum_endpoints
    endpoints = minimum_endpoints + torch.floor(
        torch.rand(len(rows), generator=generator) * endpoint_support
    ).long()
    span_starts = endpoints - span_lengths + 1
    offsets_a = torch.floor(
        torch.rand(len(rows), generator=generator) * span_lengths
    ).long()
    offsets_b = torch.floor(
        torch.rand(len(rows), generator=generator) * (span_lengths - 1)
    ).long()
    offsets_b = offsets_b + (offsets_b >= offsets_a).long()
    positions_a = span_starts + offsets_a
    positions_b = span_starts + offsets_b
    return {
        "view_a": sequences[rows, positions_a],
        "view_b": sequences[rows, positions_b],
        "span_length": span_lengths,
        "distance": (positions_a - positions_b).abs(),
        "span_start_index": span_starts,
        "span_end_index": endpoints,
        "position_a": positions_a,
        "position_b": positions_b,
    }
