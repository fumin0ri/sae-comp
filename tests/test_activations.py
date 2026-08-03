import json
from pathlib import Path

import torch

from sae_comp.activations import FORMAT, ActivationStore


def test_minimum_sequence_length_filters_training_pool(tmp_path: Path) -> None:
    shard = tmp_path / "train.pt"
    torch.save(
        {
            "activations": torch.stack([torch.zeros(8, 3), torch.ones(8, 3)]),
            "attention_mask": torch.tensor(
                [
                    [True, True, True, True, False, False, False, False],
                    [True, True, True, True, True, True, True, True],
                ]
            ),
        },
        shard,
    )
    manifest = {
        "format": FORMAT,
        "sequence_length": 8,
        "min_span_length": 2,
        "max_span_length": 4,
        "max_horizon": 3,
        "burn_in_tokens": 2,
        "minimum_valid_length": 6,
        "train": {"shards": [{"path": shard.name}]},
        "validation": {"shards": []},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    store = ActivationStore(manifest_path, seed=0)

    token_batch = next(store.token_batches(4, minimum_sequence_length=8))
    current, previous = next(store.temporal_pair_batches(4, minimum_sequence_length=8))

    assert bool((token_batch == 1).all())
    assert bool((current == 1).all())
    assert bool((previous == 1).all())


def test_random_view_pairs_are_boundary_safe_and_exchangeable(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "train.pt"
    sequences = torch.arange(4 * 12, dtype=torch.float32).reshape(4, 12, 1)
    torch.save(
        {
            "activations": sequences,
            "attention_mask": torch.ones(4, 12, dtype=torch.bool),
            "valid_lengths": torch.full((4,), 12, dtype=torch.int32),
        },
        shard,
    )
    manifest = {
        "format": FORMAT,
        "sequence_length": 12,
        "min_span_length": 2,
        "max_span_length": 8,
        "max_horizon": 7,
        "burn_in_tokens": 2,
        "minimum_valid_length": 10,
        "train": {"shards": [{"path": shard.name}]},
        "validation": {"shards": [{"path": shard.name}]},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    batch = next(
        ActivationStore(manifest_path, seed=3).random_view_pair_batches(
            4,
            max_span_length=4,
            boundary_max_horizon=7,
        )
    )
    assert bool((batch["distance"] >= 1).all())
    assert bool((batch["distance"] < batch["span_length"]).all())
    assert bool((batch["span_length"] <= 4).all())
    assert bool((batch["position_a"] >= batch["span_start_index"]).all())
    assert bool((batch["position_b"] >= batch["span_start_index"]).all())
    assert bool((batch["position_a"] <= batch["span_end_index"]).all())
    assert bool((batch["position_b"] <= batch["span_end_index"]).all())
    assert bool((batch["position_a"] >= 2).all())
    assert bool((batch["position_b"] >= 2).all())
    assert bool((batch["span_end_index"] >= 9).all())
    torch.testing.assert_close(
        (batch["position_a"] - batch["position_b"]).abs(), batch["distance"]
    )
