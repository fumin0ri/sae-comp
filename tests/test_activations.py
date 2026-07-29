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
        "train": {"shards": [{"path": shard.name}]},
        "validation": {"shards": []},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    store = ActivationStore(manifest_path, seed=0)

    token_batch = next(store.token_batches(8, minimum_sequence_length=8))
    current, previous = next(store.temporal_pair_batches(7, minimum_sequence_length=8))

    assert bool((token_batch == 1).all())
    assert bool((current == 1).all())
    assert bool((previous == 1).all())
