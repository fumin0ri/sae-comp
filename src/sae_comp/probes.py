from __future__ import annotations

import csv
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from .config import ExperimentConfig
from .evaluation import load_method


def _load_spacy_model() -> Any:
    try:
        spacy = importlib.import_module("spacy")
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        if missing == "spacy":
            raise RuntimeError(
                "MMLU syntax probes require `pip install -e '.[probe]'`"
            ) from exc
        raise RuntimeError(
            "spaCy could not start because runtime dependency "
            f"`{missing}` is missing. Repair the probe environment with "
            "`python -m pip install --upgrade -e '.[probe]'`."
        ) from exc
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
    except OSError as exc:
        raise RuntimeError(
            "spaCy model en_core_web_sm is missing; repair the probe "
            "environment with "
            "`python -m pip install --upgrade -e '.[probe]'`."
        ) from exc


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _prompt(example: dict[str, Any]) -> str:
    choices = example.get("choices", [])
    choice_text = "\n".join(
        f"{letter}. {text}" for letter, text in zip("ABCD", choices, strict=False)
    )
    return f"{example['question']}\n{choice_text}\nAnswer:"


def _pos_labels(text: str, offsets: list[tuple[int, int]], nlp: Any) -> list[str]:
    document = nlp(text)
    labels = []
    for start, end in offsets:
        if end <= start:
            labels.append("SPECIAL")
            continue
        span = document.char_span(start, end, alignment_mode="expand")
        labels.append(span[0].pos_ if span and len(span) else "X")
    return labels


@torch.inference_mode()
def extract_mmlu_probe_cache(cfg: ExperimentConfig, overwrite: bool = False) -> Path:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(cfg.run_dir) / "evaluation" / "mmlu_probe_cache.pt"
    if output.exists() and not overwrite:
        return output
    nlp = _load_spacy_model()

    device = torch.device(cfg.train.device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, revision=cfg.model.revision, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision,
        torch_dtype=_dtype(cfg.model.dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    dataset = load_dataset(
        cfg.evaluation.mmlu_dataset,
        "all",
        split="test",
        revision=cfg.evaluation.mmlu_revision,
    ).shuffle(seed=cfg.train.seed)
    allowed = set(cfg.evaluation.probe_subjects)
    selected = [row for row in dataset if row["subject"] in allowed]
    per_subject = max(1, cfg.evaluation.probe_questions // max(len(allowed), 1))
    counts: Counter[str] = Counter()
    balanced = []
    for row in selected:
        subject = row["subject"]
        if counts[subject] >= per_subject:
            continue
        counts[subject] += 1
        balanced.append(row)
        if len(balanced) >= cfg.evaluation.probe_questions:
            break
    if len(balanced) < len(allowed) * 2:
        raise RuntimeError(
            "too few MMLU questions after subject filtering: " f"{len(balanced)}"
        )

    activations: list[torch.Tensor] = []
    semantic: list[str] = []
    context: list[int] = []
    syntax: list[str] = []
    question_groups: list[int] = []
    for question_id, example in enumerate(
        tqdm(balanced, desc="MMLU probe activations")
    ):
        text = _prompt(example)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
        labels = _pos_labels(text, offsets, nlp)
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        output_values = model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = output_values.hidden_states[cfg.model.layer + 1][0]
        valid_indices = [
            index for index, (start, end) in enumerate(offsets) if end > start
        ][-cfg.evaluation.probe_tokens_per_question :]
        if not valid_indices:
            continue
        activations.append(hidden[valid_indices].to("cpu", torch.float32))
        semantic.extend([example["subject"]] * len(valid_indices))
        context.extend([question_id] * len(valid_indices))
        syntax.extend([labels[index] for index in valid_indices])
        question_groups.extend([question_id] * len(valid_indices))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "sae-comp-mmlu-probes-v1",
            "activations": torch.cat(activations),
            "semantic": semantic,
            "context": context,
            "syntax": syntax,
            "question_groups": question_groups,
            "subjects": sorted(counts),
            "questions_per_subject": dict(counts),
            "model": cfg.model.name,
            "revision": cfg.model.revision,
            "layer": cfg.model.layer,
        },
        output,
    )
    return output


def _encode_labels(values: list[Any]) -> tuple[np.ndarray, list[str]]:
    classes = sorted({str(value) for value in values})
    mapping = {value: index for index, value in enumerate(classes)}
    return np.array([mapping[str(value)] for value in values]), classes


def _stratified_token_split(
    labels: np.ndarray, train_rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    fraction = min(0.8, train_rows / max(len(labels), 1))
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
        train.extend(indices[:count])
        test.extend(indices[count:])
    rng.shuffle(train)
    rng.shuffle(test)
    if len(train) > train_rows:
        overflow = train[train_rows:]
        train = train[:train_rows]
        test.extend(overflow)
    return np.asarray(train), np.asarray(test)


def _feature_rankings(
    x: np.ndarray, labels: np.ndarray, maximum_per_class: int
) -> dict[int, np.ndarray]:
    """Rank mean-difference features once and reuse them at every sparsity."""

    rankings: dict[int, np.ndarray] = {}
    total = x.sum(axis=0, dtype=np.float64)
    rows = len(x)
    for label in np.unique(labels):
        mask = labels == label
        positive_rows = int(mask.sum())
        negative_rows = rows - positive_rows
        positive_sum = x[mask].sum(axis=0, dtype=np.float64)
        positive = positive_sum / max(positive_rows, 1)
        negative = (total - positive_sum) / max(negative_rows, 1)
        scores = positive - negative
        count = min(maximum_per_class, x.shape[1])
        indices = np.argpartition(scores, -count)[-count:]
        rankings[int(label)] = indices[np.argsort(scores[indices])[::-1]].astype(
            np.int64
        )
    return rankings


def _select_ranked_features(
    rankings: dict[int, np.ndarray], per_class: int
) -> np.ndarray:
    return np.unique(
        np.concatenate([indices[:per_class] for indices in rankings.values()])
    )


def _fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    selected: np.ndarray | None,
    max_iter: int,
    tolerance: float,
) -> tuple[float, int, int]:
    train_x = features[train_indices]
    test_x = features[test_indices]
    train_y = labels[train_indices]
    test_y = labels[test_indices]
    if selected is not None:
        train_x = train_x[:, selected]
        test_x = test_x[:, selected]
    else:
        selected = np.arange(features.shape[1])
    train_x = csr_matrix(train_x, dtype=np.float32)
    test_x = csr_matrix(test_x, dtype=np.float32)
    class_count = len(np.unique(train_y))
    validation_fraction = max(0.1, min(0.25, (class_count + 1) / max(len(train_y), 1)))
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=max_iter,
        tol=tolerance,
        early_stopping=True,
        validation_fraction=validation_fraction,
        n_iter_no_change=5,
        average=True,
        class_weight="balanced",
        random_state=0,
    )
    classifier.fit(train_x, train_y)
    return (
        float(classifier.score(test_x, test_y)),
        len(selected),
        int(classifier.n_iter_),
    )


@torch.inference_mode()
def evaluate_probes(cfg: ExperimentConfig) -> list[dict[str, Any]]:
    cache_path = extract_mmlu_probe_cache(cfg)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    activations = cache["activations"]
    label_sets = {
        "semantics": cache["semantic"],
        "context": cache["context"],
        "syntax": cache["syntax"],
    }
    device = torch.device(cfg.train.device)
    run_dir = Path(cfg.run_dir)
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "probes_progress.json"
    progress_data: dict[str, Any] = {}
    if progress_path.exists():
        progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress_data.get("config_fingerprint") == cfg.fingerprint():
        results: list[dict[str, Any]] = progress_data.get("results", [])
    else:
        results = []
    completed = {
        (
            result["method"],
            result["split"],
            result["task"],
            str(result["sparsity"]),
        )
        for result in results
    }
    sparsities: tuple[int | None, ...] = tuple(cfg.evaluation.probe_sparsities)
    if cfg.evaluation.probe_include_dense:
        sparsities = (*sparsities, None)
    jobs_per_split = len(label_sets) * len(sparsities)
    total_jobs = jobs_per_split * 5
    progress = tqdm(
        total=total_jobs,
        initial=len(completed),
        desc="linear probes",
        unit="probe",
    )
    for method in ("standard", "temporal", "proposal"):
        sae, _, _ = load_method(run_dir / "checkpoints" / f"{method}.pt", device)
        encoded_batches = []
        for start in range(0, len(activations), cfg.train.token_batch_size):
            batch = activations[start : start + cfg.train.token_batch_size].to(device)
            encoded_batches.append(sae.encode(batch, method).cpu())
        all_features = torch.cat(encoded_batches).float().numpy()
        high = sae.cfg.high_size
        splits = {"full": all_features}
        if method == "temporal":
            splits.update(
                {
                    "high": all_features[:, :high],
                    "low": all_features[:, high:],
                }
            )
        for task, raw_labels in label_sets.items():
            labels, classes = _encode_labels(raw_labels)
            if task == "syntax":
                counts = Counter(labels.tolist())
                keep = np.array([counts[int(label)] >= 10 for label in labels])
            else:
                keep = np.ones(len(labels), dtype=bool)
            task_labels = labels[keep]
            train_indices, test_indices = _stratified_token_split(
                task_labels,
                cfg.evaluation.probe_train_rows,
                cfg.train.seed,
            )
            for split, features in splits.items():
                task_features = features[keep]
                maximum = max(cfg.evaluation.probe_sparsities, default=1)
                rankings = _feature_rankings(
                    task_features[train_indices], task_labels[train_indices], maximum
                )
                for sparsity in sparsities:
                    display_sparsity = "dense" if sparsity is None else sparsity
                    key = (method, split, task, str(display_sparsity))
                    if key in completed:
                        continue
                    selected_indices = (
                        None
                        if sparsity is None
                        else _select_ranked_features(rankings, sparsity)
                    )
                    progress.set_postfix_str(
                        f"{method}/{split}/{task}/k={display_sparsity}"
                    )
                    accuracy, selected, iterations = _fit_probe(
                        task_features,
                        task_labels,
                        train_indices,
                        test_indices,
                        selected_indices,
                        cfg.evaluation.probe_max_iter,
                        cfg.evaluation.probe_tolerance,
                    )
                    result = {
                        "method": method,
                        "split": split,
                        "task": task,
                        "sparsity": display_sparsity,
                        "selected_features": selected,
                        "accuracy": accuracy,
                        "classes": len(classes),
                        "train_rows": len(train_indices),
                        "test_rows": len(test_indices),
                        "solver": "SGDClassifier(log_loss)",
                        "iterations": iterations,
                    }
                    results.append(result)
                    completed.add(key)
                    progress_path.write_text(
                        json.dumps(
                            {
                                "config_fingerprint": cfg.fingerprint(),
                                "results": results,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    progress.update(1)
    progress.close()
    results.sort(
        key=lambda item: (
            item["method"],
            item["split"],
            item["task"],
            str(item["sparsity"]),
        )
    )
    (output_dir / "probes.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "probes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    progress_path.unlink(missing_ok=True)
    return results
