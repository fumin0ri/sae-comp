from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str = "EleutherAI/pythia-160m"
    revision: str = "main"
    layer: int = 8
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "EleutherAI/the_pile_deduplicated"
    dataset_config: str = "default"
    revision: str = "fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb"
    split: str = "train"
    text_field: str = "text"
    sequence_length: int = 128
    min_valid_tokens: int = 32
    train_sequences: int = 40_960
    validation_sequences: int = 1_024
    shard_sequences: int = 256
    extraction_batch_size: int = 8
    validation_fraction: float = 0.05
    shuffle_buffer: int = 10_000


@dataclass(frozen=True)
class SAEConfig:
    dictionary_size: int = 16_384
    k: int = 20
    high_fraction: float = 0.2
    temporal_alpha: float = 1.0
    contrastive_temperature: float = 1.0
    high_reconstruction_weight: float = 0.2
    full_reconstruction_weight: float = 0.8
    auxiliary_weight: float = 0.03125
    threshold_beta: float = 0.999
    dead_token_threshold: int = 10_000_000


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    standard_steps: int = 12_000
    branch_steps: int = 6_000
    token_batch_size: int = 512
    window_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    standard_lr: float = 2e-4
    temporal_lr: float = 2e-4
    proposal_sae_lr: float = 1e-4
    proposal_predictor_lr: float = 3e-4
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    log_every: int = 100
    amp_dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass(frozen=True)
class ProposalConfig:
    window_size: int = 10
    predictor_width: int = 256
    predictor_expansion: int = 2
    predictor_warmup_steps: int = 800
    prediction_ramp_steps: int = 800
    prediction_weight: float = 1.0
    residual_prediction_weight: float = 0.1
    variance_weight: float = 0.01
    variance_target: float = 1.0
    ema_decay: float = 0.996


@dataclass(frozen=True)
class EvalConfig:
    batch_sequences: int = 8
    max_sequences: int = 1_024
    probe_questions: int = 160
    probe_tokens_per_question: int = 25
    probe_train_rows: int = 3_000
    probe_sparsities: tuple[int, ...] = (1, 5, 10, 20)
    mmlu_dataset: str = "cais/mmlu"
    mmlu_revision: str = "c30699e8356da336a370243923dbaf21066bb9fe"
    probe_subjects: tuple[str, ...] = (
        "world_religions",
        "professional_law",
        "high_school_european_history",
        "high_school_macroeconomics",
        "high_school_biology",
        "high_school_mathematics",
        "professional_medicine",
        "prehistory",
        "moral_disputes",
        "business_ethics",
    )
    bootstrap_samples: int = 2_000


@dataclass(frozen=True)
class ExperimentConfig:
    run_dir: str = "runs/paper-pythia160m"
    activation_dir: str = "data/pythia160m-layer8"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> None:
        if self.model.layer < 0:
            raise ValueError("model.layer must be non-negative")
        if self.data.sequence_length < self.proposal.window_size:
            raise ValueError("sequence_length must be at least proposal.window_size")
        if not 0 < self.data.validation_fraction < 1:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if self.data.min_valid_tokens < self.proposal.window_size:
            raise ValueError("min_valid_tokens must be at least proposal.window_size")
        if self.sae.k < 1 or self.sae.k > self.sae.dictionary_size:
            raise ValueError("sae.k must lie in [1, dictionary_size]")
        if not 0 < self.sae.high_fraction < 1:
            raise ValueError("sae.high_fraction must lie in (0, 1)")
        if self.train.standard_steps < 1 or self.train.branch_steps < 1:
            raise ValueError("training step counts must be positive")
        if not 0 <= self.proposal.predictor_warmup_steps < self.train.branch_steps:
            raise ValueError("predictor_warmup_steps must be smaller than branch_steps")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evaluation"]["probe_sparsities"] = list(self.evaluation.probe_sparsities)
        value["evaluation"]["probe_subjects"] = list(self.evaluation.probe_subjects)
        return value

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _section(cls: type[Any], raw: dict[str, Any], name: str) -> Any:
    return cls(**raw.get(name, {}))


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    cfg = ExperimentConfig(
        run_dir=raw.get("run_dir", ExperimentConfig.run_dir),
        activation_dir=raw.get("activation_dir", ExperimentConfig.activation_dir),
        model=_section(ModelConfig, raw, "model"),
        data=_section(DataConfig, raw, "data"),
        sae=_section(SAEConfig, raw, "sae"),
        train=_section(TrainConfig, raw, "train"),
        proposal=_section(ProposalConfig, raw, "proposal"),
        evaluation=_section(EvalConfig, raw, "evaluation"),
    )
    cfg.validate()
    return cfg
