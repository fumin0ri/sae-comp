from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, replace
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
    min_valid_tokens: int = 64
    burn_in_tokens: int = 32
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
    temporal_pairs_per_step: int = 448
    window_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    standard_lr: float = 2e-4
    temporal_lr: float = 2e-4
    proposal_sae_lr: float = 2e-4
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    log_every: int = 100
    amp_dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass(frozen=True)
class ProposalConfig:
    window_size: int = 32
    window_sizes: tuple[int, ...] = (2, 4, 8, 16, 32)
    min_span_length: int = 2
    sweep_pairs_per_step: int = 512
    high_fraction: float = 0.2
    low_k: int = 20
    high_reconstruction_weight: float = 0.1
    rgg_p: float = 1.0
    target_active_fraction: float = 0.025
    target_sigma: float = 0.0
    invariance_weight: float = 1.0
    rdm_weight: float = 5.0
    rdm_projections: int = 1_024
    rdm_projection_chunk_size: int = 128
    axis_rdm_features: int = 512
    axis_rdm_weight: float = 1.0
    sae_warmup_steps: int = 1_000
    regularization_ramp_steps: int = 1_000

    def sweep_budget(self, window_size: int) -> dict[str, int]:
        return {
            "window_size": window_size,
            "pair_batch_size": self.sweep_pairs_per_step,
            "sampled_pairs_per_step": self.sweep_pairs_per_step,
            "residual_values_per_step": 2 * self.sweep_pairs_per_step,
            "reconstructions_per_step": 2 * self.sweep_pairs_per_step,
            "minimum_distance": 1,
            "maximum_distance": window_size - 1,
        }


@dataclass(frozen=True)
class EvalConfig:
    batch_sequences: int = 8
    max_sequences: int = 1_024
    probe_questions: int = 160
    probe_tokens_per_question: int = 25
    probe_train_rows: int = 3_000
    probe_sparsities: tuple[int, ...] = (1, 5, 10, 20)
    probe_max_iter: int = 200
    probe_tolerance: float = 1e-3
    probe_include_dense: bool = True
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
class SAEBenchConfig:
    enabled: bool = False
    version: str = "0.6.0"
    model_name: str = "pythia-160m-deduped"
    eval_types: tuple[str, ...] = (
        "core",
        "sparse_probing",
        "sparse_probing_sae_probes",
        "ravel",
    )
    excluded_eval_types: tuple[str, ...] = ("scr", "tpp")
    llm_batch_size: int = 256
    llm_dtype: str = "float32"
    force_rerun: bool = False
    core_reconstruction_batches: int = 200
    core_sparsity_variance_batches: int = 2_000
    core_prompt_batch_size: int = 16
    core_dataset: str = "Skylion007/openwebtext"
    context_size: int = 128
    ravel_entity_attribute_selection: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "city": ("Country", "Continent", "Language"),
        }
    )


@dataclass(frozen=True)
class ExperimentConfig:
    run_dir: str = "runs/paper-pythia160m-rectified-lpjepa-v4"
    activation_dir: str = "data/pythia160m-layer8-exchangeable-v3"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    sae_bench: SAEBenchConfig = field(default_factory=SAEBenchConfig)

    def validate(self) -> None:
        if self.model.layer < 0:
            raise ValueError("model.layer must be non-negative")
        if self.data.sequence_length < self.proposal.window_size:
            raise ValueError("sequence_length must be at least proposal.window_size")
        if not 0 < self.data.validation_fraction < 1:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if self.data.burn_in_tokens < 0:
            raise ValueError("burn_in_tokens must be non-negative")
        if self.data.min_valid_tokens > self.data.sequence_length:
            raise ValueError("min_valid_tokens must not exceed sequence_length")
        if not self.proposal.window_sizes:
            raise ValueError("proposal.window_sizes must not be empty")
        if len(set(self.proposal.window_sizes)) != len(self.proposal.window_sizes):
            raise ValueError("proposal.window_sizes must be unique")
        maximum_window = max(self.proposal.window_sizes)
        if not 2 <= self.proposal.min_span_length <= min(self.proposal.window_sizes):
            raise ValueError("min_span_length must lie in [2, min(window_sizes)]")
        if self.data.min_valid_tokens < self.data.burn_in_tokens + maximum_window:
            raise ValueError(
                "min_valid_tokens must cover burn_in_tokens + max(window_sizes)"
            )
        for window_size in self.proposal.window_sizes:
            if window_size < 2:
                raise ValueError("proposal window sizes must be at least 2")
            if window_size > self.data.sequence_length:
                raise ValueError(
                    "sequence_length must be at least every proposal window size"
                )
            self.proposal.sweep_budget(window_size)
        if self.sae.k < 1 or self.sae.k > self.sae.dictionary_size:
            raise ValueError("sae.k must lie in [1, dictionary_size]")
        if not 0 < self.sae.high_fraction < 1:
            raise ValueError("sae.high_fraction must lie in (0, 1)")
        if not 0 <= self.sae.high_reconstruction_weight <= 1:
            raise ValueError("sae.high_reconstruction_weight must lie in [0, 1]")
        if not 0 < self.proposal.high_fraction < 1:
            raise ValueError("proposal.high_fraction must lie in (0, 1)")
        if not 0 <= self.proposal.high_reconstruction_weight <= 1:
            raise ValueError(
                "proposal.high_reconstruction_weight must lie in [0, 1]"
            )
        proposal_high = round(
            self.sae.dictionary_size * self.proposal.high_fraction
        )
        proposal_low = self.sae.dictionary_size - proposal_high
        if not 1 <= self.proposal.low_k <= proposal_low:
            raise ValueError("proposal.low_k must lie in [1, proposal d_low]")
        if self.proposal.rgg_p not in {1.0, 2.0}:
            raise ValueError("proposal.rgg_p must be 1 or 2")
        if not 0 < self.proposal.target_active_fraction < 1:
            raise ValueError("proposal.target_active_fraction must lie in (0, 1)")
        if self.proposal.target_sigma < 0:
            raise ValueError("proposal.target_sigma must be non-negative")
        if self.proposal.invariance_weight < 0 or self.proposal.rdm_weight < 0:
            raise ValueError("proposal regularization weights must be non-negative")
        if self.proposal.rdm_projections < 1:
            raise ValueError("proposal.rdm_projections must be positive")
        if self.proposal.rdm_projection_chunk_size < 1:
            raise ValueError("proposal.rdm_projection_chunk_size must be positive")
        if self.proposal.axis_rdm_features < 0:
            raise ValueError("proposal.axis_rdm_features must be non-negative")
        if self.proposal.axis_rdm_weight < 0:
            raise ValueError("proposal.axis_rdm_weight must be non-negative")
        if self.train.standard_steps < 1 or self.train.branch_steps < 1:
            raise ValueError("training step counts must be positive")
        if not 1 <= self.train.temporal_pairs_per_step <= self.train.token_batch_size:
            raise ValueError(
                "temporal_pairs_per_step must lie in [1, token_batch_size]"
            )
        if not 0 <= self.proposal.sae_warmup_steps < self.train.branch_steps:
            raise ValueError("sae_warmup_steps must be smaller than branch_steps")
        if self.proposal.sweep_pairs_per_step < 1:
            raise ValueError("sweep_pairs_per_step must be positive")
        if self.proposal.regularization_ramp_steps < 1:
            raise ValueError("regularization_ramp_steps must be positive")
        if not self.sae_bench.enabled:
            return
        allowed_saebench_evals = {
            "core",
            "sparse_probing",
            "sparse_probing_sae_probes",
            "ravel",
        }
        requested_saebench_evals = set(self.sae_bench.eval_types)
        forbidden_saebench_evals = {"scr", "tpp"}
        if requested_saebench_evals & forbidden_saebench_evals:
            raise ValueError("SCR and TPP are explicitly excluded from this experiment")
        unknown_saebench_evals = requested_saebench_evals - allowed_saebench_evals
        if unknown_saebench_evals:
            raise ValueError(
                f"unsupported SAEBench eval types: {sorted(unknown_saebench_evals)}"
            )
        if set(self.sae_bench.excluded_eval_types) != forbidden_saebench_evals:
            raise ValueError("sae_bench.excluded_eval_types must be exactly ['scr', 'tpp']")
        if not requested_saebench_evals:
            raise ValueError("sae_bench.eval_types must not be empty")
        expected_model = f"EleutherAI/{self.sae_bench.model_name}"
        if self.model.name != expected_model:
            raise ValueError(
                "training and SAEBench models must match exactly: "
                f"expected model.name={expected_model!r}"
            )
        if self.model.revision != "main":
            raise ValueError(
                "SAEBench comparison uses the final deduplicated Pythia checkpoint; "
                "model.revision must be 'main'"
            )
        if self.data.sequence_length != self.sae_bench.context_size:
            raise ValueError(
                "data.sequence_length and sae_bench.context_size must be identical"
            )
        if self.sae_bench.llm_batch_size < 4:
            raise ValueError("sae_bench.llm_batch_size must be at least 4")
        ravel_selection = self.sae_bench.ravel_entity_attribute_selection
        if "ravel" in requested_saebench_evals and not ravel_selection:
            raise ValueError(
                "sae_bench.ravel_entity_attribute_selection must not be empty"
            )
        for entity_class, attributes in ravel_selection.items():
            if not entity_class:
                raise ValueError("RAVEL entity class names must not be empty")
            if len(attributes) < 2:
                raise ValueError(
                    f"RAVEL entity class {entity_class!r} needs at least two attributes"
                )
            if len(set(attributes)) != len(attributes):
                raise ValueError(
                    f"RAVEL attributes for {entity_class!r} must be unique"
                )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["proposal"]["window_sizes"] = list(self.proposal.window_sizes)
        value["evaluation"]["probe_sparsities"] = list(self.evaluation.probe_sparsities)
        value["evaluation"]["probe_subjects"] = list(self.evaluation.probe_subjects)
        value["sae_bench"]["eval_types"] = list(self.sae_bench.eval_types)
        value["sae_bench"]["excluded_eval_types"] = list(
            self.sae_bench.excluded_eval_types
        )
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
        sae_bench=_section(SAEBenchConfig, raw, "sae_bench"),
    )
    cfg.validate()
    return cfg


def apply_training_overrides(
    cfg: ExperimentConfig,
    *,
    training_scale: float = 1.0,
    standard_steps: int | None = None,
    branch_steps: int | None = None,
    warmup_steps: int | None = None,
    sae_warmup_steps: int | None = None,
    regularization_ramp_steps: int | None = None,
    axis_rdm_features: int | None = None,
    run_dir: str | None = None,
) -> ExperimentConfig:
    """Resolve runtime training-budget overrides into a validated config.

    The scale changes optimizer-step counts and their associated schedules, not
    batch sizes. Explicit step values take precedence over the scale. When the
    budget changes and no run directory is supplied, a deterministic suffix is
    added so SAEBench cannot silently reuse results from another budget.
    """

    if not math.isfinite(training_scale) or training_scale <= 0:
        raise ValueError("training_scale must be a finite positive number")

    def scaled(value: int, *, allow_zero: bool = False) -> int:
        result = round(value * training_scale)
        return max(0 if allow_zero else 1, result)

    resolved_standard = (
        scaled(cfg.train.standard_steps)
        if standard_steps is None
        else standard_steps
    )
    resolved_branch = (
        scaled(cfg.train.branch_steps) if branch_steps is None else branch_steps
    )
    resolved_warmup = (
        scaled(cfg.train.warmup_steps, allow_zero=True)
        if warmup_steps is None
        else warmup_steps
    )
    resolved_sae_warmup = (
        scaled(cfg.proposal.sae_warmup_steps, allow_zero=True)
        if sae_warmup_steps is None
        else sae_warmup_steps
    )
    resolved_regularization_ramp = (
        scaled(cfg.proposal.regularization_ramp_steps)
        if regularization_ramp_steps is None
        else regularization_ramp_steps
    )
    if resolved_standard < 1 or resolved_branch < 1:
        raise ValueError("standard_steps and branch_steps must be positive")
    if resolved_warmup < 0 or resolved_sae_warmup < 0:
        raise ValueError("warmup step counts must be non-negative")
    if resolved_regularization_ramp < 1:
        raise ValueError("regularization_ramp_steps must be positive")
    resolved_axis_features = (
        cfg.proposal.axis_rdm_features
        if axis_rdm_features is None
        else axis_rdm_features
    )
    if resolved_axis_features < 0:
        raise ValueError("axis_rdm_features must be non-negative")

    base_budget = (
        cfg.train.standard_steps,
        cfg.train.branch_steps,
        cfg.train.warmup_steps,
        cfg.proposal.sae_warmup_steps,
        cfg.proposal.regularization_ramp_steps,
        cfg.proposal.axis_rdm_features,
    )
    resolved_budget = (
        resolved_standard,
        resolved_branch,
        resolved_warmup,
        resolved_sae_warmup,
        resolved_regularization_ramp,
        resolved_axis_features,
    )
    budget_changed = resolved_budget != base_budget

    resolved_run_dir = run_dir
    if resolved_run_dir is None:
        resolved_run_dir = cfg.run_dir
        if budget_changed:
            only_scaled = all(
                value is None
                for value in (
                    standard_steps,
                    branch_steps,
                    warmup_steps,
                    sae_warmup_steps,
                    regularization_ramp_steps,
                )
            ) and resolved_axis_features == cfg.proposal.axis_rdm_features
            if only_scaled:
                scale_tag = f"{training_scale:.8g}".replace(".", "p")
                suffix = f"trainx{scale_tag}"
            else:
                suffix = (
                    f"budget-s{resolved_standard}-b{resolved_branch}"
                    f"-w{resolved_warmup}-sw{resolved_sae_warmup}"
                    f"-rr{resolved_regularization_ramp}"
                    f"-axis{resolved_axis_features}"
                )
            resolved_run_dir = f"{cfg.run_dir}-{suffix}"

    resolved = replace(
        cfg,
        run_dir=resolved_run_dir,
        train=replace(
            cfg.train,
            standard_steps=resolved_standard,
            branch_steps=resolved_branch,
            warmup_steps=resolved_warmup,
        ),
        proposal=replace(
            cfg.proposal,
            sae_warmup_steps=resolved_sae_warmup,
            regularization_ramp_steps=resolved_regularization_ramp,
            axis_rdm_features=resolved_axis_features,
        ),
    )
    resolved.validate()
    return resolved
