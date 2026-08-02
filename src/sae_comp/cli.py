from __future__ import annotations

import argparse
from collections.abc import Callable

from .activations import extract_activations
from .config import ExperimentConfig, apply_training_overrides, load_config
from .evaluation import (
    evaluate_all,
    evaluate_controlled_comparison,
    evaluate_window_sweep,
)
from .probes import (
    evaluate_controlled_probes,
    evaluate_probes,
    evaluate_window_sweep_probes,
    extract_mmlu_probe_cache,
)
from .report import (
    build_controlled_report,
    build_report,
    build_window_sweep_report,
)
from .saebench import run_saebench
from .saebench_report import build_saebench_report
from .training import train_all, train_controls, train_proposal_window_sweep

Stage = tuple[str, Callable[[ExperimentConfig], object]]
STAGES: tuple[Stage, ...] = (
    ("extract", extract_activations),
    ("train", train_all),
    ("evaluate", evaluate_all),
    ("probes", evaluate_probes),
    ("report", build_report),
    ("train-window-sweep", train_proposal_window_sweep),
    ("evaluate-window-sweep", evaluate_window_sweep),
    ("probe-window-sweep", evaluate_window_sweep_probes),
    ("report-window-sweep", build_window_sweep_report),
    ("train-controls", train_controls),
    ("evaluate-controlled", evaluate_controlled_comparison),
    ("probe-controlled", evaluate_controlled_probes),
    ("report-controlled", build_controlled_report),
    ("saebench", run_saebench),
    ("report-saebench", build_saebench_report),
)


def _add_training_override_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--training-scale",
        type=float,
        default=1.0,
        help=(
            "multiply shared/branch steps and warm-up/ramp schedules; "
            "batch sizes remain fixed"
        ),
    )
    parser.add_argument("--standard-steps", type=int)
    parser.add_argument("--branch-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--sae-warmup-steps", type=int)
    parser.add_argument("--prediction-ramp-steps", type=int)
    parser.add_argument(
        "--run-dir",
        help=(
            "override the output directory; otherwise a changed training budget "
            "gets a deterministic suffix"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare standard, temporal, and transition-JEPA SAEs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, _ in STAGES:
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        _add_training_override_arguments(child)
        if command in {"extract", "probes"}:
            child.add_argument("--overwrite", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    _add_training_override_arguments(run)
    run.add_argument(
        "--stages",
        default="extract,train,evaluate,probes,report",
        help="comma-separated ordered stage names",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = apply_training_overrides(
        load_config(args.config),
        training_scale=args.training_scale,
        standard_steps=args.standard_steps,
        branch_steps=args.branch_steps,
        warmup_steps=args.warmup_steps,
        sae_warmup_steps=args.sae_warmup_steps,
        prediction_ramp_steps=args.prediction_ramp_steps,
        run_dir=args.run_dir,
    )
    print(
        "training budget: "
        f"shared={cfg.train.standard_steps} steps, "
        f"branch={cfg.train.branch_steps} steps, "
        f"warmup={cfg.train.warmup_steps}, "
        f"sae_warmup={cfg.proposal.sae_warmup_steps}, "
        f"prediction_ramp={cfg.proposal.prediction_ramp_steps}, "
        f"run_dir={cfg.run_dir}"
    )
    if args.command == "extract":
        print(extract_activations(cfg, overwrite=args.overwrite))
    elif args.command == "train":
        print(train_all(cfg))
    elif args.command == "evaluate":
        print(evaluate_all(cfg))
    elif args.command == "probes":
        if args.overwrite:
            extract_mmlu_probe_cache(cfg, overwrite=True)
        print(evaluate_probes(cfg))
    elif args.command == "report":
        print(build_report(cfg))
    elif args.command == "train-window-sweep":
        print(train_proposal_window_sweep(cfg))
    elif args.command == "evaluate-window-sweep":
        print(evaluate_window_sweep(cfg))
    elif args.command == "probe-window-sweep":
        print(evaluate_window_sweep_probes(cfg))
    elif args.command == "report-window-sweep":
        print(build_window_sweep_report(cfg))
    elif args.command == "train-controls":
        print(train_controls(cfg))
    elif args.command == "evaluate-controlled":
        print(evaluate_controlled_comparison(cfg))
    elif args.command == "probe-controlled":
        print(evaluate_controlled_probes(cfg))
    elif args.command == "report-controlled":
        print(build_controlled_report(cfg))
    elif args.command == "saebench":
        print(run_saebench(cfg))
    elif args.command == "report-saebench":
        print(build_saebench_report(cfg))
    elif args.command == "run":
        requested = [name.strip() for name in args.stages.split(",")]
        known = dict(STAGES)
        unknown = set(requested) - set(known)
        if unknown:
            raise ValueError(f"unknown stages: {sorted(unknown)}")
        for name in requested:
            print(f"== {name} ==")
            print(known[name](cfg))


if __name__ == "__main__":
    main()
