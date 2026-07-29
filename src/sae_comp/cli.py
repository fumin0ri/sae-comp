from __future__ import annotations

import argparse
from collections.abc import Callable

from .activations import extract_activations
from .config import ExperimentConfig, load_config
from .evaluation import evaluate_all, evaluate_window_sweep
from .probes import (
    evaluate_probes,
    evaluate_window_sweep_probes,
    extract_mmlu_probe_cache,
)
from .report import build_report, build_window_sweep_report
from .training import train_all, train_proposal_window_sweep


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
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare standard, temporal, and transition-JEPA SAEs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, _ in STAGES:
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        if command in {"extract", "probes"}:
            child.add_argument("--overwrite", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument(
        "--stages",
        default="extract,train,evaluate,probes,report",
        help="comma-separated ordered stage names",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
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
