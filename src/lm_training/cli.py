"""lmt: train a language model from a GPU-hour budget.

lmt train 8                  measure, show the plan, ask to confirm, then train
lmt train 8 --yes            the same, but start without asking
lmt plan 8                   measure and show the plan only
lmt prepare [--tokens N]     train the tokenizer here, and optionally tokenize N tokens
lmt resume NAME              continue a run from its last checkpoint
lmt generate NAME            continue prompts with a finished run's model
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from . import backends, config, data
from .planner import format_plan


def _plan(backend, gpu_hours: float) -> dict:
    plan = backend.plan(gpu_hours)
    print("\nPlan\n" + format_plan(plan) + "\n")
    return plan


def cmd_train(args, cfg):
    backend = backends.make(
        cfg, args.backend, args.gpus, args.gpu, args.detach, args.prepare_on
    )
    name = args.name or f"{datetime.now():%Y%m%d-%H%M%S}-{args.gpu_hours:g}h"
    with backend.session():
        plan = _plan(backend, args.gpu_hours)
        answer = "y" if args.yes else input("Start training? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Not started.")
            return
        # Planning only needs the vocab size from the config, so data work waits
        # until the run is confirmed.
        backend.prepare_tokenizer()
        backend.prepare_tokens(plan["data"]["train_tokens"])
        backend.train(name, plan)


def cmd_plan(args, cfg):
    backend = backends.make(cfg, args.backend, args.gpus, args.gpu)
    with backend.session():
        plan = _plan(backend, args.gpu_hours)
    if args.save:
        Path(args.save).write_text(json.dumps(plan, indent=1))


def cmd_prepare(args, cfg):
    data.train_tokenizer(cfg)
    if args.tokens:
        data.prepare_tokens(cfg, int(float(args.tokens)))


def cmd_resume(args, cfg):
    backend = backends.make(cfg, args.backend, detach=args.detach)
    with backend.session():
        backend.resume(args.name)


def cmd_generate(args, cfg):
    # Imported here so other commands don't pay for loading transformers.
    from .generate import loop

    loop(
        cfg.runs_dir / args.name / "model",
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lmt",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="lmt.yaml", help="run configuration (default: lmt.yaml)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def budget_args(p):
        p.add_argument(
            "gpu_hours",
            type=float,
            help="compute budget in GPU-hours (wall-clock hours x GPUs)",
        )
        p.add_argument(
            "--backend", choices=["local", "modal"], help="override compute.backend"
        )
        p.add_argument(
            "--gpus", type=int, help="local: number of GPUs (default: all visible)"
        )
        p.add_argument(
            "--gpu",
            help='modal: GPU spec such as "H100!:4" (overrides compute.modal.gpu)',
        )

    p = sub.add_parser("train", help="plan, confirm, and train")
    budget_args(p)
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="start without asking after showing the plan",
    )
    p.add_argument("--name", help="run name (default: timestamp and budget)")
    p.add_argument(
        "--detach",
        action="store_true",
        help="modal: keep training if this terminal closes",
    )
    p.add_argument(
        "--prepare-on",
        choices=["auto", "local", "modal"],
        help="modal: where to tokenize",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("plan", help="measure and print the plan without training")
    budget_args(p)
    p.add_argument("--save", help="also write the plan to this JSON file")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser(
        "prepare", help="train the tokenizer and optionally tokenize, on this machine"
    )
    p.add_argument("--tokens", help="also tokenize this many training tokens, e.g. 2e9")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("resume", help="continue a run from its last checkpoint")
    p.add_argument("name")
    p.add_argument("--backend", choices=["local", "modal"])
    p.add_argument("--detach", action="store_true")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser(
        "generate", help="continue prompts with a run's model, each one fresh"
    )
    p.add_argument("name")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument(
        "--temperature", type=float, default=0.8, help="0 for greedy decoding"
    )
    p.add_argument("--top-p", type=float, default=0.95)
    p.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    args.func(args, config.load(args.config))
