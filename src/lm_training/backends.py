"""Backends decide where each step runs: on which machine, and with which paths.

The rest of the package never needs to know which backend is in use. Each one has:
    prepare_tokenizer()      train the tokenizer if it doesn't exist yet
    plan(gpu_hours)          benchmark the GPUs that will train, and return the plan
    prepare_tokens(n)        make sure n training tokens are tokenized
    train(name, plan)        create the run directory and train
    resume(name)             continue a run from its last checkpoint
"""

import json
from contextlib import contextmanager
from pathlib import Path

from . import config, data, launch, planner


def write_run(cfg: config.Config, run_dir: Path, plan: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    config.save(cfg, run_dir / "config.yaml")
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=1))


def run_training(
    cfg: config.Config, run_dir: Path, gpus: int, jit_checkpoint: bool = False
) -> None:
    args = [run_dir, cfg.data_dir] + (["--jit-checkpoint"] if jit_checkpoint else [])
    launch.run("lm_training.train", args, gpus)


class LocalBackend:
    def __init__(self, cfg: config.Config, gpus: int | None = None):
        self.cfg, self.gpus = cfg, gpus or cfg.compute.gpus

    @contextmanager
    def session(self):
        yield

    def prepare_tokenizer(self) -> None:
        data.train_tokenizer(self.cfg)

    def plan(self, gpu_hours: float) -> dict:
        return planner.make_plan(self.cfg, gpu_hours, self.gpus)

    def prepare_tokens(self, n: int) -> None:
        data.prepare_tokens(self.cfg, n)

    def train(self, name: str, plan: dict) -> None:
        run_dir = self.cfg.runs_dir / name
        write_run(self.cfg, run_dir, plan)
        print(f"Run directory: {run_dir}")
        run_training(self.cfg, run_dir, plan["hardware"]["gpus"])

    def resume(self, name: str) -> None:
        run_dir = self.cfg.runs_dir / name
        plan = json.loads((run_dir / "plan.json").read_text())
        # The run's own config, not today's lmt.yaml.
        saved = config.load(run_dir / "config.yaml")
        saved.root = self.cfg.root
        run_training(saved, run_dir, plan["hardware"]["gpus"])


def make(
    cfg: config.Config,
    backend: str | None = None,
    gpus: int | None = None,
    gpu: str | None = None,
    detach: bool = False,
    prepare_on: str | None = None,
):
    backend = backend or cfg.compute.backend
    if backend == "local":
        return LocalBackend(cfg, gpus)
    if backend == "modal":
        try:
            from .modal_backend import ModalBackend
        except ImportError as e:
            raise SystemExit(
                "The modal backend needs the modal extra: uv sync --extra modal"
            ) from e
        return ModalBackend(
            cfg,
            gpu or cfg.compute.modal.gpu,
            detach,
            prepare_on or cfg.compute.prepare_on,
        )
    raise SystemExit(f"Unknown backend {backend!r}; use local or modal")
