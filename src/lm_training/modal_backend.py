"""Run the GPU steps, and optionally tokenization, on Modal.

This is the only module that imports modal. Everything that persists lives on one
Modal Volume mounted at /vol, with the same layout as the local project: data under
/vol/<paths.data> and runs under /vol/<paths.runs>. The confirm prompt runs on your
machine, between the remote planning step and the remote training run.
"""

import json
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import modal

from . import backends, config, data, planner

ROOT = PurePosixPath("/vol")
PROJECT = Path(__file__).resolve().parents[2]
# Past this many tokens, tokenizing on Modal is quicker than uploading local shards.
LOCAL_TOKENIZE_LIMIT = 1_000_000_000

app = modal.App("lm-training")
image = (
    modal.Image.debian_slim(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    .uv_sync(str(PROJECT), extras=["modal"])
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("lm_training")
)


def _remote_cfg(raw: dict) -> config.Config:
    return config.from_dict(raw, root=Path(ROOT))


@app.function(image=image, timeout=24 * 3600)
def remote_prepare_tokens(raw: dict, n: int) -> dict:
    return data.prepare_tokens(_remote_cfg(raw), n)


@app.function(image=image, gpu="T4", timeout=3600)
def remote_plan(raw: dict, gpu_hours: float) -> dict:
    return planner.make_plan(_remote_cfg(raw), gpu_hours)


@app.function(image=image, gpu="T4", timeout=24 * 3600, single_use_containers=True)
def remote_train(raw: dict, name: str, plan: dict | None) -> None:
    cfg = _remote_cfg(raw)
    run_dir = cfg.runs_dir / name
    # First attempt writes the run; retries and resumes reuse its files.
    if not (run_dir / "plan.json").exists():
        assert plan is not None, f"no run named {name} to resume"
        backends.write_run(cfg, run_dir, plan)
    saved = config.load(run_dir / "config.yaml")
    saved.root = cfg.root
    plan = json.loads((run_dir / "plan.json").read_text())
    backends.run_training(saved, run_dir, plan["hardware"]["gpus"], jit_checkpoint=True)


class ModalBackend:
    def __init__(self, cfg: config.Config, gpu: str, detach: bool, prepare_on: str):
        self.cfg, self.gpu, self.detach, self.prepare_on = cfg, gpu, detach, prepare_on
        m = cfg.compute.modal
        self.volume = modal.Volume.from_name(m.volume, create_if_missing=True)
        self.volumes: dict[
            str | PurePosixPath, modal.Volume | modal.CloudBucketMount
        ] = {ROOT: self.volume}
        self.secrets = [modal.Secret.from_name(s) for s in m.secrets]
        self.raw = cfg.to_dict()

    def _remote_path(self, local: Path) -> str:
        return local.relative_to(self.cfg.root).as_posix()

    def _remote_manifest(self) -> dict | None:
        try:
            raw = b"".join(
                self.volume.read_file(
                    self._remote_path(self.cfg.data_dir / "manifest.json")
                )
            )
        except FileNotFoundError:
            return None
        return json.loads(raw)

    def _remote_files(self, local_dir: Path) -> set[str]:
        try:
            return {
                e.path
                for e in self.volume.listdir(
                    self._remote_path(local_dir), recursive=True
                )
            }
        except FileNotFoundError, modal.exception.NotFoundError:
            return set()

    def _upload(self, paths: list[Path]) -> None:
        if not paths:
            return
        print(
            f"Uploading {len(paths)} file(s) ({sum(p.stat().st_size for p in paths) / 1e9:.2f} GB) to volume {self.cfg.compute.modal.volume}"
        )
        with self.volume.batch_upload(force=True) as batch:
            for p in paths:
                batch.put_file(p, self._remote_path(p))

    @contextmanager
    def session(self):
        with modal.enable_output(), app.run(detach=self.detach):
            yield

    def prepare_tokenizer(self) -> None:
        """Train the tokenizer locally, since it's small, and upload it if needed."""
        tok_dir = data.train_tokenizer(self.cfg)
        remote = self._remote_files(tok_dir)
        self._upload(
            [p for p in tok_dir.iterdir() if self._remote_path(p) not in remote]
        )

    def plan(self, gpu_hours: float) -> dict:
        fn = remote_plan.with_options(
            gpu=self.gpu, volumes=self.volumes, secrets=self.secrets
        )
        plan = fn.remote(self.raw, gpu_hours)
        plan["modal_gpu"] = self.gpu
        return plan

    def prepare_tokens(self, n: int) -> None:
        remote = self._remote_manifest()
        if remote and (remote["train_tokens"] >= n or remote["exhausted"]):
            return
        where = self.prepare_on
        if where == "auto":
            where = "local" if n <= LOCAL_TOKENIZE_LIMIT else "modal"
        if where == "modal":
            print(f"Tokenizing on a {self.cfg.compute.modal.cpu:g}-CPU Modal container")
            remote_prepare_tokens.with_options(
                cpu=self.cfg.compute.modal.cpu,
                volumes=self.volumes,
                secrets=self.secrets,
            ).remote(self.raw, n)
            return
        manifest = data.prepare_tokens(self.cfg, n)
        files = [manifest["val"]["file"]] + [s["file"] for s in manifest["shards"]]
        on_volume = self._remote_files(self.cfg.data_dir)
        self._upload(
            [
                self.cfg.data_dir / f
                for f in files
                if self._remote_path(self.cfg.data_dir / f) not in on_volume
            ]
        )
        # Upload the manifest last, so it never lists a shard the volume doesn't have.
        self._upload([self.cfg.data_dir / "manifest.json"])

    def _train(self, name: str, plan: dict | None, gpu: str) -> None:
        m = self.cfg.compute.modal
        fn = remote_train.with_options(
            gpu=gpu,
            timeout=int(m.timeout_hours * 3600),
            retries=modal.Retries(max_retries=m.max_retries, initial_delay=0.0),
            volumes=self.volumes,
            secrets=self.secrets,
        )
        # Spawn rather than .remote(), because .remote() calls expire after 24 hours.
        call = fn.spawn(self.raw, name, plan)
        print(
            f"Training {name} on {gpu}. Resume after an interruption with: lmt resume {name} --backend modal"
        )
        if self.detach:
            print(
                "Detached: training continues if you close this terminal. Follow it with: modal app logs lm-training"
            )
        call.get()
        runs = self._remote_path(self.cfg.runs_dir / name)
        print(
            f"Download the model with: modal volume get {m.volume} {runs}/model ./{runs}/model"
        )

    def train(self, name: str, plan: dict) -> None:
        self._train(name, plan, plan["modal_gpu"])

    def resume(self, name: str) -> None:
        raw = b"".join(
            self.volume.read_file(
                self._remote_path(self.cfg.runs_dir / name / "plan.json")
            )
        )
        self._train(name, None, json.loads(raw)["modal_gpu"])
