"""Load lmt.yaml into typed dataclasses.

Loading is strict: every key must be present, and unknown keys are an error.
"""

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Data:
    dataset: str
    subset: str | None
    split: str
    text_column: str
    val_tokens: int
    token_margin: float


@dataclass
class Tokenizer:
    vocab_size: int
    train_chars: int


@dataclass
class Model:
    aspect_ratio: float
    head_dim: int
    width_multiple: int
    kv_groups: int
    ffn_ratio: float
    ffn_multiple: int
    min_layers: int
    max_layers: int
    llama: dict[str, Any]


@dataclass
class PowerLaw:
    """A power law in compute: coef * C**exp, where C is the total training FLOPs."""

    coef: float
    exp: float

    def __call__(self, compute: float) -> float:
        return self.coef * compute**self.exp


@dataclass
class Scaling:
    tokens_per_param: float
    learning_rate: PowerLaw
    batch_tokens: PowerLaw
    initial_mfu: float
    max_plan_iterations: int
    bench_warmup_steps: int
    bench_steps: int
    memory_headroom: float


@dataclass
class Training:
    seq_len: int
    learning_rate: float | None
    batch_tokens: int | None
    micro_batch_size: int | None
    warmup_frac: float
    max_warmup_steps: int
    decay_frac: float
    decay_shape: str
    final_lr_ratio: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float
    checkpoint_every_minutes: float
    keep_checkpoints: int
    eval_every_minutes: float
    log_every_steps: int
    torch_compile: str | bool
    attn_implementation: str
    dataloader_workers: int
    seed: int
    report_to: str


@dataclass
class ModalOptions:
    gpu: str
    volume: str
    cpu: float
    timeout_hours: float
    max_retries: int
    secrets: list[str]


@dataclass
class Compute:
    backend: str
    gpus: int | None
    prepare_on: str
    modal: ModalOptions


@dataclass
class Paths:
    data: str
    runs: str


@dataclass
class Config:
    data: Data
    tokenizer: Tokenizer
    model: Model
    scaling: Scaling
    training: Training
    compute: Compute
    paths: Paths
    # The directory that `paths` are relative to. Set when loading, not from the YAML.
    root: Path = field(default=Path("."), compare=False)

    @property
    def data_dir(self) -> Path:
        """Where the tokenizer and token shards for this dataset and vocab size live."""
        subset = f"--{self.data.subset}" if self.data.subset else ""
        slug = f"{self.data.dataset.replace('/', '--')}{subset}--v{self.tokenizer.vocab_size}"
        return self.root / self.paths.data / slug

    @property
    def runs_dir(self) -> Path:
        return self.root / self.paths.runs

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("root")
        return d


def _build(cls, raw: Any, where: str):
    if not dataclasses.is_dataclass(cls):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"{where}: expected a mapping, got {raw!r}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.name != "root"}
    if unknown := set(raw) - names:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    if missing := names - set(raw):
        raise ValueError(f"{where}: missing keys {sorted(missing)}")
    kwargs = {}
    for name in names:
        hint = hints[name]
        # For an optional field (`X | None`), build an X unless the value is null.
        if isinstance(hint, types.UnionType):
            hint = next(a for a in typing.get_args(hint) if a is not type(None))
        kwargs[name] = (
            None if raw[name] is None else _build(hint, raw[name], f"{where}.{name}")
        )
    return cls(**kwargs)


def from_dict(raw: dict, root: Path = Path(".")) -> Config:
    cfg = _build(Config, raw, "config")
    cfg.root = root
    return cfg


def load(path: str | Path) -> Config:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        return from_dict(yaml.safe_load(f), root=path.parent)


def save(cfg: Config, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
