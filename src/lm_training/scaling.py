"""Scaling formulas: model shape from depth, parameter and FLOP counts, and the solver.

Counts are computed from the shape alone, so the solver can try every depth without
building a model.
"""

import math
from dataclasses import dataclass

from .config import Config

SHAPE_KEYS = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
)


def _round_up(x: float, multiple: int) -> int:
    return max(multiple, math.ceil(x / multiple) * multiple)


@dataclass(frozen=True)
class Shape:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    tied: bool
    seq_len: int

    @property
    def layer_params(self) -> int:
        attn = (
            self.hidden * self.head_dim * (2 * self.heads + 2 * self.kv_heads)
        )  # q, o + k, v
        mlp = 3 * self.hidden * self.ffn  # gate, up, down (SwiGLU)
        return attn + mlp + 2 * self.hidden  # + two RMSNorms

    @property
    def embedding_params(self) -> int:
        return self.vocab * self.hidden * (1 if self.tied else 2)

    @property
    def params(self) -> int:
        """Total unique parameters, which is what Chinchilla's tokens/param ratio counts."""
        return self.layers * self.layer_params + self.embedding_params + self.hidden

    @property
    def non_embedding_params(self) -> int:
        return self.params - self.embedding_params

    @property
    def flops_per_token(self) -> int:
        """Training FLOPs per token, forward and backward.

        Six FLOPs per matmul parameter, plus attention, as in PaLM and nanochat.
        """
        matmul = (
            self.layers * (self.layer_params - 2 * self.hidden)
            + self.vocab * self.hidden
        )  # + LM head
        attention = 12 * self.layers * self.heads * self.head_dim * self.seq_len
        return 6 * matmul + attention

    def llama_kwargs(self) -> dict:
        return {
            "num_hidden_layers": self.layers,
            "hidden_size": self.hidden,
            "num_attention_heads": self.heads,
            "num_key_value_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.ffn,
            "vocab_size": self.vocab,
            "tie_word_embeddings": self.tied,
            "max_position_embeddings": self.seq_len,
        }


def shape_for_depth(cfg: Config, layers: int) -> Shape:
    m, pins = cfg.model, cfg.model.llama
    layers = pins.get("num_hidden_layers", layers)
    hidden = pins.get(
        "hidden_size", _round_up(m.aspect_ratio * layers, m.width_multiple)
    )
    head_dim = pins.get("head_dim", m.head_dim)
    heads = pins.get("num_attention_heads", max(1, hidden // head_dim))
    kv_heads = pins.get("num_key_value_heads", max(1, heads // m.kv_groups))
    if heads % kv_heads:
        raise ValueError(
            f"{heads} attention heads are not divisible by {kv_heads} KV heads"
        )
    ffn = pins.get("intermediate_size", _round_up(m.ffn_ratio * hidden, m.ffn_multiple))
    return Shape(
        layers=layers,
        hidden=hidden,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        ffn=ffn,
        vocab=cfg.tokenizer.vocab_size,
        tied=pins.get("tie_word_embeddings", True),
        seq_len=cfg.training.seq_len,
    )


def ladder(cfg: Config) -> list[Shape]:
    """Every candidate shape, smallest first."""
    if "num_hidden_layers" in cfg.model.llama:
        return [shape_for_depth(cfg, cfg.model.llama["num_hidden_layers"])]
    return [
        shape_for_depth(cfg, n)
        for n in range(cfg.model.min_layers, cfg.model.max_layers + 1)
    ]


def compute_needed(cfg: Config, shape: Shape) -> float:
    """FLOPs to train `shape` on tokens_per_param × params tokens."""
    return shape.flops_per_token * cfg.scaling.tokens_per_param * shape.params


def solve_shape(cfg: Config, compute: float) -> Shape:
    """Return the shape whose compute-optimal training cost is closest to `compute`.

    Closeness is compared as a ratio (in log space). Depths are discrete, and loss
    barely changes within a small factor of the compute-optimal size.
    """
    return min(
        ladder(cfg), key=lambda s: abs(math.log(compute_needed(cfg, s) / compute))
    )
