"""Turn a Hugging Face dataset into a trained tokenizer and token shards on disk.

Layout of cfg.data_dir:
    tokenizer/        the tokenizer, saved with save_pretrained
    val.npy           held-out tokens for evaluation
    train-00000.npy   training tokens, about 134M per shard
    manifest.json     what the shards hold, and where reading the dataset stopped

Shards are only ever added, so a larger run tokenizes just the tokens it's missing.
"""

import bisect
import hashlib
import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from .config import Config

EOT = "<|endoftext|>"
# GPT-4's pre-tokenization regex, which Llama 3 and nanochat also use.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SHARD_TOKENS = 2**27


def tokenizer_dir(cfg: Config) -> Path:
    return cfg.data_dir / "tokenizer"


def _stream(cfg: Config):
    from datasets import load_dataset

    return load_dataset(
        cfg.data.dataset, cfg.data.subset, split=cfg.data.split, streaming=True
    )


def _texts(dataset, column: str) -> Iterator[str]:
    for row in dataset:
        if text := row[column]:
            yield text


def train_tokenizer(cfg: Config) -> Path:
    """Train a byte-level BPE tokenizer on the start of the dataset, and save it."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    out = tokenizer_dir(cfg)
    if (out / "tokenizer.json").exists():
        return out

    def limited_texts():
        chars, start = 0, time.time()
        for text in _texts(_stream(cfg), cfg.data.text_column):
            yield text
            chars += len(text)
            if chars >= cfg.tokenizer.train_chars:
                print(
                    f"  read {chars / 1e6:.0f}M characters in {time.time() - start:.0f}s",
                    flush=True,
                )
                return

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.tokenizer.vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(
        f"Training a {cfg.tokenizer.vocab_size}-token BPE tokenizer on {cfg.tokenizer.train_chars / 1e6:.0f}M characters"
    )
    tok.train_from_iterator(limited_texts(), trainer)

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token=EOT, eos_token=EOT, pad_token=EOT
    )
    wrapped.save_pretrained(out)
    print(f"Saved tokenizer to {out}")
    return out


def tokenizer_sha(cfg: Config) -> str:
    return hashlib.sha256(
        (tokenizer_dir(cfg) / "tokenizer.json").read_bytes()
    ).hexdigest()[:16]


def read_manifest(cfg: Config) -> dict | None:
    path = cfg.data_dir / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def train_tokens_available(cfg: Config) -> int:
    manifest = read_manifest(cfg)
    return manifest["train_tokens"] if manifest else 0


def prepare_tokens(cfg: Config, train_tokens: int) -> dict:
    """Make sure at least `train_tokens` training tokens are on disk, plus a val split."""
    from tokenizers import Tokenizer

    out = cfg.data_dir
    sha = tokenizer_sha(cfg)
    manifest = read_manifest(cfg) or {
        "dataset": cfg.data.dataset,
        "subset": cfg.data.subset,
        "split": cfg.data.split,
        "tokenizer_sha": sha,
        "dtype": "uint16" if cfg.tokenizer.vocab_size <= 2**16 else "uint32",
        "val": None,
        "shards": [],
        "train_tokens": 0,
        "stream_state": None,
        "exhausted": False,
    }
    if manifest["tokenizer_sha"] != sha:
        raise RuntimeError(
            f"Token shards in {out} were made with a different tokenizer; delete them to re-tokenize"
        )
    if manifest["train_tokens"] >= train_tokens or manifest["exhausted"]:
        return manifest

    tok = Tokenizer.from_file(str(tokenizer_dir(cfg) / "tokenizer.json"))
    eot = tok.token_to_id(EOT)
    dtype = np.dtype(manifest["dtype"])
    dataset = _stream(cfg)
    if manifest["stream_state"]:
        dataset.load_state_dict(manifest["stream_state"])

    def save_manifest():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    def flush(chunks: list[np.ndarray], name: str) -> int:
        tokens = np.concatenate(chunks).astype(dtype)
        np.save(out / name, tokens)
        return len(tokens)

    need = train_tokens - manifest["train_tokens"]
    print(f"Tokenizing {need / 1e6:,.0f}M tokens from {cfg.data.dataset} into {out}")
    buffer, buffered, done, start, last_report = [], 0, 0, time.time(), time.time()
    rows = iter(dataset)
    while True:
        batch = [row[cfg.data.text_column] for _, row in zip(range(1024), rows)]
        if not batch:
            manifest["exhausted"] = True
            break
        for ids in tok.encode_batch_fast(batch, add_special_tokens=False):
            buffer.append(np.asarray(ids.ids + [eot]))
            buffered += len(ids.ids) + 1

        if manifest["val"] is None:
            if buffered >= cfg.data.val_tokens:
                manifest["val"] = {
                    "file": "val.npy",
                    "tokens": flush(buffer, "val.npy"),
                }
                buffer, buffered = [], 0
            continue
        if buffered >= SHARD_TOKENS or done + buffered >= need:
            name = f"train-{len(manifest['shards']):05d}.npy"
            n = flush(buffer, name)
            manifest["shards"].append({"file": name, "tokens": n})
            manifest["train_tokens"] += n
            manifest["stream_state"] = dataset.state_dict()
            save_manifest()
            done += n
            buffer, buffered = [], 0
            if done >= need:
                break
        if time.time() - last_report > 15:
            rate = (done + buffered) / (time.time() - start)
            print(
                f"  {(done + buffered) / 1e6:,.0f}M / {need / 1e6:,.0f}M tokens ({rate / 1e6:.2f}M tok/s)",
                flush=True,
            )
            last_report = time.time()

    if buffer and manifest["val"] is not None:  # dataset ran out mid-shard
        name = f"train-{len(manifest['shards']):05d}.npy"
        n = flush(buffer, name)
        manifest["shards"].append({"file": name, "tokens": n})
        manifest["train_tokens"] += n
    save_manifest()
    if manifest["exhausted"]:
        print(
            f"  dataset exhausted at {manifest['train_tokens'] / 1e6:,.0f}M training tokens; training will repeat data"
        )
    return manifest


class TokenBlocks(torch.utils.data.Dataset):
    """Fixed-length blocks of concatenated documents.

    Blocks are read lazily from memory-mapped shards, which each DataLoader worker
    opens for itself.
    """

    def __init__(self, files: list[Path], seq_len: int):
        self.files = [str(f) for f in files]
        self.seq_len = seq_len
        sizes = [np.load(f, mmap_mode="r").shape[0] // seq_len for f in self.files]
        self.offsets = np.cumsum([0] + sizes).tolist()
        self._arrays = None  # opened on first use, in each worker process

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> dict:
        if self._arrays is None:
            self._arrays = [np.load(f, mmap_mode="r") for f in self.files]
        shard = bisect.bisect_right(self.offsets, index) - 1
        start = (index - self.offsets[shard]) * self.seq_len
        ids = torch.from_numpy(
            self._arrays[shard][start : start + self.seq_len].astype(np.int64)
        )
        return {"input_ids": ids, "labels": ids}

    def __getstate__(self):
        return {**self.__dict__, "_arrays": None}


def datasets_for_training(
    data_dir: Path, seq_len: int
) -> tuple[TokenBlocks, TokenBlocks]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    train = TokenBlocks([data_dir / s["file"] for s in manifest["shards"]], seq_len)
    val = TokenBlocks([data_dir / manifest["val"]["file"]], seq_len)
    return train, val
