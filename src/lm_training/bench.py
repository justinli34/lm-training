"""GPU measurements for the planner.

These run as subprocesses, so a crash or out-of-memory error can't take the planner
down with it.

python -m lm_training.bench probe OUT.json
    Report the GPU's name, count, and memory, and measure bf16 matmul speed.
torchrun ... -m lm_training.bench run SPEC.json OUT.json
    Find the largest micro-batch that fits in memory. Time real Trainer steps on
    random tokens at that size and at half of it, and report the faster one.
"""

import json
import os
import re
import sys
import tempfile
import time

import torch

from . import launch

# Dense bf16 tensor-core peaks in TFLOP/s, from NVIDIA datasheets. Datasheets headline
# the 2:4-sparse figure, which is twice these. Patterns are tried in order against
# torch's device name. GeForce cards are left out on purpose: their published figures
# don't say which accumulation precision they assume.
SPEC_BF16_TFLOPS = [
    (r"\bH100\b.*\bPCIe\b", 756),
    (r"\bH100\b(?!.*\bNVL\b)", 989),  # SXM
    (r"\bH200\b(?!.*\bNVL\b)", 989),  # SXM
    (r"\bB200\b", 2250),
    (r"\bA100\b", 312),
    (r"\bL40S\b", 362),
    (r"\bA10\b", 125),
    (r"\bL4\b", 121),
]


def spec_tflops(gpu_name: str) -> float | None:
    """The datasheet's dense bf16 peak for this GPU, or None if it isn't in the table."""
    for pattern, tflops in SPEC_BF16_TFLOPS:
        if re.search(pattern, gpu_name):
            return tflops
    return None


def probe() -> dict:
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    iters, start = 20, time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": props.name,
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory_gb": props.total_memory / 1e9,
        "matmul_tflops": 2 * n**3 * iters / seconds / 1e12,
        "spec_tflops": spec_tflops(props.name),
    }


def _model(spec: dict):
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(LlamaConfig(**spec["llama"]))
    model.set_attn_implementation(spec["attn_implementation"])
    return model


def find_micro_batch(spec: dict) -> tuple[int, float]:
    """Return the largest power-of-2 micro-batch that trains within the memory headroom.

    Also returns its peak memory in GB.
    """
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    limit = (
        spec["memory_headroom"] * torch.cuda.get_device_properties(device).total_memory
    )
    micro = spec["micro_cap"]
    while micro >= 1:
        model = optimizer = None
        try:
            torch.cuda.reset_peak_memory_stats(device)
            model = _model(spec).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), fused=True)
            ids = torch.randint(
                0, spec["llama"]["vocab_size"], (micro, spec["seq_len"]), device=device
            )
            for _ in range(2):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(input_ids=ids, labels=ids).loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(device)
            if peak <= limit or spec.get("micro_fixed"):
                return micro, peak / 1e9
        except torch.OutOfMemoryError:
            if spec.get("micro_fixed"):
                raise
        finally:
            del model, optimizer
            torch.cuda.empty_cache()
        micro //= 2
    raise RuntimeError("Even a micro-batch of 1 does not fit in GPU memory")


def time_trainer(spec: dict, micro: int) -> dict:
    """Throughput of real Trainer steps on random tokens."""
    from transformers import (
        PrinterCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    world = int(os.environ.get("WORLD_SIZE", "1"))
    warmup, steps = spec["warmup_steps"], spec["steps"]

    class Random(torch.utils.data.Dataset):
        def __len__(self):
            return micro * world * (warmup + steps)

        def __getitem__(self, index):
            ids = torch.randint(0, spec["llama"]["vocab_size"], (spec["seq_len"],))
            return {"input_ids": ids, "labels": ids}

    class Timer(TrainerCallback):
        def __init__(self):
            self.marks: list[float] = []

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in (warmup, warmup + steps):
                torch.cuda.synchronize()
                self.marks.append(time.perf_counter())

    timer = Timer()
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            per_device_train_batch_size=micro,
            max_steps=warmup + steps,
            learning_rate=1e-4,
            weight_decay=0.1,
            optim="adamw_torch_fused",
            bf16=True,
            tf32=True,
            torch_compile=spec["torch_compile"],
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
            logging_strategy="no",
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False,
        )
        trainer = Trainer(
            model=_model(spec), args=args, train_dataset=Random(), callbacks=[timer]
        )
        trainer.remove_callback(PrinterCallback)
        trainer.train()
        del trainer
        torch.cuda.empty_cache()
    seconds = timer.marks[1] - timer.marks[0]
    return {
        "micro_batch_size": micro,
        "world_size": world,
        "tokens_per_second": steps * micro * world * spec["seq_len"] / seconds,
        "step_seconds": seconds / steps,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def run(spec: dict) -> dict:
    """Time the largest micro-batch that fits and half of it, and return the faster.

    Fitting in memory isn't enough. Close to the limit, throughput can collapse without
    any out-of-memory error, for example when the Windows driver starts spilling into
    system RAM. Timing both sizes catches this.
    """
    micro, _ = find_micro_batch(spec)
    options = [micro] if spec.get("micro_fixed") or micro == 1 else [micro, micro // 2]
    return max(
        (time_trainer(spec, m) for m in options), key=lambda r: r["tokens_per_second"]
    )


def main() -> None:
    launch.silence_known_warnings()
    mode, *paths = sys.argv[1:]
    if mode == "probe":
        result = probe()
    else:
        with open(paths[0]) as f:
            spec = json.load(f)
        result = run(spec)
    if int(os.environ.get("RANK", "0")) == 0:
        with open(paths[-1], "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
