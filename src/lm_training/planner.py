"""Turn a GPU-hour budget into a full training plan, measuring the GPU along the way.

Throughput depends on the model's shape, so the shape can't be chosen before
measuring. The planner works in rounds:
    1. Guess the achieved FLOP/s as initial_mfu times the measured matmul peak.
    2. Compute the budget C = FLOP/s x seconds, and pick the shape whose
       compute-optimal training cost is closest to C.
    3. Benchmark that shape to get its real FLOP/s, and go back to step 2.
It stops when the shape stops changing. Of the shapes it benchmarked, it keeps the
one whose tokens per parameter come closest to the target.
"""

import importlib.util
import json
import math
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from . import launch, scaling
from .config import Config


def _bench(args: list, gpus: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.json"
        if args[0] == "run":
            spec = Path(tmp) / "spec.json"
            spec.write_text(json.dumps(args[1]))
            args = ["run", spec]
        launch.run(
            "lm_training.bench",
            [*args, out],
            gpus,
            env={"TORCH_CPP_LOG_LEVEL": "ERROR"},
        )  # quiet OOM probing
        return json.loads(out.read_text())


def _resolve_compile(setting) -> bool:
    if setting == "auto":
        return importlib.util.find_spec("triton") is not None
    return bool(setting)


def _batch_seqs(cfg: Config, compute: float) -> int:
    tokens = cfg.training.batch_tokens or cfg.scaling.batch_tokens(compute)
    return max(1, round(tokens / cfg.training.seq_len))


def _micro_cap(cfg: Config, seqs: int, gpus: int) -> int:
    if cfg.training.micro_batch_size:
        return cfg.training.micro_batch_size
    per_gpu = max(1, seqs // gpus)
    return min(64, 2 ** int(math.log2(per_gpu)))


def make_plan(
    cfg: Config, gpu_hours: float, gpus: int | None = None, log=print
) -> dict:
    started = time.time()
    log("Measuring the GPU...")
    hw = _bench(["probe"], 1)
    gpus = gpus or hw["gpu_count"]
    if gpus > hw["gpu_count"]:
        raise ValueError(
            f"Asked for {gpus} GPUs but only {hw['gpu_count']} are visible"
        )
    hw["gpus"] = gpus
    log(f"  {gpus} x {hw['gpu_name']}, {hw['gpu_memory_gb']:.0f} GB, {_matmul(hw)}")

    wall_seconds = gpu_hours * 3600 / gpus
    compile_ = _resolve_compile(cfg.training.torch_compile)
    # Achieved FLOP/s across all GPUs together.
    flops_per_second = cfg.scaling.initial_mfu * hw["matmul_tflops"] * 1e12 * gpus
    benched: dict[int, tuple[scaling.Shape, dict]] = {}
    search = []

    for _ in range(cfg.scaling.max_plan_iterations):
        compute = flops_per_second * wall_seconds
        shape = scaling.solve_shape(cfg, compute)
        if shape.layers in benched:
            break
        seqs = _batch_seqs(cfg, compute)
        log(
            f"Benchmarking {shape.layers} layers x {shape.hidden} wide ({shape.params / 1e6:.1f}M params)..."
        )
        result = _bench(
            [
                "run",
                {
                    "llama": {**shape.llama_kwargs(), **cfg.model.llama},
                    "attn_implementation": cfg.training.attn_implementation,
                    "seq_len": cfg.training.seq_len,
                    "micro_cap": _micro_cap(cfg, seqs, gpus),
                    "micro_fixed": cfg.training.micro_batch_size is not None,
                    "memory_headroom": cfg.scaling.memory_headroom,
                    "torch_compile": compile_,
                    "warmup_steps": cfg.scaling.bench_warmup_steps,
                    "steps": cfg.scaling.bench_steps,
                },
            ],
            gpus,
        )
        flops_per_second = result["tokens_per_second"] * shape.flops_per_token
        tokens = result["tokens_per_second"] * wall_seconds
        achieved = flops_per_second / gpus / 1e12
        benched[shape.layers] = (shape, result)
        search.append(
            {
                "layers": shape.layers,
                "params": shape.params,
                "tokens_per_second": result["tokens_per_second"],
                "micro_batch_size": result["micro_batch_size"],
                "tokens_per_param": tokens / shape.params,
                **_utilization_fields(achieved, hw),
            }
        )
        log(
            f"  {result['tokens_per_second']:,.0f} tok/s, {utilization(achieved, hw)}"
            f" -> {tokens / shape.params:.1f} tokens/param in budget"
        )

    # Keep the benchmarked shape whose tokens/param comes closest to the target,
    # comparing ratios (in log space) rather than differences.
    def miss(item):
        shape, result = item
        ratio = result["tokens_per_second"] * wall_seconds / shape.params
        return abs(math.log(ratio / cfg.scaling.tokens_per_param))

    shape, result = min(benched.values(), key=miss)

    tps = result["tokens_per_second"]
    tokens = int(tps * wall_seconds)
    compute = shape.flops_per_token * tokens
    micro = result["micro_batch_size"]
    accum = max(1, round(_batch_seqs(cfg, compute) / (micro * gpus)))
    batch_tokens = micro * gpus * accum * cfg.training.seq_len
    steps = tokens // batch_tokens
    t = cfg.training
    achieved_tflops = tps * shape.flops_per_token / gpus / 1e12

    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpu_hours": gpu_hours,
        "wall_seconds": wall_seconds,
        "hardware": hw,
        "model": {
            "llama": {**shape.llama_kwargs(), **cfg.model.llama},
            "params": shape.params,
            "non_embedding_params": shape.non_embedding_params,
            "flops_per_token": shape.flops_per_token,
        },
        "throughput": {
            "tokens_per_second": tps,
            **_utilization_fields(achieved_tflops, hw),
            "peak_memory_gb": result["peak_memory_gb"],
        },
        "training": {
            "tokens": tokens,
            "tokens_per_param": tokens / shape.params,
            "compute_flops": compute,
            "batch_tokens": batch_tokens,
            "micro_batch_size": micro,
            "grad_accum": accum,
            "steps": steps,
            "step_seconds": batch_tokens / tps,
            "warmup_steps": max(
                1, min(t.max_warmup_steps, round(t.warmup_frac * steps))
            ),
            "learning_rate": t.learning_rate or cfg.scaling.learning_rate(compute),
            "torch_compile": compile_,
        },
        "data": {"train_tokens": int(tokens * (1 + cfg.data.token_margin))},
        "search": search,
        "planning_seconds": time.time() - started,
    }


def _utilization_fields(achieved_tflops: float, hw: dict) -> dict:
    spec = hw.get("spec_tflops")
    return {
        "achieved_tflops_per_gpu": achieved_tflops,
        "mfu": achieved_tflops / spec if spec else None,
        "fraction_of_matmul": achieved_tflops / hw["matmul_tflops"],
    }


def utilization(achieved_tflops: float, hw: dict) -> str:
    """MFU against the datasheet peak, plus the share of the measured matmul peak.

    For GPUs missing from bench.SPEC_BF16_TFLOPS, only the matmul share is shown.
    """
    fields = _utilization_fields(achieved_tflops, hw)
    of_matmul = f"{fields['fraction_of_matmul']:.0%} of matmul"
    if fields["mfu"] is None:
        return of_matmul
    return f"MFU {fields['mfu']:.0%} ({of_matmul})"


def _matmul(hw: dict) -> str:
    measured = f"bf16 matmul {hw['matmul_tflops']:.0f} TFLOP/s per GPU"
    if spec := hw.get("spec_tflops"):
        return f"{measured} (datasheet peak {spec:g})"
    return f"{measured} (no datasheet peak on file, so no MFU)"


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _si(x: float) -> str:
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= scale:
            return f"{x / scale:.2f}{unit}"
    return f"{x:.0f}"


def format_plan(plan: dict) -> str:
    hw, tp, tr = plan["hardware"], plan["throughput"], plan["training"]
    m = plan["model"]
    llama = m["llama"]
    budget = (
        f"{plan['gpu_hours']:g} GPU-hours = {_hms(plan['wall_seconds'])} "
        f"on {hw['gpus']} x {hw['gpu_name']}"
    )
    matmul = _matmul(hw)
    achieved = tp["achieved_tflops_per_gpu"]
    training = (
        f"training {achieved:.0f} TFLOP/s per GPU, {utilization(achieved, hw)}, "
        f"{tp['tokens_per_second']:,.0f} tok/s, peak memory {tp['peak_memory_gb']:.1f} GB"
    )
    params = (
        f"{_si(m['params'])} params ({_si(m['non_embedding_params'])} non-embedding)"
    )
    tied = "tied" if llama["tie_word_embeddings"] else "untied"
    shape = (
        f"{llama['num_hidden_layers']} layers, hidden {llama['hidden_size']}, "
        f"{llama['num_attention_heads']} heads / {llama['num_key_value_heads']} KV "
        f"x {llama['head_dim']}, ffn {llama['intermediate_size']}, "
        f"vocab {llama['vocab_size']}, {tied} embeddings"
    )
    tokens = (
        f"{_si(tr['tokens'])} tokens = {tr['tokens_per_param']:.1f} tokens/param, "
        f"{tr['compute_flops']:.2e} FLOPs"
    )
    batch = (
        f"{_si(tr['batch_tokens'])} tokens = {tr['micro_batch_size']} seqs "
        f"x {hw['gpus']} GPUs x {tr['grad_accum']} accum "
        f"x {llama['max_position_embeddings']}"
    )
    schedule = (
        f"~{tr['steps']:,} steps of {tr['step_seconds']:.2f}s, "
        f"peak LR {tr['learning_rate']:.2e}, {tr['warmup_steps']} warmup steps, "
        "WSD decay by wall-clock"
    )
    compile_ = f"torch.compile {'on' if tr['torch_compile'] else 'off'}"
    planning = (
        f"took {plan['planning_seconds']:.0f}s over {len(plan['search'])} benchmark(s)"
    )
    rows = [
        ("Budget", budget),
        ("Measured", matmul),
        ("", training),
        ("Model", params),
        ("", shape),
        ("Data", tokens),
        ("Batch", batch),
        ("Schedule", schedule),
        ("Compile", compile_),
        ("Planning", planning),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k:<{width}}  {v}" for k, v in rows)
