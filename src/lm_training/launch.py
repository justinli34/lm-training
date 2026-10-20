"""Run a module on one GPU with plain Python, or on several with torchrun.

Trainer detects torchrun's environment and sets up distributed training itself.
"""

import logging
import os
import subprocess
import sys
import warnings


def command(module: str, args: list[str], gpus: int) -> list[str]:
    if gpus <= 1:
        return [sys.executable, "-m", module, *args]
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={gpus}",
        "-m", module, *args,
    ]  # fmt: skip


def silence_known_warnings() -> None:
    """Hide three library warnings that appear on multi-GPU runs and are harmless.

    The launched modules call this first. Each filter matches a single message, so
    anything else these libraries report still prints.
    """
    # accelerate warns about Linux kernels older than 5.5. Modal's gVisor sandbox
    # reports its own kernel version (4.19), not the host's.
    logging.getLogger("accelerate.utils.other").addFilter(
        lambda record: "Detected kernel version" not in record.getMessage()
    )
    # torch 2.14 deprecated all_gather_into_tensor, which accelerate's gather still
    # calls when Trainer averages the loss and token counts across GPUs.
    warnings.filterwarnings(
        "ignore", message=r".*all_gather_into_tensor", category=FutureWarning
    )
    # torch.compile drops the profiler label that DDP wraps around forward(). That
    # only matters when profiling.
    logging.getLogger("torch._dynamo.variables.ctx_manager").addFilter(
        lambda record: "Profiler record function" not in record.getMessage()
    )


def run(module: str, args: list, gpus: int, env: dict | None = None) -> None:
    # Otherwise, when stdout is a pipe, our buffered output lands after the child's.
    sys.stdout.flush()
    subprocess.run(
        command(module, [str(a) for a in args], gpus),
        env={**os.environ, **(env or {})},
        check=True,
    )
