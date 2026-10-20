"""The training process, run as `python -m lm_training.train RUN_DIR DATA_DIR`.

With several GPUs it runs under torchrun. It is a thin layer over transformers.Trainer
that adds what Trainer lacks: a wall-clock budget. The learning rate warms up, holds
steady, and starts decaying once (1 - decay_frac) of the budget's training time is
used. Training stops when the budget is used up. Each checkpoint stores the elapsed
training time, so resuming after a crash or preemption continues the same clock.
"""

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    PrinterCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging

from . import config, data, launch
from .planner import utilization

# Written last into each checkpoint, so it also shows that the checkpoint is complete.
CLOCK_FILE = "budget.json"
# With several GPUs, the ranks agree on elapsed time about this often. Each agreement
# makes the CPU wait for the GPU, so doing it every step would stall the GPU.
CLOCK_SYNC_SECONDS = 5

# TrainingArguments that affect speed. bench.py uses them too, so the planner measures
# the throughput that training will get.
SPEED_ARGS: dict[str, Any] = {
    "optim": "adamw_torch_fused",
    "bf16": True,
    "tf32": True,
    "ddp_find_unused_parameters": False,
    # Checking each micro-batch's loss for NaN makes the CPU wait for the GPU to finish it.
    "logging_nan_inf_filter": False,
    # Every block is full with no masked labels, so all ranks count the same tokens and
    # the per-step gather that averages the count across GPUs changes nothing.
    "average_tokens_across_devices": False,
}


class Clock:
    def __init__(self, budget_seconds: float, elapsed: float = 0.0):
        self.budget = budget_seconds
        self.elapsed = elapsed

    @property
    def progress(self) -> float:
        return self.elapsed / self.budget


def wsd_schedule(cfg: config.Training, warmup_steps: int, clock: Clock):
    decay_start = 1 - cfg.decay_frac

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if clock.progress <= decay_start:
            return 1.0
        x = min(1.0, (clock.progress - decay_start) / cfg.decay_frac)
        remaining = 1 - math.sqrt(x) if cfg.decay_shape == "1-sqrt" else 1 - x
        return cfg.final_lr_ratio + (1 - cfg.final_lr_ratio) * remaining

    return multiplier


class BudgetTrainer(Trainer):
    def __init__(self, *args, schedule, clock: Clock, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedule, self.clock = schedule, clock

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            optimizer = optimizer or self.optimizer
            assert optimizer is not None, (
                "Trainer creates the optimizer before the scheduler"
            )
            self.lr_scheduler = LambdaLR(optimizer, self.schedule)
        return self.lr_scheduler

    def _save_checkpoint(self, model, trial):
        # Every save passes through here, including the SIGTERM checkpoint, which
        # skips on_save. The clock file goes in last to mark the checkpoint complete.
        ckpt = (
            Path(self._get_output_dir(trial=trial))
            / f"checkpoint-{self.state.global_step}"
        )
        if self.is_world_process_zero():
            # A save can overwrite a checkpoint of the same step; it's incomplete until done.
            (ckpt / CLOCK_FILE).unlink(missing_ok=True)
        super()._save_checkpoint(model, trial)
        if self.is_world_process_zero():
            (ckpt / CLOCK_FILE).write_text(json.dumps({"elapsed": self.clock.elapsed}))


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class BudgetCallback(TrainerCallback):
    """Track training time, trigger checkpoints and evals, and print progress.

    Time spent evaluating and checkpointing doesn't count against the budget.
    """

    def __init__(self, clock: Clock, cfg: config.Training, plan: dict):
        self.clock, self.cfg, self.plan = clock, cfg, plan
        self.final_eval_loss = None
        self.distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        # Between syncs the clock stands still, so every rank sees the same clock, and
        # with it the same learning rate.
        self.sync_every = (
            max(1, round(CLOCK_SYNC_SECONDS / plan["training"]["step_seconds"]))
            if self.distributed
            else 1
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.mark = time.perf_counter()
        self.unsynced = 0.0
        self.last_save = self.last_eval = self.clock.elapsed
        self.log_step, self.log_elapsed = state.global_step, self.clock.elapsed

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        self.unsynced += now - self.mark
        self.mark = now
        if state.global_step % self.sync_every:
            return
        dt, self.unsynced = self.unsynced, 0.0
        if self.distributed:
            t = torch.tensor([dt], device="cuda")
            # Every rank must agree on the elapsed time, or they won't stop together.
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            dt = t.item()
        self.clock.elapsed += dt

        if self.clock.elapsed >= self.clock.budget:
            control.should_training_stop = control.should_save = (
                control.should_evaluate
            ) = True
        if (
            self.clock.elapsed - self.last_save
            >= self.cfg.checkpoint_every_minutes * 60
        ):
            control.should_save = True
        if self.clock.elapsed - self.last_eval >= self.cfg.eval_every_minutes * 60:
            control.should_evaluate = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.mark = time.perf_counter()
        self.last_eval = self.clock.elapsed
        self.final_eval_loss = (metrics or {}).get("eval_loss")

    def on_save(self, args, state, control, **kwargs):
        self.mark = time.perf_counter()
        self.last_save = self.clock.elapsed

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        clock = f"{_hms(self.clock.elapsed)} / {_hms(self.clock.budget)} ({self.clock.progress:.0%})"
        if "eval_loss" in logs:
            print(
                f"step {state.global_step:>6} | eval loss {logs['eval_loss']:.4f} | {clock}",
                flush=True,
            )
            return
        if "loss" not in logs:
            return
        steps = state.global_step - self.log_step
        # This rank's own time, which includes steps the clock hasn't synced yet.
        elapsed = self.clock.elapsed + self.unsynced
        seconds = elapsed - self.log_elapsed
        self.log_step, self.log_elapsed = state.global_step, elapsed
        tps = (
            steps * self.plan["training"]["batch_tokens"] / seconds
            if seconds > 0
            else 0.0
        )
        hw = self.plan["hardware"]
        achieved = tps * self.plan["model"]["flops_per_token"] / hw["gpus"] / 1e12
        print(
            f"step {state.global_step:>6}/~{self.plan['training']['steps']} | loss {logs['loss']:.4f} | "
            f"lr {logs.get('learning_rate', 0):.2e} | grad {logs.get('grad_norm', 0):.2f} | "
            f"{tps:,.0f} tok/s | {utilization(achieved, hw)} | {clock}",
            flush=True,
        )


def latest_checkpoint(output_dir: Path) -> tuple[Path, float] | None:
    """Return the newest complete checkpoint and its elapsed training time.

    A checkpoint without a clock file was cut off mid-save, so it's skipped.
    """
    found = []
    for path in output_dir.glob("checkpoint-*"):
        if (match := re.fullmatch(r"checkpoint-(\d+)", path.name)) and (
            path / CLOCK_FILE
        ).exists():
            found.append((int(match[1]), path))
    if not found:
        return None
    _, path = max(found)
    return path, json.loads((path / CLOCK_FILE).read_text())["elapsed"]


def main(run_dir: Path, data_dir: Path, jit_checkpoint: bool = False) -> None:
    launch.silence_known_warnings()
    hf_logging.disable_progress_bar()  # per-checkpoint "Writing model shards" bars
    if (run_dir / "summary.json").exists():
        print(f"{run_dir.name} already finished; see {run_dir / 'summary.json'}")
        return
    cfg = config.load(run_dir / "config.yaml")
    plan = json.loads((run_dir / "plan.json").read_text())
    t, p = cfg.training, plan["training"]

    tokenizer = PreTrainedTokenizerFast.from_pretrained(data_dir / "tokenizer")
    model_config = LlamaConfig(
        **plan["model"]["llama"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    train_set, val_set = data.datasets_for_training(data_dir, t.seq_len)

    output_dir = run_dir / "checkpoints"
    resume = latest_checkpoint(output_dir)
    clock = Clock(plan["wall_seconds"], elapsed=resume[1] if resume else 0.0)
    max_steps = (
        2 * p["steps"] + 1000
    )  # only a safety cap; the clock decides when to stop
    # Budget already spent: only the final save is missing.
    if resume and clock.elapsed >= clock.budget:
        max_steps = int(resume[0].name.split("-")[1])

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=p["micro_batch_size"],
        per_device_eval_batch_size=p["micro_batch_size"],
        gradient_accumulation_steps=p["grad_accum"],
        max_steps=max_steps,
        learning_rate=p["learning_rate"],
        weight_decay=t.weight_decay,
        adam_beta1=t.adam_beta1,
        adam_beta2=t.adam_beta2,
        adam_epsilon=t.adam_epsilon,
        max_grad_norm=t.max_grad_norm,
        torch_compile=p["torch_compile"],
        save_strategy="no",  # BudgetCallback schedules checkpoints by training time,
        eval_strategy="no",  # and evals too
        save_total_limit=t.keep_checkpoints,
        logging_strategy="steps",
        logging_steps=t.log_every_steps,
        prediction_loss_only=True,
        report_to=t.report_to,
        run_name=run_dir.name,
        disable_tqdm=True,
        dataloader_num_workers=t.dataloader_workers,
        dataloader_pin_memory=True,
        enable_jit_checkpoint=jit_checkpoint,
        seed=t.seed,
        **SPEED_ARGS,
    )
    callback = BudgetCallback(clock, t, plan)
    model = LlamaForCausalLM(model_config)
    model.set_attn_implementation(t.attn_implementation)
    trainer = BudgetTrainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=tokenizer,
        callbacks=[callback],
        schedule=wsd_schedule(t, p["warmup_steps"], clock),
        clock=clock,
    )
    trainer.remove_callback(PrinterCallback)

    if resume:
        print(
            f"Resuming from {resume[0].name} at {_hms(clock.elapsed)} of {_hms(clock.budget)}",
            flush=True,
        )
    trainer.train(resume_from_checkpoint=str(resume[0]) if resume else None)

    # The SIGTERM checkpoint also ends train() normally, with budget left. That run
    # isn't finished: exit with an error so Modal retries it from the checkpoint.
    if clock.elapsed < clock.budget and trainer.state.global_step < max_steps:
        if trainer.is_world_process_zero():
            print(
                f"Stopped at {_hms(clock.elapsed)} of {_hms(clock.budget)}; "
                "resuming continues from the last checkpoint",
                flush=True,
            )
        sys.exit(1)

    trainer.save_model(str(run_dir / "model"))
    if trainer.is_world_process_zero():
        summary = {
            "steps": trainer.state.global_step,
            "tokens": trainer.state.global_step * p["batch_tokens"],
            "training_seconds": clock.elapsed,
            "final_eval_loss": callback.final_eval_loss,
            "model_dir": str(run_dir / "model"),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"Done: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        jit_checkpoint="--jit-checkpoint" in sys.argv,
    )
