# Language Model Training

Train a Llama-style language model from one number: the compute budget in GPU-hours.

```bash
uv sync
uv run lmt train 8
```

`lmt` measures your GPU, works out the model size, token count, batch size, and learning
rate that fit 8 GPU-hours, shows you the plan, and trains after you confirm. It is a thin
layer over Hugging Face `transformers` (`LlamaConfig`, `Trainer`), `tokenizers`, and
`datasets`. All settings live in [`lmt.yaml`](lmt.yaml).

## Commands

```bash
uv run lmt train 8                  # measure, show the plan, ask, train
uv run lmt train 8 --yes            # same, without asking
uv run lmt plan 8                   # measure and show the plan only
uv run lmt prepare --tokens 2e9     # train the tokenizer and tokenize ahead of time
uv run lmt resume <run name>        # continue a run from its last checkpoint
uv run lmt train 8 --backend modal --gpu "H100!:4"   # run on Modal (see below)
```

GPU-hours mean wall-clock hours × GPUs: 8 GPU-hours on 4 GPUs is 2 hours of training.
Every run gets a folder in `runs/` with its frozen `config.yaml`, `plan.json`, checkpoints,
the final `model/` (loadable with `AutoModelForCausalLM.from_pretrained`), and `summary.json`.

## How the budget becomes a plan

With a fixed tokens-per-parameter ratio `r`, compute `C`, parameters `N`, and tokens `D`:

```
C = seconds × GPUs × measured FLOP/s per GPU
C = FLOPs_per_token(shape) × D,   D = r × N
```

Model shape comes from depth: `hidden = aspect_ratio × layers` (rounded to 128),
`heads = hidden / head_dim`, `ffn = 8/3 × hidden`. The solver scans depths for the shape
whose compute-optimal cost is closest to `C`.

Throughput depends on the shape, so the planner measures as it goes:

1. Measure bf16 matmul speed and guess achieved FLOP/s as `initial_mfu` of it.
2. Solve for a shape and benchmark real `Trainer` steps on it, at the largest micro-batch
   that fits and at half that size, keeping the faster.
3. Use the measured FLOP/s to solve again. Repeat until the shape stops changing.

Batch size and learning rate then follow from `C`. Every derived value can be pinned in
`lmt.yaml`, for example `training.learning_rate: 1e-3` or `model.llama.num_hidden_layers: 12`,
and the solver fills in the rest.

**The budget is enforced at runtime, not just predicted.** The LR schedule is
warmup-stable-decay. The decay starts when 80% of the budget's *training time* is used,
and training stops when all of it is used. Plans that are off, slow data, or interruptions
change how many tokens are trained, not how long training runs. Time spent on evaluation
and checkpointing is not counted.

## Checkpoints and resuming

A checkpoint is saved every `checkpoint_every_minutes` of training time with the full
`Trainer` state: model, optimizer, scheduler, RNG, data position, and the training clock.
`lmt resume <run>` continues from it exactly, with the same data order and the same place
in the LR schedule. A checkpoint cut off mid-save is skipped.

## Data and tokenizer

`lmt` trains a byte-level BPE tokenizer (GPT-4 split pattern) on the first
`tokenizer.train_chars` characters of the dataset. It then streams the dataset into
`uint16` token shards in `data/<dataset>--<subset>--v<vocab>/`. Shards only grow: a
bigger run tokenizes just the tokens it is missing. Changing the dataset, subset, or
vocab size gives a new folder.

## Running on Modal

```bash
uv sync --extra modal
uv run modal setup                  # one-time login
uv run lmt train 8 --backend modal --gpu "H100!:4"
```

- **Container.** Built from this project's `uv.lock`. Your code is uploaded on each run,
  and logs stream to your terminal.
- **What runs where.** Planning runs on the same GPU type as training. The confirm prompt
  runs locally, so no GPU waits on you.
- **Storage.** Tokenizer, data, and runs live on a Modal Volume (`compute.modal.volume`)
  with the same layout as locally.
- **Tokenization.** The tokenizer is trained locally and uploaded. Token shards are made
  locally and uploaded when the run needs at most 1B tokens, otherwise on a CPU container
  (`--prepare-on` overrides this).
- **Preemption and long runs.** GPU containers can be preempted. Modal retries the call
  (`max_retries`), and training resumes from the last checkpoint. Runs longer than 24 hours
  continue the same way after each 24-hour attempt times out.
- **Detach.** With `--detach`, training keeps going if you close the terminal. Follow it
  with `modal app logs lm-training`.
- **Getting the model.** Download it with
  `modal volume get lm-training runs/<run>/model ./runs/<run>`, which creates
  `./runs/<run>/model/`. The local `./runs/<run>` folder must already exist; if it
  doesn't, Modal writes every file over a single file at that path.

Pin the GPU type: a plain `H100` may be upgraded to an H200, which would no longer match
the benchmark. `H100!` prevents this.

## Where the defaults come from

| Setting | Default | Source |
|---|---|---|
| tokens per parameter | 20, total params | Chinchilla ([Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)). Count total params, not non-embedding ([Pearce & Song 2024](https://arxiv.org/abs/2406.12907)) |
| learning rate | `0.3118 · C^-0.125` | [DeepSeek LLM](https://arxiv.org/abs/2401.02954), fit for AdamW at 1e17–2e19 FLOPs |
| batch size | `0.2920 · C^0.3271` tokens | DeepSeek LLM (gives 9.2M tokens for their 7B run, which used 9.4M) |
| schedule | warmup-stable-decay, 20% decay, 1-sqrt to 0 | [Hägele et al. 2024](https://arxiv.org/abs/2405.18392): 20% cooldown matches cosine, 1-sqrt beats linear |
| warmup | 2% of steps, at most 2000 | DeepSeek LLM and SmolLM2 use 2000 steps |
| AdamW | β = (0.9, 0.95), wd 0.1, clip 1.0 | Llama 2/3, DeepSeek LLM, OLMo 2 |
| width / depth | 64 | [nanochat](https://github.com/karpathy/nanochat). Real models span 19 (SmolLM2-135M) to 128 (Llama 3 8B), and Kaplan et al. found loss is flat across that range |
| head_dim | 64 | SmolLM2, Llama 3.2 1B, Qwen2.5-0.5B |
| FFN | 8/3 × hidden, SwiGLU | Llama, SmolLM2 |
| KV heads | equal to query heads | nanochat, OLMo 2, SmolLM2-1.7B. GQA mainly saves inference memory |
| tied embeddings | yes | SmolLM2, Qwen2.5/3 small models, Llama 3.2 1B/3B |
| vocab | 32768 | nanochat. Fits `uint16` |
| sequence length | 2048 | nanochat, SmolLM2 pre-training |
| FLOPs per token | 6 × matmul params + 12·L·d·T | PaLM appendix B, nanochat |

## Example plans

```bash
uv run lmt plan 8

Measuring the GPU...
  1 x NVIDIA GeForce RTX 5070 Ti, 17 GB, bf16 matmul 99 TFLOP/s per GPU (no datasheet peak on file, so no MFU)
Benchmarking 10 layers x 640 wide (71.8M params)...
  68,421 tok/s, 40% of matmul -> 27.5 tokens/param in budget

Plan
  Budget    8 GPU-hours = 8:00:00 on 1 x NVIDIA GeForce RTX 5070 Ti
  Measured  bf16 matmul 99 TFLOP/s per GPU (no datasheet peak on file, so no MFU)
            training 40 TFLOP/s per GPU, 40% of matmul, 68,421 tok/s, peak memory 7.3 GB
  Model     71.78M params (50.80M non-embedding)
            10 layers, hidden 640, 10 heads / 10 KV x 64, ffn 1792, vocab 32768, tied embeddings
  Data      1.97B tokens = 27.5 tokens/param, 1.16e+18 FLOPs
  Batch     237.57K tokens = 4 seqs x 1 GPUs x 29 accum x 2048
  Schedule  ~8,294 steps of 3.47s, peak LR 1.72e-03, 166 warmup steps, WSD decay by wall-clock
  Compile   torch.compile off
  Planning  took 38s over 1 benchmark(s)
```

```bash
uv run lmt plan 8 --backend modal --gpu "H100!:4"

Measuring the GPU...
  4 x NVIDIA H100 80GB HBM3, 85 GB, bf16 matmul 799 TFLOP/s per GPU (datasheet peak 989)
Benchmarking 16 layers x 1024 wide (239.1M params)...
  716,002 tok/s, MFU 33% (41% of matmul) -> 21.6 tokens/param in budget

Plan
  Budget    8 GPU-hours = 2:00:00 on 4 x NVIDIA H100 80GB HBM3
  Measured  bf16 matmul 799 TFLOP/s per GPU (datasheet peak 989)
            training 329 TFLOP/s per GPU, MFU 33% (41% of matmul), 716,002 tok/s, peak memory 33.9 GB
  Model     239.11M params (205.55M non-embedding)
            16 layers, hidden 1024, 16 heads / 16 KV x 64, ffn 2816, vocab 32768, tied embeddings
  Data      5.16B tokens = 21.6 tokens/param, 9.47e+18 FLOPs
  Batch     524.29K tokens = 16 seqs x 4 GPUs x 4 accum x 2048
  Schedule  ~9,832 steps of 0.73s, peak LR 1.32e-03, 197 warmup steps, WSD decay by wall-clock
  Compile   torch.compile on
  Planning  took 116s over 1 benchmark(s)
```

## Limits

- **Small budgets have coarse shapes.** Depths are whole numbers, so at small budgets
  neighbouring shapes differ by up to 2× in size. The planner picks the one closest to
  the target ratio, and the plan shows the ratio it actually achieved.
- **Single node only.** Multi-GPU uses `torchrun` on one machine, up to 8 GPUs on Modal.
  On Windows, use one GPU (no NCCL); for more, use Linux or WSL2.
- **No `torch.compile` on Windows.** Triton isn't installed there, so `torch_compile: auto`
  turns it off. It is on for Linux and Modal.
