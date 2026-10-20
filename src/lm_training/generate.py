"""Continue prompts with a trained model, one at a time.

Each prompt starts a new document: nothing carries over from earlier prompts.
The model is a base model, so it continues text rather than answering it.
"""

from pathlib import Path

import torch
from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    TextStreamer,
)


def loop(
    model_dir: Path, max_new_tokens: int, temperature: float, top_p: float
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(model_dir, dtype=dtype, device_map=device)
    model.eval()
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )

    print("Empty line or Ctrl+D (Ctrl+Z, Enter on Windows) quits; Ctrl+C stops early.")
    while True:
        try:
            prompt = input("\n> ")
        except EOFError:
            break
        if not prompt:
            break
        # Training data separates documents with <|endoftext|>, so leading with it
        # tells the model a new document starts here.
        ids = tok(str(tok.eos_token) + prompt, return_tensors="pt").input_ids.to(device)
        print(prompt, end="", flush=True)
        try:
            # transformers' self-type for generate() doesn't match its own models.
            model.generate(  # ty: ignore[invalid-argument-type]
                ids,
                generation_config=gen,
                attention_mask=torch.ones_like(ids),
                streamer=streamer,
            )
        except KeyboardInterrupt:
            print()
