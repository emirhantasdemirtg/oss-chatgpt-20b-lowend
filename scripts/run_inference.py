#!/usr/bin/env python3
"""
Run text generation with:
- bitsandbytes 4-bit quantization (bnb-4bit) for arbitrary HF models (default: EleutherAI/gpt-neox-20b)
- or pre-quantized GPTQ models by passing their repo id and --quantization gptq

Note: 20B on 4GB VRAM will offload most weights to CPU RAM. Ensure sufficient system RAM.
"""
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)


def load_bnb_4bit(model_id: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def load_gptq(model_id: str):
    # For many GPTQ repos, loading via transformers works if the repo exposes the right files.
    # Otherwise, consider using text-generation-webui or AutoGPTQ APIs.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def generate(model, tok, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    import threading
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    full = ""
    for chunk in streamer:
        print(chunk, end="", flush=True)
        full += chunk
    thread.join()
    print()
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--quantization", choices=["bnb-4bit", "gptq"], default="bnb-4bit")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    if args.quantization == "bnb-4bit":
        model, tok = load_bnb_4bit(args.model)
    else:
        model, tok = load_gptq(args.model)

    generate(model, tok, args.prompt, args.max_new_tokens, args.temperature)


if __name__ == "__main__":
    main()
