#!/usr/bin/env python3
import argparse
import os
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import torch


def load_model(model_id: str, quantization: str = "bnb-4bit"):
    device_map = "auto"
    torch_dtype = torch.float16

    if quantization == "bnb-4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    else:
        # Fallback: no in-graph quantization (use pre-quantized weights like GPTQ via model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
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

    # Stream tokens
    import threading
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    text = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        text += new_text
    thread.join()
    print()
    return text


def main():
    ap = argparse.ArgumentParser(description="Simple CLI for OSS 20B chat/inference")
    ap.add_argument("--model", default="EleutherAI/gpt-neox-20b", help="HF model id (or local path)")
    ap.add_argument("--quantization", choices=["bnb-4bit", "none"], default="bnb-4bit",
                    help="Use bitsandbytes 4-bit or load model as-is (for pre-quantized GPTQ, pass its repo id and choose 'none')")
    ap.add_argument("--prompt", required=True, help="Prompt text")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    model, tok = load_model(args.model, args.quantization)
    generate(model, tok, args.prompt, args.max_new_tokens, args.temperature)


if __name__ == "__main__":
    main()
