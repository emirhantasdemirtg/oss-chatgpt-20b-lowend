# QLoRA and Quantization on Colab (Practical Guide)

Reality check
- 20B models are too large for 4GB VRAM. Use heavy quantization and CPU offload locally, or use Colab GPU to prepare models.
- Colab T4 (16GB) can often run 20B in 4-bit for inference and sometimes light LoRA experiments with careful settings, but it will be slow and may OOM. 7B/13B are far more practical.

Setup
- In Colab: Runtime -> Change runtime type -> GPU.
- Install deps:
  !pip install -U transformers accelerate bitsandbytes peft datasets auto-gptq optimum safetensors huggingface_hub sentencepiece protobuf<5

Download model (example)
- Base 20B: EleutherAI/gpt-neox-20b
  from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
  bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True)
  tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=True)
  model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", quantization_config=bnb, device_map="auto", trust_remote_code=True)

Inference
  from transformers import TextStreamer
  prompt = "Hello!"
  inputs = tok(prompt, return_tensors="pt").to(model.device)
  out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
  print(tok.decode(out[0], skip_special_tokens=True))

LoRA notes
- For 20B, even QLoRA can be fragile on T4. Prefer 7B/13B for fine-tuning and then distill/compress.
- If you attempt QLoRA: use low ranks (r=8/16), gradient checkpointing, paged optimizers, small batch sizes, short sequences.
