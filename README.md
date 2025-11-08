# OSS 20B Chat Model for Low-End Laptops

This project explores running and adapting an open 20B-parameter LLM on commodity hardware with minimal VRAM using extreme quantization and CPU offload. It also includes a Colab path for quantization and light adaptation.

Key points
- 20B on 4GB VRAM is not feasible in full precision. We rely on 4-bit quantization and heavy CPU offload, or use Colab (T4/other GPUs) for setup.
- Recommended 20B base: EleutherAI/gpt-neox-20b. Pre-quantized variants (GPTQ/GGUF) from the community are preferred for low-VRAM inference.
- Fine-tuning 20B on 4GB VRAM is impractical. If adapting is required, use LoRA/QLoRA on Colab and/or target smaller backbones (7B/13B) and distill back.

What’s included
- scripts/run_inference.py: Load a model in 4-bit (bitsandbytes) or use a pre-quantized GPTQ variant with CPU/GPU offload.
- scripts/download_prequantized.py: Fetch pre-quantized weights from Hugging Face.
- src/oss_chat20b/cli.py: Simple CLI for prompting models.
- colab/QLoRA_20B_Colab.md: Practical notes and steps for Colab.

Quickstart
1) Create a venv and install deps:
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
2) Run 4-bit inference (CPU offload + any available VRAM):
   python scripts/run_inference.py --model EleutherAI/gpt-neox-20b --quantization bnb-4bit --prompt "Hello!"
3) Or use a pre-quantized GPTQ model:
   python scripts/run_inference.py --model TheBloke/gpt-neox-20b-GPTQ --quantization gptq --prompt "Hello!"

Notes
- Expect slow inference on laptops; ensure you have enough system RAM (>=24–32GB recommended for 20B at 4-bit with offload).
- For fine-tuning, see the Colab guide; consider 7B/13B backbones for practicality.
