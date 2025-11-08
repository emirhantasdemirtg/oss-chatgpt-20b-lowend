# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This project runs 20B-parameter LLMs on commodity hardware through extreme quantization (4-bit) and CPU offload. The primary development path is **Google Colab** for quantization and adaptation, with local inference scripts for testing on low-VRAM systems.

**Key constraint**: 20B models at full precision require far more than 4GB VRAM. All workflows use 4-bit quantization (bitsandbytes or GPTQ). Fine-tuning 20B on low-end hardware is impractical—use Colab or target smaller models (7B/13B) for adaptation.

## Environment Setup

### Local Development
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Colab Setup
In notebook cells:
```python
!pip install -U transformers accelerate bitsandbytes peft datasets auto-gptq optimum safetensors huggingface_hub sentencepiece protobuf<5
```

## Common Commands

### Run Inference (Local)

**4-bit quantized inference with bitsandbytes:**
```bash
python scripts/run_inference.py --model EleutherAI/gpt-neox-20b --quantization bnb-4bit --prompt "Your prompt here"
```

**Using pre-quantized GPTQ model:**
```bash
python scripts/run_inference.py --model TheBloke/gpt-neox-20b-GPTQ --quantization gptq --prompt "Your prompt here"
```

**Additional options:**
- `--max-new-tokens <int>` (default: 256)
- `--temperature <float>` (default: 0.7)

### Download Pre-Quantized Models
```bash
python scripts/download_prequantized.py --repo TheBloke/gpt-neox-20b-GPTQ --out models/gpt-neox-20b-GPTQ
```

### CLI Interface
```bash
python -m src.oss_chat20b.cli --model EleutherAI/gpt-neox-20b --quantization bnb-4bit --prompt "Hello"
```

## Architecture

### Core Components

**scripts/run_inference.py**: Main inference script with two quantization paths:
- `load_bnb_4bit()`: Dynamic 4-bit quantization using bitsandbytes (NF4 format with double quantization)
- `load_gptq()`: Load pre-quantized GPTQ models from HuggingFace
- `generate()`: Streaming text generation with threading

**scripts/download_prequantized.py**: Downloads pre-quantized checkpoints from HuggingFace using `snapshot_download()`

**src/oss_chat20b/cli.py**: Alternative CLI interface, mirrors run_inference.py functionality with slightly different argument handling

### Quantization Strategy

All model loading uses `device_map="auto"` for automatic CPU/GPU memory management. With 4GB VRAM:
- Most model weights offload to system RAM (requires 24-32GB recommended)
- GPU handles compute for active layers
- Expect slow inference on laptops due to CPU/GPU memory transfers

Both quantization paths use:
- `torch.float16` for compute dtype
- `trust_remote_code=True` for model loading
- Tokenizer padding token fallback to EOS token
- Streaming generation via `TextIteratorStreamer`

### Colab Workflow

See `colab/QLoRA_20B_Colab.md` for detailed Colab setup. Key points:
- Colab T4 (16GB) can handle 20B in 4-bit for inference
- QLoRA fine-tuning on 20B is fragile even on T4—prefer 7B/13B models
- For adaptation: use low LoRA ranks (r=8/16), gradient checkpointing, paged optimizers, small batches

## Development Notes

### Working with Models

- **Base model**: EleutherAI/gpt-neox-20b (default)
- **Pre-quantized sources**: TheBloke's GPTQ repos on HuggingFace
- Models are not committed to git (ignored in `.gitignore`)

### Memory Requirements

- **Local inference**: 24-32GB system RAM minimum for CPU offload
- **Colab T4**: 16GB VRAM sufficient for 4-bit inference
- **Fine-tuning**: Use Colab or smaller models; 4GB VRAM insufficient for 20B adaptation

### Generation Parameters

Default parameters across scripts:
- `max_new_tokens=256`
- `temperature=0.7`
- `top_p=0.95`
- `repetition_penalty=1.1`
- `do_sample=True`

### Directory Structure

- `scripts/`: Standalone Python scripts for inference and model management
- `src/oss_chat20b/`: Package module with CLI interface
- `colab/`: Colab-specific guides and notebooks
- `configs/`: Empty, intended for future config files
- `data/`, `models/`, `notebooks/`: Ignored by git, for local artifacts
