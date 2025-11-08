# OSS 20B Chat Model for Low-End Laptops

This project explores running and adapting an open 20B-parameter LLM on commodity hardware with minimal VRAM using extreme quantization and CPU offload. It also includes a Colab path for quantization and light adaptation.

## Key Points
- 20B on 4GB VRAM is not feasible in full precision. We rely on 4-bit quantization and heavy CPU offload, or use Colab (T4/other GPUs) for setup.
- Recommended 20B base: EleutherAI/gpt-neox-20b. Pre-quantized variants (GPTQ/GGUF) from the community are preferred for low-VRAM inference.
- Fine-tuning 20B on 4GB VRAM is impractical. If adapting is required, use LoRA/QLoRA on Colab and/or target smaller backbones (7B/13B) and distill back.

## What's Included
- `scripts/run_inference.py`: Load a model in 4-bit (bitsandbytes) or use a pre-quantized GPTQ variant with CPU/GPU offload.
- `scripts/download_prequantized.py`: Fetch pre-quantized weights from Hugging Face.
- `src/oss_chat20b/cli.py`: Simple CLI for prompting models.
- `colab/QLoRA_20B_Colab.md`: Practical notes and steps for Colab.

## Quick Start

### Option 1: Run on Google Colab (Recommended for 20B models)

Colab provides free GPU access (T4 with 16GB VRAM) which is ideal for running 20B models in 4-bit quantization.

#### Method A: Clone and Run from GitHub

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Create a new notebook**: Click "New Notebook"

3. **Enable GPU runtime**:
   - Click `Runtime` → `Change runtime type`
   - Select `T4 GPU` (or any available GPU)
   - Click `Save`

4. **Clone this repository and install dependencies**:
   ```python
   !git clone https://github.com/emirhantasdemirtg/oss-chatgpt-20b-lowend.git
   %cd oss-chatgpt-20b-lowend
   !pip install -q -r requirements.txt
   ```

5. **Run inference**:
   ```python
   !python scripts/run_inference.py --model EleutherAI/gpt-neox-20b --quantization bnb-4bit --prompt "Explain quantum computing in simple terms:"
   ```

#### Method B: Quick Start with Code Snippet

If you just want to experiment without cloning the repo:

1. **Open Google Colab** and enable GPU (see steps 1-3 above)

2. **Install dependencies**:
   ```python
   !pip install -q transformers accelerate bitsandbytes torch
   ```

3. **Run this code**:
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
   import torch

   # Load model in 4-bit
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.float16,
       bnb_4bit_use_double_quant=True,
   )

   print("Loading tokenizer...")
   tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
   
   print("Loading model (this may take a few minutes)...")
   model = AutoModelForCausalLM.from_pretrained(
       "EleutherAI/gpt-neox-20b",
       quantization_config=bnb_config,
       device_map="auto",
       trust_remote_code=True
   )

   # Generate text
   prompt = "Explain quantum computing in simple terms:"
   inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
   outputs = model.generate(
       **inputs, 
       max_new_tokens=256, 
       temperature=0.7,
       do_sample=True
   )
   
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

#### Using Pre-quantized GPTQ Models (Faster Loading)

```python
!python scripts/run_inference.py --model TheBloke/gpt-neox-20b-GPTQ --quantization gptq --prompt "Hello!"
```

### Option 2: Run Locally (Requires 24-32GB RAM)

1. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run 4-bit inference with CPU offload**:
   ```bash
   python scripts/run_inference.py --model EleutherAI/gpt-neox-20b --quantization bnb-4bit --prompt "Hello!"
   ```

3. **Or use a pre-quantized GPTQ model**:
   ```bash
   python scripts/run_inference.py --model TheBloke/gpt-neox-20b-GPTQ --quantization gptq --prompt "Hello!"
   ```

## Advanced Usage

### Download Pre-quantized Models
```bash
python scripts/download_prequantized.py --repo TheBloke/gpt-neox-20b-GPTQ --out models/gpt-neox-20b-GPTQ
```

### Command-line Options
```bash
python scripts/run_inference.py \
  --model EleutherAI/gpt-neox-20b \
  --quantization bnb-4bit \
  --prompt "Your prompt here" \
  --max-new-tokens 512 \
  --temperature 0.8
```

### Alternative CLI
```bash
python -m src.oss_chat20b.cli --model EleutherAI/gpt-neox-20b --quantization bnb-4bit --prompt "Hello"
```

## Notes
- **Colab users**: First model load takes 5-10 minutes to download weights (~40GB). Subsequent runs are faster.
- **Local users**: Expect slow inference on laptops; ensure you have enough system RAM (≥24–32GB recommended for 20B at 4-bit with offload).
- **Fine-tuning**: See `colab/QLoRA_20B_Colab.md` for details. Consider 7B/13B backbones for practicality.
- **Memory management**: The scripts use `device_map="auto"` to automatically distribute model layers between GPU and CPU RAM.

## Troubleshooting

- **Out of Memory on Colab**: Try using a pre-quantized GPTQ model or restart the runtime
- **Slow local inference**: This is expected with CPU offload. Use Colab for better performance.
- **Model download issues**: Ensure stable internet connection; downloads are resumed automatically if interrupted.
