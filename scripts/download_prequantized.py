#!/usr/bin/env python3
"""
Download pre-quantized checkpoints (e.g., GPTQ) from Hugging Face.
Example:
  python scripts/download_prequantized.py --repo TheBloke/gpt-neox-20b-GPTQ --out models/gpt-neox-20b-GPTQ
"""
import argparse
from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id with pre-quantized weights (e.g., TheBloke/...) ")
    ap.add_argument("--out", required=True, help="Local output directory")
    args = ap.parse_args()

    snapshot_download(repo_id=args.repo, local_dir=args.out, resume_download=True)
    print(f"Downloaded {args.repo} to {args.out}")


if __name__ == "__main__":
    main()
