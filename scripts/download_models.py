"""Download base models needed for LogLLM training.

Defaults:
    - google-bert/bert-base-uncased   -> models/bert-base-uncased   (~440 MB)
    - meta-llama/Llama-3.1-8B         -> models/Llama-3.1-8B        (~16 GB, gated)

Llama-3.1-8B is a gated repo. You must:
    1. Request access at https://huggingface.co/meta-llama/Llama-3.1-8B
    2. Provide your HF token in ONE of these ways (in order of priority):
       a. `HF_TOKEN` in a `.env` file at repo root  (copy from .env.example)
       b. `HF_TOKEN` exported as an env var
       c. Cached token from `hf auth login` / `huggingface-cli login`

Usage:
    uv run python scripts/download_models.py
    uv run python scripts/download_models.py --llama meta-llama/Meta-Llama-3-8B-Instruct
    uv run python scripts/download_models.py --skip-llama
    uv run python scripts/download_models.py --skip-bert
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

load_dotenv(ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

DEFAULT_BERT = "google-bert/bert-base-uncased"
DEFAULT_LLAMA = "meta-llama/Llama-3.1-8B"


def folder_has_files(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.safetensors")) or any(path.rglob("*.bin"))


def download_model(repo_id: str, local_dir: Path) -> None:
    if folder_has_files(local_dir):
        size_mb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1024**2
        print(f"[skip] {local_dir.name} already exists ({size_mb:.1f} MB)")
        return
    print(f"[download] {repo_id} -> {local_dir.relative_to(ROOT)}")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), token=HF_TOKEN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LogLLM base models.")
    parser.add_argument("--bert", default=DEFAULT_BERT, help="BERT repo id.")
    parser.add_argument("--llama", default=DEFAULT_LLAMA, help="Llama repo id.")
    parser.add_argument("--skip-bert", action="store_true", help="Skip BERT download.")
    parser.add_argument("--skip-llama", action="store_true", help="Skip Llama download.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_bert:
        download_model(args.bert, MODELS_DIR / args.bert.split("/")[-1])

    if not args.skip_llama:
        download_model(args.llama, MODELS_DIR / args.llama.split("/")[-1])

    print(f"\n[done] Models under {MODELS_DIR.relative_to(ROOT)}/:")
    for d in sorted(MODELS_DIR.iterdir()):
        if d.is_dir():
            size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024**2
            print(f"  {d.name:35s}  {size_mb:9.1f} MB")


if __name__ == "__main__":
    main()
