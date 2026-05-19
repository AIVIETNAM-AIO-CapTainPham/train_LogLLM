"""Download and extract the BGL log dataset.

Usage:
    uv run python scripts/download_bgl.py
    uv run python scripts/download_bgl.py --url https://example.com/BGL.zip

Outputs (under <repo>/data/):
    - BGL.zip / BGL.tar.gz  (downloaded archive)
    - BGL.log               (extracted log)

To train on only a slice of the log, set `start_line` / `end_line` in
prepareData/sliding_window.py — no need to pre-cut the file here.
"""
import argparse
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DEFAULT_URL = "https://zenodo.org/records/8196385/files/BGL.zip"


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size / 1024**2:.1f} MB)")
        return
    print(f"[download] {url}")

    def progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            sys.stdout.write(
                f"\r  {downloaded / 1024**2:.1f} / {total_size / 1024**2:.1f} MB ({pct:.1f}%)"
            )
        else:
            sys.stdout.write(f"\r  {downloaded / 1024**2:.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def extract(archive: Path, out_dir: Path) -> Path:
    log_path = out_dir / "BGL.log"
    if log_path.exists():
        print(f"[skip] {log_path.name} already extracted ({log_path.stat().st_size / 1024**2:.1f} MB)")
        return log_path

    print(f"[extract] {archive.name}")
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as tf:
            tf.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive.name}")

    candidates = [p for p in out_dir.rglob("BGL.log") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"BGL.log not found inside {archive.name}")
    found = candidates[0]
    if found != log_path:
        found.rename(log_path)
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & extract BGL dataset.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Archive URL to fetch.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    archive_name = Path(args.url).name or "BGL_archive"
    archive = DATA_DIR / archive_name
    download(args.url, archive)
    extract(archive, DATA_DIR)

    print(f"\n[done] Files in {DATA_DIR.relative_to(ROOT)}/:")
    for f in sorted(DATA_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name:30s}  {f.stat().st_size / 1024**2:8.1f} MB")


if __name__ == "__main__":
    main()
