"""Download datasets for DensityChunker experiments.

Usage:
    python scripts/download_data.py              # Download all datasets
    python scripts/download_data.py --legalbench # Download LegalBench-RAG only
    python scripts/download_data.py --narrativeqa # Download NarrativeQA only
    python scripts/download_data.py --mini        # Download LegalBench-RAG-mini only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Project root is one level up from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def download_legalbench_rag(mini_only: bool = False):
    """Download LegalBench-RAG from GitHub + Dropbox."""
    target_dir = DATASETS_DIR / "legalbench_rag"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Clone the repo (evaluation code + structure)
    repo_dir = target_dir / "legalbenchrag_repo"
    if not repo_dir.exists():
        print("Cloning LegalBench-RAG repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/zeroentropy-ai/legalbenchrag.git", str(repo_dir)],
            check=True,
        )
    else:
        print(f"Repository already exists at {repo_dir}")

    # Download data from Dropbox
    data_dir = target_dir / "data"
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"Data directory already exists at {data_dir}")
        print("Checking for corpus and benchmarks...")
    else:
        print("Downloading LegalBench-RAG data from Dropbox...")
        print("Note: If automatic download fails, please manually download from:")
        print("  https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=0")
        print("  and extract into: datasets/legalbench_rag/data/")

        # Try using the repo's download script if available
        download_script = repo_dir / "download_data.sh"
        if download_script.exists():
            print(f"Found download script at {download_script}, running it...")
            try:
                subprocess.run(
                    ["bash", str(download_script)],
                    cwd=str(target_dir),
                    check=True,
                )
            except subprocess.CalledProcessError:
                print("Download script failed. Manual download required.")
        else:
            # Try direct download via Python
            _try_dropbox_download(target_dir, data_dir)

    # Also try loading via HuggingFace as a fallback/supplement
    _try_huggingface_legalbench(target_dir, mini_only)

    # Verify download
    _verify_legalbench(target_dir)


def _try_dropbox_download(target_dir: Path, data_dir: Path):
    """Attempt to download LegalBench-RAG data from Dropbox."""
    import zipfile
    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)

    # Dropbox direct download link (dl=1 forces download)
    dropbox_url = (
        "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/"
        "AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=1"
    )

    zip_path = target_dir / "legalbenchrag_data.zip"

    try:
        print(f"Downloading from Dropbox to {zip_path}...")
        urllib.request.urlretrieve(dropbox_url, str(zip_path))
        print("Download complete. Extracting...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(data_dir))
        print("Extraction complete.")
        zip_path.unlink()
    except Exception as e:
        print(f"Dropbox download failed: {e}")
        print("Please download manually from the Dropbox link.")
        if zip_path.exists():
            zip_path.unlink()


def _try_huggingface_legalbench(target_dir: Path, mini_only: bool):
    """Try loading LegalBench-RAG from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("HuggingFace datasets not installed. Skipping HuggingFace download.")
        return

    hf_dir = target_dir / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)

    domains = ["cuad", "contractnli", "maud", "privacyqa"]

    for domain in domains:
        hf_name = f"orgrctera/legalbenchrag_{domain}"
        out_file = hf_dir / f"{domain}.json"
        if out_file.exists():
            print(f"  {domain} already downloaded, skipping.")
            continue

        try:
            print(f"  Downloading {domain} from HuggingFace...")
            ds = load_dataset(hf_name, split="test")
            # Save as JSON for easy loading
            records = []
            for row in ds:
                records.append(dict(row))
            out_file.write_text(json.dumps(records, indent=2))
            print(f"  Saved {len(records)} records to {out_file}")
        except Exception as e:
            print(f"  Failed to download {domain}: {e}")
            print(f"  Continuing with other domains...")

    # Also download the mini versions if they exist
    if mini_only:
        for domain in domains:
            hf_name = f"orgrctera/legalbenchrag_{domain}_mini"
            out_file = hf_dir / f"{domain}_mini.json"
            if out_file.exists():
                continue
            try:
                ds = load_dataset(hf_name, split="test")
                records = [dict(row) for row in ds]
                out_file.write_text(json.dumps(records, indent=2))
                print(f"  Saved {len(records)} mini records to {out_file}")
            except Exception:
                pass


def _verify_legalbench(target_dir: Path):
    """Verify that LegalBench-RAG data is available."""
    data_dir = target_dir / "data"
    hf_dir = target_dir / "huggingface"

    corpus_dir = data_dir / "corpus" if data_dir.exists() else None
    benchmarks_dir = data_dir / "benchmarks" if data_dir.exists() else None

    print("\n--- LegalBench-RAG Verification ---")

    # Check corpus
    if corpus_dir and corpus_dir.exists():
        corpus_files = []
        for sub in corpus_dir.iterdir():
            if sub.is_dir():
                corpus_files.extend(sub.glob("*.txt"))
            elif sub.suffix == ".txt":
                corpus_files.append(sub)
        print(f"Corpus: {len(corpus_files)} text files")
        if corpus_files:
            total_chars = sum(f.stat().st_size for f in corpus_files)
            print(f"Total corpus size: {total_chars / 1e6:.1f}M characters")
    else:
        print("Corpus directory not found (will use HuggingFace data)")

    # Check benchmarks
    if benchmarks_dir and benchmarks_dir.exists():
        bench_files = list(benchmarks_dir.glob("*.json"))
        print(f"Benchmarks: {len(bench_files)} JSON files")
        for bf in bench_files:
            data = json.loads(bf.read_text())
            print(f"  {bf.name}: {len(data)} queries")
    else:
        print("Benchmarks directory not found (will use HuggingFace data)")

    # Check HuggingFace downloads
    if hf_dir.exists():
        hf_files = list(hf_dir.glob("*.json"))
        print(f"HuggingFace cache: {len(hf_files)} domain files")
        for hf in hf_files:
            data = json.loads(hf.read_text())
            print(f"  {hf.stem}: {len(data)} records")


def download_narrativeqa(subsample: int = 0):
    """Download NarrativeQA from LongBench (preprocessed, no broken URLs).

    Args:
        subsample: If > 0, only keep this many samples. 0 means keep all.
    """
    from datasets import load_dataset

    target_dir = DATASETS_DIR / "narrativeqa"
    target_dir.mkdir(parents=True, exist_ok=True)

    out_file = target_dir / "longbench_narrativeqa.json"

    if out_file.exists():
        existing = json.loads(out_file.read_text())
        print(f"NarrativeQA (LongBench) already downloaded: {len(existing)} samples")
        print(f"  Saved at: {out_file}")
        return

    print("Downloading NarrativeQA from THUDM/LongBench (narrativeqa subset)...")
    ds = load_dataset("THUDM/LongBench", "narrativeqa", split="test")

    samples = []
    for row in ds:
        sample = {
            "id": row.get("_id", ""),
            "question": row["input"],
            "context": row["context"],
            "answers": row["answers"],
            "dataset": row.get("dataset", "narrativeqa"),
            "length": row.get("length", 0),
        }
        samples.append(sample)

    if subsample > 0 and subsample < len(samples):
        print(f"  Subsampling from {len(samples)} to {subsample} samples...")
        import random
        random.seed(42)
        samples = random.sample(samples, subsample)

    out_file.write_text(json.dumps(samples, indent=2))
    print(f"Saved {len(samples)} samples to {out_file}")

    # Stats
    total_chars = sum(len(s["context"]) for s in samples)
    avg_chars = total_chars / len(samples) if samples else 0
    print(f"  Total context text: {total_chars / 1e6:.1f}M characters")
    print(f"  Average context length: {avg_chars:.0f} characters ({avg_chars / 4:.0f} tokens approx)")
    print(f"  Total questions: {len(samples)}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for DensityChunker")
    parser.add_argument("--legalbench", action="store_true", help="Download LegalBench-RAG only")
    parser.add_argument("--narrativeqa", action="store_true", help="Download NarrativeQA only")
    parser.add_argument("--mini", action="store_true", help="Download mini version only (LegalBench)")
    parser.add_argument("--narrativeqa-samples", type=int, default=0,
                        help="Subsample NarrativeQA to N samples (0 = all)")
    args = parser.parse_args()

    # If no specific flag, download all
    download_all = not args.legalbench and not args.narrativeqa

    if download_all or args.legalbench:
        print("=" * 60)
        print("Downloading LegalBench-RAG...")
        print("=" * 60)
        download_legalbench_rag(mini_only=args.mini)

    if download_all or args.narrativeqa:
        print("\n" + "=" * 60)
        print("Downloading NarrativeQA (LongBench)...")
        print("=" * 60)
        download_narrativeqa(subsample=args.narrativeqa_samples)

    print("\nDone!")


if __name__ == "__main__":
    main()