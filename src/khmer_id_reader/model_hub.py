"""Download models from HuggingFace on first use."""

import os
import sys
from pathlib import Path

HF_REPO = "vattanac/khmer-ocr-crnn-ctc"
MODEL_FILES = [
    "khmer_ocr_document.pt",
    "khmer_ocr_handwriting.pt",
    "khmer_lm.pkl.gz",
    "vocab.json",
    "khmer_ocr.py",
]


def _cache_dir() -> Path:
    d = Path(os.environ.get(
        "KHMER_ID_READER_CACHE",
        Path.home() / ".cache" / "khmer-id-reader" / "models"
    ))
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_models() -> Path:
    """Download models if not cached. Returns the cache directory."""
    cache = _cache_dir()
    missing = [f for f in MODEL_FILES if not (cache / f).exists()]

    if not missing:
        return cache

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Install huggingface-hub to auto-download models:\n"
            "  pip install huggingface-hub\n"
            "Or manually download from https://huggingface.co/vattanac/khmer-ocr-crnn-ctc",
            file=sys.stderr,
        )
        raise

    print(f"Downloading {len(missing)} model file(s) from {HF_REPO}...")
    for fname in missing:
        print(f"  > {fname}")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=fname,
            local_dir=str(cache),
        )
    print("Done. Models cached at", cache)
    return cache
