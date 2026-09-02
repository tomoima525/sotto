"""Model download and cache management.

Whisper/LLM models come from the HuggingFace hub cache; the ReazonSpeech
ONNX model is a sherpa-onnx GitHub release tarball extracted under the app's
config directory (it is not distributed on the hub in that form).
"""

from __future__ import annotations

import glob
import logging
import shutil
import tarfile
import threading
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

log = logging.getLogger(__name__)


def is_cached(repo_id: str) -> bool:
    """Check if a model repo is fully available in the local HF cache (offline probe)."""
    try:
        snapshot_download(repo_id, local_files_only=True)
        return True
    except (LocalEntryNotFoundError, FileNotFoundError, OSError):
        return False


def download(repo_id: str) -> str:
    """Return the local snapshot path, downloading only if not fully cached.

    Cache-first so launches never touch the network (or fail offline) once
    the models are present.
    """
    try:
        return snapshot_download(repo_id, local_files_only=True)
    except (LocalEntryNotFoundError, FileNotFoundError, OSError):
        log.info("Model not cached, downloading: %s", repo_id)
        return snapshot_download(repo_id)


def resolve_whisper_path(repo_id: str) -> str:
    """Resolve a whisper repo to a local path mlx-whisper can load.

    mlx-whisper 0.4.x only looks for weights.safetensors/weights.npz, but some
    newer mlx-community repos ship model.safetensors — symlink it into place.
    """
    from pathlib import Path

    path = Path(download(repo_id))
    if not (path / "weights.safetensors").exists() and not (path / "weights.npz").exists():
        model_file = path / "model.safetensors"
        if model_file.exists():
            (path / "weights.safetensors").symlink_to(model_file)
            log.info("Symlinked weights.safetensors -> model.safetensors in %s", path)
    return str(path)


def ensure_cached(repo_ids: list[str], progress_cb=None) -> None:
    """Download any missing repos. progress_cb(repo_id, i, total) is called before each."""
    missing = [r for r in repo_ids if not is_cached(r)]
    for i, repo_id in enumerate(missing):
        if progress_cb:
            progress_cb(repo_id, i, len(missing))
        download(repo_id)


# -- ReazonSpeech (sherpa-onnx GitHub release tarball, ~440MB) --

REAZON_DIR_NAME = "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17"
REAZON_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{REAZON_DIR_NAME}.tar.bz2"
)
# The four files sherpa-onnx's transducer loader needs; exact basenames vary
# between exports, so resolve them by pattern like hayamimi does.
_REAZON_FILE_PATTERNS = {
    "encoder": "encoder-*.int8.onnx",
    "decoder": "decoder-*.int8.onnx",
    "joiner": "joiner-*.int8.onnx",
    "tokens": "tokens.txt",
}
# Serializes concurrent download attempts (menu selection racing a dictation's
# lazy load) so two writers never share one .part file.
_reazon_lock = threading.Lock()


def reazon_model_dir() -> Path:
    from .config import CONFIG_DIR

    return CONFIG_DIR / "models" / REAZON_DIR_NAME


def reazon_files() -> dict[str, str] | None:
    """Resolve the model's files, or None if any is missing (not downloaded)."""
    d = reazon_model_dir()
    out: dict[str, str] = {}
    for key, pattern in _REAZON_FILE_PATTERNS.items():
        hits = sorted(glob.glob(str(d / pattern)))
        if not hits:
            return None
        out[key] = hits[0]
    return out


def reazon_is_cached() -> bool:
    return reazon_files() is not None


def download_reazon(progress_cb=None) -> Path:
    """Download and extract the ReazonSpeech tarball. Idempotent and safe to
    call from multiple threads; progress_cb(bytes_read, bytes_total) if given.
    """
    with _reazon_lock:
        target = reazon_model_dir()
        if reazon_is_cached():
            return target
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp = parent / f".{REAZON_DIR_NAME}.tar.bz2.part"
        log.info("Downloading ReazonSpeech model (~440MB) from %s", REAZON_MODEL_URL)
        req = urllib.request.Request(
            REAZON_MODEL_URL, headers={"User-Agent": "sotto-download/1.0"}
        )
        try:
            with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
                total = int(resp.headers.get("Content-Length", 0))
                read = 0
                while True:
                    buf = resp.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    read += len(buf)
                    if progress_cb:
                        progress_cb(read, total)
            log.info("Extracting to %s", target)
            # Extract into a staging dir and rename into place, so an
            # interrupted extraction can never leave a truncated .onnx at the
            # final path (sherpa-onnx would then fail with an opaque
            # onnxruntime/protobuf error every launch).
            staging = parent / f".{REAZON_DIR_NAME}.extracting"
            shutil.rmtree(staging, ignore_errors=True)
            try:
                with tarfile.open(tmp, "r:bz2") as tf:
                    # Single top-level directory named after the tarball; the
                    # "data" filter refuses path traversal and other surprises.
                    top = tf.getmembers()[0].name.split("/")[0]
                    tf.extractall(staging, filter="data")
                shutil.rmtree(target, ignore_errors=True)  # stale partial dir
                (staging / top).replace(target)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        finally:
            tmp.unlink(missing_ok=True)
        if not reazon_is_cached():
            raise RuntimeError(
                f"ReazonSpeech download finished but model files are missing "
                f"under {target}"
            )
        return target
