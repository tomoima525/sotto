"""Model download and cache management (HuggingFace hub)."""

from __future__ import annotations

import logging

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
    """Download a model repo (no-op if cached). Returns the local snapshot path."""
    log.info("Ensuring model is cached: %s", repo_id)
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
