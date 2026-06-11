"""App configuration: TOML file in ~/Library/Application Support/local-dictation/."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import tomli_w

log = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / "Library" / "Application Support" / "local-dictation"
CONFIG_PATH = CONFIG_DIR / "config.toml"

DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_LLM_MODEL = "mlx-community/Qwen3.5-4B-4bit"

WHISPER_MODEL_CHOICES = [
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-large-v3-turbo-4bit",
    "mlx-community/whisper-large-v3-mlx",
]

HOTKEY_CHOICES = ["alt_r", "cmd_r", "f13"]


@dataclass
class Config:
    hotkey: str = "alt_r"
    whisper_model: str = DEFAULT_WHISPER_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    cleanup_enabled: bool = True
    language: str = "auto"  # "auto" | whisper language code ("en", "ja", ...)

    @classmethod
    def load(cls) -> "Config":
        if not CONFIG_PATH.exists():
            return cls()
        try:
            with open(CONFIG_PATH, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            log.warning("Could not read %s (%s); using defaults", CONFIG_PATH, e)
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "wb") as f:
            tomli_w.dump(asdict(self), f)
