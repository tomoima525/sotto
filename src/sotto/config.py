"""App configuration: TOML file in ~/Library/Application Support/sotto/."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import tomli_w

log = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / "Library" / "Application Support" / "sotto"
CONFIG_PATH = CONFIG_DIR / "config.toml"

# Pre-rename location (the app used to be called local-dictation).
_LEGACY_CONFIG_PATH = (
    Path.home() / "Library" / "Application Support" / "local-dictation" / "config.toml"
)


def _migrate_legacy_config() -> None:
    if CONFIG_PATH.exists() or not _LEGACY_CONFIG_PATH.exists():
        return
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_bytes(_LEGACY_CONFIG_PATH.read_bytes())
        log.info("Migrated config from %s", _LEGACY_CONFIG_PATH)
    except OSError as e:
        log.warning("Could not migrate legacy config (%s)", e)

DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_LLM_MODEL = "mlx-community/Qwen3.5-4B-4bit"

WHISPER_MODEL_CHOICES = [
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-large-v3-turbo-4bit",
    "mlx-community/whisper-large-v3-mlx",
]

HOTKEY_CHOICES = ["alt_r", "cmd_r", "f13"]

# value stored in config -> menu label
INPUT_MODE_CHOICES = {
    "hold": "Hold to talk",
    "toggle": "Toggle (press to start / stop)",
}

# value stored in config -> menu label ("auto" lets Whisper detect from audio)
LANGUAGE_CHOICES = {
    "auto": "Universal (detect from audio)",
    "en": "English",
    "ja": "Japanese",
}


@dataclass
class Config:
    hotkey: str = "alt_r"
    whisper_model: str = DEFAULT_WHISPER_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    cleanup_enabled: bool = True
    language: str = "auto"  # "auto" | whisper language code ("en", "ja", ...)
    input_device: str = "default"  # device name, or "default" for system default
    input_mode: str = "hold"  # "hold" | "toggle"

    @classmethod
    def load(cls) -> "Config":
        _migrate_legacy_config()
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
