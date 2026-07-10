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
# Small, fast, multilingual model for streaming mode (much cheaper per call than
# turbo — preview lag drops to ~1-2s, at the cost of accuracy).
DEFAULT_STREAMING_WHISPER_MODEL = "mlx-community/whisper-small-mlx"

# repo -> menu label (with a short description of the speed/accuracy tradeoff).
WHISPER_MODEL_CHOICES = {
    "mlx-community/whisper-large-v3-turbo": "Turbo — balanced, recommended",
    "mlx-community/whisper-large-v3-turbo-4bit": "Turbo (4-bit) — less memory, similar speed",
    "mlx-community/whisper-large-v3-mlx": "Large v3 — most accurate, slower",
}

# repo -> menu label (with a short description of the speed/accuracy tradeoff).
STREAMING_WHISPER_MODEL_CHOICES = {
    "mlx-community/whisper-small-mlx": "Small — balanced, for daily use",
    "mlx-community/whisper-base-mlx": "Base — faster, lower accuracy",
}

HOTKEY_CHOICES = ["alt_r", "cmd_r", "f13"]

# value stored in config -> menu label
INPUT_MODE_CHOICES = {
    "hold": "Hold to talk",
    "toggle": "Toggle (press to start / stop)",
    "streaming": "Streaming (live preview)",
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
    input_mode: str = "hold"  # "hold" | "toggle" | "streaming"
    streaming_whisper_model: str = DEFAULT_STREAMING_WHISPER_MODEL
    streaming_silence_ms: int = 700  # trailing pause that ends a phrase
    streaming_max_segment_s: float = 12.0  # hard cut for a long monologue

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
