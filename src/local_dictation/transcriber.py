"""Whisper transcription via mlx-whisper, with silence gating.

Silence gating matters: Whisper hallucinates plausible text from empty audio
("Thank you.", "ご視聴ありがとうございました"), so we reject too-short or
too-quiet recordings before inference and low-confidence results after.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .recorder import SAMPLE_RATE

log = logging.getLogger(__name__)

MIN_DURATION_S = 0.3
MIN_RMS = 0.003
MAX_NO_SPEECH_PROB = 0.6


class Transcriber:
    def __init__(self, model_repo: str, language: str = "auto") -> None:
        self.model_repo = model_repo
        self.language = language
        self._model_path: str | None = None

    def _path(self) -> str:
        if self._model_path is None:
            from .models import resolve_whisper_path

            self._model_path = resolve_whisper_path(self.model_repo)
        return self._model_path

    def warmup(self) -> None:
        """Trigger model load + Metal kernel compilation so the first real
        dictation isn't slow."""
        import mlx_whisper

        t0 = time.monotonic()
        mlx_whisper.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=self._path(),
            language="en",
        )
        log.info("Whisper warmup done in %.1fs", time.monotonic() - t0)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio; returns "" for silence/noise."""
        import mlx_whisper

        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_DURATION_S:
            log.debug("Rejected: too short (%.2fs)", duration)
            return ""
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < MIN_RMS:
            log.debug("Rejected: too quiet (rms=%.5f)", rms)
            return ""

        t0 = time.monotonic()
        language = None if self.language == "auto" else self.language
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._path(),
            language=language,
        )
        elapsed = time.monotonic() - t0

        segments = result.get("segments", [])
        if segments:
            no_speech = float(np.mean([s.get("no_speech_prob", 0.0) for s in segments]))
            if no_speech > MAX_NO_SPEECH_PROB:
                log.debug("Rejected: no_speech_prob=%.2f", no_speech)
                return ""

        text = result.get("text", "").strip()
        log.info(
            "Transcribed %.1fs audio in %.1fs (lang=%s, %d chars)",
            duration, elapsed, result.get("language"), len(text),
        )
        return text
