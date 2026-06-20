"""In-memory microphone recorder. 16 kHz mono float32 — Whisper's native format.

Audio never touches disk; the buffer is dropped after transcription.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

DEFAULT_DEVICE = "default"


def list_input_devices() -> list[str]:
    """Names of all devices that can record, deduplicated, in device order."""
    names: list[str] = []
    for d in sd.query_devices():
        if d["max_input_channels"] > 0 and d["name"] not in names:
            names.append(d["name"])
    return names


def default_input_device() -> str | None:
    """Name of the system default input device."""
    try:
        return sd.query_devices(sd.default.device[0])["name"]
    except (sd.PortAudioError, ValueError, TypeError):
        return None


def resolve_input_device(name: str | None) -> int | None:
    """Map a stored device name to a PortAudio index; None means system default."""
    if name in (None, "", DEFAULT_DEVICE):
        return None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and d["name"] == name:
            return i
    log.warning("Input device %r not found; using system default", name)
    return None


class Recorder:
    def __init__(self, device_name: str = DEFAULT_DEVICE) -> None:
        self.device_name = device_name
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        # Smoothed input level (0..~1) for the menu-bar meter. Written on the
        # PortAudio callback thread, read on the main thread — a lone float, so
        # the GIL makes that safe without a lock.
        self._level = 0.0

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def current_level(self) -> float:
        """Most recent smoothed input level (RMS). 0 when not recording."""
        return self._level

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            self._chunks = []
            self._level = 0.0
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=resolve_input_device(self.device_name),
                callback=self._callback,
            )
            self._stream.start()
        log.debug("Recording started")

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("Audio stream status: %s", status)
        self._chunks.append(indata[:, 0].copy())
        rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
        # Fast attack, slow decay so the meter jumps to speech and eases back.
        self._level = rms if rms > self._level else self._level * 0.85 + rms * 0.15

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a 1-D float32 array."""
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        self._level = 0.0
        audio = (
            np.concatenate(self._chunks)
            if self._chunks
            else np.zeros(0, dtype=np.float32)
        )
        self._chunks = []
        log.debug("Recording stopped: %.2fs", len(audio) / SAMPLE_RATE)
        return audio
