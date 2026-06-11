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


class Recorder:
    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            self._chunks = []
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        log.debug("Recording started")

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("Audio stream status: %s", status)
        self._chunks.append(indata[:, 0].copy())

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a 1-D float32 array."""
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        audio = (
            np.concatenate(self._chunks)
            if self._chunks
            else np.zeros(0, dtype=np.float32)
        )
        self._chunks = []
        log.debug("Recording stopped: %.2fs", len(audio) / SAMPLE_RATE)
        return audio
