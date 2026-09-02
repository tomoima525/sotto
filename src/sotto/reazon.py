"""ReazonSpeech Japanese transcription via sherpa-onnx (CPU-only INT8 ONNX).

The k2 Zipformer transducer from reazon-research/reazonspeech-k2-v2 (Apache-2.0,
packaged as ONNX by the sherpa-onnx project). On real Japanese speech it
measures well under half whisper-large-v3-turbo's character error rate while
decoding many times faster than realtime on CPU — so it also frees the GPU
for the MLX models.

Two behavioral differences from the Whisper transcriber that callers rely on:
- Output is UNPUNCTUATED. The LLM cleanup pass restores 。/、 (its gate always
  fires on unpunctuated text), so ja dictations always pay the LLM pass.
- There is no no_speech_prob to gate hallucinations on; transducers return
  empty text on silence instead of hallucinating, so the pre-inference
  duration/RMS gates are the only ones needed.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .recorder import SAMPLE_RATE
from .transcriber import MIN_DURATION_S, MIN_RMS

log = logging.getLogger(__name__)

NUM_THREADS = 4

# ReazonSpeech emits TV-subtitle annotation brackets around boundary words;
# they carry no speech content (same stripping hayamimi applies).
_JUNK_CHARS = "［］〈〉"

# Zipformer offline decoding is meant for utterance-scale audio (hayamimi caps
# merged groups at 25s); a long toggle-mode recording is segmented by the
# energy VAD and decoded piecewise instead of in one pass.
MAX_UTTERANCE_S = 30.0


class ReazonTranscriber:
    """Duck-type drop-in for Transcriber (warmup/transcribe), Japanese only."""

    def __init__(self) -> None:
        self._recognizer = None
        self._warmed = False

    def load(self) -> None:
        """Build the recognizer, downloading the model first if missing."""
        if self._recognizer is not None:
            return
        import sherpa_onnx

        from . import models

        files = models.reazon_files()
        if files is None:
            models.download_reazon()
            files = models.reazon_files()
        if files is None:
            raise RuntimeError("ReazonSpeech model files unavailable")
        t0 = time.monotonic()
        # modified_beam_search: CER 8.6% -> 5.8% on real broadcast ja for +25%
        # decode time, still far faster than realtime (hayamimi's measurement).
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=files["encoder"],
            decoder=files["decoder"],
            joiner=files["joiner"],
            tokens=files["tokens"],
            num_threads=NUM_THREADS,
            model_type="zipformer",
            decoding_method="modified_beam_search",
        )
        log.info("ReazonSpeech loaded in %.1fs", time.monotonic() - t0)

    def warmup(self) -> None:
        """Load + first decode; idempotent, so safe to call per stream start."""
        if self._warmed:
            return
        self.load()
        t0 = time.monotonic()
        self._decode(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self._warmed = True
        log.info("ReazonSpeech warmup done in %.1fs", time.monotonic() - t0)

    def _decode(self, audio: np.ndarray) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self._recognizer.decode_stream(stream)
        text = stream.result.text
        for junk in _JUNK_CHARS:
            text = text.replace(junk, "")
        return text.strip()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio; returns "" for silence/noise."""
        self.load()
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_DURATION_S:
            log.debug("Rejected: too short (%.2fs)", duration)
            return ""
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < MIN_RMS:
            log.debug("Rejected: too quiet (rms=%.5f)", rms)
            return ""

        t0 = time.monotonic()
        if duration <= MAX_UTTERANCE_S:
            text = self._decode(audio)
        else:
            text = "".join(t for t in map(self._decode, _split(audio)) if t)
        log.info(
            "ReazonSpeech transcribed %.1fs audio in %.1fs (%d chars)",
            duration, time.monotonic() - t0, len(text),
        )
        return text


def _split(audio: np.ndarray) -> list[np.ndarray]:
    """Cut a long recording into utterance-scale pieces at speech pauses."""
    from .vad import EnergyVADSegmenter

    vad = EnergyVADSegmenter()
    segments = vad.feed(audio)
    tail = vad.flush()
    if tail is not None:
        segments.append(tail)
    return segments or [audio]
