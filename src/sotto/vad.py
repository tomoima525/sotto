"""Energy-based voice-activity segmenter for streaming dictation.

Pure numpy, no torch/onnx/silero — it classifies ~30 ms frames by RMS and cuts
a segment after a trailing pause (or a hard max length). It runs single-
threaded on the stream loop; feed it the recorder's drained audio and it
returns finalized speech segments to transcribe.

The thresholds sit above the transcriber's MIN_RMS (0.003) so room tone does
not open a segment, and every emitted segment still passes the transcriber's
own gates as a second line of defense against Whisper silence-hallucination.
"""

from __future__ import annotations

import logging

import numpy as np

from .recorder import SAMPLE_RATE

log = logging.getLogger(__name__)

FRAME_S = 0.03  # classify audio in ~30 ms frames for deterministic timing
SPEECH_RMS = 0.006  # frame RMS above this opens/continues a segment
SILENCE_RMS = 0.004  # below this is silence; the band between adds hysteresis
DEFAULT_MIN_SILENCE_S = 0.7  # trailing silence that finalizes a segment
DEFAULT_MIN_SEGMENT_S = 0.4  # drop segments shorter than this (hallucination guard)
DEFAULT_MAX_SEGMENT_S = 12.0  # hard cut so a monologue still streams (< Whisper's 30 s)
TRIM_KEEP_S = 0.2  # silence kept around speech when emitting (natural word edges)


class EnergyVADSegmenter:
    def __init__(
        self,
        min_silence_s: float = DEFAULT_MIN_SILENCE_S,
        min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
        max_segment_s: float = DEFAULT_MAX_SEGMENT_S,
    ) -> None:
        self.min_silence_s = min_silence_s
        self.min_segment_s = min_segment_s
        self.max_segment_s = max_segment_s
        self._frame = max(1, int(FRAME_S * SAMPLE_RATE))
        self.reset()

    def reset(self) -> None:
        self._pending: list[np.ndarray] = []
        self._carry = np.zeros(0, dtype=np.float32)  # leftover < one frame
        self._silence_run_s = 0.0
        self._in_speech = False
        self._seg_len_s = 0.0

    # -- internal --

    def _seg_audio(self) -> np.ndarray:
        return (
            np.concatenate(self._pending)
            if self._pending
            else np.zeros(0, dtype=np.float32)
        )

    def _finalize(self) -> np.ndarray | None:
        audio = self._seg_audio()
        self._pending = []
        self._silence_run_s = 0.0
        self._in_speech = False
        self._seg_len_s = 0.0
        if len(audio) / SAMPLE_RATE < self.min_segment_s:
            return None
        return _trim_silence(audio)

    # -- public --

    def feed(self, audio: np.ndarray) -> list[np.ndarray]:
        """Consume a block of audio; return any segments finalized within it."""
        segments: list[np.ndarray] = []
        buf = np.concatenate([self._carry, audio]) if self._carry.size else audio
        n_frames = len(buf) // self._frame
        self._carry = buf[n_frames * self._frame :].copy()

        for i in range(n_frames):
            frame = buf[i * self._frame : (i + 1) * self._frame]
            rms = float(np.sqrt(np.mean(frame**2)))
            self._pending.append(frame)
            self._seg_len_s += FRAME_S
            if rms >= SPEECH_RMS:
                self._in_speech = True
                self._silence_run_s = 0.0
            elif rms < SILENCE_RMS:
                self._silence_run_s += FRAME_S

            ended_by_pause = (
                self._in_speech
                and self._silence_run_s >= self.min_silence_s
                and self._seg_len_s >= self.min_segment_s
            )
            if ended_by_pause or self._seg_len_s >= self.max_segment_s:
                seg = self._finalize()
                if seg is not None:
                    segments.append(seg)
        return segments

    def flush(self) -> np.ndarray | None:
        """Force-finalize the in-progress segment on stop (e.g. mid-phrase)."""
        if self._carry.size:
            self._pending.append(self._carry)
            self._carry = np.zeros(0, dtype=np.float32)
        if not self._in_speech:
            self._finalize()  # reset; drop silence-only tail
            return None
        return self._finalize()


def _trim_silence(audio: np.ndarray) -> np.ndarray:
    """Trim leading/trailing silence beyond TRIM_KEEP_S using a coarse envelope."""
    frame = max(1, int(FRAME_S * SAMPLE_RATE))
    n = len(audio) // frame
    if n == 0:
        return audio
    framed = audio[: n * frame].reshape(n, frame)
    loud = np.sqrt((framed**2).mean(axis=1)) >= SPEECH_RMS
    if not loud.any():
        return audio
    keep = max(1, int(TRIM_KEEP_S / FRAME_S))
    first = max(0, int(np.argmax(loud)) - keep)
    last = min(n, n - int(np.argmax(loud[::-1])) + keep)
    return audio[first * frame : last * frame]
