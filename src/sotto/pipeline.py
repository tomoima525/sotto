"""Dictation pipeline: state machine + worker thread.

Hotkey callbacks must return immediately (they run on the poller thread), so
they only flip state and enqueue commands. All inference and pasteboard work
happens on the single worker thread, which also serializes overlapping
dictations (release-while-processing, rapid double-taps).
"""

from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from typing import Callable

import numpy as np

from .cleaner import Cleaner
from .config import Config
from .injector import inject
from .recorder import SAMPLE_RATE, Recorder
from .transcriber import Transcriber
from .vad import EnergyVADSegmenter

log = logging.getLogger(__name__)


class State(enum.Enum):
    LOADING = "loading"
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    STREAMING = "streaming"


# How often the stream loop drains audio and runs the VAD.
DRAIN_INTERVAL_S = 0.25

# Draft previews of the in-progress phrase (pattern borrowed from hayamimi):
# while speech is ongoing, periodically re-transcribe the unfinalized segment
# and show the result as a throwaway draft that the next draft or the phrase's
# final text replaces. The throttle is measured from the END of the previous
# draft decode, so a decode slower than the interval self-paces instead of
# starving segment finalization.
DRAFT_EVERY_S = 0.5   # min gap between draft decodes
DRAFT_WINDOW_S = 8.0  # decode at most this much of the phrase tail per draft
DRAFT_MIN_AUDIO_S = 0.5  # don't draft until the phrase has this much audio

# Refine pass (hayamimi's two-pass idea, collapsed to one pass at stop): the
# live preview comes from the small model, but the pasted text is a fresh
# decode of the whole dictation by the accurate main model. Beyond this much
# speech the re-decode is skipped — stop latency grows with dictation length
# (roughly transcriber RTF x duration) and a very long dictation gets the
# small-model transcript plus LLM cleanup, as before.
REFINE_MAX_S = 120.0


class Pipeline:
    def __init__(
        self,
        config: Config,
        on_state_change: Callable[[State], None] | None = None,
        on_partial: Callable[[str, str], None] | None = None,
    ) -> None:
        self.config = config
        self._on_state_change = on_state_change
        # on_partial(kind, text): kind in
        # {"start","draft","append","status","commit","end"}
        # "status" carries a label for the post-stop work (refine, cleanup) so
        # the overlay can show a spinner instead of freezing on the last draft.
        # "draft" carries the transcript plus an in-progress preview of the
        # current phrase; each draft (or the phrase's final "append") replaces
        # the previous one.
        self._on_partial = on_partial
        self.state = State.LOADING

        self.recorder = Recorder(device_name=config.input_device)
        self.transcriber = Transcriber(config.whisper_model, config.language)
        self.cleaner = Cleaner(config.llm_model, gate_enabled=config.cleanup_gate)

        # Streaming: a small fast model + an energy VAD. Warmed lazily on first
        # stream so normal startup isn't slowed by a second model load.
        self.stream_transcriber = Transcriber(
            config.streaming_whisper_model, config.language
        )
        self._stream_warmed = False
        self.vad = EnergyVADSegmenter(
            min_silence_s=config.streaming_silence_ms / 1000,
            max_segment_s=config.streaming_max_segment_s,
        )
        self._transcript_parts: list[str] = []
        # Finalized segments' audio, kept (in RAM only, like everything else)
        # for the refine pass at stop; dropped once the dictation ends.
        self._segment_audio: list[np.ndarray] = []
        self._stream_audio_s = 0.0
        self._last_draft = 0.0  # monotonic time the last draft decode finished
        self._stream_stop: threading.Event | None = None
        self._stream_thread: threading.Thread | None = None

        self._commands: queue.Queue[str] = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="pipeline")

    def start(self) -> None:
        self._worker.start()

    def shutdown(self) -> None:
        self._commands.put("quit")

    # -- called from the hotkey thread --

    def begin_recording(self) -> None:
        self._commands.put("start")

    def end_recording(self) -> None:
        self._commands.put("stop")

    def toggle_recording(self) -> None:
        """Start if idle, stop if recording (for toggle input mode)."""
        self._commands.put("toggle")

    def toggle_streaming(self) -> None:
        """Start if idle, stop if streaming (for streaming input mode)."""
        self._commands.put("toggle_stream")

    # -- worker thread --

    def _set_state(self, state: State) -> None:
        self.state = state
        log.info("State: %s", state.value)
        if self._on_state_change:
            try:
                self._on_state_change(state)
            except Exception:
                log.exception("State-change callback failed")

    def _emit_partial(self, kind: str, text: str = "") -> None:
        if self._on_partial:
            try:
                self._on_partial(kind, text)
            except Exception:
                log.exception("Partial-text callback failed")

    def _join_transcript(self, draft: str = "") -> str:
        parts = self._transcript_parts + ([draft] if draft else [])
        sep = "" if self.config.language == "ja" else " "
        return sep.join(parts).strip()

    def _run(self) -> None:
        try:
            self.transcriber.warmup()
            self.cleaner.warmup()
        except Exception:
            log.exception("Model warmup failed")
        self._set_state(State.IDLE)

        while True:
            cmd = self._commands.get()
            if cmd == "quit":
                if self._stream_stop is not None:
                    self._stream_stop.set()
                if self.recorder.recording:
                    self.recorder.stop()
                return
            if cmd == "start":
                self._start_recording()
            elif cmd == "stop":
                self._stop_and_process()
            elif cmd == "toggle":
                if self.state == State.IDLE:
                    self._start_recording()
                elif self.state == State.RECORDING:
                    self._stop_and_process()
            elif cmd == "toggle_stream":
                if self.state == State.IDLE:
                    self._start_streaming()
                elif self.state == State.STREAMING:
                    self._stop_streaming()

    def _start_recording(self) -> None:
        if self.state != State.IDLE:
            return
        self._set_state(State.RECORDING)
        self.recorder.start()

    def _stop_and_process(self) -> None:
        if self.state != State.RECORDING:
            return
        audio = self.recorder.stop()
        self._set_state(State.PROCESSING)
        self._process(audio)
        self._set_state(State.IDLE)

    def _process(self, audio) -> None:
        try:
            text = self.transcriber.transcribe(audio)
            if not text:
                log.info("Nothing transcribed; skipping")
                return
            if self.config.cleanup_enabled:
                text = self.cleaner.clean(text)
            inject(text)
        except Exception:
            log.exception("Dictation failed")

    # -- streaming mode --

    def _start_streaming(self) -> None:
        if self.state != State.IDLE:
            return
        if not self._stream_warmed:
            try:
                self.stream_transcriber.warmup()
            except Exception:
                log.exception("Streaming model warmup failed")
            self._stream_warmed = True
        self.vad.reset()
        self._transcript_parts = []
        self._segment_audio = []
        self._stream_audio_s = 0.0
        self._stream_stop = threading.Event()
        self._set_state(State.STREAMING)
        self.recorder.start()
        self._emit_partial("start")
        self._stream_thread = threading.Thread(
            target=self._stream_loop, daemon=True, name="stream"
        )
        self._stream_thread.start()

    def _stream_loop(self) -> None:
        """Drain audio, segment it, transcribe each segment. Runs on its own
        thread so the worker stays responsive to stop/quit; it is the only
        transcriber caller while streaming, so no concurrent inference."""
        stop = self._stream_stop
        assert stop is not None
        while not stop.is_set():
            audio = self.recorder.drain_chunks()
            if audio is not None:
                for seg in self.vad.feed(audio):
                    self._transcribe_and_emit(seg)
            self._maybe_emit_draft()
            stop.wait(DRAIN_INTERVAL_S)
        # Final flush: capture and transcribe the in-progress phrase.
        tail = self.recorder.drain_chunks()
        if tail is not None:
            for seg in self.vad.feed(tail):
                self._transcribe_and_emit(seg)
        final = self.vad.flush()
        if final is not None:
            self._transcribe_and_emit(final)

    def _maybe_emit_draft(self) -> None:
        """Preview the in-progress phrase by transcribing its unfinalized audio.

        Finalized segments always come first (the caller feeds the VAD before
        drafting), and a finalize empties the pending buffer, so a draft never
        races the phrase's own final text.
        """
        if not self.vad.in_speech:
            return
        if time.monotonic() - self._last_draft < DRAFT_EVERY_S:
            return
        cur = self.vad.current_audio(DRAFT_WINDOW_S)
        if len(cur) < int(DRAFT_MIN_AUDIO_S * SAMPLE_RATE):
            return
        try:
            draft = self.stream_transcriber.transcribe(cur)
        except Exception:
            log.exception("Draft transcription failed")
            draft = ""
        finally:
            self._last_draft = time.monotonic()
        if draft:
            self._emit_partial("draft", self._join_transcript(draft))

    def _collect_for_refine(self, seg: np.ndarray) -> None:
        """Keep a finalized segment's audio for the refine pass at stop.

        Kept even when the small model transcribes it to nothing — the main
        model may still recover that speech. Past REFINE_MAX_S the refine pass
        won't run, so the collected audio is freed instead of accumulating.
        """
        if not self.config.streaming_refine:
            return
        self._stream_audio_s += len(seg) / SAMPLE_RATE
        if self._stream_audio_s <= REFINE_MAX_S:
            self._segment_audio.append(seg)
        elif self._segment_audio:
            log.info("Refine off for this dictation: over %.0fs of speech", REFINE_MAX_S)
            self._segment_audio = []

    def _refine_or_join(self) -> str:
        """The transcript to clean and paste: a fresh main-model decode of the
        whole dictation when possible, else the joined small-model parts."""
        fast = self._join_transcript()
        audio, self._segment_audio = self._segment_audio, []
        if not self.config.streaming_refine or not audio:
            return fast
        try:
            t0 = time.monotonic()
            self._emit_partial("status", "Transcribing with the main model…")
            refined = self.transcriber.transcribe(np.concatenate(audio))
            log.info("Refine pass took %.1fs", time.monotonic() - t0)
        except Exception:
            log.exception("Refine pass failed; using the streaming transcript")
            return fast
        # A re-decode must never lose content (hayamimi's guard): coming back
        # much shorter than the fast parts means it dropped speech.
        if len(refined.strip()) < 0.7 * len(fast):
            log.info("Refine dropped content (%d vs %d chars); using the "
                     "streaming transcript", len(refined.strip()), len(fast))
            return fast
        return refined

    def _transcribe_and_emit(self, seg) -> None:
        self._collect_for_refine(seg)
        try:
            text = self.stream_transcriber.transcribe(seg)
        except Exception:
            log.exception("Streaming segment transcription failed")
            return
        if not text:
            # Still refresh the overlay: a stale draft of this phrase may be
            # showing, and the phrase just finalized to nothing (noise).
            self._emit_partial("append", self._join_transcript())
            return
        self._transcript_parts.append(text)
        self._emit_partial("append", self._join_transcript())

    def _stop_streaming(self) -> None:
        if self.state != State.STREAMING:
            return
        if self._stream_stop is not None:
            self._stream_stop.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=30)
        self.recorder.stop()  # teardown; chunks already drained
        self._stream_stop = None
        self._stream_thread = None

        self._set_state(State.PROCESSING)
        try:
            full = self._refine_or_join()
            if not full:
                log.info("Nothing transcribed; skipping")
            else:
                if self.config.cleanup_enabled:
                    self._emit_partial("status", "Cleaning up…")
                    cleaned = self.cleaner.clean(full)
                else:
                    cleaned = full
                self._emit_partial("commit", cleaned)
                inject(cleaned)
        except Exception:
            log.exception("Streaming dictation failed")
        finally:
            self._emit_partial("end")
            self._set_state(State.IDLE)
