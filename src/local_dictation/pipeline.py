"""Dictation pipeline: state machine + worker thread.

Hotkey callbacks must return immediately (they run on pynput's thread), so
they only flip state and enqueue commands. All inference and pasteboard work
happens on the single worker thread, which also serializes overlapping
dictations (release-while-processing, rapid double-taps).
"""

from __future__ import annotations

import enum
import logging
import queue
import threading
from typing import Callable

from .cleaner import Cleaner
from .config import Config
from .injector import inject
from .recorder import Recorder
from .transcriber import Transcriber

log = logging.getLogger(__name__)


class State(enum.Enum):
    LOADING = "loading"
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class Pipeline:
    def __init__(
        self,
        config: Config,
        on_state_change: Callable[[State], None] | None = None,
    ) -> None:
        self.config = config
        self._on_state_change = on_state_change
        self.state = State.LOADING

        self.recorder = Recorder()
        self.transcriber = Transcriber(config.whisper_model, config.language)
        self.cleaner = Cleaner(config.llm_model)

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

    # -- worker thread --

    def _set_state(self, state: State) -> None:
        self.state = state
        log.info("State: %s", state.value)
        if self._on_state_change:
            try:
                self._on_state_change(state)
            except Exception:
                log.exception("State-change callback failed")

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
                if self.recorder.recording:
                    self.recorder.stop()
                return
            if cmd == "start" and self.state == State.IDLE:
                self._set_state(State.RECORDING)
                self.recorder.start()
            elif cmd == "stop" and self.state == State.RECORDING:
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
