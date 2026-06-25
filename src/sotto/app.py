"""Menu bar app (rumps). AppKit is main-thread-only, so all status-item
updates coming from worker threads are dispatched onto the main queue."""

from __future__ import annotations

import logging
from collections import deque

import rumps
from libdispatch import dispatch_async, dispatch_get_main_queue

from . import models
from .config import (
    Config,
    HOTKEY_CHOICES,
    INPUT_MODE_CHOICES,
    LANGUAGE_CHOICES,
    STREAMING_WHISPER_MODEL_CHOICES,
    WHISPER_MODEL_CHOICES,
)
from .hotkey import make_listener
from .hud import HUDController
from .pipeline import Pipeline, State
from .recorder import DEFAULT_DEVICE, default_input_device, list_input_devices

log = logging.getLogger(__name__)

STATE_TITLES = {
    State.LOADING: "⏳",
    State.IDLE: "🎤",
    State.RECORDING: "🔴",
    State.PROCESSING: "✍️",
    State.STREAMING: "🟢",
}

# Live recording meter: a scrolling waveform driven by the mic level. Block
# glyphs share an advance width, so the status item width stays fixed (no
# jitter). The lowest glyph is the resting baseline — a flat line while you
# speak means the mic isn't being picked up.
WAVE_BARS = "▁▂▃▄▅▆▇█"
WAVE_WIDTH = 7
WAVE_INTERVAL_S = 0.07  # ~14 Hz
# RMS range mapped onto the bars. Below the floor (room tone / a mic that isn't
# being heard) rests at ▁; normal speech lands mid-range, loud speech tops out.
WAVE_FLOOR = 0.004
WAVE_CEIL = 0.18


def level_to_bar(level: float) -> str:
    span = max(0.0, level - WAVE_FLOOR) / (WAVE_CEIL - WAVE_FLOOR)
    norm = min(1.0, span) ** 0.5  # sqrt curve so quiet speech still registers
    return WAVE_BARS[round(norm * (len(WAVE_BARS) - 1))]

HOTKEY_LABELS = {
    "alt_r": "Right Option (⌥)",
    "cmd_r": "Right Command (⌘)",
    "f13": "F13",
}


class DictationApp(rumps.App):
    def __init__(self, config: Config | None = None) -> None:
        super().__init__("⏳", quit_button=None)
        self.config = config if config is not None else Config.load()

        self.cleanup_item = rumps.MenuItem(
            "Cleanup with LLM", callback=self._toggle_cleanup
        )
        self.cleanup_item.state = self.config.cleanup_enabled

        hotkey_menu = rumps.MenuItem("Hotkey")
        for name in HOTKEY_CHOICES:
            item = rumps.MenuItem(
                HOTKEY_LABELS.get(name, name), callback=self._pick_hotkey
            )
            item._hotkey_name = name
            item.state = name == self.config.hotkey
            hotkey_menu.add(item)

        model_menu = rumps.MenuItem("Whisper Model")
        for repo in WHISPER_MODEL_CHOICES:
            item = rumps.MenuItem(repo.split("/")[-1], callback=self._pick_model)
            item._model_repo = repo
            item.state = repo == self.config.whisper_model
            model_menu.add(item)

        stream_model_menu = rumps.MenuItem("Streaming Model")
        for repo in STREAMING_WHISPER_MODEL_CHOICES:
            item = rumps.MenuItem(repo.split("/")[-1], callback=self._pick_stream_model)
            item._model_repo = repo
            item.state = repo == self.config.streaming_whisper_model
            stream_model_menu.add(item)

        language_menu = rumps.MenuItem("Language")
        for code, label in LANGUAGE_CHOICES.items():
            item = rumps.MenuItem(label, callback=self._pick_language)
            item._language_code = code
            item.state = code == self.config.language
            language_menu.add(item)

        input_mode_menu = rumps.MenuItem("Input Mode")
        for mode, label in INPUT_MODE_CHOICES.items():
            item = rumps.MenuItem(label, callback=self._pick_input_mode)
            item._input_mode = mode
            item.state = mode == self.config.input_mode
            input_mode_menu.add(item)

        self.mic_menu = rumps.MenuItem("Microphone")
        self._populate_mic_menu()

        self.menu = [
            self.cleanup_item,
            input_mode_menu,
            language_menu,
            self.mic_menu,
            hotkey_menu,
            model_menu,
            stream_model_menu,
            None,
            rumps.MenuItem("Quit", callback=self._quit),
        ]

        self.hud = HUDController()
        self.pipeline = Pipeline(
            self.config,
            on_state_change=self._state_changed,
            on_partial=self._on_partial,
        )
        self.hotkey = make_listener(
            self.config.hotkey, self.config.input_mode, self.pipeline
        )

        self._wave: deque[str] = deque(maxlen=WAVE_WIDTH)
        self._meter_timer = rumps.Timer(self._animate_meter, WAVE_INTERVAL_S)

        self._ensure_models_then_start()

    def _ensure_models_then_start(self) -> None:
        import threading

        def boot():
            repos = [self.config.whisper_model, self.config.llm_model]
            missing = [r for r in repos if not models.is_cached(r)]
            if missing:
                self._set_title("⬇️")
                for repo in missing:
                    log.info("Downloading %s ...", repo)
                    models.download(repo)
            self.pipeline.start()
            self.hotkey.start()

        threading.Thread(target=boot, daemon=True, name="boot").start()

    # -- state display --

    def _set_title(self, title: str) -> None:
        def update():
            self.title = title

        dispatch_async(dispatch_get_main_queue(), update)

    def _state_changed(self, state: State) -> None:
        # Timer start/stop and title writes must happen on the main thread.
        def update():
            if state in (State.RECORDING, State.STREAMING):
                self._start_meter()
            else:
                self._stop_meter()
                self.title = STATE_TITLES.get(state, "🎤")

        dispatch_async(dispatch_get_main_queue(), update)

    def _on_partial(self, kind: str, text: str) -> None:
        # Streaming preview events from the pipeline. HUDController hops to the
        # main thread itself, so we can call it directly.
        if kind == "start":
            self.hud.show()
        elif kind in ("append", "commit"):
            self.hud.set_text(text)
        elif kind == "end":
            self.hud.hide()

    def _start_meter(self) -> None:
        self._wave.clear()
        self._wave.extend(WAVE_BARS[0] * WAVE_WIDTH)  # flat resting baseline
        self.title = "".join(self._wave)
        if not self._meter_timer.is_alive():
            self._meter_timer.start()

    def _stop_meter(self) -> None:
        if self._meter_timer.is_alive():
            self._meter_timer.stop()

    def _animate_meter(self, _timer) -> None:
        self._wave.append(level_to_bar(self.pipeline.recorder.current_level()))
        self.title = "".join(self._wave)

    def _pick_language(self, sender) -> None:
        code = sender._language_code
        for item in self.menu["Language"].values():
            item.state = item._language_code == code
        self.config.language = code
        self.config.save()
        self.pipeline.transcriber.language = code
        log.info("Language set to %r", code)

    # -- microphone menu --

    def _populate_mic_menu(self) -> None:
        menu = self.mic_menu
        for key in list(menu.keys()):
            del menu[key]

        system_default = default_input_device()
        label = "System Default"
        if system_default:
            label = f"System Default ({system_default})"
        item = rumps.MenuItem(label, callback=self._pick_mic)
        item._device_name = DEFAULT_DEVICE
        item.state = self.config.input_device == DEFAULT_DEVICE
        menu.add(item)

        for name in list_input_devices():
            item = rumps.MenuItem(name, callback=self._pick_mic)
            item._device_name = name
            item.state = name == self.config.input_device
            menu.add(item)

        menu.add(None)
        menu.add(rumps.MenuItem("Refresh Devices", callback=self._refresh_mics))

    def _pick_mic(self, sender) -> None:
        name = sender._device_name
        for item in self.mic_menu.values():
            if hasattr(item, "_device_name"):
                item.state = item._device_name == name
        self.config.input_device = name
        self.config.save()
        self.pipeline.recorder.device_name = name
        log.info("Input device set to %r", name)

    def _refresh_mics(self, sender) -> None:
        self._populate_mic_menu()

    # -- menu callbacks (run on the main thread) --

    def _toggle_cleanup(self, sender) -> None:
        sender.state = not sender.state
        self.config.cleanup_enabled = bool(sender.state)
        self.config.save()

    def _restart_listener(self) -> None:
        self.hotkey.stop()
        self.hotkey = make_listener(
            self.config.hotkey, self.config.input_mode, self.pipeline
        )
        self.hotkey.start()

    def _pick_hotkey(self, sender) -> None:
        name = sender._hotkey_name
        if name == self.config.hotkey:
            return
        for item in self.menu["Hotkey"].values():
            item.state = item._hotkey_name == name
        self.config.hotkey = name
        self.config.save()
        self._restart_listener()

    def _pick_input_mode(self, sender) -> None:
        mode = sender._input_mode
        if mode == self.config.input_mode:
            return
        for item in self.menu["Input Mode"].values():
            item.state = item._input_mode == mode
        self.config.input_mode = mode
        self.config.save()
        self._restart_listener()
        log.info("Input mode set to %r", mode)

    def _pick_model(self, sender) -> None:
        repo = sender._model_repo
        if repo == self.config.whisper_model:
            return
        for item in self.menu["Whisper Model"].values():
            item.state = item._model_repo == repo
        self.config.whisper_model = repo
        self.config.save()
        rumps.notification(
            "Sotto", "", "Whisper model changed — restart the app to apply."
        )

    def _pick_stream_model(self, sender) -> None:
        repo = sender._model_repo
        if repo == self.config.streaming_whisper_model:
            return
        for item in self.menu["Streaming Model"].values():
            item.state = item._model_repo == repo
        self.config.streaming_whisper_model = repo
        self.config.save()
        rumps.notification(
            "Sotto", "", "Streaming model changed — restart the app to apply."
        )

    def _quit(self, sender) -> None:
        self._stop_meter()
        self.hud.teardown()
        self.hotkey.stop()
        self.pipeline.shutdown()
        rumps.quit_application()


def run_app(config: Config | None = None) -> None:
    DictationApp(config).run()
