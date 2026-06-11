"""Menu bar app (rumps). AppKit is main-thread-only, so all status-item
updates coming from worker threads are dispatched onto the main queue."""

from __future__ import annotations

import logging

import rumps
from libdispatch import dispatch_async, dispatch_get_main_queue

from . import models
from .config import Config, HOTKEY_CHOICES, WHISPER_MODEL_CHOICES
from .hotkey import HotkeyListener
from .pipeline import Pipeline, State
from .recorder import DEFAULT_DEVICE, default_input_device, list_input_devices

log = logging.getLogger(__name__)

STATE_TITLES = {
    State.LOADING: "⏳",
    State.IDLE: "🎤",
    State.RECORDING: "🔴",
    State.PROCESSING: "✍️",
}

HOTKEY_LABELS = {
    "alt_r": "Right Option (⌥)",
    "cmd_r": "Right Command (⌘)",
    "f13": "F13",
}


class DictationApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("⏳", quit_button=None)
        self.config = Config.load()

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

        self.mic_menu = rumps.MenuItem("Microphone")
        self._populate_mic_menu()

        self.menu = [
            self.cleanup_item,
            self.mic_menu,
            hotkey_menu,
            model_menu,
            None,
            rumps.MenuItem("Quit", callback=self._quit),
        ]

        self.pipeline = Pipeline(self.config, on_state_change=self._state_changed)
        self.hotkey = HotkeyListener(
            self.config.hotkey,
            on_hold_start=self.pipeline.begin_recording,
            on_hold_end=self.pipeline.end_recording,
        )

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
        self._set_title(STATE_TITLES.get(state, "🎤"))

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

    def _pick_hotkey(self, sender) -> None:
        name = sender._hotkey_name
        if name == self.config.hotkey:
            return
        for item in self.menu["Hotkey"].values():
            item.state = item._hotkey_name == name
        self.config.hotkey = name
        self.config.save()
        self.hotkey.stop()
        self.hotkey = HotkeyListener(
            name,
            on_hold_start=self.pipeline.begin_recording,
            on_hold_end=self.pipeline.end_recording,
        )
        self.hotkey.start()

    def _pick_model(self, sender) -> None:
        repo = sender._model_repo
        if repo == self.config.whisper_model:
            return
        for item in self.menu["Whisper Model"].values():
            item.state = item._model_repo == repo
        self.config.whisper_model = repo
        self.config.save()
        rumps.notification(
            "Local Dictation", "", "Whisper model changed — restart the app to apply."
        )

    def _quit(self, sender) -> None:
        self.hotkey.stop()
        self.pipeline.shutdown()
        rumps.quit_application()


def run_app() -> None:
    DictationApp().run()
