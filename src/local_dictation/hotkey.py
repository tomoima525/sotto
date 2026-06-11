"""Global hold-to-talk hotkey via pynput.

Requires the host process (your terminal during development) to have
Input Monitoring permission — without it pynput silently receives no events.
Key autorepeat is debounced: on_press fires repeatedly while a key is held.
"""

from __future__ import annotations

import logging
from typing import Callable

from pynput import keyboard

log = logging.getLogger(__name__)

KEY_MAP = {
    "alt_r": keyboard.Key.alt_r,
    "cmd_r": keyboard.Key.cmd_r,
    "f13": keyboard.Key.f13,
}


class HotkeyListener:
    def __init__(
        self,
        key_name: str,
        on_hold_start: Callable[[], None],
        on_hold_end: Callable[[], None],
    ) -> None:
        if key_name not in KEY_MAP:
            raise ValueError(f"Unknown hotkey {key_name!r}; choose from {list(KEY_MAP)}")
        self._key = KEY_MAP[key_name]
        self._on_hold_start = on_hold_start
        self._on_hold_end = on_hold_end
        self._held = False
        self._listener: keyboard.Listener | None = None

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()
        log.debug("Hotkey listener started for %s", self._key)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _on_press(self, key) -> None:
        if key == self._key and not self._held:
            self._held = True
            try:
                self._on_hold_start()
            except Exception:
                log.exception("on_hold_start failed")

    def _on_release(self, key) -> None:
        if key == self._key and self._held:
            self._held = False
            try:
                self._on_hold_end()
            except Exception:
                log.exception("on_hold_end failed")
