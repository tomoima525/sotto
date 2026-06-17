"""Global hold-to-talk hotkey via hardware key-state polling.

We poll the key's live state with Quartz instead of installing a CGEventTap
(as pynput does). A listen-only event tap is silently disabled by macOS when
its callback is starved — which happens on a fanless Mac while the worker
thread runs Whisper + the LLM — and pynput never re-enables it, so key events
(notably the release) just stop arriving with no error. Polling reads the
current state every tick, so heavy CPU load can at most delay an edge by one
poll interval; it can never drop one.

Modifier keys (Option/Command) are read from the modifier-flags state rather
than CGEventSourceKeyState, which does not reliably report modifiers. The
flags word carries device-specific bits that distinguish left from right, so a
right-hand hotkey never triggers on the left key. Regular keys (F13) use
CGEventSourceKeyState by virtual keycode.

Still requires the host process to have Input Monitoring + Accessibility.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import Quartz

log = logging.getLogger(__name__)

# Device-specific modifier-flag bits (IOKit NX_DEVICE*KEYMASK): these appear in
# the low bits of the modifier-flags word and distinguish left vs right.
_NX_DEVICE_RCMD = 0x10
_NX_DEVICE_RALT = 0x40

# name -> (kind, value):
#   "flag": value is a device-specific bit tested against CGEventSourceFlagsState
#   "key":  value is a virtual keycode tested against CGEventSourceKeyState
KEY_MAP: dict[str, tuple[str, int]] = {
    "alt_r": ("flag", _NX_DEVICE_RALT),  # Right Option
    "cmd_r": ("flag", _NX_DEVICE_RCMD),  # Right Command
    "f13": ("key", 0x69),
}

POLL_INTERVAL_S = 0.02  # 50 Hz; negligible CPU, sub-frame latency for a hold

_SOURCE = Quartz.kCGEventSourceStateHIDSystemState


def key_is_down(kind: str, value: int) -> bool:
    if kind == "flag":
        return bool(int(Quartz.CGEventSourceFlagsState(_SOURCE)) & value)
    return bool(Quartz.CGEventSourceKeyState(_SOURCE, value))


class HotkeyListener:
    def __init__(
        self,
        key_name: str,
        on_hold_start: Callable[[], None],
        on_hold_end: Callable[[], None],
    ) -> None:
        if key_name not in KEY_MAP:
            raise ValueError(f"Unknown hotkey {key_name!r}; choose from {list(KEY_MAP)}")
        self._kind, self._value = KEY_MAP[key_name]
        self._on_hold_start = on_hold_start
        self._on_hold_end = on_hold_end
        self._held = False
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True, name="hotkey")
        self._thread.start()
        log.debug("Hotkey poller started (%s=0x%02X)", self._kind, self._value)

    def stop(self) -> None:
        self._running = False
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)

    def _poll(self) -> None:
        while self._running:
            down = key_is_down(self._kind, self._value)
            if down and not self._held:
                self._held = True
                self._fire(self._on_hold_start)
            elif not down and self._held:
                self._held = False
                self._fire(self._on_hold_end)
            time.sleep(POLL_INTERVAL_S)

    @staticmethod
    def _fire(callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            log.exception("Hotkey callback failed")
