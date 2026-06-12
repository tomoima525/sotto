"""Insert text at the cursor of the frontmost app: pasteboard + simulated Cmd+V.

Paste (not synthetic typing) because CGEvent unicode typing is unreliable for
long text and IME-active Japanese fields. The previous clipboard string is
restored afterwards, but only if the pasteboard still holds our text
(changeCount check) so we never clobber something the user copied meanwhile.
"""

from __future__ import annotations

import logging
import time

import Quartz
from AppKit import NSPasteboard, NSPasteboardTypeString

log = logging.getLogger(__name__)

V_KEYCODE = 9
PASTE_SETTLE_S = 0.4


def accessibility_trusted() -> bool:
    from ApplicationServices import AXIsProcessTrusted

    return bool(AXIsProcessTrusted())


def _post_cmd_v() -> None:
    for key_down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(None, V_KEYCODE, key_down)
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def inject(text: str) -> None:
    """Paste text into the focused field, then restore the prior clipboard."""
    if not text:
        return
    pasteboard = NSPasteboard.generalPasteboard()
    saved = pasteboard.stringForType_(NSPasteboardTypeString)

    pasteboard.clearContents()
    pasteboard.setString_forType_(text, NSPasteboardTypeString)
    our_change_count = pasteboard.changeCount()

    _post_cmd_v()
    time.sleep(PASTE_SETTLE_S)

    if saved is not None and pasteboard.changeCount() == our_change_count:
        pasteboard.clearContents()
        pasteboard.setString_forType_(saved, NSPasteboardTypeString)
    log.debug("Injected %d chars", len(text))
