"""Floating live-preview overlay for streaming dictation.

A borderless, non-activating NSPanel that floats above other windows and never
takes key focus — so the final paste still lands in the app the user was typing
in. All AppKit calls run on the main thread (via libdispatch), mirroring the
menu-bar meter's dispatch pattern, so the streaming thread can call show/
set_text/hide directly without touching AppKit off-thread.
"""

from __future__ import annotations

import logging

from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSLineBreakByWordWrapping,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSStatusWindowLevel,
    NSTextField,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from libdispatch import dispatch_async, dispatch_get_main_queue

log = logging.getLogger(__name__)

_WIDTH = 520.0
_HEIGHT = 56.0
_MARGIN_TOP = 12.0  # gap below the menu bar
_FONT_SIZE = 14.0
_PAD_X = 16.0
_PAD_Y = 8.0
_MAX_CHARS = 200  # show the tail of long dictations
_PLACEHOLDER = "Listening…"


class _HUDPanel(NSPanel):
    # Never become key/main: that's what keeps focus on the user's app.
    def canBecomeKeyWindow(self):  # noqa: N802
        return False

    def canBecomeMainWindow(self):  # noqa: N802
        return False


class HUDController:
    def __init__(self) -> None:
        self._panel = None
        self._label = None

    # -- main-thread builders --

    def _build(self) -> None:
        screen = NSScreen.mainScreen()
        if screen is None:
            return
        vf = screen.visibleFrame()
        # Top-center, just below the menu bar (visibleFrame excludes the menu bar).
        x = vf.origin.x + (vf.size.width - _WIDTH) / 2.0
        y = vf.origin.y + vf.size.height - _HEIGHT - _MARGIN_TOP
        rect = NSMakeRect(x, y, _WIDTH, _HEIGHT)

        panel = _HUDPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(NSStatusWindowLevel)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)

        content = panel.contentView()
        content.setWantsLayer_(True)
        layer = content.layer()
        layer.setBackgroundColor_(
            NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.78).CGColor()
        )
        layer.setCornerRadius_(12.0)

        label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(_PAD_X, _PAD_Y, _WIDTH - 2 * _PAD_X, _HEIGHT - 2 * _PAD_Y)
        )
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setTextColor_(NSColor.whiteColor())
        label.setFont_(NSFont.systemFontOfSize_(_FONT_SIZE))
        label.setLineBreakMode_(NSLineBreakByWordWrapping)
        label.cell().setWraps_(True)
        label.setStringValue_(_PLACEHOLDER)
        content.addSubview_(label)

        self._panel = panel
        self._label = label

    # -- public API (thread-safe; each hops to the main queue) --

    def show(self) -> None:
        def work():
            if self._panel is None:
                self._build()
            if self._panel is not None:
                self._label.setStringValue_(_PLACEHOLDER)
                self._panel.orderFrontRegardless()

        dispatch_async(dispatch_get_main_queue(), work)

    def set_text(self, text: str) -> None:
        def work():
            if self._label is None:
                return
            shown = text.strip() or _PLACEHOLDER
            if len(shown) > _MAX_CHARS:
                shown = "…" + shown[-_MAX_CHARS:]
            self._label.setStringValue_(shown)

        dispatch_async(dispatch_get_main_queue(), work)

    def hide(self) -> None:
        def work():
            if self._panel is not None:
                self._panel.orderOut_(None)

        dispatch_async(dispatch_get_main_queue(), work)

    def teardown(self) -> None:
        def work():
            if self._panel is not None:
                self._panel.close()
                self._panel = None
                self._label = None

        dispatch_async(dispatch_get_main_queue(), work)
