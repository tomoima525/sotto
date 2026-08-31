"""Floating live-preview overlay for streaming dictation.

A borderless, non-activating NSPanel that floats above other windows and never
takes key focus — so the final paste still lands in the app the user was typing
in. All AppKit calls run on the main thread (via libdispatch), mirroring the
menu-bar meter's dispatch pattern, so the streaming thread can call show/
set_text/hide directly without touching AppKit off-thread.

The panel is pinned by its top edge and grows downward as the transcript wraps
onto more lines, up to a fraction of the screen height; past that it shows the
tail. A status row (spinner + label) appears while the post-stop refine and
cleanup passes run, so the wait after releasing the hotkey isn't silent.
"""

from __future__ import annotations

import logging

from AppKit import (
    NSAnimationContext,
    NSAttributedString,
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSLineBreakByWordWrapping,
    NSMakeRect,
    NSMutableParagraphStyle,
    NSPanel,
    NSParagraphStyleAttributeName,
    NSScreen,
    NSStatusWindowLevel,
    NSStringDrawingUsesFontLeading,
    NSStringDrawingUsesLineFragmentOrigin,
    NSTextField,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from libdispatch import (
    DISPATCH_TIME_NOW,
    dispatch_after,
    dispatch_async,
    dispatch_get_main_queue,
    dispatch_time,
)

log = logging.getLogger(__name__)

_WIDTH = 520.0
_MIN_HEIGHT = 56.0
_MAX_HEIGHT_FRAC = 0.40  # of the screen's visible height
_MARGIN_TOP = 12.0  # gap below the menu bar
_FONT_SIZE = 14.0
_STATUS_FONT_SIZE = 11.0
_STATUS_H = 14.0
_STATUS_GAP = 6.0
_PAD_X = 16.0
_PAD_Y = 8.0
_PLACEHOLDER = "Listening…"

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.09
_FADE_DURATION = 0.35


def _nanos(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


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
        self._status_label = None
        self._text = _PLACEHOLDER
        self._status = ""
        self._spin_frame = 0
        self._spin_gen = 0  # bumps to cancel an in-flight spinner loop
        self._dismiss_gen = 0  # bumps to cancel an in-flight dismissal

    # -- main-thread builders --

    def _build(self) -> None:
        screen = NSScreen.mainScreen()
        if screen is None:
            return
        vf = screen.visibleFrame()
        # Top-center, just below the menu bar (visibleFrame excludes the menu bar).
        x = vf.origin.x + (vf.size.width - _WIDTH) / 2.0
        y = vf.origin.y + vf.size.height - _MIN_HEIGHT - _MARGIN_TOP
        rect = NSMakeRect(x, y, _WIDTH, _MIN_HEIGHT)

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
            NSMakeRect(_PAD_X, _PAD_Y, _WIDTH - 2 * _PAD_X, _MIN_HEIGHT - 2 * _PAD_Y)
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

        status = NSTextField.alloc().initWithFrame_(
            NSMakeRect(_PAD_X, _PAD_Y, _WIDTH - 2 * _PAD_X, _STATUS_H)
        )
        status.setBezeled_(False)
        status.setDrawsBackground_(False)
        status.setEditable_(False)
        status.setSelectable_(False)
        status.setTextColor_(NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.6))
        status.setFont_(NSFont.systemFontOfSize_(_STATUS_FONT_SIZE))
        status.setStringValue_("")
        status.setHidden_(True)
        content.addSubview_(status)

        self._panel = panel
        self._label = label
        self._status_label = status

    # -- layout (main thread only) --

    def _text_attrs(self):
        para = NSMutableParagraphStyle.alloc().init()
        para.setLineBreakMode_(NSLineBreakByWordWrapping)
        return {
            NSFontAttributeName: NSFont.systemFontOfSize_(_FONT_SIZE),
            NSParagraphStyleAttributeName: para,
        }

    def _measure(self, text: str) -> float:
        attrs = self._text_attrs()
        s = NSAttributedString.alloc().initWithString_attributes_(text, attrs)
        rect = s.boundingRectWithSize_options_(
            (_WIDTH - 2 * _PAD_X, 1.0e6),
            NSStringDrawingUsesLineFragmentOrigin | NSStringDrawingUsesFontLeading,
        )
        return float(rect.size.height)

    def _max_height(self) -> float:
        screen = NSScreen.mainScreen()
        if screen is None:
            return 400.0
        return screen.visibleFrame().size.height * _MAX_HEIGHT_FRAC

    def _fit_tail(self, text: str, budget: float) -> str:
        """Longest tail of `text` that renders within `budget` points, "…"-prefixed."""
        if self._measure(text) <= budget:
            return text
        # Binary search the earliest start offset whose tail still fits.
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._measure("…" + text[mid:]) <= budget:
                hi = mid
            else:
                lo = mid + 1
        return "…" + text[lo:]

    def _relayout(self) -> None:
        """Resize the panel to the current text/status and place the subviews."""
        if self._panel is None:
            return
        has_status = bool(self._status)
        chrome = 2 * _PAD_Y + ((_STATUS_H + _STATUS_GAP) if has_status else 0.0)
        max_text_h = max(self._max_height() - chrome, self._measure("X"))
        shown = self._fit_tail(self._text, max_text_h)
        text_h = min(self._measure(shown), max_text_h)
        height = max(_MIN_HEIGHT, chrome + text_h)
        text_h = height - chrome  # absorb the min-height slack into the text area

        self._label.setStringValue_(shown)
        # AppKit's origin is bottom-left: the status row sits below the text.
        if has_status:
            self._status_label.setFrame_(
                NSMakeRect(_PAD_X, _PAD_Y, _WIDTH - 2 * _PAD_X, _STATUS_H)
            )
            self._status_label.setHidden_(False)
            text_y = _PAD_Y + _STATUS_H + _STATUS_GAP
        else:
            self._status_label.setHidden_(True)
            text_y = _PAD_Y
        self._label.setFrame_(
            NSMakeRect(_PAD_X, text_y, _WIDTH - 2 * _PAD_X, text_h)
        )

        frame = self._panel.frame()
        if abs(frame.size.height - height) > 0.5:
            # Keep the top edge pinned; grow downward.
            top = frame.origin.y + frame.size.height
            self._panel.setFrame_display_(
                NSMakeRect(frame.origin.x, top - height, _WIDTH, height), True
            )

    # -- spinner (main thread only) --

    def _render_status(self) -> None:
        if self._status_label is None or not self._status:
            return
        frame = _SPINNER_FRAMES[self._spin_frame % len(_SPINNER_FRAMES)]
        self._status_label.setStringValue_(f"{frame}  {self._status}")

    def _spin(self, gen: int) -> None:
        if gen != self._spin_gen or not self._status:
            return
        self._spin_frame += 1
        self._render_status()
        dispatch_after(
            dispatch_time(DISPATCH_TIME_NOW, _nanos(_SPINNER_INTERVAL)),
            dispatch_get_main_queue(),
            lambda: self._spin(gen),
        )

    def _stop_spinner(self) -> None:
        self._spin_gen += 1

    # -- public API (thread-safe; each hops to the main queue) --

    def show(self) -> None:
        def work():
            self._dismiss_gen += 1  # cancel a pending fade-out
            if self._panel is None:
                self._build()
            if self._panel is None:
                return
            self._text = _PLACEHOLDER
            self._status = ""
            self._stop_spinner()
            self._status_label.setHidden_(True)
            self._relayout()
            self._panel.setAlphaValue_(1.0)
            self._panel.orderFrontRegardless()

        dispatch_async(dispatch_get_main_queue(), work)

    def set_text(self, text: str) -> None:
        def work():
            if self._label is None:
                return
            self._text = text.strip() or _PLACEHOLDER
            self._relayout()

        dispatch_async(dispatch_get_main_queue(), work)

    def set_status(self, status: str) -> None:
        """Show (or, with an empty string, clear) the spinner + status row."""

        def work():
            if self._status_label is None:
                return
            was_running = bool(self._status)
            self._status = status.strip()
            if not self._status:
                self._stop_spinner()
                self._status_label.setStringValue_("")
                self._relayout()
                return
            self._render_status()
            self._relayout()
            if not was_running:
                self._spin_gen += 1
                self._spin(self._spin_gen)

        dispatch_async(dispatch_get_main_queue(), work)

    def dismiss(self, hold: float = 1.0) -> None:
        """Hold the final text briefly, then fade the panel out."""

        def work():
            if self._panel is None:
                return
            self._status = ""
            self._stop_spinner()
            self._status_label.setHidden_(True)
            self._relayout()
            self._dismiss_gen += 1
            gen = self._dismiss_gen
            dispatch_after(
                dispatch_time(DISPATCH_TIME_NOW, _nanos(hold)),
                dispatch_get_main_queue(),
                lambda: self._fade_out(gen),
            )

        dispatch_async(dispatch_get_main_queue(), work)

    def _fade_out(self, gen: int) -> None:
        if gen != self._dismiss_gen or self._panel is None:
            return

        def animate(ctx):
            ctx.setDuration_(_FADE_DURATION)
            self._panel.animator().setAlphaValue_(0.0)

        def done():
            if gen != self._dismiss_gen or self._panel is None:
                return
            self._panel.orderOut_(None)
            self._panel.setAlphaValue_(1.0)

        NSAnimationContext.runAnimationGroup_completionHandler_(animate, done)

    def hide(self) -> None:
        def work():
            self._dismiss_gen += 1
            self._stop_spinner()
            self._status = ""
            if self._panel is not None:
                self._panel.orderOut_(None)
                self._panel.setAlphaValue_(1.0)

        dispatch_async(dispatch_get_main_queue(), work)

    def teardown(self) -> None:
        def work():
            self._dismiss_gen += 1
            self._stop_spinner()
            if self._panel is not None:
                self._panel.close()
                self._panel = None
                self._label = None
                self._status_label = None

        dispatch_async(dispatch_get_main_queue(), work)
