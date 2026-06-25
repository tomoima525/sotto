"""CLI entry point with headless test subcommands for each pipeline stage.

Transcripts printed here go to your terminal only; nothing is written to disk.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import Config, LANGUAGE_CHOICES


def _setup_logging(debug: bool) -> None:
    # Without --debug, stay quiet: the menu bar already shows state, so only
    # surface warnings/errors. --debug brings back the full INFO/DEBUG stream.
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


def _disable_tqdm_mp_lock() -> None:
    """Give tqdm a plain threading lock instead of its default
    multiprocessing.RLock.

    mlx_whisper creates a tqdm bar on every transcribe; tqdm's default write
    lock is a multiprocessing semaphore whose finalizer doesn't run on
    Ctrl+C, leaving a 'leaked semaphore' resource_tracker warning at exit.
    We never fork workers, so a thread lock is sufficient.
    """
    import threading

    from tqdm import tqdm

    tqdm.set_lock(threading.RLock())


def cmd_download(args, config: Config) -> None:
    from . import models

    for repo in (config.whisper_model, config.llm_model):
        print(f"Downloading {repo} ...")
        path = models.download(repo)
        print(f"  -> {path}")


def cmd_devices(args, config: Config) -> None:
    from .recorder import DEFAULT_DEVICE, default_input_device, list_input_devices

    system_default = default_input_device()
    for name in list_input_devices():
        markers = []
        if name == system_default:
            markers.append("system default")
        if name == config.input_device or (
            config.input_device == DEFAULT_DEVICE and name == system_default
        ):
            markers.append("selected")
        suffix = f"  ({', '.join(markers)})" if markers else ""
        print(f"{name}{suffix}")


def _record_seconds(seconds: float, config: Config):
    from .recorder import Recorder, resolve_input_device

    recorder = Recorder(device_name=config.input_device)
    import sounddevice as sd

    index = resolve_input_device(config.input_device)
    device = sd.query_devices(index if index is not None else sd.default.device[0])
    print(f"Recording for {seconds:.0f}s from {device['name']!r} — speak now...")
    recorder.start()
    time.sleep(seconds)
    audio = recorder.stop()
    return audio


def cmd_record(args, config: Config) -> None:
    import numpy as np

    from .recorder import SAMPLE_RATE

    audio = _record_seconds(args.seconds, config)
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    print(f"Captured {len(audio) / SAMPLE_RATE:.2f}s  rms={rms:.5f}  peak={peak:.3f}")
    if rms < 0.003:
        print("WARNING: very low level — check mic permission/input device")


def cmd_transcribe(args, config: Config) -> None:
    from .transcriber import Transcriber

    transcriber = Transcriber(config.whisper_model, args.language or config.language)
    print("Warming up Whisper...")
    transcriber.warmup()
    audio = _record_seconds(args.seconds, config)
    t0 = time.monotonic()
    text = transcriber.transcribe(audio)
    print(f"({time.monotonic() - t0:.1f}s) Transcript: {text!r}")


def cmd_stream(args, config: Config) -> None:
    """Headless streaming test: speak for N seconds, print live segments and
    the final cleaned text. Validates VAD + small-model latency without the GUI."""
    import numpy as np

    from .recorder import SAMPLE_RATE, Recorder
    from .transcriber import Transcriber
    from .vad import EnergyVADSegmenter

    lang = args.language or config.language
    model = args.model or config.streaming_whisper_model
    print(f"Streaming model: {model}  (language={lang})")
    transcriber = Transcriber(model, lang)
    print("Warming up streaming model...")
    transcriber.warmup()

    vad = EnergyVADSegmenter(
        min_silence_s=config.streaming_silence_ms / 1000,
        max_segment_s=config.streaming_max_segment_s,
    )
    recorder = Recorder(device_name=config.input_device)
    parts: list[str] = []

    def handle(seg):
        t0 = time.monotonic()
        text = transcriber.transcribe(seg)
        dt = time.monotonic() - t0
        if text:
            parts.append(text)
            print(f"  [seg {len(seg)/SAMPLE_RATE:.1f}s -> {dt:.1f}s] {text!r}")

    print(f"Speak now for {args.seconds:.0f}s (pause between phrases)...")
    recorder.start()
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        audio = recorder.drain_chunks()
        if audio is not None:
            for seg in vad.feed(audio):
                handle(seg)
        time.sleep(0.25)
    tail = recorder.drain_chunks()
    if tail is not None:
        for seg in vad.feed(tail):
            handle(seg)
    final = vad.flush()
    if final is not None:
        handle(final)
    recorder.stop()

    sep = "" if lang == "ja" else " "
    full = sep.join(parts).strip()
    print(f"\nRaw transcript: {full!r}")
    if full and not args.no_cleanup:
        from .cleaner import Cleaner

        cleaner = Cleaner(config.llm_model)
        cleaner.warmup()
        t0 = time.monotonic()
        print(f"Cleaned ({time.monotonic()-t0:.1f}s): {cleaner.clean(full)!r}")


def cmd_clean(args, config: Config) -> None:
    from .cleaner import Cleaner

    cleaner = Cleaner(config.llm_model)
    print("Loading LLM...")
    cleaner.warmup()
    t0 = time.monotonic()
    cleaned = cleaner.clean(args.text)
    print(f"({time.monotonic() - t0:.1f}s) Cleaned: {cleaned!r}")


def cmd_inject(args, config: Config) -> None:
    from .injector import accessibility_trusted, inject

    if not accessibility_trusted():
        print("WARNING: process is not Accessibility-trusted; paste may not work.")
    print(f"Injecting in {args.delay:.0f}s — focus a text field now...")
    time.sleep(args.delay)
    inject(args.text)
    print("Done. Check the focused field and that your old clipboard is restored.")


def cmd_hotkey_test(args, config: Config) -> None:
    import Quartz

    from .hotkey import HotkeyListener
    from .injector import accessibility_trusted

    print(f"Accessibility trusted: {accessibility_trusted()}")
    print(
        f"Watching hotkey {config.hotkey!r}. Hold and release it; Ctrl+C to exit.\n"
        "Prints DOWN/UP on detection, plus the raw modifier-flags word so you\n"
        "can see which bits your keyboard sets. If nothing changes when you\n"
        "press it, grant your terminal Input Monitoring in System Settings >\n"
        "Privacy & Security, then restart the terminal."
    )
    src = Quartz.kCGEventSourceStateHIDSystemState
    listener = HotkeyListener(
        config.hotkey,
        on_press=lambda: print(f"DOWN   flags=0x{int(Quartz.CGEventSourceFlagsState(src)):08X}"),
        on_release=lambda: print("UP"),
    )
    listener.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        listener.stop()


def cmd_run(args, config: Config) -> None:
    if args.language:
        # Per-invocation override; persisted only if the user later changes
        # settings via the menu.
        config.language = args.language
    if args.no_menubar:
        from .hotkey import make_listener
        from .pipeline import Pipeline

        pipeline = Pipeline(config)
        pipeline.start()
        listener = make_listener(config.hotkey, config.input_mode, pipeline)
        listener.start()
        verb = "press" if config.input_mode == "toggle" else "hold"
        print(
            f"Loading models, then {verb} {config.hotkey!r} to dictate. Ctrl+C to exit."
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            listener.stop()
            pipeline.shutdown()
    else:
        from .app import run_app

        run_app(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sotto", description="Sotto — fully-local dictation for macOS"
    )
    from . import __version__

    parser.add_argument("--version", action="version", version=f"sotto {__version__}")
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("download", help="download models into the HF cache")

    sub.add_parser("devices", help="list audio input devices")

    p = sub.add_parser("record", help="test mic capture")
    p.add_argument("--seconds", type=float, default=3)

    p = sub.add_parser("transcribe", help="record then transcribe")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument(
        "--language",
        choices=list(LANGUAGE_CHOICES),
        help="transcription language (default: config value; auto = detect from audio)",
    )

    p = sub.add_parser("stream", help="test streaming dictation headlessly")
    p.add_argument("--seconds", type=float, default=15)
    p.add_argument(
        "--language",
        choices=list(LANGUAGE_CHOICES),
        help="transcription language (default: config value)",
    )
    p.add_argument("--model", help="streaming whisper repo (default: config value)")
    p.add_argument("--no-cleanup", action="store_true", help="skip the LLM cleanup pass")

    p = sub.add_parser("clean", help="run LLM cleanup on a string")
    p.add_argument("text")

    p = sub.add_parser("inject", help="paste text into the focused app")
    p.add_argument("text")
    p.add_argument("--delay", type=float, default=3)

    sub.add_parser("hotkey-test", help="test hotkey capture + permission doctor")

    p = sub.add_parser("run", help="run the app (default: menu bar)")
    p.add_argument("--no-menubar", action="store_true", help="headless terminal mode")
    p.add_argument(
        "--language",
        choices=list(LANGUAGE_CHOICES),
        help="transcription language for this run (default: config value)",
    )

    args = parser.parse_args()
    _setup_logging(args.debug)
    _disable_tqdm_mp_lock()
    config = Config.load()

    commands = {
        "download": cmd_download,
        "devices": cmd_devices,
        "record": cmd_record,
        "transcribe": cmd_transcribe,
        "stream": cmd_stream,
        "clean": cmd_clean,
        "inject": cmd_inject,
        "hotkey-test": cmd_hotkey_test,
        "run": cmd_run,
    }
    if args.command is None:
        args.no_menubar = False
        args.language = None
        cmd_run(args, config)
    else:
        commands[args.command](args, config)


if __name__ == "__main__":
    main()
