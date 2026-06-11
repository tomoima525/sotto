# Local Dictation

Fully-local dictation for macOS (Apple Silicon). Hold a hotkey, speak in English or Japanese, release — your words are transcribed with Whisper, lightly cleaned up by a small local LLM (punctuation, filler-word removal), and pasted at the cursor of whatever app you're in.

Everything runs on-device via [MLX](https://github.com/ml-explore/mlx). No audio or text ever leaves your Mac, and nothing is persisted — audio and transcripts live only in RAM.

- **ASR**: `mlx-community/whisper-large-v3-turbo` (EN/JA auto-detect)
- **Cleanup**: `mlx-community/Qwen3.5-4B-4bit` (can be toggled off)
- **Memory**: ~4.5 GB resident while running
- **Latency**: ~2–3 s for a 10 s utterance after warmup

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an Apple Silicon Mac.

```sh
uv sync
uv run local-dictation download   # one-time, ~4 GB into the HuggingFace cache
```

### macOS permissions

macOS attributes permissions to the **app you launch from** — during development that's your terminal (Terminal.app, iTerm2, Ghostty, ...). Grant your terminal all three in **System Settings → Privacy & Security**:

| Permission | Used for | Symptom if missing |
|---|---|---|
| **Microphone** | recording your voice | recordings are silent (rms ≈ 0) |
| **Input Monitoring** | the global hold-to-talk hotkey | hotkey silently does nothing |
| **Accessibility** | simulated ⌘V to paste at the cursor | text never appears |

Microphone prompts automatically on first recording. The other two usually need to be added manually (+ button → select your terminal app). **Restart the terminal after granting.** If you switch terminal apps, re-grant.

Run the permission doctor to check:

```sh
uv run local-dictation hotkey-test
```

It prints Accessibility status and `DOWN`/`UP` when you press the hotkey; silence on keypress means Input Monitoring is missing.

## Usage

```sh
uv run local-dictation run        # menu bar app
```

Hold **Right Option (⌥)**, speak, release. The cleaned text is pasted into the focused text field and your previous clipboard is restored.

Menu bar: 🎤 idle · 🔴 recording · ✍️ processing. The menu lets you toggle LLM cleanup, change the hotkey (Right Option / Right Command / F13), and switch Whisper models.

Config lives at `~/Library/Application Support/local-dictation/config.toml`.

### Testing each stage

```sh
uv run local-dictation record --seconds 3     # mic level check
uv run local-dictation transcribe --seconds 5 # record + Whisper
uv run local-dictation clean "um so I think uh we should ship it"
uv run local-dictation inject "テスト ✅" --delay 3  # focus a text field within 3s
uv run local-dictation run --no-menubar       # full pipeline, headless with logs
```

## Limitations

- Doesn't work in secure input fields (passwords) — macOS blocks synthetic events there by design.
- Non-text clipboard contents (images, files) are not restored after pasting.
- The `fn` key can't be used as the hotkey (not visible to event taps).
