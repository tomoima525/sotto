# Sotto

> _sotto voce_ — in a quiet voice · **そっと** — softly, unobtrusively

Fully-local dictation for macOS (Apple Silicon). Hold a hotkey, speak in English or Japanese, release — your words are transcribed with Whisper, lightly cleaned up by a small local LLM (punctuation, filler-word removal), and pasted at the cursor of whatever app you're in.

Everything runs on-device via [MLX](https://github.com/ml-explore/mlx). No audio or text ever leaves your Mac, and nothing is persisted — audio and transcripts live only in RAM.

- **ASR**: `mlx-community/whisper-large-v3-turbo` (EN/JA auto-detect)
- **Cleanup**: `mlx-community/Qwen3.5-4B-4bit` (can be toggled off)
- **Memory**: ~4.5 GB resident while running
- **Latency**: ~1 s when the transcript needs no cleanup, ~2–3 s when the LLM runs (after warmup)

## Setup

### From HomeBrew

```
brew install tomoima525/sotto/sotto
sotto download   # one-time model fetch (~4 GB)
sotto run        # menu bar app
```

Tap repo: https://github.com/tomoima525/homebrew-sotto

### From source

Requires [uv](https://docs.astral.sh/uv/) and an Apple Silicon Mac.

```sh
uv sync
uv run sotto download   # one-time, ~4 GB into the HuggingFace cache
```

### macOS permissions

macOS attributes permissions to the **app you launch from** — during development that's your terminal (Terminal.app, iTerm2, Ghostty, ...). Grant your terminal all three in **System Settings → Privacy & Security**:

| Permission           | Used for                            | Symptom if missing              |
| -------------------- | ----------------------------------- | ------------------------------- |
| **Microphone**       | recording your voice                | recordings are silent (rms ≈ 0) |
| **Input Monitoring** | the global hold-to-talk hotkey      | hotkey silently does nothing    |
| **Accessibility**    | simulated ⌘V to paste at the cursor | text never appears              |

Microphone prompts automatically on first recording. The other two usually need to be added manually (+ button → select your terminal app). **Restart the terminal after granting.** If you switch terminal apps, re-grant.

Run the permission doctor to check:

```sh
uv run sotto hotkey-test
```

It prints Accessibility status and `DOWN`/`UP` when you press the hotkey; silence on keypress means Input Monitoring is missing.

## Usage

```sh
uv run sotto run        # menu bar app
```

Hold **Right Option (⌥)**, speak, release. The cleaned text is pasted into the focused text field and your previous clipboard is restored. (Prefer not to hold the key? Switch **Input Mode** to **Toggle** and press once to start, again to stop.)

### Input modes

- **Hold to talk** — hold the hotkey while speaking, release to transcribe.
- **Toggle** — press once to start, again to stop.
- **Streaming (live preview)** — press once to start; a floating overlay shows your words **as you speak**, transcribed phrase-by-phrase by a small fast Whisper model. Press again to stop; the full transcript is cleaned up by the LLM and pasted once into the focused app. The overlay never steals focus. Streaming uses a separate, smaller model for low latency — pick it under **Streaming Model**: **Small** (balanced, for daily use) or **Base** (faster, lower accuracy). Accuracy is lower than the default turbo model, so the final LLM cleanup compensates somewhat.

Menu bar: 🎤 idle · live waveform while recording · ✍️ processing · 🟢 streaming. The recording waveform is driven by your mic level, so a flat line while you speak means the wrong input device is selected. The menu lets you toggle LLM cleanup, choose the input mode, set the language (**Universal** auto-detects from the audio; force **English** or **Japanese** for short utterances that auto-detect gets wrong), pick the microphone (the **Microphone** submenu shows which device is in use — virtual devices from Loom/Zoom/etc. can silently become the system default), change the hotkey (Right Option / Right Command / F13), and switch the Whisper / Streaming models.

Config lives at `~/Library/Application Support/sotto/config.toml`.

### Cleanup gate

Whisper already returns punctuated, capitalized text in both languages, so on a
filler-free utterance the cleanup LLM spends ~1 s to hand back what it was given
— and occasionally rewrites Japanese politeness forms it was told to preserve.
Sotto therefore runs the LLM only when the transcript actually needs it: when it
contains filler words, or when it lacks sentence-final punctuation (a fragment).
On clean dictation this removes roughly half the post-release latency.

The tradeoff is that skipped transcripts also miss the LLM's incidental fixes
(e.g. "ten thirty" → "10:30"). To always run the model, set `cleanup_gate = false`
in `config.toml`. To turn cleanup off entirely, use the menu bar toggle.

### Testing each stage

```sh
uv run sotto devices                # list input devices, show selected
uv run sotto record --seconds 3     # mic level check
uv run sotto transcribe --seconds 5 # record + Whisper
uv run sotto transcribe --language ja  # force Japanese (auto/en/ja)
uv run sotto stream --seconds 15    # streaming: live segments + final cleaned text
uv run sotto stream --model mlx-community/whisper-base-mlx  # try a faster model
uv run sotto clean "um so I think uh we should ship it"
uv run sotto clean --force "Already clean."  # bypass the gate, always run the LLM
uv run sotto gate-test              # cleanup gate: decisions + time saved (no mic needed)
uv run sotto gate-test --no-verify  # gate decisions only, instant (skips the model load)
uv run sotto inject "テスト ✅" --delay 3  # focus a text field within 3s
uv run sotto run --no-menubar       # full pipeline, headless with logs
```

## Limitations

- Doesn't work in secure input fields (passwords) — macOS blocks synthetic events there by design.
- Non-text clipboard contents (images, files) are not restored after pasting.
- The `fn` key can't be used as the hotkey (not visible to event taps).
