"""Light transcript cleanup with a small local LLM (mlx-lm).

Fixes punctuation/capitalization and removes filler words while preserving
wording and language. Falls back to the raw transcript if the model output
looks wrong (length ratio out of bounds) so a misbehaving LLM can never eat
a dictation.

The system prompt + few-shot prefix is constant, so its KV cache is computed
once at warmup and reused for every dictation (trimmed back after each
generation). On low-bandwidth chips (M2 Air) this saves several seconds per
dictation.
"""

from __future__ import annotations

import logging
import re
import time

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You clean up speech-to-text transcripts. Rules:
1. Remove filler words and verbal tics everywhere they appear: um, uh, like, \
you know; えーと, えっと, えー, あの, あのー, その(filler), なんか, まあ, ですね(filler).
2. Fix punctuation and capitalization. Use 。and 、for Japanese.
3. Fix obvious transcription errors.
4. Keep the original wording, meaning, and language. Never translate, never \
summarize, never answer questions contained in the text.
Output only the cleaned text, nothing else."""

FEW_SHOT = [
    (
        "um so i think uh we should like ship the new feature on friday you know",
        "So I think we should ship the new feature on Friday.",
    ),
    (
        "えっと 来週のえーと金曜日なんですけど あの 午後ならなんか空いてますので まあ ご都合いかがでしょうか",
        "来週の金曜日なんですけど、午後なら空いてますので、ご都合いかがでしょうか。",
    ),
    (
        "えーとですね あの 来週の予定なんですけど まあ 火曜日ならなんか大丈夫そうです",
        "来週の予定なんですけど、火曜日なら大丈夫そうです。",
    ),
]

MIN_LENGTH_RATIO = 0.4
MAX_LENGTH_RATIO = 1.5

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class Cleaner:
    def __init__(self, model_repo: str) -> None:
        self.model_repo = model_repo
        self._model = None
        self._tokenizer = None
        self._chat_kwargs: dict = {"enable_thinking": False}
        self._prefix_tokens: list[int] | None = None
        self._prefix_state: list | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load

        t0 = time.monotonic()
        self._model, self._tokenizer = load(self.model_repo)
        log.info("LLM loaded in %.1fs", time.monotonic() - t0)
        self._build_prefix_cache()

    def warmup(self) -> None:
        self.load()
        t0 = time.monotonic()
        self._generate("Warm up.", max_tokens=8)
        log.info("LLM warmup done in %.1fs", time.monotonic() - t0)

    # -- prompt construction --

    def _prefix_messages(self) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for raw, cleaned in FEW_SHOT:
            messages.append({"role": "user", "content": raw})
            messages.append({"role": "assistant", "content": cleaned})
        return messages

    def _apply_template(self, messages: list[dict], add_generation_prompt: bool):
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                **self._chat_kwargs,
            )
        except TypeError:
            # Tokenizer doesn't accept enable_thinking; drop it and rely on
            # <think> stripping.
            self._chat_kwargs = {}
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt
            )

    def _full_prompt_tokens(self, transcript: str) -> list[int]:
        messages = self._prefix_messages()
        messages.append({"role": "user", "content": transcript})
        return list(self._apply_template(messages, add_generation_prompt=True))

    def _build_prefix_cache(self) -> None:
        """Pre-compute the KV cache for the constant prompt prefix.

        The prefix is derived as the longest common prefix of two probe
        prompts rather than by rendering the prefix messages alone — chat
        templates treat the final turn specially (e.g. Qwen injects an empty
        <think> block into the last assistant message), so a solo render is
        not a token-prefix of the full prompt.
        """
        try:
            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache

            a = self._full_prompt_tokens("AAAA totally different probe")
            b = self._full_prompt_tokens("zzzz 全く別のプローブです")
            n = next(
                (i for i in range(min(len(a), len(b))) if a[i] != b[i]),
                min(len(a), len(b)),
            )
            prefix = a[:n]
            if len(prefix) < 8:
                log.warning("Chat template is not prefix-stable; prompt cache disabled")
                return
            cache = make_prompt_cache(self._model)
            self._model(mx.array(prefix)[None], cache=cache)
            mx.eval([
                arr
                for c in cache
                for arr in (c.state if isinstance(c.state, (list, tuple)) else [c.state])
                if arr is not None
            ])
            # Snapshot states so each generation can start from a fresh cache.
            # Copy list-typed states (ArraysCache mutates its list in place);
            # the arrays themselves are immutable under MLX slice-assignment.
            self._prefix_state = [
                (
                    list(c.state) if isinstance(c.state, list) else c.state,
                    getattr(c, "meta_state", ""),
                )
                for c in cache
            ]
            self._prefix_tokens = prefix
            log.info("Prompt prefix cached (%d tokens)", len(prefix))
        except Exception:
            log.exception("Prompt prefix caching failed; continuing without it")
            self._prefix_tokens = None
            self._prefix_state = None

    def _restore_prefix_cache(self):
        """Build a cache pre-filled with the prefix KV state (cheap: refs only)."""
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self._model)
        for c, (state, meta) in zip(cache, self._prefix_state):
            c.state = list(state) if isinstance(state, list) else state
            try:
                if meta:
                    c.meta_state = meta
            except AttributeError:
                pass
        return cache

    # -- generation --

    def _generate(self, transcript: str, max_tokens: int) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        full = self._full_prompt_tokens(transcript)
        prompt = full
        prompt_cache = None
        if self._prefix_state is not None and full[: len(self._prefix_tokens)] == self._prefix_tokens:
            prompt = full[len(self._prefix_tokens):]
            prompt_cache = self._restore_prefix_cache()

        text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
            prompt_cache=prompt_cache,
        )
        return _THINK_RE.sub("", text).strip()

    def clean(self, transcript: str) -> str:
        """Clean a transcript; returns the raw transcript on any failure."""
        transcript = transcript.strip()
        if not transcript:
            return transcript
        try:
            self.load()
            t0 = time.monotonic()
            n_input_tokens = len(self._tokenizer.encode(transcript))
            cleaned = self._generate(transcript, max_tokens=2 * n_input_tokens + 32)
            elapsed = time.monotonic() - t0
        except Exception:
            log.exception("Cleanup failed; using raw transcript")
            return transcript

        if not cleaned:
            log.warning("Cleanup returned empty; using raw transcript")
            return transcript
        ratio = len(cleaned) / len(transcript)
        if not (MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO):
            log.warning(
                "Cleanup length ratio %.2f out of bounds; using raw transcript", ratio
            )
            return transcript
        log.info("Cleaned %d -> %d chars in %.1fs", len(transcript), len(cleaned), elapsed)
        return cleaned
