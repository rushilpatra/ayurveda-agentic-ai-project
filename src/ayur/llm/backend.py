"""Frozen local LLM behind a swappable interface.

The primary system uses a FROZEN model - no fine-tuning. The LLM is used only
for: LLM baselines, optional NL normalisation, final evidence-grounded
explanations, and a bounded language-quality subset. The planner never calls it.

Model choice comes from detected hardware (see ayur.env_detect), so changing
machines changes the tier without touching code.
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ayur.env_detect import MODEL_REPOS, detect


#: Qwen3 emits a <think>...</think> block before its answer unless thinking is
#: disabled. Measured on the first smoke run: a 64-token budget was consumed
#: entirely by reasoning, leaving no answer. Baselines must either disable
#: thinking or strip it - silently truncating mid-thought scores the model as
#: wrong for a formatting reason and would understate every LLM baseline.
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def strip_thinking(text: str) -> str:
    """Remove a leading reasoning block, closed or truncated."""
    if THINK_CLOSE in text:
        return text.split(THINK_CLOSE, 1)[1].strip()
    if text.lstrip().startswith(THINK_OPEN):
        return ""  # budget exhausted before any answer was produced
    return text.strip()


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    raw: str = ""
    truncated_in_thinking: bool = False

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds > 0 else 0.0


class LLMBackend(Protocol):
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> Generation: ...


@dataclass
class MLXBackend:
    """Apple MLX backend. Loads lazily so importing this module is cheap."""

    repo: str | None = None
    max_kv_size: int | None = None
    #: Qwen3-style models reason before answering. Off by default: the planner
    #: never calls the LLM, and for baselines an explicit budget beats a hidden one.
    thinking: bool = False
    _model: object = field(default=None, repr=False)
    _tokenizer: object = field(default=None, repr=False)

    def __post_init__(self):
        if self.repo is None:
            env = detect()
            self.repo = MODEL_REPOS[env.model_tier]
            if self.repo is None:
                raise RuntimeError(
                    f"detected memory ({env.memory_gb} GB) is below the smallest "
                    "supported MLX tier; run non-LLM stages only"
                )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.loaded:
            return
        try:
            from mlx_lm import load
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("mlx-lm not installed - run `make setup`") from e
        t0 = time.perf_counter()
        self._model, self._tokenizer = load(self.repo)
        self.load_seconds = time.perf_counter() - t0

    def unload(self) -> None:
        """Release unified memory. Matters on a 16 GB machine."""
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.0
    ) -> Generation:
        self.load()
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        messages = [{"role": "user", "content": prompt}]
        try:
            text_prompt = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.thinking,
            )
        except TypeError:
            # Tokenizer template predates the enable_thinking kwarg.
            text_prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            text_prompt = prompt

        sampler = make_sampler(temp=temperature)
        t0 = time.perf_counter()
        out = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=text_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        cleaned = strip_thinking(out)
        return Generation(
            text=cleaned,
            raw=out,
            truncated_in_thinking=(cleaned == "" and THINK_OPEN in out),
            prompt_tokens=len(self._tokenizer.encode(text_prompt)),
            completion_tokens=len(self._tokenizer.encode(out)),
            seconds=elapsed,
        )


@dataclass
class EchoBackend:
    """Deterministic stand-in for tests and for CI without a model download."""

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> Generation:
        return Generation(text=f"[echo] {prompt[:80]}", prompt_tokens=0, completion_tokens=0, seconds=0.0)


def get_backend(name: str = "mlx", **kw) -> LLMBackend:
    if name == "mlx":
        return MLXBackend(**kw)
    if name == "echo":
        return EchoBackend()
    raise ValueError(f"unknown backend: {name}")


def smoke() -> int:
    """Load the detected model and run one real generation. Records timings."""
    env = detect()
    print("=" * 64)
    print("MLX SMOKE TEST")
    print("=" * 64)
    print(f"  tier   {env.model_tier}")
    print(f"  repo   {MODEL_REPOS[env.model_tier]}")
    print(f"  memory {env.memory_gb} GB unified")
    print("-" * 64)

    backend = MLXBackend()
    print(f"loading {backend.repo} (first run downloads ~2-3 GB) ...")
    backend.load()
    print(f"  loaded in {backend.load_seconds:.1f}s")

    prompt = (
        "In Ayurveda, name the three doshas. "
        "Answer with only the three names, comma separated."
    )
    gen = backend.generate(prompt, max_tokens=64, temperature=0.0)
    print("-" * 64)
    print(f"  prompt      : {prompt}")
    print(f"  completion  : {gen.text.strip()[:300]}")
    print(f"  tokens      : {gen.prompt_tokens} in / {gen.completion_tokens} out")
    print(f"  time        : {gen.seconds:.2f}s  ({gen.tokens_per_second:.1f} tok/s)")

    out = Path("results/mlx_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "repo": backend.repo,
                "tier": env.model_tier,
                "load_seconds": round(backend.load_seconds, 2),
                "prompt": prompt,
                "completion": gen.text.strip(),
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
                "generate_seconds": round(gen.seconds, 3),
                "tokens_per_second": round(gen.tokens_per_second, 2),
            },
            indent=2,
        )
    )
    backend.unload()
    print("=" * 64)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(smoke())
