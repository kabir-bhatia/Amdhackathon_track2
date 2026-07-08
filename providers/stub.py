"""Deterministic stub providers for harness/Docker testing without a real model.

Lets us validate the full container contract (read tasks -> sample frames ->
produce all 4 styles -> write valid results.json) on a 4 GB-VRAM laptop.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import LLMProvider, VisionProvider


class StubVisionProvider(VisionProvider):
    def describe(self, frames: List["np.ndarray"]) -> str:
        n = len(frames)
        return (
            f"A short video clip summarized from {n} sampled frames. "
            "The scene shows a subject moving through an everyday setting with "
            "natural lighting and clear background detail."
        )


class StubLLMProvider(LLMProvider):
    _TEMPLATES = {
        "formal": "The footage depicts {d}",
        "sarcastic": "Oh good, more riveting content: {d}",
        "humorous_tech": "Shipped straight to prod with zero tests: {d}",
        "humorous_non_tech": "Peak entertainment, honestly: {d}",
    }

    def style(self, description: str, style: str) -> str:
        d = description.strip().rstrip(".").lower()
        return self._TEMPLATES[style].format(d=d) + "."
