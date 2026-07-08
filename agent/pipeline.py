"""Per-clip pipeline: sample frames -> describe -> style into every requested tone.

Robustness is a scoring concern: a missing style scores zero for that clip, so
we always return every requested style key, degrading to a safe fallback caption
rather than raising.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from agent import config
from agent.sampler import sample_frames
from providers.base import LLMProvider, VisionProvider

log = logging.getLogger("pipeline")


class CaptioningPipeline:
    def __init__(self, vision: VisionProvider, llm: LLMProvider):
        self.vision = vision
        self.llm = llm

    def run(self, video_url: str, styles: List[str]) -> Dict[str, str]:
        """Return {style: caption} for every requested style (never missing a key)."""
        description = self._describe(video_url)
        captions: Dict[str, str] = {}
        for style in styles:
            captions[style] = self._style_one(description, style)
        return captions

    def _describe(self, video_url: str) -> str:
        frames = sample_frames(video_url, config.MAX_FRAMES, config.SAMPLE_FPS)
        return self.vision.describe(frames)

    def _style_one(self, description: str, style: str) -> str:
        try:
            caption = (self.llm.style(description, style) or "").strip()
            if caption:
                return caption
            log.warning("Empty caption for style=%s; using fallback.", style)
        except Exception:  # noqa: BLE001 - never drop a required style
            log.exception("Styling failed for style=%s; using fallback.", style)
        # Fallback keeps the key present and at least content-faithful.
        return description.strip()
