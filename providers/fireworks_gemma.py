"""Gemma via the Fireworks AI API (OpenAI-compatible chat completions).

Wired for Stage 2, once the $50 coupon is redeemed and FIREWORKS_* env vars are
set. Uses plain `requests` to avoid an extra SDK dependency.
"""
from __future__ import annotations

import requests

from agent import config

from .base import LLMProvider
from .fireworks_common import extract_text, post_chat
from .prompts import build_style_prompt


class FireworksGemmaProvider(LLMProvider):
    def __init__(self):
        if not config.FIREWORKS_API_KEY or not config.FIREWORKS_MODEL_ID:
            raise RuntimeError(
                "FIREWORKS_API_KEY and FIREWORKS_MODEL_ID must be set for the fireworks provider."
            )
        self.url = config.FIREWORKS_BASE_URL.rstrip("/") + "/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {config.FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
        }
        self.model = config.FIREWORKS_MODEL_ID

    def style(self, description: str, style: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": build_style_prompt(description, style)}],
            # Headroom so reasoning models finish thinking and still emit the caption.
            "max_tokens": 1536,
            "temperature": 0.8,
            "top_p": 0.95,
        }
        return extract_text(post_chat(self.url, self.headers, payload))
