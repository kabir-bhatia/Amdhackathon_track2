"""Vision-stage provider using a Fireworks serverless multimodal model.

Sends sampled frames (as base64 JPEG data URIs) to the OpenAI-compatible
chat/completions endpoint and returns a neutral factual description. Frames are
downscaled to bound payload size, latency, and token cost.
"""
from __future__ import annotations

import base64
from typing import List

import cv2
import numpy as np
import requests

from agent import config

from .base import VisionProvider
from .prompts import VISION_PROMPT


class FireworksVisionProvider(VisionProvider):
    def __init__(self):
        if not config.FIREWORKS_API_KEY:
            raise RuntimeError("FIREWORKS_API_KEY must be set for the fireworks vision provider.")
        self.url = config.FIREWORKS_BASE_URL.rstrip("/") + "/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {config.FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
        }
        self.model = config.FIREWORKS_VISION_MODEL_ID
        self.max_side = config.MAX_IMAGE_SIDE

    def _to_data_uri(self, frame: "np.ndarray") -> str:
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest > self.max_side:
            scale = self.max_side / longest
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode frame")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def describe(self, frames: List["np.ndarray"]) -> str:
        content = [{"type": "text", "text": VISION_PROMPT}]
        for f in frames:
            content.append({"type": "image_url", "image_url": {"url": self._to_data_uri(f)}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 200,
            "temperature": 0.2,
        }
        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
