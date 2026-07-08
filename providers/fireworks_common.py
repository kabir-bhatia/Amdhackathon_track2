"""Shared helpers for Fireworks chat/completions responses.

Many serverless models are 'reasoning' models: the final answer is in
`message.content`, with chain-of-thought in `message.reasoning_content`. We
prefer content, fall back to reasoning if content is empty, and strip the
wrapping quotes models often add around a single caption.
"""
from __future__ import annotations

import re
import time

import requests

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def post_chat(url: str, headers: dict, payload: dict, timeout: int = 60, retries: int = 4) -> dict:
    """POST to chat/completions with backoff on rate limits / transient errors.

    Retries on 429, 5xx, and timeouts (the free tier rate-limits rapid calls).
    """
    delay = 2.0
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = requests.HTTPError(f"{resp.status_code} {resp.reason}")
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            time.sleep(delay)
            delay *= 2
    raise last_exc if last_exc else RuntimeError("post_chat failed")


def extract_text(resp_json: dict) -> str:
    """Return the model's final answer only.

    Reasoning models keep chain-of-thought in `reasoning_content` (ignored) and
    the answer in `content`. We never surface reasoning as output — if `content`
    is empty the caller treats it as a failure (usually means too few tokens).
    """
    msg = resp_json["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    text = _THINK.sub("", text).strip()  # strip inline <think>...</think> if present
    # Strip a single pair of wrapping quotes if the whole thing is quoted.
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text
