"""Placeholder for a hosted VLM API vision provider (Stage 2).

Kept as a named impl so the factory wiring is complete. Fill in with a concrete
hosted vision endpoint (e.g. a Fireworks vision model or other API) when chosen.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import VisionProvider


class HostedVLMProvider(VisionProvider):
    def __init__(self):
        raise NotImplementedError(
            "HostedVLMProvider is a Stage 2 placeholder; use 'local_vlm' or 'stub' for now."
        )

    def describe(self, frames: List["np.ndarray"]) -> str:  # pragma: no cover
        raise NotImplementedError
