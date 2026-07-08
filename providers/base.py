"""Provider interfaces.

Two stages, two interfaces:
  - VisionProvider: frames -> one neutral, factual scene description.
  - LLMProvider:    (description, style) -> a caption in that style.

Implementations live alongside this file and are chosen at runtime by
`providers.factory` based on env config.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class VisionProvider(ABC):
    """Turns sampled video frames into a factual scene description."""

    @abstractmethod
    def describe(self, frames: List["np.ndarray"]) -> str:
        """Return a neutral, style-free description of what the frames show."""
        raise NotImplementedError


class LLMProvider(ABC):
    """Rewrites a factual description into a caption of a requested style."""

    @abstractmethod
    def style(self, description: str, style: str) -> str:
        """Return a caption for `description` in the given `style`."""
        raise NotImplementedError
