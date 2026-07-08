"""Select provider implementations from runtime config.

Heavy providers (local GPU models, hosted API clients) are imported lazily so
the slim container / laptop never needs torch or transformers just to run stubs.
"""
from __future__ import annotations

from agent import config

from .base import LLMProvider, VisionProvider


def get_vision_provider() -> VisionProvider:
    name = config.VISION_PROVIDER.lower()
    if name == "stub":
        from .stub import StubVisionProvider
        return StubVisionProvider()
    if name == "local_vlm":
        from .local_vlm import LocalVLMProvider
        return LocalVLMProvider(config.VLM_MODEL_ID)
    if name == "fireworks":
        from .fireworks_vision import FireworksVisionProvider
        return FireworksVisionProvider()
    if name == "hosted":
        from .hosted_vlm import HostedVLMProvider
        return HostedVLMProvider()
    raise ValueError(f"Unknown VISION_PROVIDER: {config.VISION_PROVIDER!r}")


def get_llm_provider() -> LLMProvider:
    name = config.LLM_PROVIDER.lower()
    if name == "stub":
        from .stub import StubLLMProvider
        return StubLLMProvider()
    if name == "local_gemma":
        from .local_gemma import LocalGemmaProvider
        return LocalGemmaProvider(config.GEMMA_MODEL_ID)
    if name == "fireworks":
        from .fireworks_gemma import FireworksGemmaProvider
        return FireworksGemmaProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER!r}")
