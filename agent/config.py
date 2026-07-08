"""Runtime configuration, read purely from environment variables.

Keeps *what* the pipeline does separate from *where* models run, so the same
image can point at a stub, a local GPU model, or a hosted API with no code change.
"""
import os

# The four styles every clip must be captioned in (order is stable for prompts).
STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")

# Container I/O contract (fixed by the hackathon harness).
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Provider selection.
VISION_PROVIDER = _get("VISION_PROVIDER", "stub")   # stub | local_vlm | hosted
LLM_PROVIDER = _get("LLM_PROVIDER", "stub")         # stub | local_gemma | fireworks

# Frame sampling.
MAX_FRAMES = int(_get("MAX_FRAMES", "8"))
SAMPLE_FPS = float(_get("SAMPLE_FPS", "1.0"))
# Cap the longest side of each frame before it reaches the VLM. UHD frames
# otherwise expand into an enormous number of visual tokens and OOM the GPU.
MAX_IMAGE_SIDE = int(_get("MAX_IMAGE_SIDE", "768"))

# Local model IDs (GPU pod).
VLM_MODEL_ID = _get("VLM_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
GEMMA_MODEL_ID = _get("GEMMA_MODEL_ID", "google/gemma-2-2b-it")

# Fireworks (populated once the coupon is redeemed).
FIREWORKS_API_KEY = _get("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = _get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_MODEL_ID = _get("FIREWORKS_MODEL_ID", "")
