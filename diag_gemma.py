#!/usr/bin/env python3
"""Standalone Gemma styling smoke test — surfaces the real error the pipeline hides.

Run on the pod:
    python diag_gemma.py
"""
import traceback

from agent import config

DESC = (
    "A ginger kitten walks toward the camera through green foliage in a sunny garden, "
    "with dry leaves and dirt on the ground."
)

print("GEMMA_MODEL_ID =", config.GEMMA_MODEL_ID)

try:
    from providers.local_gemma import LocalGemmaProvider
    p = LocalGemmaProvider(config.GEMMA_MODEL_ID)
    print("Model loaded OK. Trying one style() call...\n")
    out = p.style(DESC, "sarcastic")
    print("STYLE OUTPUT (sarcastic):\n", repr(out))
except Exception:
    print("---- FULL TRACEBACK ----")
    traceback.print_exc()
