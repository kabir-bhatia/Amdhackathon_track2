#!/usr/bin/env python3
"""Container entrypoint for Track 2 (Video Captioning Agent).

Contract:
  - read tasks from /input/tasks.json
  - write results to /output/results.json (valid JSON, every requested style present)
  - exit 0 on success, non-zero on failure
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

# Reduce GPU memory fragmentation (applies to ROCm/HIP too). Set before torch loads.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from agent import config
from agent.pipeline import CaptioningPipeline
from providers.factory import get_llm_provider, get_vision_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("run")


def load_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("tasks.json must be a JSON array")
    return tasks


def write_results(path: str, results) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(tmp, path)  # atomic: never leave a half-written results.json


def main() -> int:
    start = time.time()
    log.info(
        "Providers: vision=%s llm=%s | max_frames=%d sample_fps=%.2f",
        config.VISION_PROVIDER, config.LLM_PROVIDER, config.MAX_FRAMES, config.SAMPLE_FPS,
    )

    try:
        tasks = load_tasks(config.INPUT_PATH)
    except Exception:  # noqa: BLE001
        log.exception("Failed to read %s", config.INPUT_PATH)
        return 1

    # Build providers once (model load is the expensive part).
    vision = get_vision_provider()
    llm = get_llm_provider()
    pipeline = CaptioningPipeline(vision, llm)

    results = []
    for task in tasks:
        task_id = task.get("task_id")
        video_url = task.get("video_url")
        styles = task.get("styles") or list(config.STYLES)
        log.info("Task %s: %d styles", task_id, len(styles))
        try:
            captions = pipeline.run(video_url, styles)
        except Exception:  # noqa: BLE001 - one bad clip must not sink the run
            log.exception("Task %s failed; emitting fallback captions.", task_id)
            captions = {s: "A short video clip." for s in styles}
        results.append({"task_id": task_id, "captions": captions})

    try:
        write_results(config.OUTPUT_PATH, results)
    except Exception:  # noqa: BLE001
        log.exception("Failed to write %s", config.OUTPUT_PATH)
        return 1

    log.info("Done: %d tasks in %.1fs -> %s", len(results), time.time() - start, config.OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
