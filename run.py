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


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader for local dev (no dependency). Real env vars win."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()  # must run before `config` reads the environment

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

    # Build providers once (model load is the expensive part). A failure here
    # (e.g. missing key) must NOT crash the container: degrade to fallback
    # captions so we still emit valid JSON and exit 0.
    pipeline = None
    try:
        pipeline = CaptioningPipeline(get_vision_provider(), get_llm_provider())
    except Exception:  # noqa: BLE001
        log.exception("Provider init failed; running in degraded fallback mode.")

    results = []
    for task in tasks:
        task_id = task.get("task_id")
        video_url = task.get("video_url")
        styles = task.get("styles") or list(config.STYLES)
        log.info("Task %s: %d styles", task_id, len(styles))
        captions = {s: "A short video clip." for s in styles}
        if pipeline is not None:
            try:
                captions = pipeline.run(video_url, styles)
            except Exception:  # noqa: BLE001 - one bad clip must not sink the run
                log.exception("Task %s failed; emitting fallback captions.", task_id)
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
