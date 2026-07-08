"""Download a clip and sample a capped set of evenly-spaced frames.

Uses OpenCV only (no ffmpeg CLI dependency for decoding), so it works the same
on the laptop and in the slim container. Frames are returned as RGB numpy arrays.
"""
from __future__ import annotations

import os
import tempfile
from typing import List

import cv2
import numpy as np
import requests

_DOWNLOAD_TIMEOUT = 60  # seconds; clips are <=2 min, keep well under the 10-min budget


def _download(video_url: str) -> str:
    """Fetch the clip to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    with requests.get(video_url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return path


def _resolve(video_url: str):
    """Return (path, is_temp). Accepts http(s) URLs, file:// URLs, and local paths."""
    if video_url.startswith(("http://", "https://")):
        return _download(video_url), True
    local = video_url[7:] if video_url.startswith("file://") else video_url
    if os.path.exists(local):
        return local, False
    raise FileNotFoundError(f"video not found: {video_url}")


def sample_frames(video_url: str, max_frames: int, sample_fps: float) -> List["np.ndarray"]:
    """Return up to `max_frames` RGB frames, evenly spaced across the clip.

    `sample_fps` bounds how densely we *consider* frames; the even-spacing cap of
    `max_frames` is what actually limits how many reach the vision model.
    """
    path, is_temp = _resolve(video_url)
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_url}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if total <= 0:
            # Fallback: read sequentially if metadata is missing.
            return _sample_sequential(cap, max_frames)

        duration = total / fps if fps > 0 else 0
        # Candidate count bounded by both the requested fps and the hard frame cap.
        n = max(1, min(max_frames, int(duration * sample_fps) or max_frames))
        indices = np.linspace(0, total - 1, num=n, dtype=int)

        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames decoded from: {video_url}")
        return frames
    finally:
        if is_temp:
            try:
                os.remove(path)
            except OSError:
                pass


def _sample_sequential(cap, max_frames: int) -> List["np.ndarray"]:
    frames: List[np.ndarray] = []
    read = 0
    while read < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        read += 1
    cap.release()
    if not frames:
        raise RuntimeError("No frames decoded (sequential fallback).")
    return frames
