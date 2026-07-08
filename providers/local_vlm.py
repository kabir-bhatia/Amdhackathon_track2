"""Open vision-language model for scene description, run on the AMD GPU pod.

Default: Qwen2-VL-2B-Instruct (small, strong at multi-frame description). Loaded
lazily via transformers so the laptop/stub path never imports torch.
"""
from __future__ import annotations

from typing import List

import numpy as np

from agent import config

from .base import VisionProvider
from .prompts import VISION_PROMPT


class LocalVLMProvider(VisionProvider):
    def __init__(self, model_id: str):
        import torch  # noqa: PLC0415 - lazy heavy import
        from PIL import Image  # noqa: PLC0415
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # noqa: PLC0415

        self._torch = torch
        self._Image = Image
        self.model_id = model_id
        self.max_side = config.MAX_IMAGE_SIDE
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        # Bound the visual-token budget so UHD frames can't OOM the GPU.
        max_pixels = self.max_side * self.max_side
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=256 * 28 * 28, max_pixels=max_pixels
        )

    def _resize(self, img):
        w, h = img.size
        longest = max(w, h)
        if longest <= self.max_side:
            return img
        scale = self.max_side / longest
        return img.resize((int(w * scale), int(h * scale)), self._Image.BILINEAR)

    def describe(self, frames: List["np.ndarray"]) -> str:
        images = [self._resize(self._Image.fromarray(f)) for f in frames]
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": VISION_PROMPT})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=160, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        desc = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        # Release activation memory so it can't accumulate across clips.
        del inputs, out, trimmed
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        return desc.strip()
