"""Open vision-language model for scene description, run on the AMD GPU pod.

Default: Qwen2-VL-2B-Instruct (small, strong at multi-frame description). Loaded
lazily via transformers so the laptop/stub path never imports torch.
"""
from __future__ import annotations

from typing import List

import numpy as np

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
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def describe(self, frames: List["np.ndarray"]) -> str:
        images = [self._Image.fromarray(f) for f in frames]
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
        return desc.strip()
