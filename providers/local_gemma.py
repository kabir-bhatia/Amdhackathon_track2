"""Open Gemma model for style rewriting, run on the AMD GPU pod.

Default: gemma-2-2b-it. Also positions us for the $3,000 "Best Use of Gemma"
prize. Loaded lazily so the stub/laptop path stays torch-free.
"""
from __future__ import annotations

from .base import LLMProvider
from .prompts import build_style_prompt


class LocalGemmaProvider(LLMProvider):
    def __init__(self, model_id: str):
        import torch  # noqa: PLC0415 - lazy heavy import
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self.model_id = model_id
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )

    def style(self, description: str, style: str) -> str:
        prompt = build_style_prompt(description, style)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with self._torch.no_grad():
            out = self.model.generate(
                inputs, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.95
            )
        text = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        )
        return text.strip()
