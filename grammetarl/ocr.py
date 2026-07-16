from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path

import requests

from .prompt_manager import render_prompt


class OCRProvider(ABC):
    @abstractmethod
    def extract_page_text(self, image_path: Path, page_number: int) -> str:
        raise NotImplementedError


class NoOCRProvider(OCRProvider):
    def extract_page_text(self, image_path: Path, page_number: int) -> str:
        return ""


class OpenAICompatibleVisionOCR(OCRProvider):
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 180,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def extract_page_text(self, image_path: Path, page_number: int) -> str:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        system_prompt = render_prompt("ocr_openai_system.j2")
        user_text = render_prompt("ocr_openai_user_text.j2", page_number=page_number)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        r = requests.post(
            f"{self.endpoint}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()


class DeepSeekOCRLocalProvider(OCRProvider):
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR-2") -> None:
        self.model_name = model_name
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "DeepSeek local OCR requires: torch, pillow, transformers"
            ) from exc

        self._torch = torch
        self._Image = Image
        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def extract_page_text(self, image_path: Path, page_number: int) -> str:
        self._lazy_load()
        image = self._Image.open(image_path).convert("RGB")
        prompt = render_prompt("ocr_deepseek_local_user_text.j2")
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=2200)
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()
