"""
Inference engine for NL2SQL.

Wraps base model + LoRA adapter loading. Used by:
  - eval/run_benchmark.py
  - api/main.py
  - app/streamlit_demo.py

Defaults to 4-bit quantization for low-VRAM inference (e.g., HF Spaces, Colab CPU+GPU).
"""
from __future__ import annotations

import os
from typing import Any


class InferenceEngine:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
        }

        if load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = bnb_config
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"loaded adapter: {adapter_path}")

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Greedy by default — SQL gen wants determinism, not creativity."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = temperature > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Only decode new tokens, not the prompt
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


# Module-level singleton — one model load per process
_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")
        adapter = os.getenv("ADAPTER_PATH") or None
        _engine = InferenceEngine(model_name=model_name, adapter_path=adapter)
    return _engine
