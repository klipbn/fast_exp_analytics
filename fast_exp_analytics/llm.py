from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class OpenAICompatConfig:
    api_key: str
    base_url: str
    model: str
    timeout: int | None = None

def build_llm_review(*, config: OpenAICompatConfig, system_prompt: str, user_prompt: str) -> str:
    if not config.api_key:
        raise ValueError("config.api_key is required")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise ImportError("Package 'openai' is required for LLM review. Install it separately.") from exc

    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.timeout)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""
