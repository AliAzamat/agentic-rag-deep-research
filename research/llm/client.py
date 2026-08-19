"""One LLM entry point for the whole project.

Every call in this codebase goes through here, which is what makes cost attribution
possible later: there is exactly one place that knows how many tokens were spent.
The base_url points at an OpenAI-compatible server, so the same code runs against a
hosted endpoint today and our own vLLM server in a later step.
"""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

MODEL = os.environ.get("RESEARCH_MODEL", "Qwen/Qwen3-8B-Instruct")

_client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("LLM_API_KEY", "not-needed-for-local-vllm"),
)


def chat_json(system: str, user: str, temperature: float = 0.0) -> tuple[dict[str, Any], dict[str, int]]:
    """Call the model and parse JSON. Returns (payload, usage).

    Temperature defaults to 0 because every number this project reports is a
    comparison; sampling noise would make the eval unreadable.
    """
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }
    try:
        return json.loads(resp.choices[0].message.content or "{}"), usage
    except json.JSONDecodeError:
        # Fail soft: an unparseable generation degrades one answer, it does not
        # crash a 120-question eval run halfway through.
        return {"answer": "", "citations": []}, usage
