"""Compare two request profiles against the same vLLM server.

  long:  30 requests, ~3,000 prompt tokens, ~600 completion tokens each
  agent: 30 questions x 3.4 hops = 102 requests, ~1,400 prompt / ~40 completion each

Same model, same GPU, wildly different throughput characteristics. This is the
measurement that justifies the flags in launch.sh.
"""
from __future__ import annotations

import asyncio
import time

from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
MODEL = "Qwen/Qwen3-8B-Instruct"

SHARED_PREFIX = (
    "You are a research assistant. Answer the sub-question using ONLY the numbered "
    "passages. Be terse. Return ONLY JSON: {\"answer\": str, \"citations\": [int]}\n\n"
)


async def one(prompt: str, max_tokens: int) -> float:
    started = time.perf_counter()
    await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return time.perf_counter() - started


async def profile(name: str, count: int, prompt: str, max_tokens: int) -> None:
    started = time.perf_counter()
    latencies = await asyncio.gather(*(one(prompt, max_tokens) for _ in range(count)))
    wall = time.perf_counter() - started
    latencies.sort()
    print(
        f"{name:6s} n={count:4d}  wall={wall:6.2f}s  "
        f"p50={latencies[len(latencies) // 2]:5.2f}s  "
        f"p99={latencies[int(len(latencies) * 0.99)]:5.2f}s  "
        f"throughput={count / wall:6.2f} req/s"
    )


async def main() -> None:
    long_prompt = SHARED_PREFIX + ("passage text " * 700)
    agent_prompt = SHARED_PREFIX + ("passage text " * 320)
    await profile("long", 30, long_prompt, 600)
    await profile("agent", 102, agent_prompt, 40)


if __name__ == "__main__":
    asyncio.run(main())
