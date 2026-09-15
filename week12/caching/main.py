import time

import requests

from .cache import LLMCache


MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"


cache = LLMCache(
    ttl_seconds=60,
)


def call_llm(prompt: str) -> tuple[dict, bool, float]:
    start = time.perf_counter()

    cached = cache.get(
        model=MODEL,
        prompt=prompt,
    )

    if cached is not None:
        elapsed = time.perf_counter() - start

        return cached, True, elapsed

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        },
        timeout=300,
    )

    response.raise_for_status()

    data = response.json()

    cache.set(
        model=MODEL,
        prompt=prompt,
        response=data,
    )

    elapsed = time.perf_counter() - start

    return data, False, elapsed


def main():
    prompt = "Explain visual odometry in 200 words."

    print("=== Request 1 ===")

    response, cached, elapsed = call_llm(prompt)

    print(f"Cache hit: {cached}")
    print(f"Time: {elapsed:.3f}s")

    print("\n=== Request 2 ===")

    response, cached, elapsed = call_llm(prompt)

    print(f"Cache hit: {cached}")
    print(f"Time: {elapsed:.3f}s")

    print("\n=== Cache statistics ===")

    print(cache.stats())

    print("\n=== Waiting for TTL ===")

    print(
        f"TTL is {cache.ttl_seconds} seconds."
    )


if __name__ == "__main__":
    main()
