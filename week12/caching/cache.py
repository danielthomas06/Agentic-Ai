import hashlib
import json
import time
from pathlib import Path
from typing import Any


class LLMCache:
    def __init__(
        self,
        cache_dir: str = "week12/caching/cache",
        ttl_seconds: int = 3600,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ttl_seconds = ttl_seconds

        self.hits = 0
        self.misses = 0
        self.expired = 0

    def _make_key(
        self,
        model: str,
        prompt: str,
    ) -> str:
        """
        Create a deterministic cache key.

        The model is part of the key so that responses generated
        by different models are never mixed.
        """

        raw = f"{model}:{prompt}"

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    def get(
        self,
        model: str,
        prompt: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve a cached response.

        Returns None when:
        - no cache entry exists
        - cache entry has expired
        """

        key = self._make_key(model, prompt)
        cache_file = self.cache_dir / f"{key}.json"

        # Cache miss
        if not cache_file.exists():
            self.misses += 1
            return None

        try:
            with cache_file.open("r", encoding="utf-8") as f:
                cached = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None

        created_at = cached.get("created_at")

        if created_at is None:
            self.misses += 1
            return None

        age = time.time() - created_at

        # Cache expired
        if age > self.ttl_seconds:
            self.expired += 1
            self.misses += 1

            try:
                cache_file.unlink()
            except OSError:
                pass

            return None

        # Cache hit
        self.hits += 1

        return cached["response"]

    def set(
        self,
        model: str,
        prompt: str,
        response: dict[str, Any],
    ) -> None:
        """
        Store an LLM response in the cache.
        """

        key = self._make_key(model, prompt)
        cache_file = self.cache_dir / f"{key}.json"

        payload = {
            "created_at": time.time(),
            "model": model,
            "prompt": prompt,
            "response": response,
        }

        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                indent=2,
            )

    def stats(self) -> dict[str, Any]:
        """
        Return cache performance statistics.
        """

        total_requests = self.hits + self.misses

        hit_rate = (
            self.hits / total_requests
            if total_requests > 0
            else 0.0
        )

        return {
            "hits": self.hits,
            "misses": self.misses,
            "expired": self.expired,
            "total_requests": total_requests,
            "hit_rate": round(hit_rate, 3),
            "ttl_seconds": self.ttl_seconds,
        }

    def clear(self) -> None:
        """
        Delete all cached responses.
        """

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except OSError:
                pass
