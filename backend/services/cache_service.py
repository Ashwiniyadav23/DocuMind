"""Optional Redis cache; all Redis failures degrade to an uncached request."""

from __future__ import annotations

import logging
from functools import lru_cache

from ..core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client():
    url = get_settings().redis_url
    if not url:
        return None
    try:
        from redis import Redis

        return Redis.from_url(url, decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
    except Exception:
        logger.warning("Redis setup failed; continuing without cache.", exc_info=True)
        return None


def get(key: str) -> str | None:
    try:
        client = _client()
        return client.get(key) if client else None
    except Exception:
        logger.warning("Redis read failed; treating it as a cache miss.", exc_info=True)
        return None


def set(key: str, value: str, ttl: int) -> None:
    try:
        client = _client()
        if client:
            client.set(key, value, ex=ttl)
    except Exception:
        logger.warning("Redis write failed; returning the generated answer.", exc_info=True)


def invalidate_answers() -> None:
    """Remove chat entries after the indexed document set changes."""
    try:
        client = _client()
        if client:
            for key in client.scan_iter(match="documind:answer:v1:*", count=200):
                client.delete(key)
    except Exception:
        logger.warning("Redis invalidation failed; cached answers will expire by TTL.", exc_info=True)
