"""Backward-compatible FastAPI entry point for existing deployment commands."""

from backend.app_factory import app

__all__ = ["app"]
