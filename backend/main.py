"""ASGI entry point supporting both package and legacy startup commands."""

from __future__ import annotations

if __package__:
    from .app_factory import app
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.app_factory import app

__all__ = ["app"]
