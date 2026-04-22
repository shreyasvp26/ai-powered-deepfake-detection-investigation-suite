"""Auth dependencies (V2A-01: no-op; wire JWT / sessions in a later task)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request


async def _noop_principal(request: Request) -> None:
    return None


OptionalAuth = Annotated[Any | None, Depends(_noop_principal)]
