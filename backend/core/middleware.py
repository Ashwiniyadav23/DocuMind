from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .exceptions import AIServiceError, DocumentProcessingError


class RequestContextMiddleware:
    """Attach a request ID and return consistent domain error responses."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except (AIServiceError, DocumentProcessingError) as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content={"detail": str(exc)},
                headers={"X-Request-ID": request_id},
            )
            await response(scope, receive, send)


def get_request_id(request: Request) -> str:
    return request.scope.get("state", {}).get("request_id", "")
