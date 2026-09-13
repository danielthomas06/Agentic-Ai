import time
import uuid

from fastapi import Request


async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    request.state.request_id = request_id

    print(
        f"[REQUEST] {request_id} "
        f"{request.method} {request.url.path}"
    )

    try:
        response = await call_next(request)

        elapsed = time.perf_counter() - start_time

        response.headers["X-Request-ID"] = request_id

        print(
            f"[RESPONSE] {request_id} "
            f"status={response.status_code} "
            f"time={elapsed:.3f}s"
        )

        return response

    except Exception as exc:
        elapsed = time.perf_counter() - start_time

        print(
            f"[ERROR] {request_id} "
            f"time={elapsed:.3f}s "
            f"error={type(exc).__name__}"
        )

        raise