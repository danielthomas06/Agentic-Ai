from fastapi import FastAPI

from .middleware import request_logging_middleware
from .routes import router


app = FastAPI(
    title="Multi-Agent Research API",
    description="FastAPI interface for the Week 7 multi-agent supervisor.",
    version="1.0.0",
)


app.middleware("http")(request_logging_middleware)

app.include_router(router)