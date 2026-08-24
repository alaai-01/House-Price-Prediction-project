"""FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload
Then open http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import prediction
from app.core.config import get_settings
from app.services.inference import model_service
from app.utils.logging_config import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: configure logging and load the model exactly once.
    configure_logging()
    model_service.load()
    yield
    # Shutdown: nothing to clean up for a joblib model.


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Predicts Indian residential property prices from listing features.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(prediction.router)

    @app.get("/", tags=["root"])
    def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
