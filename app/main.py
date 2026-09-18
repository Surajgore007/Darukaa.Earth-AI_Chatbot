"""
FastAPI Main Application Entrypoint.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app import __version__
from app.core.config import settings
from app.api.routes import router as api_router

app = FastAPI(
    title="Darukaa.Earth Biodiversity Intelligence API",
    description="Scientific, knowledge-grounded AI environmental scientist for biodiversity and land management.",
    version=__version__
)

# Enable CORS for local testing and integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes (/health, /chat)
app.include_router(api_router)

# Mount static frontend directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=(settings.environment == "development")
    )
