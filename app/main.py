"""
FastAPI Main Application Entrypoint.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app import __version__
from app.core.config import settings
from app.api.routes import router as api_router

app = FastAPI(
    title="Darukaa.Earth Biodiversity Intelligence API",
    description="Scientific, knowledge-grounded AI environmental scientist for biodiversity and land management.",
    version=__version__
)

# Enable CORS for cross-origin API calls and testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes for both root and /api prefix
app.include_router(api_router)
app.include_router(api_router, prefix="/api")

# Static directory and frontend HTML loading
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


def load_frontend_html() -> str:
    """Safely load index.html from static dir or root fallbacks."""
    possible_paths = [
        os.path.join(static_dir, "index.html"),
        os.path.join(os.getcwd(), "app", "static", "index.html"),
        os.path.join(os.path.dirname(__file__), "..", "app", "static", "index.html"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass

    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Darukaa.Earth AI Chatbot</title>
</head>
<body style="font-family:sans-serif;padding:24px;background:#0d1310;color:#e6edf3;">
    <h1>🌱 Darukaa.Earth — AI Biodiversity Intelligence</h1>
    <p>API is operational. Visit <a href="/docs" style="color:#3fb950;">/docs</a> for interactive Swagger API.</p>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
@app.get("/api", response_class=HTMLResponse, include_in_schema=False)
@app.get("/api/index.py", response_class=HTMLResponse, include_in_schema=False)
def serve_frontend():
    return HTMLResponse(content=load_frontend_html())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=(settings.environment == "development")
    )
