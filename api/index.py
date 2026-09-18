"""
Vercel Serverless Entrypoint for Darukaa.Earth AI Chatbot.
"""
import sys
import os

# Ensure repository root is in sys.path so 'app' and 'data' modules import cleanly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.main import app

# Export app for Vercel WSGI/ASGI handler
__all__ = ["app"]
