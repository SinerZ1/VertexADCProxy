"""ADC-authenticated reverse proxy for Vertex AI."""

from .app import app, create_app

__all__ = ["app", "create_app"]
