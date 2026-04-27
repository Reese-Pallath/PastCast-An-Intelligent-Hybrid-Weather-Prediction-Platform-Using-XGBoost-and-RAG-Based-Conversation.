"""
WSGI entry point for production deployment (Gunicorn / uWSGI).

Usage:
    gunicorn wsgi:app --workers 4 --timeout 120 --bind 0.0.0.0:$PORT
"""

from app import create_app

app = create_app()
