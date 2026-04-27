"""
Health Routes — /health endpoint as a Flask Blueprint.
"""

from datetime import datetime

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/health")
def health():
    from flask import current_app
    rag_engine = current_app.config.get("RAG_ENGINE")
    lstm_memory = current_app.config.get("LSTM_MEMORY")
    app_config = current_app.config.get("APP_CONFIG")

    return jsonify({
        "status": "healthy",
        "model": "Qwen2.5-1.5B + MarianMT + RAG + LSTM",
        "weather_api": bool(app_config and app_config.OPENWEATHER_API),
        "rag_ready": rag_engine is not None and rag_engine.is_ready,
        "lstm_ready": lstm_memory is not None and lstm_memory.is_ready,
        "environment": app_config.__class__.__name__ if app_config else "unknown",
        "timestamp": datetime.now().isoformat(),
    })
