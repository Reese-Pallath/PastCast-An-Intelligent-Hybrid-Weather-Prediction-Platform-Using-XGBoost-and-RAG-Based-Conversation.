"""
PastCast Flask Application — App Factory Pattern

This is the production-grade entry point. All business logic lives in
services/*, route handlers in routes/*, and configuration in config.py.
"""

import logging
import os
import threading

# Must be set before ANY torch/numpy import to prevent BLAS/OpenMP thread
# deadlocks on macOS when multiple PyTorch-based components load in sequence.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from config import get_config
from extensions import limiter
from utils.logging_config import setup_logging
from utils.db import init_db, cleanup_old_sessions

logger = logging.getLogger(__name__)


def create_app(config=None):
    """
    Flask application factory.

    Parameters
    ----------
    config : Config | None
        Override config (used by tests). If None, auto-detect from FLASK_ENV.
    """
    if config is None:
        config = get_config()

    # ── Logging ─────────────────────────────────────────────────
    setup_logging(level=config.LOG_LEVEL, fmt=config.LOG_FORMAT)

    # ── Flask ───────────────────────────────────────────────────
    app = Flask(__name__)
    app.config["APP_CONFIG"] = config

    # ── CORS ────────────────────────────────────────────────────
    CORS(
        app,
        resources={r"/*": {"origins": config.CORS_ORIGINS}},
    )

    # ── Rate Limiting ───────────────────────────────────────────
    limiter.init_app(app)

    # ── Database ────────────────────────────────────────────────
    init_db()
    try:
        deleted = cleanup_old_sessions(max_age_days=config.MAX_SESSION_AGE_DAYS)
        if deleted:
            logger.info("Startup cleanup: removed %d expired sessions", deleted)
    except Exception as e:
        logger.warning("Session cleanup failed at startup: %s", e)

    # ── ML Model ────────────────────────────────────────────────
    ml_predictor = None
    ml_model_path = os.path.join(config.ML_MODEL_DIR, "rain_predictor.pkl")
    try:
        if os.path.exists(ml_model_path):
            from ml.predictor import WeatherRainfallPredictor
            ml_predictor = WeatherRainfallPredictor.load(ml_model_path)
            logger.info("ML model loaded successfully from %s", ml_model_path)
        else:
            logger.warning("ML model not found at %s — using rule-based fallback", ml_model_path)
    except Exception as e:
        logger.error("ML model load failed: %s — using rule-based fallback", e)
    app.config["ML_PREDICTOR"] = ml_predictor

    # ── RAG Engine ──────────────────────────────────────────────
    rag_engine = None
    try:
        from chatbot.rag.rag_engine import get_rag_engine
        rag_engine = get_rag_engine()
        if rag_engine.is_ready:
            logger.info("RAG engine loaded successfully")
        else:
            logger.warning("RAG engine loaded but index not ready — run rag_indexer.py")
    except Exception as e:
        logger.warning("RAG engine not available: %s", e)
    app.config["RAG_ENGINE"] = rag_engine

    # ── LSTM Memory ─────────────────────────────────────────────
    # Loaded in a daemon thread with a 10 s timeout: nn.LSTM init can
    # deadlock on macOS when BLAS threads are already live from the
    # sentence-transformers load above.  The timeout keeps startup fast
    # and lets the rest of the app work even if LSTM isn't available.
    lstm_memory = None
    _lstm_result: list = [None]

    def _load_lstm():
        try:
            from chatbot.memory.lstm_memory import get_lstm_memory
            _lstm_result[0] = get_lstm_memory()
        except Exception as e:
            logger.warning("LSTM memory not available: %s", e)

    _t = threading.Thread(target=_load_lstm, daemon=True)
    _t.start()
    _t.join(timeout=10)
    if _t.is_alive():
        logger.warning("LSTM memory timed out during loading — skipping")
    else:
        lstm_memory = _lstm_result[0]
        if lstm_memory and lstm_memory.is_ready:
            logger.info("LSTM memory loaded successfully")
        elif lstm_memory:
            logger.warning("LSTM memory loaded but encoder not ready")
    app.config["LSTM_MEMORY"] = lstm_memory

    # ── Register Blueprints ─────────────────────────────────────
    from routes.weather import weather_bp
    from routes.chat import chat_bp
    from routes.health import health_bp

    app.register_blueprint(weather_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(health_bp)

    # ── Global Error Handlers ───────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "status": 404}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed", "status": 405}), 405

    @app.errorhandler(500)
    def internal_error(e):
        logger.exception("Unhandled server error")
        return jsonify({"error": "Internal server error", "status": 500}), 500

    @app.errorhandler(Exception)
    def unhandled_exception(e):
        logger.exception("Unhandled exception: %s", e)
        return jsonify({
            "error": "Internal server error",
            "details": str(e) if config.DEBUG else "An unexpected error occurred.",
            "status": 500,
        }), 500

    logger.info(
        "PastCast app created [env=%s, debug=%s, ML=%s, RAG=%s, LSTM=%s]",
        config.__class__.__name__,
        config.DEBUG,
        ml_predictor is not None,
        rag_engine is not None and rag_engine.is_ready,
        lstm_memory is not None and lstm_memory.is_ready,
    )

    return app


# ── Direct execution (development only) ────────────────────────
if __name__ == "__main__":
    app = create_app()
    # use_reloader=False: Flask's reloader forks the process, which deadlocks
    # on macOS after PyTorch has spawned threads (ML model + embeddings).
    app.run(host="0.0.0.0", port=8000, debug=app.config["APP_CONFIG"].DEBUG, use_reloader=False)
