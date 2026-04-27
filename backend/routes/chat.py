"""
Chat Routes — /api/* endpoints as a Flask Blueprint.
"""

import logging
from datetime import datetime

from flask import Blueprint, request, jsonify

from extensions import limiter
from utils.db import (
    add_message, get_recent_messages, clear_history,
    create_session, get_session, touch_session,
    save_lstm_state, load_lstm_state, update_context_summary,
)
from utils.nlp_model import generate_nlm_reply
from utils.validators import sanitise_text
from services.chat_service import full_response

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/api/session", methods=["POST"])
@limiter.limit("20 per minute")
def api_create_session():
    """Create a new chat session with LSTM memory."""
    from flask import current_app
    lstm_memory = current_app.config.get("LSTM_MEMORY")
    rag_engine = current_app.config.get("RAG_ENGINE")

    session_id = create_session()
    if lstm_memory and lstm_memory.is_ready:
        lstm_memory.init_session(session_id)

    return jsonify({
        "session_id": session_id,
        "status": "created",
        "features": {
            "rag_enabled": rag_engine is not None and rag_engine.is_ready,
            "lstm_enabled": lstm_memory is not None and lstm_memory.is_ready,
        },
        "timestamp": datetime.now().isoformat(),
    })


@chat_bp.route("/api/session/<session_id>/context", methods=["GET"])
def api_session_context(session_id):
    """Get LSTM memory context summary for a session."""
    from flask import current_app
    lstm_memory = current_app.config.get("LSTM_MEMORY")

    session = get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    context_summary = ""
    message_count = 0

    if lstm_memory and lstm_memory.is_ready:
        context_summary = lstm_memory.get_context_summary(session_id)
        message_count = lstm_memory.get_message_count(session_id)

    return jsonify({
        "session_id": session_id,
        "context_summary": context_summary,
        "message_count": message_count,
        "created_at": session.get("created_at"),
        "last_active": session.get("last_active"),
    })


@chat_bp.route("/api/message", methods=["POST"])
@limiter.limit("30 per minute")
def api_message():
    from flask import current_app
    app_config = current_app.config.get("APP_CONFIG")
    rag_engine = current_app.config.get("RAG_ENGINE")
    lstm_memory = current_app.config.get("LSTM_MEMORY")

    data = request.get_json() or {}
    raw_input = (data.get("text") or "").strip()
    session_id = data.get("session_id")

    # ── Input validation ────────────────────────────────────────
    max_len = app_config.MAX_INPUT_LENGTH if app_config else 500
    user_input = sanitise_text(raw_input, max_length=max_len)

    if not user_input:
        return jsonify({"reply": "Please enter a message."})

    # Auto-create session if not provided
    if not session_id:
        session_id = create_session()
        if lstm_memory and lstm_memory.is_ready:
            lstm_memory.init_session(session_id)
    else:
        # Restore LSTM state from DB if needed
        if lstm_memory and lstm_memory.is_ready:
            if lstm_memory.get_message_count(session_id) == 0:
                state_bytes = load_lstm_state(session_id)
                if state_bytes:
                    lstm_memory.restore_state(session_id, state_bytes)
            touch_session(session_id)

    add_message("user", user_input, session_id=session_id)

    # Generate response with RAG + LSTM
    api_key = app_config.OPENWEATHER_API if app_config else None
    weather_url = app_config.WEATHER_URL if app_config else "https://api.openweathermap.org/data/2.5"
    data_path = app_config.DATA_PATH if app_config else "data/trends.csv"

    reply = full_response(
        user_input,
        session_id=session_id,
        rag_engine=rag_engine,
        lstm_memory=lstm_memory,
        get_recent_messages_fn=get_recent_messages,
        api_key=api_key,
        weather_url=weather_url,
        data_path=data_path,
    ).strip()

    if not reply:
        retry_prompt = (
            "<|im_start|>system\n Provide a short correct answer.\n"
            f"<|im_start|>user\n {user_input}\n"
            "<|im_start|>assistant\n"
        )
        reply = generate_nlm_reply(retry_prompt)

    add_message("ai", reply, session_id=session_id)

    # Update LSTM memory
    if lstm_memory and lstm_memory.is_ready and session_id:
        lstm_memory.update(session_id, user_input, reply)
        state_bytes = lstm_memory.serialize_state(session_id)
        save_lstm_state(session_id, state_bytes)
        context_summary = lstm_memory.get_context_summary(session_id)
        update_context_summary(session_id, context_summary)

    return jsonify({
        "reply": reply,
        "session_id": session_id,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "memory": {
            "rag_used": rag_engine is not None and rag_engine.is_ready,
            "lstm_active": lstm_memory is not None and lstm_memory.is_ready,
            "message_count": lstm_memory.get_message_count(session_id) if lstm_memory and lstm_memory.is_ready else 0,
        },
    })


@chat_bp.route("/api/history")
def api_history():
    session_id = request.args.get("session_id")
    msgs = get_recent_messages(20, session_id=session_id)
    return jsonify([{"role": r, "content": c} for r, c in msgs])


@chat_bp.route("/api/clear", methods=["POST"])
def api_clear():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    clear_history(session_id=session_id)
    logger.info("Chat history cleared (session=%s)", session_id or "all")
    return jsonify({"message": "Chat memory cleared."})
