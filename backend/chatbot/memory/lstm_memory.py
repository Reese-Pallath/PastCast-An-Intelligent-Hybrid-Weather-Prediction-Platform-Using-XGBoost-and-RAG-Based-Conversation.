"""
LSTM Conversation Memory for PastCast Chatbot
Encodes multi-turn dialogue into persistent context vectors for session continuity.
"""

import json
import logging
import os
import io
import numpy as np
import torch
import torch.nn as nn

from utils.embeddings import get_encoder, EMBEDDING_DIM

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_memory.pt")

HIDDEN_DIM = 256           # LSTM hidden state size
NUM_LAYERS = 2
CONTEXT_SUMMARY_DIM = 128  # Compressed context vector
MAX_HISTORY = 50           # Max messages to consider
BIDIRECTIONAL = True


class ConversationLSTM(nn.Module):
    """
    Bidirectional LSTM that processes a sequence of message embeddings
    and produces a fixed-size context vector summarising the conversation.
    """

    def __init__(self, input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection (normalize different embedding dimensions)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Core LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Context compression: LSTM output → fixed context vector
        lstm_out_dim = hidden_dim * self.num_directions
        self.context_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, CONTEXT_SUMMARY_DIM),
            nn.Tanh(),
        )

        # Attention mechanism for weighting message importance
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embeddings, lengths=None):
        """
        Args:
            embeddings: (batch, seq_len, input_dim) message embeddings
            lengths: (batch,) actual sequence lengths (optional)
        Returns:
            context_vector: (batch, CONTEXT_SUMMARY_DIM)
        """
        # Project input
        x = self.input_proj(embeddings)
        x = self.input_norm(x)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Attention-weighted pooling
        attn_weights = self.attention(lstm_out)       # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attended = (lstm_out * attn_weights).sum(dim=1)  # (B, lstm_out_dim)

        # Project to context vector
        context = self.context_proj(attended)
        return context

    def get_hidden_state(self, embeddings):
        """Get the raw LSTM hidden state for persistence."""
        x = self.input_proj(embeddings)
        x = self.input_norm(x)
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class LSTMMemoryManager:
    """
    Manages conversation memory using the LSTM model.
    Handles encoding, state persistence, and context summarisation.
    """

    def __init__(self):
        self.model = ConversationLSTM()
        self.encoder = None
        self._ready = False

        # Load trained model if available
        if os.path.exists(MODEL_PATH):
            try:
                state = torch.load(MODEL_PATH, map_location="cpu",
                                   weights_only=True)
                self.model.load_state_dict(state)
                logger.info("LSTM memory model loaded")
            except Exception as e:
                logger.warning("LSTM model load failed (%s), using untrained", e)
        else:
            logger.warning("LSTM model not found — using untrained (run train_lstm.py)")

        self.model.eval()

        # Load sentence encoder (shared singleton with RAG)
        self.encoder = get_encoder()
        if self.encoder is not None:
            self._ready = True

        # In-memory session states: {session_id: {hidden, context, messages}}
        self._sessions = {}

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _encode_messages(self, messages: list) -> torch.Tensor:
        """Encode a list of message strings to embeddings tensor."""
        if not messages or self.encoder is None:
            return torch.zeros(1, 1, EMBEDDING_DIM)

        # Truncate to max history
        messages = messages[-MAX_HISTORY:]
        embeddings = self.encoder.encode(messages, show_progress_bar=False,
                                          convert_to_numpy=True)
        return torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)  # (1, T, D)

    def get_context_vector(self, session_id: str) -> np.ndarray:
        """Get the current context vector for a session."""
        if session_id not in self._sessions:
            return np.zeros(CONTEXT_SUMMARY_DIM)

        session = self._sessions[session_id]
        messages = session.get("messages", [])
        if not messages:
            return np.zeros(CONTEXT_SUMMARY_DIM)

        embeddings = self._encode_messages(messages)
        with torch.no_grad():
            context = self.model(embeddings)
        return context.squeeze(0).numpy()

    def update(self, session_id: str, user_msg: str, bot_msg: str):
        """Update session state with new messages."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {"messages": [], "context": None}

        session = self._sessions[session_id]
        session["messages"].append(f"User: {user_msg}")
        session["messages"].append(f"Assistant: {bot_msg}")

        # Update context vector
        session["context"] = self.get_context_vector(session_id)

    def get_context_summary(self, session_id: str) -> str:
        """
        Generate a natural language summary of the conversation context.
        Uses the context vector to identify key conversation themes.
        """
        if session_id not in self._sessions:
            return ""

        session = self._sessions[session_id]
        messages = session.get("messages", [])
        if not messages:
            return ""

        # Extract key topics from recent messages
        recent = messages[-10:]  # Last 5 turns

        # Simple extractive summary: identify key nouns/topics
        topics = set()
        weather_terms = {"rain", "temperature", "humidity", "wind", "storm",
                         "forecast", "monsoon", "heat", "cold", "cloud",
                         "pressure", "cyclone", "flood", "drought"}
        city_terms = {"mumbai", "delhi", "chennai", "bengaluru", "kolkata",
                     "london", "tokyo", "new york", "sydney", "dubai",
                     "pune", "hyderabad", "goa", "jaipur"}

        for msg in recent:
            words = msg.lower().split()
            for word in words:
                clean = word.strip(".,!?:;")
                if clean in weather_terms:
                    topics.add(clean)
                if clean in city_terms:
                    topics.add(clean.title())

        if not topics:
            return f"Conversation has {len(messages)} messages."

        topic_str = ", ".join(sorted(topics))
        return f"Previous conversation topics: {topic_str}. Total messages: {len(messages)}."

    def init_session(self, session_id: str):
        """Initialize a new session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {"messages": [], "context": None}

    def serialize_state(self, session_id: str) -> bytes:
        """Serialize session state for database storage using JSON (safe)."""
        if session_id not in self._sessions:
            return b""

        session = self._sessions[session_id]
        data = {
            "messages": session["messages"],
        }
        return json.dumps(data).encode("utf-8")

    def restore_state(self, session_id: str, state_bytes: bytes):
        """Restore session state from database (JSON-based, safe)."""
        if not state_bytes:
            return

        try:
            data = json.loads(state_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Backward compat: try old torch format
            try:
                buf = io.BytesIO(state_bytes)
                data = torch.load(buf, map_location="cpu", weights_only=False)
                logger.warning("Restored LSTM state from legacy torch format for session %s", session_id)
            except Exception:
                logger.error("Failed to restore LSTM state for session %s", session_id)
                return

        self._sessions[session_id] = {
            "messages": data.get("messages", []),
            "context": data.get("context"),
        }

    def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        if session_id not in self._sessions:
            return 0
        return len(self._sessions[session_id].get("messages", []))


# Singleton
_lstm_memory = None

def get_lstm_memory() -> LSTMMemoryManager:
    """Get or create the global LSTM memory manager."""
    global _lstm_memory
    if _lstm_memory is None:
        _lstm_memory = LSTMMemoryManager()
    return _lstm_memory
