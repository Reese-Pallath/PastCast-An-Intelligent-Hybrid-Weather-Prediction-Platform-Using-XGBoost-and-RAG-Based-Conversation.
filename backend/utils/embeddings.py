"""
Shared SentenceTransformer singleton.
Both RAG engine and LSTM memory import from here to avoid
loading the ~90 MB model twice.
"""

import logging

logger = logging.getLogger(__name__)

_encoder = None
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

try:
    from sentence_transformers import SentenceTransformer as _ST
except ImportError:
    _ST = None
    logger.warning("sentence-transformers not installed — embeddings disabled")


def get_encoder():
    """Return (or lazily create) the shared SentenceTransformer instance."""
    global _encoder
    if _encoder is None and _ST is not None:
        logger.info("Loading shared embedding model: %s", EMBEDDING_MODEL)
        _encoder = _ST(EMBEDDING_MODEL)
        logger.info("Shared embedding model loaded")
    return _encoder
