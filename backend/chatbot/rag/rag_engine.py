"""
RAG Engine — Retrieval-Augmented Generation for PastCast Chatbot
Uses FAISS + sentence-transformers for semantic search over weather knowledge.
"""

import logging
import os
import json
import pickle
import numpy as np

from utils.embeddings import get_encoder, EMBEDDING_DIM

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None
    logger.warning("faiss-cpu not installed — RAG will use fallback cosine search")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(SCRIPT_DIR, "index")
INDEX_PATH = os.path.join(INDEX_DIR, "rag_index.faiss")
META_PATH = os.path.join(INDEX_DIR, "rag_metadata.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.45


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    Encodes documents into FAISS vector index and retrieves
    the most relevant chunks for any user query.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = get_encoder()  # Shared singleton
        self.index = None
        self.metadata = []
        self.model_name = model_name
        self._ready = False

        if self.model is not None:
            logger.info("RAG engine using shared embedding model")
        else:
            logger.warning("RAG engine: no embedding model available")

        self._load_index()

    def _load_index(self):
        """Load existing FAISS index and metadata from disk."""
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            try:
                if faiss is not None:
                    self.index = faiss.read_index(INDEX_PATH)
                else:
                    with open(INDEX_PATH + ".npy", "rb") as f:
                        self._fallback_vectors = np.load(f)
                with open(META_PATH, "rb") as f:
                    self.metadata = pickle.load(f)
                self._ready = True
                logger.info("RAG index loaded: %d chunks", len(self.metadata))
            except Exception as e:
                logger.error("RAG index load failed: %s", e)
        else:
            logger.warning("RAG index not found — run rag_indexer.py to build it")

    @property
    def is_ready(self) -> bool:
        return self._ready and self.model is not None

    def encode(self, texts: list) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.model is None:
            return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        embeddings = self.model.encode(texts, show_progress_bar=False,
                                        convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        return self.encode([text])[0]

    def build_index(self, documents: list):
        """
        Build FAISS index from a list of document dicts.

        Each doc should have: 'content', 'title', 'category', 'id'
        """
        if self.model is None:
            logger.error("Cannot build index without embedding model")
            return

        texts = []
        meta = []
        for doc in documents:
            content = doc.get("content", "")
            title = doc.get("title", "")
            # Combine title + content for richer embeddings
            combined = f"{title}. {content}" if title else content
            texts.append(combined)
            meta.append({
                "id": doc.get("id", ""),
                "title": title,
                "content": content,
                "category": doc.get("category", ""),
                "region": doc.get("region", "global"),
                "chunk_type": doc.get("chunk_type", "document"),
            })

        logger.info("Encoding %d documents...", len(texts))
        embeddings = self.encode(texts)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_norm = embeddings / norms

        if faiss is not None:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product = cosine on normalized
            self.index.add(embeddings_norm)
        else:
            self._fallback_vectors = embeddings_norm

        self.metadata = meta
        self._ready = True

        # Save to disk
        os.makedirs(INDEX_DIR, exist_ok=True)
        if faiss is not None:
            faiss.write_index(self.index, INDEX_PATH)
        else:
            np.save(INDEX_PATH + ".npy", embeddings_norm)
        with open(META_PATH, "wb") as f:
            pickle.dump(meta, f)

        logger.info("RAG index built: %d chunks -> %s", len(meta), INDEX_DIR)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        """
        Retrieve top-K relevant chunks for a query.

        Returns list of dicts: {content, title, category, score}
        """
        if not self.is_ready:
            return []

        query_vec = self.encode_single(query)
        # Normalize query
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        query_vec = query_vec.reshape(1, -1)

        if faiss is not None and self.index is not None:
            scores, indices = self.index.search(query_vec, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Fallback: numpy cosine similarity
            sims = np.dot(self._fallback_vectors, query_vec.T).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            indices = top_indices
            scores = sims[top_indices]

        results = []
        for i, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0 or idx >= len(self.metadata):
                continue
            if score < SIMILARITY_THRESHOLD:
                continue
            meta = self.metadata[idx]
            results.append({
                "content": meta["content"],
                "title": meta["title"],
                "category": meta["category"],
                "score": float(score),
                "rank": i + 1,
            })

        return results

    def format_context(self, results: list) -> str:
        """Format retrieved chunks into a context string for the LLM prompt."""
        if not results:
            return ""

        parts = []
        for r in results:
            parts.append(f"[{r['title']}] {r['content']}")
        return "\n\n".join(parts)


# Singleton for import
_rag_engine = None

def get_rag_engine() -> RAGEngine:
    """Get or create the global RAG engine singleton."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
