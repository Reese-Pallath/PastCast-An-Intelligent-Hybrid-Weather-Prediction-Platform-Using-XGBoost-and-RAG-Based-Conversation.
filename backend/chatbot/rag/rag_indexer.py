"""
RAG Indexer — Builds the FAISS index from chatbot knowledge datasets.
Run this once after generating datasets, or whenever data changes.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chatbot.rag.rag_engine import RAGEngine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "datasets", "data")


def load_knowledge_documents():
    """Load and merge all knowledge sources into indexable documents."""
    documents = []

    # 1. Climate Knowledge (primary RAG source)
    ck_path = os.path.join(DATASETS_DIR, "climate_knowledge.json")
    if os.path.exists(ck_path):
        with open(ck_path, "r") as f:
            chunks = json.load(f)
        for chunk in chunks:
            documents.append({
                "id": chunk["id"],
                "title": chunk["title"],
                "content": chunk["content"],
                "category": chunk["category"],
                "region": chunk.get("region", "global"),
                "chunk_type": "knowledge",
            })
        print(f"  📚 Loaded {len(chunks)} climate knowledge chunks")

    # 2. Weather QA (index answers for retrieval)
    qa_path = os.path.join(DATASETS_DIR, "weather_qa.json")
    if os.path.exists(qa_path):
        with open(qa_path, "r") as f:
            qa_data = json.load(f)
        qa_count = 0
        for qa in qa_data:
            documents.append({
                "id": qa["id"],
                "title": qa["question"],
                "content": qa["answer"],
                "category": qa["category"],
                "region": "global",
                "chunk_type": "qa",
            })
            qa_count += 1
        print(f"  ❓ Loaded {qa_count} weather QA pairs")

    return documents


def build_index():
    """Build the full RAG index."""
    print("=" * 60)
    print("  🔍 PastCast RAG — Index Builder")
    print("=" * 60)

    # Check datasets exist
    if not os.path.exists(DATASETS_DIR):
        print("❌ Datasets not found. Run chatbot_datasets.py first.")
        print(f"   Expected at: {DATASETS_DIR}")
        return False

    documents = load_knowledge_documents()
    if not documents:
        print("❌ No documents found to index")
        return False

    print(f"\n📊 Total documents to index: {len(documents)}")

    # Build index
    engine = RAGEngine()
    engine.build_index(documents)

    # Test retrieval
    print("\n🧪 Testing retrieval...")
    test_queries = [
        "What causes monsoon rain?",
        "How hot does Delhi get?",
        "What is the Beaufort scale?",
        "How do barometers work?",
        "El Niño effects on weather",
    ]

    for query in test_queries:
        results = engine.retrieve(query, top_k=2)
        if results:
            top = results[0]
            print(f"  Q: {query}")
            print(f"  → [{top['score']:.3f}] {top['title']}")
        else:
            print(f"  Q: {query} → No results above threshold")

    print(f"\n✅ RAG index ready with {len(documents)} chunks")
    return True


if __name__ == "__main__":
    build_index()
