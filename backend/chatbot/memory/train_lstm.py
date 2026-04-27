"""
LSTM Memory Training Pipeline
Trains the conversation memory model on multi-turn dialogue patterns.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chatbot.memory.lstm_memory import ConversationLSTM, EMBEDDING_DIM, CONTEXT_SUMMARY_DIM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_memory.pt")
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "datasets", "data", "conversation_patterns.json")

EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 20


class ConversationDataset(Dataset):
    """Dataset of conversation sequences for contrastive LSTM training."""

    def __init__(self, conversations, encoder):
        self.conversations = conversations
        self.encoder = encoder
        self.embeddings_cache = {}
        self._prepare()

    def _prepare(self):
        """Pre-encode all conversation messages."""
        print("  🔄 Encoding conversation messages...")
        for i, conv in enumerate(self.conversations):
            msgs = [m["content"] for m in conv["messages"]]
            if not msgs:
                continue
            embs = self.encoder.encode(msgs, show_progress_bar=False,
                                        convert_to_numpy=True)
            self.embeddings_cache[i] = embs
        print(f"  ✅ Encoded {len(self.embeddings_cache)} conversations")

    def __len__(self):
        return len(self.embeddings_cache)

    def __getitem__(self, idx):
        embs = self.embeddings_cache[idx]
        # Pad or truncate to MAX_SEQ_LEN
        if len(embs) > MAX_SEQ_LEN:
            embs = embs[:MAX_SEQ_LEN]
        elif len(embs) < MAX_SEQ_LEN:
            pad = np.zeros((MAX_SEQ_LEN - len(embs), EMBEDDING_DIM), dtype=np.float32)
            embs = np.concatenate([embs, pad], axis=0)

        topic = self.conversations[idx].get("topic", "general")
        return {
            "embeddings": torch.tensor(embs, dtype=torch.float32),
            "length": min(len(self.embeddings_cache[idx]), MAX_SEQ_LEN),
            "topic": topic,
        }


def contrastive_loss(anchors, positives, negatives, margin=1.0):
    """
    Triplet contrastive loss.
    Similar conversations (same topic) should have similar context vectors.
    """
    pos_dist = torch.norm(anchors - positives, dim=1)
    neg_dist = torch.norm(anchors - negatives, dim=1)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()


def reconstruction_loss(model, embeddings, lengths):
    """
    Self-supervised loss: context vector should capture key information.
    Uses context vector to predict a summary of the input.
    """
    context = model(embeddings)
    # Encourage context vectors to be distinct (not collapse to zero)
    variance_loss = -torch.log(context.var(dim=0).mean() + 1e-8)
    # Encourage uniformity across batch
    norm_context = torch.nn.functional.normalize(context, dim=1)
    sim_matrix = torch.mm(norm_context, norm_context.t())
    # Mask diagonal
    mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool)
    uniformity_loss = sim_matrix[mask].pow(2).mean()

    return variance_loss + uniformity_loss * 0.5


def train():
    """Full training pipeline."""
    print("=" * 60)
    print("  🧠 PastCast LSTM Memory — Training Pipeline")
    print("=" * 60)

    # Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found: {DATA_PATH}")
        print("   Run chatbot_datasets.py first")
        return

    with open(DATA_PATH, "r") as f:
        conversations = json.load(f)
    print(f"\n📊 Loaded {len(conversations)} conversation patterns")

    # Load encoder
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        print("❌ sentence-transformers required for training")
        return

    # Create dataset
    dataset = ConversationDataset(conversations, encoder)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           drop_last=True)

    # Group by topic for contrastive pairs
    topic_indices = {}
    for i, conv in enumerate(conversations):
        topic = conv.get("topic", "general")
        if topic not in topic_indices:
            topic_indices[topic] = []
        topic_indices[topic].append(i)

    # Init model
    model = ConversationLSTM()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n🚀 Training for {EPOCHS} epochs...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            embeddings = batch["embeddings"]   # (B, T, D)
            lengths = batch["length"]

            # Forward pass
            context_vectors = model(embeddings)

            # Reconstruction/self-supervised loss
            recon_loss = reconstruction_loss(model, embeddings, lengths)

            # Simple MSE loss: encourage temporal processing
            # Partial context (first half) should differ from full context
            half_len = MAX_SEQ_LEN // 2
            partial_embs = embeddings.clone()
            partial_embs[:, half_len:, :] = 0
            partial_context = model(partial_embs)

            temporal_loss = 1.0 - torch.nn.functional.cosine_similarity(
                context_vectors, partial_context
            ).mean()

            loss = recon_loss + temporal_loss * 0.3

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")
    print(f"   Best loss: {best_loss:.4f}")

    # Verify
    model.eval()
    with torch.no_grad():
        test_embs = torch.randn(1, 6, EMBEDDING_DIM)
        ctx = model(test_embs)
        print(f"   Test context shape: {ctx.shape}")
        print(f"   Test context norm: {torch.norm(ctx):.4f}")

    print("\n✅ LSTM memory training complete!")


if __name__ == "__main__":
    train()
