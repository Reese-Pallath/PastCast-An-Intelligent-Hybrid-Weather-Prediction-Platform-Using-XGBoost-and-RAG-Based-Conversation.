# PastCast: System Architecture & Technologies

## 1. Project Overview
PastCast is a production-grade, hybrid weather prediction platform. It combines a trained Machine Learning model (XGBoost) for highly accurate rainfall classification with an advanced Artificial Intelligence layer (RAG + LSTM + Qwen LLM) to provide a conversational meteorology assistant.

## 2. Core Technologies Used
* **Frontend**: React 19, TypeScript, Tailwind CSS, Recharts (for visual data mapping).
* **Backend**: Python 3.9+, Flask (using the App Factory pattern), SQLite (WAL mode).
* **Machine Learning**: 
  * XGBoost (Gradient Boosted Decision Trees for structured numerical forecasting).
  * Feature Engineering applied to synthetic historical data.
* **Artificial Intelligence**:
  * **RAG (Retrieval-Augmented Generation)**: FAISS Vector Database and `all-MiniLM-L6-v2` embeddings for factual grounding.
  * **Memory**: PyTorch LSTM customized for continuous dialogue contextual state.
  * **LLM**: Qwen 2.5-1.5B (Generative text output).
  * **Translation**: MarianMT (Helsinki NLP models for translating English to Hindi, Marathi, Tamil, Telugu).
* **DevOps**: Docker, Docker Compose, GitHub Actions, MLflow.
