# ── Stage 1: Build React Frontend ────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --production=false

COPY public/ ./public/
COPY src/ ./src/
COPY tsconfig.json postcss.config.js tailwind.config.js ./

ARG REACT_APP_API_BASE_URL=""
ENV REACT_APP_API_BASE_URL=$REACT_APP_API_BASE_URL

RUN npm run build


# ── Stage 2: Python Backend + Built Frontend ────────────────────
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Backend code
COPY backend/ ./

# Built frontend (served by Gunicorn if needed, or behind a reverse proxy)
COPY --from=frontend-build /app/build ./static/

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5).raise_for_status()"

CMD ["gunicorn", "wsgi:app", "--workers", "4", "--timeout", "120", "--bind", "0.0.0.0:8000"]
