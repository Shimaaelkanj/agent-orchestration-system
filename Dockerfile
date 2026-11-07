FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# The Chroma client must match the Chroma server image. Kept in its own small layer.
ARG CHROMA_VERSION=
RUN if [ -n "$CHROMA_VERSION" ]; then pip install "chromadb==$CHROMA_VERSION"; fi

COPY orchestrator ./orchestrator
COPY scripts ./scripts
RUN useradd -m app && mkdir -p /app/workspace /app/data /app/logs && chown -R app /app
USER app