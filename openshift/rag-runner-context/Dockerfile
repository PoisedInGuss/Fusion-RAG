# Incoming: python:3.11-slim --- {Image, OCI}
# Processing: add minimal deps + project code for RAG --- {3 jobs: deps install, copy repo, env setup}
# Outgoing: CPU-only RAG runner image --- {Image, OCI}

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    build-essential gcc g++ make \
    python3-dev \
    default-jre-headless \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Dependencies for RAG generation + retrieval (PyTerrier + cross-encoders)
RUN python -m pip install --no-cache-dir \
    pyyaml==6.0.2 \
    requests==2.32.3 \
    evaluate==0.4.1 \
    datasets==2.19.1 \
    python-terrier==0.10.0 \
    pandas==2.1.4 \
    numpy==1.26.4 \
    torch==2.5.1 \
    sentence-transformers==3.1.1 \
    transformers==4.46.3 \
    faiss-cpu==1.8.0 \
    pyserini==0.22.1 \
    sentencepiece==0.2.0 \
    ranx==0.3.20 \
    openai==1.57.4

WORKDIR /app

# Copy ONLY code and config needed for RAG; data/indexes stay on PVCs.
COPY src /app/src
COPY config /app/config
COPY scripts /app/scripts

RUN chgrp -R 0 /app && chmod -R g+rwX /app

ENV PYTHONUNBUFFERED=1
ENV PROJECT_ROOT=/app
ENV JAVA_HOME=/usr/lib/jvm/default-java

