# Incoming: docker.io/nvidia/cuda:12.1.1-runtime-ubuntu22.04 --- {Image, OCI}
# Processing: install Java+Pyserini+SPLADE deps --- {3 jobs: configuration, hardening, installation}
# Outgoing: GPU-capable Pyserini SPLADE runtime --- {Image, OCI}

FROM docker.io/nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps:
# - Java 11 runtime for Anserini/Lucene indexing/search
# - build-essential for some Python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ca-certificates curl git \
    default-jre-headless \
    openjdk-11-jdk-headless \
    build-essential gcc g++ make \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch CUDA 12.1 + core packages needed for SPLADE encoding + Pyserini for Lucene impact indexing/search.
# Keep versions compatible with our cluster driver (CUDA 12.1 runtime).
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121

# Pyserini brings Anserini + Lucene tooling; transformers provides SPLADE model.
# faiss-cpu + sentence-transformers enable dense (TCT-ColBERT) indexing on CPU/GPU.
RUN python3 -m pip install --no-cache-dir \
    pyserini==0.22.1 \
    transformers==4.46.3 \
    sentencepiece==0.2.0 \
    numpy==1.26.4 \
    faiss-cpu==1.8.0 \
    sentence-transformers==3.1.1

ENV HOME=/tmp
ENV TMPDIR=/tmp
ENV PYTHONUNBUFFERED=1
ENV CC=/usr/bin/gcc
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

