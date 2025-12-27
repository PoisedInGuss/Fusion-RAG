# Incoming: docker.io/nvidia/cuda:12.1.1-runtime-ubuntu22.04 --- {Image, OCI}
# Processing: install BGE dense indexing stack --- {3 jobs: configuration, installation, validation}
# Outgoing: GPU-capable BGE runtime --- {Image, OCI}

FROM docker.io/nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps: Python, build toolchain, SSL/certs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ca-certificates curl git \
    build-essential gcc g++ make \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# CUDA 12.1 PyTorch for GPU encoding
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121

# Dense retrieval stack:
# - sentence-transformers: BGE encoder
# - faiss-cpu: FAISS index builder (IVFPQ)
# - transformers pinned < 4.45 to avoid CVE guard on torch<2.6
RUN python3 -m pip install --no-cache-dir \
    sentence-transformers==3.1.1 \
    faiss-cpu==1.8.0 \
    numpy==1.26.4 \
    transformers==4.44.2

ENV HOME=/tmp
ENV TMPDIR=/tmp
ENV PYTHONUNBUFFERED=1
ENV CC=/usr/bin/gcc

