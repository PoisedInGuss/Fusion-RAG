FROM docker.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Minimal Python runtime + vLLM with CUDA 12.4 wheels.
# Build runs as root in OpenShift Build; runtime runs as restricted UID.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates git curl \
    build-essential gcc g++ make \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin torch to CUDA 12.4 to match the cluster driver (CUDA 12.4).
# DO NOT let pip select cu126/cu128 wheels on this cluster.
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124

# vLLM upstream hard-pins torch without "+cu124", so we install vLLM without deps,
# then install the official vLLM requirements file that does NOT pin torch.
# We also install CUDA runtime extras from requirements/cuda.txt excluding torch/audio/vision.
RUN python3 -m pip install --no-cache-dir --no-deps vllm==0.8.5.post1 \
  && curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5.post1/requirements/common.txt" -o /tmp/vllm-common.txt \
  && python3 -m pip install --no-cache-dir -r /tmp/vllm-common.txt \
  && rm -f /tmp/vllm-common.txt \
  && python3 -m pip install --no-cache-dir \
    "numba==0.61.2" \
    "ray[cgraph]>=2.43.0,!=2.44.*"

# OpenShift runs with random UID; ensure writable defaults.
ENV HOME=/tmp
ENV TMPDIR=/tmp
ENV CC=/usr/bin/gcc

EXPOSE 8000

