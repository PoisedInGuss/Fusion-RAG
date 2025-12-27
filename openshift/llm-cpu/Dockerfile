# Incoming: ./architecture/data_flow_standard.yaml --- {Dict, YAML}
# Processing: assemble CPU-only vLLM runtime with OpenVINO backend --- {2 jobs: generation, routing}
# Outgoing: image-registry/425krish/vllm-cpu:latest --- {Image, Dockerfile}
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ca-certificates curl git \
    build-essential gcc g++ make \
    pkg-config libssl-dev \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

ARG VLLM_OV_REPO=https://github.com/vllm-project/vllm-openvino.git
ARG VLLM_OV_REF=main

RUN git clone --depth=1 --branch "${VLLM_OV_REF}" "${VLLM_OV_REPO}" /tmp/vllm-openvino \
  && cd /tmp/vllm-openvino \
  && VLLM_TARGET_DEVICE=empty PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu python3 -m pip install --no-cache-dir . \
  && (python3 -m pip uninstall -y triton || true) \
  && rm -rf /tmp/vllm-openvino

ENV HOME=/tmp
ENV TMPDIR=/tmp

WORKDIR /app

EXPOSE 8000

