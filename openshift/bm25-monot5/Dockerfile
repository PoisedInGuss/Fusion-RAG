FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install Java + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install dependencies for BM25>>MonoT5:
# - PyTerrier for BM25
# - sentence-transformers for CrossEncoder
# - torch (CPU-only)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.0.3 \
    python-terrier==0.10.0

# Install torch from CPU-only index
RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install sentence-transformers (depends on torch) - use compatible versions
RUN pip install --no-cache-dir \
    sentence-transformers==3.1.1 \
    transformers==4.44.2

WORKDIR /tmp

