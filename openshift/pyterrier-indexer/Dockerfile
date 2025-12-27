FROM python:3.11-slim

# Install Java at build time (build runs with root; runtime runs restricted UID).
RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless && rm -rf /var/lib/apt/lists/*

# PyTerrier + dependencies for indexing.
# Note: PyPI package name is python-terrier; the import module is pyterrier.
# Pin numpy <2 to avoid pandas ABI mismatch in this environment.
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.0.3 python-terrier==0.10.0

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

