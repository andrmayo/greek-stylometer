FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.13 from deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg curl ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3.13-dev && \
    rm -rf /var/lib/apt/lists/*

# Make python3.13 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# Install pip
RUN python -m ensurepip --upgrade && \
    python -m pip install --no-cache-dir --upgrade pip

# Install gcloud CLI for GCS file transfers
RUN curl -fsSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz \
      | tar -xz -C /opt && \
    /opt/google-cloud-sdk/install.sh --quiet --path-update=true
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"

# Install PyTorch and dependencies (cached unless pyproject.toml changes)
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir .

# Copy source code (only this layer rebuilds on code changes)
COPY . .
RUN pip install --no-cache-dir --no-deps .

ENTRYPOINT ["python", "-m", "greek_stylometer.vertex_wrapper"]
