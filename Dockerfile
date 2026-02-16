FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# gcsfuse for transparent GCS access
RUN apt-get update && apt-get install -y --no-install-recommends gnupg curl ca-certificates && \
    echo "deb https://packages.cloud.google.com/apt gcsfuse-noble main" \
      > /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y --no-install-recommends gcsfuse && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "greek_stylometer.vertex_wrapper"]
