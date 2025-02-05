ARG BASE_IMAGE=python:3.9-slim

# Stage 1: Builder
FROM ${BASE_IMAGE} as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# --------------------------------------------
# Stage 2: Runtime
FROM ${BASE_IMAGE}

# Set environment variables
ENV BENTOML_HOME="/bentoml" \
    PYTHONPATH="/workspace" \
    BENTOML__API_SERVER__DEFAULT_PORT=3000 \
    BENTOML__CONTAINER__BUILD_CONTEXT="/workspace" \
    BENTOML_DISABLE_USAGE_STATS=True

# Install system runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to include user-local binaries
ENV PATH=/root/.local/bin:$PATH

# Create workspace directory
WORKDIR /workspace

# Copy Bento files
COPY . /workspace/

# Install Bento dependencies
RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

# Install the Bento (replace with your Bento name:version)
ARG BENTO_TAG=asl_classifier:latest
RUN bentoml retrieve ${BENTO_TAG} --working-dir /workspace

# Expose BentoML port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:3000/readyz || exit 1

# Run BentoML server
ENTRYPOINT ["bentoml", "serve", "/workspace/", "--production", "--host", "0.0.0.0", "--port", "3000", "--enable-features", "all"]