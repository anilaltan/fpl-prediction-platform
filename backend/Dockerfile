# =============================================================================
# Backend Dockerfile - Multi-stage build for FastAPI application
# Optimized for 4GB RAM constraint with memory-efficient ML model loading
# =============================================================================

FROM python:3.11-slim AS base

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Metadata labels
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="FPL Prediction Platform" \
      org.opencontainers.image.url="https://github.com/your-org/fpl-prediction-platform" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.title="FPL Prediction Platform Backend" \
      org.opencontainers.image.description="FastAPI backend for FPL point prediction using ML"

# =============================================================================
# Dependencies Stage
# =============================================================================
FROM base AS deps

# Install system dependencies
# - gcc: Required for compiling Python packages
# - postgresql-client: For database operations
# - coinor-cbc: CBC solver for PuLP (ILP optimization)
# - curl: For health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    postgresql-client \
    coinor-cbc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Reduces image size
# --user: Installs to user directory (better for non-root user)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements.txt

# =============================================================================
# Builder Stage (for future use - e.g., compiling extensions)
# =============================================================================
FROM deps AS builder

# Copy application code
COPY . .

# =============================================================================
# Production Stage
# =============================================================================
FROM base AS runner

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    coinor-cbc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --system --gid 1001 appuser && \
    useradd --system --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from deps stage
COPY --from=deps /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Set Python path to include user site-packages
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Reduce TensorFlow/CUDA log noise (memory optimization)
ENV TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_FORCE_GPU_ALLOW_GROWTH=true

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
