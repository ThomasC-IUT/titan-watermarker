# ============================================
# Titan Visual Watermarker — Production Ready
# ============================================
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================
FROM python:3.12-slim AS runtime

# System dependencies for OpenCV, PDF processing, and Fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    fonts-dejavu-core \
    fonts-liberation \
    file \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -r titan && useradd -r -g titan titan

COPY --from=builder /install /usr/local

WORKDIR /app
COPY app/ ./app/

# Create data directories and set permissions
RUN mkdir -p /data/uploads /data/processed /data/keys && \
    chown -R titan:titan /data /app

# Switch to non-root user
USER titan

ENV PYTHONUNBUFFERED=1
ENV FORTRESS_DATA_DIR=/data
ENV FORTRESS_KEYS_DIR=/data/keys

EXPOSE 8000

# Basic healthcheck for production readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
