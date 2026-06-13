# ─────────────────────────────────────────────────────────────────
# Dockerfile — Student Performance Predictor (Streamlit App)
#
# What this does, step by step:
#   1. Starts from official Python 3.11 slim image (small & secure)
#   2. Sets the working directory inside the container
#   3. Copies requirements first (Docker cache optimization)
#   4. Installs all Python dependencies
#   5. Copies the rest of the project files
#   6. Trains the model inside the container on first build
#   7. Exposes port 8501 (Streamlit default)
#   8. Runs the Streamlit app
# ─────────────────────────────────────────────────────────────────

# ── Base Image ────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────
LABEL maintainer="Aryansingh Bais"
LABEL description="Student Performance Predictor — Streamlit App"
LABEL version="1.0.0"

# ── Environment Variables ─────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Working Directory ─────────────────────────────────────────────
WORKDIR /app

# ── Install System Dependencies ───────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python Dependencies ───────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy Project Files ────────────────────────────────────────────
COPY data/  /app/data/
COPY src/   /app/src/
COPY app/   /app/app/
COPY api/   /app/api/

# ── Create model directory ────────────────────────────────────────
RUN mkdir -p /app/model

# ── Generate Dataset & Train Model ───────────────────────────────
# Using absolute paths to avoid any working directory confusion
RUN cd /app && python /app/data/generate_data.py
RUN cd /app && python -m src.train

# ── Expose Port ───────────────────────────────────────────────────
EXPOSE 8501

# ── Health Check ──────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Run Command ───────────────────────────────────────────────────
CMD ["streamlit", "run", "/app/app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]