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
# python:3.11-slim = Python 3.11 on Debian with minimal packages
# 'slim' keeps image size small (~150MB vs ~900MB for full image)
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────
LABEL maintainer="Aryansingh Bais"
LABEL description="Student Performance Predictor — Streamlit App"
LABEL version="1.0.0"

# ── Environment Variables ─────────────────────────────────────────
# Prevents Python from writing .pyc files (keeps container clean)
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr (logs appear immediately)
ENV PYTHONUNBUFFERED=1
# Streamlit config via environment
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Working Directory ─────────────────────────────────────────────
# All subsequent commands run from /app inside the container
WORKDIR /app

# ── Install System Dependencies ───────────────────────────────────
# These are needed by some Python packages (numpy, pandas internals)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/* cleans apt cache → smaller image

# ── Install Python Dependencies ───────────────────────────────────
# Copy requirements FIRST (before rest of code)
# Why? Docker caches each layer — if requirements.txt hasn't changed,
# Docker reuses the cached pip install layer → much faster rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
# --no-cache-dir prevents pip from storing download cache → smaller image

# ── Copy Project Files ────────────────────────────────────────────
COPY data/       ./data/
COPY src/        ./src/
COPY app/        ./app/
COPY api/        ./api/

# ── Generate Dataset & Train Model ───────────────────────────────
# Runs inside the container at build time so model.pkl exists
# when the app starts (model/ is in .gitignore so not copied above)
RUN python data/generate_data.py
RUN python -m src.train

# ── Expose Port ───────────────────────────────────────────────────
EXPOSE 8501

# ── Health Check ─────────────────────────────────────────────────
# Docker checks this every 30s — if it fails, container is "unhealthy"
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Run Command ───────────────────────────────────────────────────
# Starts Streamlit when container launches
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]