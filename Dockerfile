FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    RUNS_DIR=/app/runs

WORKDIR /app

# Install OS system dependencies for Playwright/Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    git \
    libglib2.0-0 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & package configuration
COPY pyproject.toml /app/

# Install python dependencies including fastapi & uvicorn
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir fastapi uvicorn[standard] gunicorn && \
    pip install --no-cache-dir -e .

# Install Playwright browser dependencies
RUN playwright install chromium --with-deps

# Copy application source
COPY . /app

# Create runs directory for mount/persistent storage
RUN mkdir -p /app/runs

EXPOSE 8080

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "deep_research.server:app", "--bind", "0.0.0.0:8080"]
