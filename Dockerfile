# ----------------------------------------------------
# Stage 1: Build Dependencies
# ----------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Upgrade OS packages to clear out known base-image vulnerabilities (CVEs)
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

COPY pyproject.toml ./

RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache -r pyproject.toml

# ----------------------------------------------------
# Stage 2: Runtime Environment
# ----------------------------------------------------
FROM python:3.12-slim-bookworm AS runner

WORKDIR /app

# Upgrade OS packages in the runtime stage as well
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]