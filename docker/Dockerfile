# Reproducible Python 3.12 + ML wheels. Optional — systemd on the host is simpler and faster.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 fonts-dejavu-core git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 ASA_CONFIG=/app/config/config.yaml
CMD ["python", "-m", "asa.core.runner", "tick"]
