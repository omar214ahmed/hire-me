FROM python:3.11-slim

# System deps:
#   - build-essential: some deps (e.g. ctranslate2's dependents) may need to
#     compile; harmless to keep even for wheel-only installs.
#   - libgomp1: OpenMP runtime used by ctranslate2 (faster-whisper).
#   - ffmpeg: faster-whisper/ctranslate2 audio decoding.
#   - curl: container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer is cached unless requirements.txt changes.
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
