FROM python:3.11-slim

# System deps: build-essential for packages that compile C extensions,
# libgomp1 for onnxruntime/torch's OpenMP threading.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer is cached unless requirements.txt changes.
COPY app/requirements.txt .

# CPU-only torch build — much smaller than the default CUDA wheel and this
# project doesn't use GPU. Remove this line if you need CUDA.

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall --no-deps "transformers==4.44.2" \
    && pip show transformers | grep -i version

COPY app/ .

# Uploaded resumes land here (see helpers/config.py STORAGE_DIR /
# preprocessing/validator.py files_dir) — created so the app doesn't need
# root to mkdir at runtime.
RUN mkdir -p files/candidates storage

EXPOSE 8000

# Default command runs the API; docker-compose overrides this for the
# celery worker service.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
