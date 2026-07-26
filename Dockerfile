# GPU image for the real Qwen3-TTS backend.
# For the mock backend, any python:3.11-slim base works too.
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TTS_BACKEND=qwen

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir torch qwen-tts soundfile

COPY app ./app

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
