# Submission image for Track 2 (Video Captioning Agent). linux/amd64, no GPU.
# All models are called over the Fireworks API, so the image stays slim.
#
# Track 2 injects NO API key ("use your own credentials inside the container"),
# so the key is baked in at build time from a build-arg (kept out of the repo;
# supplied by a CI secret). Build:
#   docker buildx build --platform linux/amd64 \
#     --build-arg FIREWORKS_API_KEY=fw_xxx -t <registry>/track2:latest --push .
FROM python:3.11-slim

# OpenCV runtime needs a couple of shared libs even for the headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/
COPY providers/ ./providers/
COPY run.py .

# API key supplied at build time (not committed). Present only in the built image.
ARG FIREWORKS_API_KEY=""
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}

# Default to the validated non-Gemma Fireworks pipeline. Overridable via env.
ENV VISION_PROVIDER=fireworks \
    LLM_PROVIDER=fireworks \
    FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
    FIREWORKS_VISION_MODEL_ID=accounts/fireworks/models/qwen3p7-plus \
    FIREWORKS_TEXT_MODEL_ID=accounts/fireworks/models/minimax-m3 \
    MAX_FRAMES=8 \
    SAMPLE_FPS=1.0 \
    MAX_IMAGE_SIDE=768 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "run.py"]
