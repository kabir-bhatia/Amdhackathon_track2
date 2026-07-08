# Slim linux/amd64 base for the harness/stub path. The heavy model deps
# (torch/transformers) are intentionally NOT installed here — that decision
# (bundle-and-run on CPU vs. call a hosted API) is deferred to Stage 2.
#
# Build for the judging VM (linux/amd64):
#   docker buildx build --platform linux/amd64 -t <registry>/track2:latest --push .
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

# Default to stub providers; the harness/operator overrides via env at runtime.
ENV VISION_PROVIDER=stub \
    LLM_PROVIDER=stub \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "run.py"]
