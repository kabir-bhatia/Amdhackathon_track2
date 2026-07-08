# Track 2 — Video Captioning Agent (AMD Developer Hackathon: ACT II)

Reads `/input/tasks.json`, watches each clip, and writes captions in four styles
(`formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`) to
`/output/results.json`.

## Architecture

```
tasks.json → sample frames (OpenCV) → VisionProvider.describe → LLMProvider.style ×4 → results.json
```

Two swappable interfaces isolate *where* models run from *what* the pipeline does:

| Interface        | stub                | Stage 1 (GPU pod)   | Stage 2 (later)        |
|------------------|---------------------|---------------------|------------------------|
| `VisionProvider` | `StubVisionProvider`| `LocalVLMProvider`  | `HostedVLMProvider`    |
| `LLMProvider`    | `StubLLMProvider`   | `LocalGemmaProvider`| `FireworksGemmaProvider` |

Selected at runtime by `VISION_PROVIDER` / `LLM_PROVIDER` env vars (see `.env.example`).

## Run locally (stub — no models, safe on 4 GB VRAM)

```bash
pip install -r requirements.txt
mkdir -p test/output
INPUT_PATH=test/input/tasks.json OUTPUT_PATH=test/output/results.json \
  VISION_PROVIDER=stub LLM_PROVIDER=stub python run.py
python validate_results.py test/output/results.json test/input/tasks.json
```

## Run on the AMD GPU pod (real models)

```bash
pip install -r requirements.txt -r requirements-gpu.txt   # torch is usually preinstalled
rocm-smi                                                   # confirm the GPU
INPUT_PATH=test/input/tasks.json OUTPUT_PATH=test/output/results.json \
  VISION_PROVIDER=local_vlm LLM_PROVIDER=local_gemma python run.py
```

## Docker (linux/amd64, judging VM target)

```bash
docker buildx build --platform linux/amd64 -t <registry>/track2:latest --push .
docker run --rm -v $PWD/test/input:/input -v $PWD/test/output:/output <registry>/track2:latest
```

## Stage 2 (deferred)
Runtime-model choice (Fireworks-hosted Gemma vs. bundled models), Gemma fine-tuning,
prompt/output-length tuning, audio/ASR, registry push + submission dry run.
