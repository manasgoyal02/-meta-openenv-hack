---
title: AgriOpsEnv
emoji: "🌾"
colorFrom: green
colorTo: yellow
sdk: docker
sdk_version: "1.0.0"
app_port: 7860
app_file: app.py
pinned: false
---

# AgriOpsEnv

AgriOpsEnv is a deterministic OpenEnv benchmark for agricultural operations and decision-making. It evaluates whether an agent can reason over structured farm context and produce practical, constraint-aware actions.

## Overview

The benchmark has 3 tasks with increasing difficulty:

1. Easy: Crop Selection Under Field Constraints
- Input: `N`, `P`, `K`, `pH`, `temperature`, `humidity`, `rainfall`, `season`, `water availability`
- Agent output: `crop_recommendation`
- Grading: exact crop match with partial credit for accepted alternatives

2. Medium: Field Issue Diagnosis and Intervention
- Input: crop type, soil conditions, visible field symptoms, weather observations
- Agent output: `diagnosis` and `intervention`
- Grading: deterministic scoring for diagnosis and intervention quality with unsafe-action penalties

3. Hard: Seasonal Farm Planning Under Constraints
- Input: soil profile, seasonal weather forecast, irrigation availability, fertilizer availability, farm size, budget
- Agent output: crop, fertilizer strategy, irrigation strategy, estimated cost
- Grading: deterministic component scoring for crop suitability, fertilizer suitability, irrigation feasibility, and budget compliance

## Reward Design

The environment uses dense rewards:
- rewards useful intermediate actions
- gives partial credit for partially correct answers
- penalizes invalid actions, repeated low-value actions, unsafe advice, and infeasible plans

All scoring is deterministic and reproducible.

## Environment API

Typed Pydantic models are preserved:
- `AgriOpsAction`
- `AgriOpsObservation`
- `AgriOpsState`

OpenEnv interface is preserved:
- `reset(task=...)`
- `step(action)`
- `state`

## Project Layout

```text
.
|-- models.py
|-- client.py
|-- inference.py
|-- openenv.yaml
|-- pyproject.toml
|-- requirements.txt
|-- server/
|   |-- app.py
|   |-- environment.py
|   `-- Dockerfile
`-- agriops_env/
    |-- models.py
    |-- client.py
    |-- openenv.yaml
    |-- pyproject.toml
    |-- requirements.txt
    `-- server/
        |-- app.py
        |-- environment.py
        `-- Dockerfile
```

The duplicated layout is intentionally kept for OpenEnv/hackathon compatibility.

## Local Run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Start environment server

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3) Run baseline agent

Linux/macOS:

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export AGRIOPS_TASK=easy
export AGRIOPS_ENV_URL=http://localhost:8000
python inference.py
```

Windows PowerShell:

```powershell
$env:HF_TOKEN="your_token"
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:AGRIOPS_TASK="easy"
$env:AGRIOPS_ENV_URL="http://localhost:8000"
python inference.py
```

## Docker

```bash
docker build -t agriops-env -f server/Dockerfile .
docker run -p 8000:7860 agriops-env
```

## OpenEnv Validation

Use your usual validator flow. The implementation keeps OpenEnv-compatible reset/step/state behavior and deterministic scoring.

## Notes

- Default baseline mode is deterministic heuristic policy for stable benchmark scores.
- Set `USE_LLM_POLICY=1` to allow LLM output to override matching action stages.
