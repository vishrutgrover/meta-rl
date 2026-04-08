---
title: GTM Strategy Optimizer
emoji: 📈
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
base_path: /web
pinned: false
tags:
  - openenv
---

# GTM Strategy Optimizer — OpenEnv Environment

An RL environment that simulates **Go-To-Market (GTM) strategy optimization** for product launches. Agents learn to allocate marketing budgets, target customer segments, craft messaging, run experiments, and adjust pricing to maximize revenue under uncertainty.

The Hugging Face Space ships with an interactive dashboard at `/web/` (the default route) that runs the **trained PPO policy**, an **equal-allocation heuristic**, and a **uniform random** agent on the same task and seed, then renders revenue / brand-health / budget-allocation comparisons via Plotly.

## Why GTM?

Every startup and growth team does GTM optimization manually — iterating on channels, messaging, and targeting through trial and error. This environment captures the real complexity: noisy metrics, delayed brand effects, diminishing returns on ad spend, and the tension between short-term revenue and long-term brand strength.

## Action Space

Each timestep (1 week), the agent chooses:

| Action | Type | Description |
|--------|------|-------------|
| `budget_allocation` | `dict[str, float]` | Channel → fraction of weekly budget (sum ≤ 1.0) |
| `segment_targeting` | `dict[str, float]` | Segment → targeting weight (sum ≈ 1.0) |
| `messaging` | `dict[str, float]` | Dimension → emphasis weight (sum ≈ 1.0) |
| `experiment` | `str \| null` | Optional experiment to launch |
| `pricing_action` | `str \| null` | Optional pricing change |

**Messaging dimensions:** cost_savings, performance, reliability, innovation, ease_of_use, security

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `week` / `total_weeks` | `int` | Current week and episode length |
| `budget_remaining` | `float` | Remaining budget |
| `channel_metrics` | `dict` | Per-channel: impressions, clicks, conversions, spend, CTR, CVR, ROI |
| `funnel` | `dict` | Visitors, signups, activations, retained users + rates |
| `segment_performance` | `dict` | Per-segment: conversion rate, engagement, churn, revenue |
| `experiment_result` | `dict \| null` | Completed experiment results |
| `brand_score` | `float` | Noisy proxy for brand health (0-100) |
| `total_revenue` | `float` | Cumulative revenue |
| `message` | `str` | Human-readable summary |

## Tasks

| Task | Difficulty | Weeks | Channels | Segments | Features |
|------|-----------|-------|----------|----------|----------|
| `channel_optimizer` | Easy | 12 | 3 | 2 | Budget + targeting only |
| `growth_strategist` | Medium | 24 | 5 | 3 | + experiments, pricing, brand management |
| `market_dominator` | Hard | 36 | 7 | 4 | + active competitor, market regime shifts, compliance traps |

## Setup & Usage

### Local Development

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t gtm-optimizer -f server/Dockerfile .
docker run -p 8000:8000 gtm-optimizer
```

### Client Usage

```python
from client import GTMEnv
from models import GTMAction

with GTMEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="channel_optimizer")
    while not result.done:
        action = GTMAction(
            budget_allocation={"paid_search": 0.5, "paid_social": 0.3, "email_lifecycle": 0.2},
            segment_targeting={"startup_founders": 0.6, "smb_owners": 0.4},
            messaging={"performance": 0.3, "innovation": 0.3, "ease_of_use": 0.2, "cost_savings": 0.1, "reliability": 0.05, "security": 0.05},
        )
        result = env.step(action)
    print(f"Score: {result.observation.reward}")
```

### Baseline Inference

```bash
export OPENAI_API_KEY=sk-...
python baseline.py --model gpt-4o-mini
```

### Train an RL Policy

A custom lightweight PPO trainer (`rl/train.py`) trains a small actor-critic
network against the simulator. One checkpoint per task.

```bash
# Train
python -m rl.train --task channel_optimizer --total-steps 200000
python -m rl.train --task growth_strategist --total-steps 300000
python -m rl.train --task market_dominator  --total-steps 500000

# Inference (greedy rollout, prints per-week actions and grader score)
python -m rl.infer --task channel_optimizer
```

Checkpoints are written to `checkpoints/<task_id>.pt`. Commit them so the
deployed Space can serve `/infer` without retraining.

### Inference via API

```bash
curl -X POST http://localhost:7860/infer \
     -H "Content-Type: application/json" \
     -d '{"task_id": "channel_optimizer", "seed": 42}'
```

Returns a JSON payload with `grader_score`, `total_revenue`, and the full
weekly action trajectory.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks` | GET | List all tasks with action schemas |
| `/baseline` | POST | Run heuristic baseline, return scores |
| `/grader` | POST | Get grader score for a task |
| `/infer` | POST | Run trained RL policy on a task, return action trajectory |
| `/reset` | POST | Reset environment for a task |
| `/step` | POST | Execute one action step |
| `/state` | GET | Get current episode state |
| `/health` | GET | Health check |
| `/ws` | WS | WebSocket endpoint for persistent sessions |

## Baseline Scores

| Task | Heuristic (equal alloc) |
|------|------------------------|
| `channel_optimizer` | ~0.51 |
| `growth_strategist` | ~0.33 |
| `market_dominator` | ~0.42 |

Scores improve with intelligent channel selection, messaging alignment, and experimentation.

## Environment Dynamics

- **Diminishing returns**: Channel effectiveness decays with cumulative spend
- **Brand evolution**: Consistent messaging builds brand; variance erodes it
- **Noisy observations**: All metrics include noise proportional to difficulty
- **Delayed effects**: Brand investment pays off over weeks, not immediately
- **Competitor response** (hard mode): Competitor increases aggression when you perform well
- **Market shifts** (hard mode): Demand shocks at weeks ~12 and ~24
