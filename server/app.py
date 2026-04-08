"""FastAPI application for the GTM Strategy Optimizer environment."""

from __future__ import annotations

import os
import sys

# Ensure parent directory is on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

import gradio as gr

from openenv.core.env_server import create_fastapi_app

from models import GTMAction, GTMObservation
from server.environment import GTMEnvironment
from server.tasks import TASKS
from server.simulation import MESSAGING_DIMS
from ui.dashboard import build_dashboard, DASHBOARD_CSS

# Create the core OpenEnv FastAPI app (REST + WebSocket endpoints, no default UI)
app = create_fastapi_app(GTMEnvironment, GTMAction, GTMObservation)

# Mount our own custom Gradio dashboard at /web — bypasses OpenEnv's default
# Playground tab so users only see the polished comparison + interactive UI.
# app = gr.mount_gradio_app(
#     app,
#     build_dashboard(),
#     path="/web",
#     theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
#     css=DASHBOARD_CSS,
# )


# ── Root: redirect to the Gradio dashboard ───────────────────────────────

from fastapi.responses import RedirectResponse


@app.get("/")
def root():
    """Send visitors straight to the dashboard."""
    return {
        "message": "GTM Strategy Optimizer API",
        "endpoints": ["/tasks", "/grader", "/baseline", "/infer"],
        "docs": "/docs"
    }


# ── Custom endpoints required by the hackathon ─────────────────────────────


class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: str
    description: str
    total_weeks: int
    total_budget: float
    channels: list[str]
    segments: list[str]
    messaging_dimensions: list[str]
    available_experiments: list[str]
    available_pricing_actions: list[str]
    action_schema: dict
    has_grader: bool = True
    grader_endpoint: str = "/grader"


@app.get("/tasks")
def list_tasks() -> list[TaskInfo]:
    """Return list of tasks and the action schema."""
    result = []
    for task_id, t in TASKS.items():
        result.append(
            TaskInfo(
                task_id=task_id,
                name=t.name,
                difficulty=t.difficulty,
                description=t.description,
                total_weeks=t.total_weeks,
                total_budget=t.total_budget,
                channels=[c.name for c in t.channels],
                segments=[s.name for s in t.segments],
                messaging_dimensions=MESSAGING_DIMS,
                available_experiments=t.available_experiments,
                available_pricing_actions=t.available_pricing_actions,
                action_schema={
                    "budget_allocation": {
                        "type": "object",
                        "description": "channel_name -> fraction of weekly budget (sum <= 1.0)",
                        "keys": [c.name for c in t.channels],
                    },
                    "segment_targeting": {
                        "type": "object",
                        "description": "segment_name -> weight (should sum to ~1.0)",
                        "keys": [s.name for s in t.segments],
                    },
                    "messaging": {
                        "type": "object",
                        "description": "dimension -> weight (should sum to ~1.0)",
                        "keys": MESSAGING_DIMS,
                    },
                    "experiment": {
                        "type": "string|null",
                        "options": t.available_experiments,
                    },
                    "pricing_action": {
                        "type": "string|null",
                        "options": t.available_pricing_actions,
                    },
                },
            )
        )
    return result


class GraderRequest(BaseModel):
    task_id: str
    episode_id: Optional[str] = None


class GraderResponse(BaseModel):
    task_id: str
    episode_id: str
    score: Optional[float]
    message: str


def _run_grader(task_id: str, episode_id: Optional[str] = None) -> GraderResponse:
    """Core grader logic shared across endpoints."""
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

    from server.tasks import create_simulator, get_task

    task_def = get_task(task_id)
    sim = create_simulator(task_id, seed=42)

    channels = list(sim.channels.keys())
    segments = list(sim.segments.keys())
    equal_budget = {ch: 1.0 / len(channels) for ch in channels}
    equal_segments = {seg: 1.0 / len(segments) for seg in segments}
    equal_messaging = {dim: 1.0 / len(MESSAGING_DIMS) for dim in MESSAGING_DIMS}

    while not sim.is_done:
        sim.step(
            budget_allocation=equal_budget,
            segment_targeting=equal_segments,
            messaging=equal_messaging,
        )

    score = task_def.grader(sim.state)
    return GraderResponse(
        task_id=task_id,
        episode_id=episode_id or "",
        score=score,
        message=f"Grader score for {task_def.name}: {score:.4f}",
    )


@app.post("/grader")
def run_grader(req: GraderRequest) -> GraderResponse:
    """Return grader score for a task. episode_id is optional."""
    return _run_grader(req.task_id, req.episode_id)


@app.post("/tasks/{task_id}/grader")
def run_task_grader(task_id: str, episode_id: Optional[str] = None) -> GraderResponse:
    """Per-task grader endpoint: POST /tasks/{task_id}/grader"""
    return _run_grader(task_id, episode_id)


class BaselineResponse(BaseModel):
    scores: dict[str, float]
    message: str


@app.post("/baseline")
def run_baseline() -> BaselineResponse:
    """Run a deterministic heuristic baseline and return scores for all 3 tasks."""
    from server.tasks import create_simulator, get_task

    scores = {}
    for task_id in TASKS:
        task_def = get_task(task_id)
        sim = create_simulator(task_id, seed=42)

        channels = list(sim.channels.keys())
        segments = list(sim.segments.keys())
        equal_budget = {ch: 1.0 / len(channels) for ch in channels}
        equal_segments = {seg: 1.0 / len(segments) for seg in segments}
        equal_messaging = {dim: 1.0 / len(MESSAGING_DIMS) for dim in MESSAGING_DIMS}

        while not sim.is_done:
            sim.step(
                budget_allocation=equal_budget,
                segment_targeting=equal_segments,
                messaging=equal_messaging,
            )

        scores[task_id] = task_def.grader(sim.state)

    return BaselineResponse(
        scores=scores,
        message="Baseline (equal-allocation heuristic) scores for all tasks",
    )


# ── RL inference endpoint ──────────────────────────────────────────────────


class InferRequest(BaseModel):
    task_id: str
    seed: Optional[int] = None


class InferResponse(BaseModel):
    task_id: str
    checkpoint_loaded: bool
    grader_score: Optional[float]
    total_revenue: float
    total_conversions: int
    average_cac: float
    brand_score: float
    actions: list[dict]
    message: str


@app.post("/infer")
def run_infer(req: InferRequest) -> InferResponse:
    """Run a trained RL policy on a task and return the action trajectory."""
    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")
    from rl.infer import run_inference

    result = run_inference(req.task_id, seed=req.seed)
    return InferResponse(**result)


# ── Server entry point ─────────────────────────────────────────────────────


def main() -> None:
    """Run the FastAPI server with uvicorn (used as a console script)."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
