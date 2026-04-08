"""Inference script for the GTM Strategy Optimizer environment.

Mandatory hackathon submission entry point. Drives the GTM environment with an
LLM agent (via the OpenAI client) and emits structured stdout logs in the
[START] / [STEP] / [END] format defined by the spec.

Environment variables:
    API_BASE_URL   OpenAI-compatible base URL (default: HF router)
    MODEL_NAME     Model identifier (default: Qwen2.5-72B-Instruct)
    HF_TOKEN       API key (or API_KEY)
    GTM_TASK       Task to run: channel_optimizer | growth_strategist |
                   market_dominator (default: channel_optimizer)
    GTM_SEED       Episode seed (default: 42)
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

# Make repo root importable when invoked from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from models import GTMAction
from server.environment import GTMEnvironment
from server.simulation import MESSAGING_DIMS
from server.tasks import TASKS, get_task

# ── Config ─────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"

TASK_NAME = os.getenv("GTM_TASK", "channel_optimizer")
SEED = int(os.getenv("GTM_SEED", "42"))
BENCHMARK = "gtm_strategy_optimizer"

TEMPERATURE = 0.3
MAX_TOKENS = 600
SUCCESS_SCORE_THRESHOLD = 0.5  # grader scores in [0,1]; >0.5 = beat random


# ── Structured stdout logging ──────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt + LLM helpers ───────────────────────────────────────────────────


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Go-To-Market (GTM) strategist running a product launch in a simulated market.
    Each week you decide:
      1. budget_allocation — fractions across the available channels (sum <= 1.0)
      2. segment_targeting — fractions across customer segments (sum ~ 1.0)
      3. messaging — emphasis across messaging dimensions (sum ~ 1.0)
      4. experiment — optional experiment id or null
      5. pricing_action — optional pricing change or null

    Goals: maximize revenue, maintain brand health, avoid waste, respect compliance.
    Strategy hints: diversify early, double down on high-ROI channels, match
    messaging to segment preferences, keep messaging consistent week-to-week.

    Reply with ONLY a single JSON object matching:
    {
      "budget_allocation": {"channel_name": fraction, ...},
      "segment_targeting": {"segment_name": fraction, ...},
      "messaging": {"dimension": fraction, ...},
      "experiment": "type" or null,
      "pricing_action": "action" or null
    }
    No prose, no code fences, no commentary.
    """
).strip()


def _format_observation(obs, task) -> str:
    parts = [
        f"Week {obs.week}/{obs.total_weeks}",
        f"Budget remaining: ${obs.budget_remaining:,.0f} (weekly ${obs.weekly_budget:,.0f})",
        f"Brand: {obs.brand_score:.0f}/100  Total revenue: ${obs.total_revenue:,.0f}  "
        f"Conversions: {obs.total_conversions}  CAC: ${obs.average_cac:,.0f}",
    ]
    if obs.channel_metrics:
        parts.append("Channels:")
        for ch, m in obs.channel_metrics.items():
            parts.append(
                f"  {ch}: {m.impressions} imp, {m.clicks} clk, {m.conversions} conv, "
                f"${m.spend:,.0f} spend, ROI={m.roi:.2f}"
            )
    if obs.segment_performance:
        parts.append("Segments:")
        for seg, m in obs.segment_performance.items():
            parts.append(
                f"  {seg}: cvr={m.conversion_rate:.4f}, eng={m.engagement_score:.1f}, "
                f"rev=${m.revenue:,.0f}"
            )
    if obs.experiment_result:
        parts.append(f"Experiment: {obs.experiment_result.recommendation}")

    parts.append(f"Available channels: {[c.name for c in task.channels]}")
    parts.append(f"Available segments: {[s.name for s in task.segments]}")
    if task.available_experiments:
        parts.append(f"Available experiments: {task.available_experiments}")
    if task.available_pricing_actions:
        parts.append(f"Available pricing actions: {task.available_pricing_actions}")
    parts.append(f"Messaging dimensions: {MESSAGING_DIMS}")
    parts.append("\nRespond with the JSON action only.")
    return "\n".join(parts)


def _equal_action_dict(task) -> Dict[str, Any]:
    return {
        "budget_allocation": {c.name: 1.0 / len(task.channels) for c in task.channels},
        "segment_targeting": {s.name: 1.0 / len(task.segments) for s in task.segments},
        "messaging": {d: 1.0 / len(MESSAGING_DIMS) for d in MESSAGING_DIMS},
        "experiment": None,
        "pricing_action": None,
    }


def _parse_llm_action(text: str, task) -> Dict[str, Any]:
    """Best-effort JSON extraction. Falls back to equal allocation."""
    fallback = _equal_action_dict(task)
    if not text:
        return fallback
    s = text.strip()
    if "```json" in s:
        s = s.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in s:
        s = s.split("```", 1)[1].split("```", 1)[0].strip()
    # Trim to first {...} block
    if "{" in s and "}" in s:
        s = s[s.index("{"): s.rindex("}") + 1]
    try:
        action = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return fallback
    for key in ("budget_allocation", "segment_targeting", "messaging"):
        if not isinstance(action.get(key), dict):
            action[key] = fallback[key]
    action.setdefault("experiment", None)
    action.setdefault("pricing_action", None)
    return action


def _ask_llm(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return ""


def _short_action_str(action_dict: Dict[str, Any]) -> str:
    """Compact one-line representation of an action for the [STEP] log."""
    budget = action_dict.get("budget_allocation", {}) or {}
    top = sorted(budget.items(), key=lambda kv: -kv[1])[:3]
    budget_str = ",".join(f"{k[:8]}={v:.2f}" for k, v in top)
    exp = action_dict.get("experiment") or "none"
    pricing = action_dict.get("pricing_action") or "none"
    return f"budget=[{budget_str}]/exp={exp}/price={pricing}"


# ── Main loop ──────────────────────────────────────────────────────────────


def main() -> int:
    if TASK_NAME not in TASKS:
        print(
            f"[DEBUG] Unknown GTM_TASK={TASK_NAME!r}, falling back to channel_optimizer",
            flush=True,
        )
        task_id = "channel_optimizer"
    else:
        task_id = TASK_NAME

    task = get_task(task_id)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env = GTMEnvironment()
    try:
        obs = env.reset(task_id=task_id, seed=SEED)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Starting task: {task.name} ({task.difficulty})\n"
                    f"Duration: {task.total_weeks} weeks  Budget: ${task.total_budget:,.0f}\n"
                    f"Channels: {[c.name for c in task.channels]}\n"
                    f"Segments: {[s.name for s in task.segments]}\n\n"
                    + _format_observation(obs, task)
                ),
            },
        ]

        step = 0
        while not obs.done:
            step += 1
            llm_text = _ask_llm(client, messages)
            action_dict = _parse_llm_action(llm_text, task)

            error: Optional[str] = None
            try:
                gtm_action = GTMAction(**action_dict)
                obs = env.step(gtm_action)
            except Exception as exc:
                error = f"step_failed:{exc}"
                # Use equal allocation as a safe fallback so the episode can continue
                obs = env.step(GTMAction(**_equal_action_dict(task)))

            reward = float(obs.reward) if obs.reward is not None else 0.0
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=_short_action_str(action_dict),
                reward=reward,
                done=bool(obs.done),
                error=error,
            )

            # Append to context for the next turn (trim aggressively to stay small)
            messages.append({"role": "assistant", "content": llm_text or "{}"})
            messages.append(
                {"role": "user", "content": _format_observation(obs, task)}
            )
            if len(messages) > 10:
                messages = [messages[0]] + messages[-8:]

            if obs.done:
                break

        # Final grader score (env's grader returns a value in [0, 1])
        grader = env.get_grader_score(env.state.episode_id)
        score = float(grader) if grader is not None else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] inference failed: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return 0


if __name__ == "__main__":
    sys.exit(main())
