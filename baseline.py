"""Baseline inference script for the GTM Strategy Optimizer.

Uses the OpenAI API to run an LLM agent against all 3 tasks.
Reads OPENAI_API_KEY from environment variables.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py [--task TASK_ID] [--model MODEL]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
import google.generativeai as genai

from models import GTMAction
from server.simulation import MESSAGING_DIMS
from server.tasks import create_simulator, get_task, TASKS

class TeeLogger:
    def __init__(self, filename="baseline.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger()



SYSTEM_PROMPT = """You are a Go-To-Market (GTM) strategy optimizer. You manage a product launch by making weekly decisions about:

1. **Budget allocation**: How to split your weekly marketing budget across available channels
2. **Segment targeting**: How to weight your targeting across customer segments
3. **Messaging**: Which value propositions to emphasize
4. **Experiments** (if available): Which experiments to run
5. **Pricing** (if available): Whether to adjust pricing

You receive weekly performance metrics and must respond with a JSON action.

Strategy tips:
- Diversify budget initially, then double down on high-performing channels
- Match messaging to segment preferences (e.g., startups care about innovation/performance)
- Maintain brand consistency — don't change messaging wildly week to week
- Use experiments to validate hypotheses before scaling
- Monitor ROI per channel and shift budget away from underperforming channels

Your response must be ONLY valid JSON matching this schema:
{
    "budget_allocation": {"channel_name": fraction, ...},  // fractions sum to <= 1.0
    "segment_targeting": {"segment_name": weight, ...},    // weights sum to ~1.0
    "messaging": {"dimension": weight, ...},               // weights sum to ~1.0
    "experiment": "experiment_type" or null,
    "pricing_action": "action" or null
}
"""


def format_observation(obs_dict: dict) -> str:
    """Format observation into a readable prompt for the LLM."""
    parts = [f"**Week {obs_dict['week']}/{obs_dict['total_weeks']}**"]
    parts.append(f"Budget remaining: ${obs_dict['budget_remaining']:,.0f} (${obs_dict['weekly_budget']:,.0f}/week)")
    parts.append(f"Brand score: {obs_dict['brand_score']:.0f}/100")
    parts.append(f"Total revenue: ${obs_dict['total_revenue']:,.0f} | Conversions: {obs_dict['total_conversions']} | Avg CAC: ${obs_dict['average_cac']:,.0f}")

    parts.append("\n**Channel Performance:**")
    for ch, m in obs_dict.get("channel_metrics", {}).items():
        parts.append(
            f"  {ch}: {m['impressions']} imp, {m['clicks']} clicks, "
            f"{m['conversions']} conv, ${m['spend']:,.0f} spend, ROI={m['roi']:.2f}"
        )

    parts.append("\n**Segment Performance:**")
    for seg, m in obs_dict.get("segment_performance", {}).items():
        parts.append(
            f"  {seg}: CVR={m['conversion_rate']:.4f}, "
            f"engagement={m['engagement_score']:.1f}, ${m['revenue']:,.0f} rev"
        )

    if obs_dict.get("experiment_result"):
        er = obs_dict["experiment_result"]
        parts.append(f"\n**Experiment Result:** {er['recommendation']}")

    parts.append(f"\nAvailable channels: {obs_dict['available_channels']}")
    parts.append(f"Available segments: {obs_dict['available_segments']}")
    if obs_dict.get("available_experiments"):
        parts.append(f"Available experiments: {obs_dict['available_experiments']}")
    if obs_dict.get("available_pricing_actions"):
        parts.append(f"Available pricing actions: {obs_dict['available_pricing_actions']}")
    parts.append(f"Messaging dimensions: {obs_dict['messaging_dimensions']}")

    return "\n".join(parts)


def parse_llm_action(response_text: str, task_id: str) -> dict:
    """Parse LLM response into an action dict. Falls back to equal allocation."""
    task_def = get_task(task_id)
    channels = [c.name for c in task_def.channels]
    segments = [s.name for s in task_def.segments]

    # Default fallback
    fallback = {
        "budget_allocation": {ch: 1.0 / len(channels) for ch in channels},
        "segment_targeting": {seg: 1.0 / len(segments) for seg in segments},
        "messaging": {dim: 1.0 / len(MESSAGING_DIMS) for dim in MESSAGING_DIMS},
        "experiment": None,
        "pricing_action": None,
    }

    try:
        # Try to extract JSON from response
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        action = json.loads(text)

        # Validate keys exist
        if "budget_allocation" not in action:
            action["budget_allocation"] = fallback["budget_allocation"]
        if "segment_targeting" not in action:
            action["segment_targeting"] = fallback["segment_targeting"]
        if "messaging" not in action:
            action["messaging"] = fallback["messaging"]

        return action
    except (json.JSONDecodeError, IndexError, KeyError):
        return fallback


def run_episode(task_id: str, provider: str = "openai", model: str = "gpt-4o-mini", seed: int = 42, verbose: bool = True) -> float:
    """Run one episode of the given task with an LLM agent."""
    if provider in ["openai", "ollama"]:
        client = OpenAI(
            base_url="http://localhost:11434/v1" if provider == "ollama" else None,
            api_key="ollama" if provider == "ollama" else os.environ.get("OPENAI_API_KEY")
        )
    elif provider == "gemini":
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    task_def = get_task(task_id)
    sim = create_simulator(task_id, seed=seed)

    channels = list(sim.channels.keys())
    segments = list(sim.segments.keys())

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Initial observation prompt
    initial_msg = (
        f"You are managing a GTM campaign: **{task_def.name}** ({task_def.difficulty})\n"
        f"{task_def.description}\n\n"
        f"Duration: {task_def.total_weeks} weeks | Budget: ${task_def.total_budget:,.0f}\n"
        f"Channels: {channels}\n"
        f"Segments: {segments}\n"
        f"Messaging dimensions: {MESSAGING_DIMS}\n"
    )
    if task_def.available_experiments:
        initial_msg += f"Experiments: {task_def.available_experiments}\n"
    if task_def.available_pricing_actions:
        initial_msg += f"Pricing actions: {task_def.available_pricing_actions}\n"
    initial_msg += "\nProvide your first week's action as JSON."

    messages.append({"role": "user", "content": initial_msg})

    while not sim.is_done:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if provider in ["openai", "ollama"]:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=500,
                    )
                    llm_text = response.choices[0].message.content or ""
                elif provider == "gemini":
                    system_instruction = next((m["content"] for m in messages if m["role"] == "system"), None)
                    contents = []
                    for m in messages:
                        if m["role"] == "system": continue
                        role = "user" if m["role"] == "user" else "model"
                        contents.append({"role": role, "parts": [m["content"]]})
                    
                    gemini_model = genai.GenerativeModel(
                        model_name=model, 
                        system_instruction=system_instruction
                    )
                    response = gemini_model.generate_content(
                        contents, 
                        generation_config={"temperature": 0.3, "max_output_tokens": 500}
                    )
                    llm_text = response.text
                break  # success
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = (attempt + 1) * 15
                    if verbose:
                        print(f"  LLM API error (attempt {attempt+1}): {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    if verbose:
                        print(f"  LLM API error: {e}, using fallback after {max_retries} attempts.")
                    llm_text = ""
                    break

        action = parse_llm_action(llm_text, task_id)

        # Step simulation
        result = sim.step(
            budget_allocation=action.get("budget_allocation", {}),
            segment_targeting=action.get("segment_targeting", {}),
            messaging=action.get("messaging", {}),
            experiment=action.get("experiment"),
            pricing_action=action.get("pricing_action"),
        )

        if verbose:
            print(
                f"  Week {sim.state.week}/{sim.state.total_weeks} | "
                f"Rev: ${result['weekly_revenue']:,.0f} | "
                f"Total: ${sim.state.total_revenue:,.0f} | "
                f"Brand: {result['brand_score_observed']:.0f}"
            )

        # Build observation for next turn
        obs_dict = {
            "week": sim.state.week,
            "total_weeks": sim.state.total_weeks,
            "budget_remaining": sim.state.budget_remaining,
            "weekly_budget": sim.state.weekly_budget,
            "brand_score": result["brand_score_observed"],
            "total_revenue": sim.state.total_revenue,
            "total_conversions": sim.state.total_conversions,
            "average_cac": sim.state.total_spend / max(sim.state.total_conversions, 1),
            "channel_metrics": result["channel_metrics"],
            "segment_performance": result["segment_performance"],
            "experiment_result": result["experiment_result"],
            "available_channels": channels,
            "available_segments": segments,
            "available_experiments": task_def.available_experiments,
            "available_pricing_actions": task_def.available_pricing_actions,
            "messaging_dimensions": MESSAGING_DIMS,
        }

        if not sim.is_done:
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({
                "role": "user",
                "content": format_observation(obs_dict) + "\n\nProvide your next action as JSON.",
            })

            # Keep context manageable — trim old turns
            if len(messages) > 12:
                messages = [messages[0]] + messages[-10:]

    score = task_def.grader(sim.state)
    return score


def main():
    parser = argparse.ArgumentParser(description="GTM Baseline Inference")
    parser.add_argument("--task", type=str, default=None, help="Run specific task (default: all)")
    parser.add_argument("--provider", type=str, choices=["openai", "ollama", "gemini"], default="openai", help="LLM Provider to use")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name (e.g., gpt-4o-mini, llama3.1:8b, gemini-1.5-flash)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set for provider openai")
        sys.exit(1)
    if args.provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set for provider gemini")
        sys.exit(1)

    tasks_to_run = [args.task] if args.task else list(TASKS.keys())
    scores = {}

    for task_id in tasks_to_run:
        print(f"\n{'='*60}")
        print(f"Running task: {task_id} (Provider: {args.provider}, Model: {args.model})")
        print(f"{'='*60}")
        score = run_episode(task_id, provider=args.provider, model=args.model, seed=args.seed, verbose=not args.quiet)
        scores[task_id] = score
        print(f"Grader score: {score:.4f}")

    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.4f}")
    print(f"  Average: {sum(scores.values()) / len(scores):.4f}")


if __name__ == "__main__":
    main()
