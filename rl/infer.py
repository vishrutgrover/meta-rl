"""Inference: load a trained checkpoint and run one episode of GTM optimization.

Usage:
    python -m rl.infer --task channel_optimizer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch

# Make repo root importable when run as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import GTMEnvironment
from server.tasks import get_task

from rl.env_adapter import (
    compute_action_dims,
    compute_obs_dim,
    obs_to_tensor,
    policy_sample_to_action,
)
from rl.policy import GTMActorCritic


def _build_policy(task_id: str) -> tuple[GTMActorCritic, Any]:
    task = get_task(task_id)
    obs_dim = compute_obs_dim(task)
    action_dims = compute_action_dims(task)
    policy = GTMActorCritic(
        obs_dim=obs_dim,
        n_channels=action_dims["budget"],
        n_segments=action_dims["segment"],
        n_messaging=action_dims["messaging"],
        n_experiments=action_dims["experiment"],
        n_pricing=action_dims["pricing"],
    )
    return policy, task


def run_inference(
    task_id: str,
    checkpoint_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one deterministic episode and return the action trajectory + metrics."""
    policy, task = _build_policy(task_id)

    if checkpoint_path is None:
        checkpoint_path = os.path.join("checkpoints", f"{task_id}.pt")

    loaded = False
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        policy.load_state_dict(ckpt["model_state"])
        loaded = True

    policy.eval()

    env = GTMEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    actions: List[Dict[str, Any]] = []

    while not obs.done:
        obs_t = obs_to_tensor(obs, task)
        with torch.no_grad():
            sample, _, _ = policy.act(obs_t.unsqueeze(0), deterministic=True)
        sample_squeezed = {k: v.squeeze(0) for k, v in sample.items()}
        action = policy_sample_to_action(sample_squeezed, task)

        next_obs = env.step(action)
        actions.append({
            "week": next_obs.week,
            "budget_allocation": action.budget_allocation,
            "segment_targeting": action.segment_targeting,
            "messaging": action.messaging,
            "experiment": action.experiment,
            "pricing_action": action.pricing_action,
            "weekly_reward": next_obs.reward,
            "total_revenue": next_obs.total_revenue,
            "brand_score": next_obs.brand_score,
        })
        obs = next_obs

    grader_score = env.get_grader_score(env.state.episode_id)
    msg = (
        f"Trained policy ({checkpoint_path})" if loaded
        else f"Untrained random policy (no checkpoint at {checkpoint_path})"
    )

    return {
        "task_id": task_id,
        "checkpoint_loaded": loaded,
        "grader_score": grader_score,
        "total_revenue": float(obs.total_revenue),
        "total_conversions": int(obs.total_conversions),
        "average_cac": float(obs.average_cac),
        "brand_score": float(obs.brand_score),
        "actions": actions,
        "message": msg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["channel_optimizer", "growth_strategist", "market_dominator"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    args = parser.parse_args()

    result = run_inference(args.task, checkpoint_path=args.checkpoint, seed=args.seed)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    print(f"Task: {result['task_id']}")
    print(f"Checkpoint loaded: {result['checkpoint_loaded']}")
    print(f"Grader score: {result['grader_score']}")
    print(f"Total revenue: ${result['total_revenue']:,.2f}")
    print(f"Total conversions: {result['total_conversions']}")
    print(f"Average CAC: ${result['average_cac']:.2f}")
    print(f"Brand score: {result['brand_score']:.1f}")
    print()
    print("Weekly actions:")
    for a in result["actions"]:
        budget_str = ", ".join(f"{k}={v:.2f}" for k, v in a["budget_allocation"].items())
        print(f"  Week {a['week']:2d}: budget=[{budget_str}]  reward={a['weekly_reward']:.3f}")


if __name__ == "__main__":
    main()
