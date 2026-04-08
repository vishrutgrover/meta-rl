"""PPO trainer for the GTM Strategy Optimizer.

Usage:
    python -m rl.train --task channel_optimizer --total-steps 200000 --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

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


# ── Hyperparameters ────────────────────────────────────────────────────────

LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
N_STEPS = 2048
N_EPOCHS = 10
BATCH_SIZE = 64
MAX_GRAD_NORM = 0.5


# ── Rollout buffer ─────────────────────────────────────────────────────────


@dataclass
class RolloutStep:
    obs: torch.Tensor
    action: Dict[str, torch.Tensor]
    log_prob: torch.Tensor
    reward: float
    value: torch.Tensor
    done: bool


def compute_gae(
    rollout: List[RolloutStep],
    last_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation."""
    advantages = torch.zeros(len(rollout), dtype=torch.float32)
    gae = 0.0
    next_value = last_value.item()
    for t in reversed(range(len(rollout))):
        nonterminal = 0.0 if rollout[t].done else 1.0
        delta = rollout[t].reward + GAMMA * next_value * nonterminal - rollout[t].value.item()
        gae = delta + GAMMA * GAE_LAMBDA * nonterminal * gae
        advantages[t] = gae
        next_value = rollout[t].value.item()
    returns = advantages + torch.tensor([s.value.item() for s in rollout], dtype=torch.float32)
    return advantages, returns


# ── Training loop ──────────────────────────────────────────────────────────


def train(
    task_id: str,
    total_steps: int,
    seed: int = 0,
    checkpoint_dir: str = "checkpoints",
    log_every: int = 1,
) -> None:
    torch.manual_seed(seed)

    env = GTMEnvironment()
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
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    obs = env.reset(task_id=task_id, seed=seed)
    obs_t = obs_to_tensor(obs, task)

    global_step = 0
    update_idx = 0
    best_mean_return = -float("inf")
    episode_returns: List[float] = []
    current_return = 0.0
    start_time = time.time()

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{task_id}.pt")

    while global_step < total_steps:
        # ── Collect rollout ──────────────────────────────────────
        rollout: List[RolloutStep] = []
        for _ in range(N_STEPS):
            with torch.no_grad():
                sample, log_prob, value = policy.act(obs_t.unsqueeze(0))
            sample_squeezed = {k: v.squeeze(0) for k, v in sample.items()}

            action = policy_sample_to_action(sample_squeezed, task)
            next_obs = env.step(action)
            reward = float(next_obs.reward) if next_obs.reward is not None else 0.0
            done = bool(next_obs.done)
            current_return += reward

            rollout.append(
                RolloutStep(
                    obs=obs_t,
                    action=sample_squeezed,
                    log_prob=log_prob.squeeze(0).detach(),
                    reward=reward,
                    value=value.squeeze(0).detach(),
                    done=done,
                )
            )

            global_step += 1
            if done:
                episode_returns.append(current_return)
                current_return = 0.0
                next_obs = env.reset(task_id=task_id, seed=seed + global_step)
            obs_t = obs_to_tensor(next_obs, task)

        # bootstrap final value
        with torch.no_grad():
            _, _, last_value = policy.act(obs_t.unsqueeze(0))
            last_value = last_value.squeeze(0).detach()

        advantages, returns = compute_gae(rollout, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Stack rollout into tensors for minibatching
        obs_batch = torch.stack([s.obs for s in rollout])
        old_log_probs = torch.stack([s.log_prob for s in rollout])
        action_batch = {
            k: torch.stack([s.action[k] for s in rollout]) for k in rollout[0].action
        }

        # ── PPO update ───────────────────────────────────────────
        n = len(rollout)
        indices = list(range(n))
        for _ in range(N_EPOCHS):
            # shuffle
            perm = torch.randperm(n).tolist()
            for start in range(0, n, BATCH_SIZE):
                mb = perm[start : start + BATCH_SIZE]
                mb_obs = obs_batch[mb]
                mb_actions = {k: v[mb] for k, v in action_batch.items()}
                mb_old_lp = old_log_probs[mb]
                mb_adv = advantages[mb]
                mb_ret = returns[mb]

                new_log_probs, entropy, value_pred = policy.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (value_pred - mb_ret).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + VF_COEF * value_loss + ENT_COEF * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        update_idx += 1

        # ── Logging + checkpoint ─────────────────────────────────
        recent = episode_returns[-20:] if episode_returns else [current_return]
        mean_return = sum(recent) / len(recent)
        elapsed = time.time() - start_time
        if update_idx % log_every == 0:
            print(
                f"[{task_id}] update={update_idx} step={global_step}/{total_steps} "
                f"episodes={len(episode_returns)} mean_return(last20)={mean_return:.3f} "
                f"policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f} "
                f"entropy={entropy.mean().item():.3f} elapsed={elapsed:.0f}s"
            )

        if mean_return > best_mean_return and len(episode_returns) >= 5:
            best_mean_return = mean_return
            torch.save(
                {
                    "model_state": policy.state_dict(),
                    "task_id": task_id,
                    "obs_dim": obs_dim,
                    "action_dims": action_dims,
                    "best_mean_return": best_mean_return,
                    "step": global_step,
                },
                checkpoint_path,
            )
            print(f"  ↳ saved checkpoint (mean_return={best_mean_return:.3f}) → {checkpoint_path}")

    print(f"Done. Best mean return: {best_mean_return:.3f}. Checkpoint: {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["channel_optimizer", "growth_strategist", "market_dominator"])
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    train(
        task_id=args.task,
        total_steps=args.total_steps,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
