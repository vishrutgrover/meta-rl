"""GTM Strategy Optimizer — OpenEnv Environment implementation."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment

from models import (
    ChannelMetrics,
    ExperimentResult,
    FunnelMetrics,
    GTMAction,
    GTMObservation,
    GTMState,
    SegmentMetrics,
)
from server.simulation import EXPERIMENT_TYPES, MESSAGING_DIMS, PRICING_ACTIONS
from server.tasks import create_simulator, get_task, TASKS


class GTMEnvironment(Environment):
    """OpenEnv environment simulating Go-To-Market strategy optimization.

    Each episode represents a product launch lifecycle. The agent makes weekly
    decisions about budget allocation, customer targeting, messaging, experiments,
    and pricing to maximize revenue under uncertainty.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._state = GTMState()
        self._sim = None
        self._task_def = None
        self._grader_scores: dict[str, float] = {}

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "channel_optimizer",
        **kwargs: Any,
    ) -> GTMObservation:
        """Start a new GTM episode for the given task."""
        task_def = get_task(task_id)
        self._task_def = task_def
        self._sim = create_simulator(task_id, seed=seed)

        self._state = GTMState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            difficulty=task_def.difficulty,
            true_brand_strength=50.0,
            true_market_demand=1.0,
            total_revenue=0.0,
            total_spend=0.0,
            total_conversions=0,
            compliance_violations=0,
            experiments_run=0,
            useful_experiments=0,
        )

        s = self._sim.state
        channels = list(self._sim.channels.keys())
        segments = list(self._sim.segments.keys())

        return GTMObservation(
            done=False,
            reward=None,
            week=0,
            total_weeks=s.total_weeks,
            budget_remaining=s.budget_remaining,
            weekly_budget=s.weekly_budget,
            channel_metrics={ch: ChannelMetrics() for ch in channels},
            funnel=FunnelMetrics(),
            segment_performance={seg: SegmentMetrics() for seg in segments},
            experiment_result=None,
            brand_score=50.0,
            total_revenue=0.0,
            total_conversions=0,
            average_cac=0.0,
            available_channels=channels,
            available_segments=segments,
            available_experiments=self._task_def.available_experiments,
            available_pricing_actions=self._task_def.available_pricing_actions,
            messaging_dimensions=MESSAGING_DIMS,
            message=self._initial_message(task_def),
        )

    def step(
        self,
        action: GTMAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GTMObservation:
        """Execute one week of GTM activity."""
        if self._sim is None:
            raise RuntimeError("Must call reset() before step()")

        self._state.step_count += 1

        # Run simulation step
        result = self._sim.step(
            budget_allocation=action.budget_allocation,
            segment_targeting=action.segment_targeting,
            messaging=action.messaging,
            experiment=action.experiment if action.experiment in self._task_def.available_experiments else None,
            pricing_action=action.pricing_action if action.pricing_action in self._task_def.available_pricing_actions else None,
        )

        s = self._sim.state
        done = self._sim.is_done

        # Update internal state
        self._state.true_brand_strength = s.brand_strength
        self._state.true_market_demand = s.market_demand
        self._state.total_revenue = s.total_revenue
        self._state.total_spend = s.total_spend
        self._state.total_conversions = s.total_conversions
        self._state.compliance_violations = s.compliance_violations
        self._state.experiments_run = s.experiments_run
        self._state.useful_experiments = s.useful_experiments

        # Compute step reward (partial progress signal)
        reward = self._compute_reward(result, s)

        # If episode done, also compute and store grader score
        if done:
            grader_score = self._task_def.grader(s)
            self._grader_scores[self._state.episode_id] = grader_score

        # Build observation
        channel_metrics = {
            ch: ChannelMetrics(**m) for ch, m in result["channel_metrics"].items()
        }
        funnel = FunnelMetrics(**result["funnel"])
        segment_perf = {
            seg: SegmentMetrics(**m) for seg, m in result["segment_performance"].items()
        }
        exp_result = None
        if result["experiment_result"]:
            exp_result = ExperimentResult(**result["experiment_result"])

        avg_cac = s.total_spend / max(s.total_conversions, 1)

        return GTMObservation(
            done=done,
            reward=round(reward, 4),
            week=s.week,
            total_weeks=s.total_weeks,
            budget_remaining=round(s.budget_remaining, 2),
            weekly_budget=round(s.weekly_budget, 2),
            channel_metrics=channel_metrics,
            funnel=funnel,
            segment_performance=segment_perf,
            experiment_result=exp_result,
            brand_score=result["brand_score_observed"],
            total_revenue=round(s.total_revenue, 2),
            total_conversions=s.total_conversions,
            average_cac=round(avg_cac, 2),
            available_channels=list(self._sim.channels.keys()),
            available_segments=list(self._sim.segments.keys()),
            available_experiments=self._task_def.available_experiments,
            available_pricing_actions=self._task_def.available_pricing_actions,
            messaging_dimensions=MESSAGING_DIMS,
            message=self._step_message(result, s, done),
        )

    @property
    def state(self) -> GTMState:
        return self._state

    def get_grader_score(self, episode_id: str) -> Optional[float]:
        """Get the grader score for a completed episode."""
        return self._grader_scores.get(episode_id)

    # ── Private helpers ────────────────────────────────────────────

    def _compute_reward(self, result: dict, s) -> float:
        """Per-step reward with partial progress signal."""
        weekly_rev = result["weekly_revenue"]
        target_weekly = self._task_def.revenue_target / self._task_def.total_weeks

        # revenue component (0-0.5)
        rev_reward = min(0.5, 0.5 * weekly_rev / max(target_weekly, 1.0))

        # efficiency bonus (0-0.2)
        weekly_spend = sum(
            m.get("spend", 0.0) for m in result["channel_metrics"].values()
        )
        if weekly_spend > 0:
            roi = weekly_rev / weekly_spend
            eff_reward = min(0.2, 0.2 * roi / 3.0)
        else:
            eff_reward = 0.0

        # brand maintenance (0-0.15)
        brand_reward = 0.15 * (s.brand_strength / 100.0)

        # penalties
        waste_penalty = 0.0
        for ch_name, m in result["channel_metrics"].items():
            if m.get("spend", 0) > 100 and m.get("conversions", 0) == 0:
                waste_penalty += 0.05

        compliance_penalty = s.compliance_violations * 0.1

        reward = rev_reward + eff_reward + brand_reward - waste_penalty - compliance_penalty
        return max(-1.0, min(1.0, reward))

    def _initial_message(self, task_def) -> str:
        channels = ", ".join(c.name for c in task_def.channels)
        segments = ", ".join(s.name for s in task_def.segments)
        return (
            f"Welcome to the GTM Strategy Optimizer — Task: {task_def.name} ({task_def.difficulty})\n"
            f"\n"
            f"{task_def.description}\n"
            f"\n"
            f"Duration: {task_def.total_weeks} weeks | Budget: ${task_def.total_budget:,.0f} "
            f"(${task_def.total_budget / task_def.total_weeks:,.0f}/week)\n"
            f"Channels: {channels}\n"
            f"Segments: {segments}\n"
            f"Product price: ${task_def.product.base_price:.0f}\n"
            f"\n"
            f"Allocate your budget wisely across channels and segments. "
            f"Craft messaging that resonates with your target customers. "
            f"Maximize revenue while building brand strength."
        )

    def _step_message(self, result: dict, s, done: bool) -> str:
        weekly_rev = result["weekly_revenue"]
        parts = [f"Week {s.week}/{s.total_weeks} | Revenue this week: ${weekly_rev:,.0f}"]
        parts.append(
            f"Cumulative: ${s.total_revenue:,.0f} revenue, "
            f"{s.total_conversions} conversions, "
            f"${s.budget_remaining:,.0f} budget remaining"
        )
        parts.append(f"Brand health: {result['brand_score_observed']:.0f}/100")

        if result["experiment_result"]:
            er = result["experiment_result"]
            parts.append(f"Experiment result: {er['recommendation']}")

        if done:
            grader = self._task_def.grader(s)
            parts.append(f"\nEpisode complete! Final grader score: {grader:.4f}")

        return " | ".join(parts) if not done else "\n".join(parts)
