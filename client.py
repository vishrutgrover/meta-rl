"""Client for the GTM Strategy Optimizer environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import (
    ChannelMetrics,
    ExperimentResult,
    FunnelMetrics,
    GTMAction,
    GTMObservation,
    GTMState,
    SegmentMetrics,
)


class GTMEnv(EnvClient[GTMAction, GTMObservation, GTMState]):
    """WebSocket client for the GTM Strategy Optimizer environment."""

    def _step_payload(self, action: GTMAction) -> Dict[str, Any]:
        """Serialize a GTMAction to JSON for the wire."""
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[GTMObservation]:
        """Parse server response into StepResult[GTMObservation]."""
        obs_data = payload.get("observation", {})

        # Parse nested channel metrics
        channel_metrics = {}
        for ch, m in obs_data.get("channel_metrics", {}).items():
            channel_metrics[ch] = ChannelMetrics(**m) if isinstance(m, dict) else m

        # Parse funnel
        funnel_data = obs_data.get("funnel", {})
        funnel = FunnelMetrics(**funnel_data) if isinstance(funnel_data, dict) else FunnelMetrics()

        # Parse segment performance
        segment_perf = {}
        for seg, m in obs_data.get("segment_performance", {}).items():
            segment_perf[seg] = SegmentMetrics(**m) if isinstance(m, dict) else m

        # Parse experiment result
        exp_data = obs_data.get("experiment_result")
        exp_result = ExperimentResult(**exp_data) if exp_data else None

        obs = GTMObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            week=obs_data.get("week", 0),
            total_weeks=obs_data.get("total_weeks", 12),
            budget_remaining=obs_data.get("budget_remaining", 0.0),
            weekly_budget=obs_data.get("weekly_budget", 0.0),
            channel_metrics=channel_metrics,
            funnel=funnel,
            segment_performance=segment_perf,
            experiment_result=exp_result,
            brand_score=obs_data.get("brand_score", 50.0),
            total_revenue=obs_data.get("total_revenue", 0.0),
            total_conversions=obs_data.get("total_conversions", 0),
            average_cac=obs_data.get("average_cac", 0.0),
            available_channels=obs_data.get("available_channels", []),
            available_segments=obs_data.get("available_segments", []),
            available_experiments=obs_data.get("available_experiments", []),
            available_pricing_actions=obs_data.get("available_pricing_actions", []),
            messaging_dimensions=obs_data.get("messaging_dimensions", []),
            message=obs_data.get("message", ""),
        )

        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> GTMState:
        """Parse server state response into GTMState."""
        return GTMState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "channel_optimizer"),
            difficulty=payload.get("difficulty", "easy"),
            true_brand_strength=payload.get("true_brand_strength", 50.0),
            true_market_demand=payload.get("true_market_demand", 1.0),
            total_revenue=payload.get("total_revenue", 0.0),
            total_spend=payload.get("total_spend", 0.0),
            total_conversions=payload.get("total_conversions", 0),
            compliance_violations=payload.get("compliance_violations", 0),
            experiments_run=payload.get("experiments_run", 0),
            useful_experiments=payload.get("useful_experiments", 0),
        )
