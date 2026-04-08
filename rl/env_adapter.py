"""Bridge between GTMEnvironment (dict-based) and tensor-based RL code."""

from __future__ import annotations

from typing import Dict, List

import torch

from models import GTMAction, GTMObservation
from server.simulation import MESSAGING_DIMS
from server.tasks import TaskDefinition

# Sentinel string used by the categorical heads when the agent picks "no action"
NONE_OPTION = "__none__"


def compute_obs_dim(task: TaskDefinition) -> int:
    """Number of scalar features in the flattened observation tensor."""
    n_channels = len(task.channels)
    n_segments = len(task.segments)
    # 5 globals + 5 per channel + 4 per segment
    return 5 + 5 * n_channels + 4 * n_segments


def compute_action_dims(task: TaskDefinition) -> Dict[str, int]:
    """Output sizes for each policy head."""
    return {
        "budget": len(task.channels),
        "segment": len(task.segments),
        "messaging": len(MESSAGING_DIMS),
        # +1 for the "none" option
        "experiment": len(task.available_experiments) + 1,
        "pricing": len(task.available_pricing_actions) + 1,
    }


def experiment_options(task: TaskDefinition) -> List[str]:
    return [NONE_OPTION] + list(task.available_experiments)


def pricing_options(task: TaskDefinition) -> List[str]:
    return [NONE_OPTION] + list(task.available_pricing_actions)


def obs_to_tensor(obs: GTMObservation, task: TaskDefinition) -> torch.Tensor:
    """Flatten a GTMObservation into a fixed-size float32 tensor.

    Layout (in order):
        - week / total_weeks
        - budget_remaining / total_budget
        - brand_score / 100
        - total_revenue / revenue_target
        - average_cac / 100
        - per channel (in task order): spend/weekly_budget, ctr, cvr, roi, conversions/100
        - per segment (in task order): conversion_rate, engagement/100, churn, revenue/10k
    """
    total_weeks = max(obs.total_weeks, 1)
    total_budget = task.total_budget if task.total_budget > 0 else 1.0
    weekly_budget = max(obs.weekly_budget, 1.0)
    revenue_target = task.revenue_target if task.revenue_target > 0 else 1.0

    feats: List[float] = [
        obs.week / total_weeks,
        obs.budget_remaining / total_budget,
        obs.brand_score / 100.0,
        obs.total_revenue / revenue_target,
        obs.average_cac / 100.0,
    ]

    for ch in task.channels:
        m = obs.channel_metrics.get(ch.name)
        if m is None:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            feats.extend([
                m.spend / weekly_budget,
                m.ctr,
                m.cvr,
                # clip ROI into a reasonable range
                max(-2.0, min(5.0, m.roi)),
                m.conversions / 100.0,
            ])

    for seg in task.segments:
        sm = obs.segment_performance.get(seg.name)
        if sm is None:
            feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            feats.extend([
                sm.conversion_rate,
                min(1.0, sm.engagement_score / 100.0),
                sm.churn_rate,
                sm.revenue / 10000.0,
            ])

    return torch.tensor(feats, dtype=torch.float32)


def policy_sample_to_action(
    sample: Dict[str, torch.Tensor],
    task: TaskDefinition,
) -> GTMAction:
    """Convert sampled policy outputs into a GTMAction.

    sample keys:
        budget   — Tensor[n_channels] on the simplex (Dirichlet sample)
        segment  — Tensor[n_segments] on the simplex
        messaging— Tensor[6]          on the simplex
        experiment — int (index into experiment_options(task))
        pricing    — int (index into pricing_options(task))
    """
    budget = sample["budget"].detach().cpu().tolist()
    segment = sample["segment"].detach().cpu().tolist()
    messaging = sample["messaging"].detach().cpu().tolist()

    budget_alloc = {ch.name: float(budget[i]) for i, ch in enumerate(task.channels)}
    segment_target = {seg.name: float(segment[i]) for i, seg in enumerate(task.segments)}
    messaging_dict = {dim: float(messaging[i]) for i, dim in enumerate(MESSAGING_DIMS)}

    exp_idx = int(sample["experiment"].item())
    exp_opts = experiment_options(task)
    experiment = exp_opts[exp_idx] if exp_idx < len(exp_opts) else NONE_OPTION
    if experiment == NONE_OPTION:
        experiment = None

    price_idx = int(sample["pricing"].item())
    price_opts = pricing_options(task)
    pricing = price_opts[price_idx] if price_idx < len(price_opts) else NONE_OPTION
    if pricing == NONE_OPTION:
        pricing = None

    return GTMAction(
        budget_allocation=budget_alloc,
        segment_targeting=segment_target,
        messaging=messaging_dict,
        experiment=experiment,
        pricing_action=pricing,
    )
