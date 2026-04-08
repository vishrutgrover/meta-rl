"""Task definitions for the GTM Strategy Optimizer.

Three tasks with increasing difficulty:
  1. channel_optimizer (easy)   — 12 weeks, 3 channels, 2 segments
  2. growth_strategist (medium) — 24 weeks, 5 channels, 3 segments
  3. market_dominator  (hard)   — 36 weeks, 7 channels, 4 segments + competitor + regime shifts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from .simulation import (
    ChannelConfig,
    EXPERIMENT_TYPES,
    MarketSimulator,
    MESSAGING_DIMS,
    PRICING_ACTIONS,
    ProductConfig,
    SegmentConfig,
    SimState,
)
from .grader import grade_channel_optimizer, grade_growth_strategist, grade_market_dominator


@dataclass
class TaskDefinition:
    """Everything needed to instantiate + grade a task."""

    task_id: str
    name: str
    difficulty: str
    description: str
    total_weeks: int
    total_budget: float
    channels: List[ChannelConfig]
    segments: List[SegmentConfig]
    product: ProductConfig
    noise_level: float
    enable_competitor: bool
    enable_regime_shifts: bool
    revenue_target: float  # for grading
    available_experiments: List[str]
    available_pricing_actions: List[str]
    grader: Callable[[SimState], float]


# ── Task configurations ───────────────────────────────────────────────────

TASK_CHANNEL_OPTIMIZER = TaskDefinition(
    task_id="channel_optimizer",
    name="Channel Optimizer",
    difficulty="easy",
    description=(
        "Maximize revenue by allocating budget across 3 marketing channels "
        "targeting 2 customer segments over 12 weeks. Focus on finding the "
        "right channel-segment fit."
    ),
    total_weeks=12,
    total_budget=50000.0,
    channels=[
        ChannelConfig(
            name="paid_search",
            base_ctr=0.012,
            base_cvr=0.025,
            saturation_alpha=1.5,
            cost_per_impression=18.0,
            min_spend_for_signal=200.0,
            segment_affinity={"startup_founders": 1.4, "smb_owners": 1.0},
        ),
        ChannelConfig(
            name="paid_social",
            base_ctr=0.008,
            base_cvr=0.015,
            saturation_alpha=2.0,
            cost_per_impression=12.0,
            min_spend_for_signal=150.0,
            segment_affinity={"startup_founders": 1.2, "smb_owners": 0.8},
        ),
        ChannelConfig(
            name="email_lifecycle",
            base_ctr=0.025,
            base_cvr=0.035,
            saturation_alpha=1.0,
            cost_per_impression=5.0,
            min_spend_for_signal=100.0,
            segment_affinity={"startup_founders": 0.9, "smb_owners": 1.5},
        ),
    ],
    segments=[
        SegmentConfig(
            name="startup_founders",
            size=0.6,
            price_sensitivity=0.7,
            message_preference={
                "cost_savings": 0.1, "performance": 0.3, "reliability": 0.1,
                "innovation": 0.3, "ease_of_use": 0.15, "security": 0.05,
            },
            base_churn=0.08,
        ),
        SegmentConfig(
            name="smb_owners",
            size=0.4,
            price_sensitivity=0.5,
            message_preference={
                "cost_savings": 0.25, "performance": 0.15, "reliability": 0.25,
                "innovation": 0.05, "ease_of_use": 0.2, "security": 0.1,
            },
            base_churn=0.05,
        ),
    ],
    product=ProductConfig(base_price=99.0, differentiation=0.7, complexity=0.3),
    noise_level=0.1,
    enable_competitor=False,
    enable_regime_shifts=False,
    revenue_target=120000.0,
    available_experiments=[],
    available_pricing_actions=[],
    grader=grade_channel_optimizer,
)

TASK_GROWTH_STRATEGIST = TaskDefinition(
    task_id="growth_strategist",
    name="Growth Strategist",
    difficulty="medium",
    description=(
        "Maximize revenue while maintaining brand health and budget efficiency. "
        "Manage 5 channels, 3 segments, run experiments, and adjust pricing "
        "over 24 weeks. Balance short-term revenue with long-term brand building."
    ),
    total_weeks=24,
    total_budget=150000.0,
    channels=[
        ChannelConfig(
            name="paid_search", base_ctr=0.012, base_cvr=0.022,
            saturation_alpha=1.5, cost_per_impression=20.0, min_spend_for_signal=200.0,
            segment_affinity={"startup_founders": 1.4, "smb_owners": 1.0, "enterprise": 0.7},
        ),
        ChannelConfig(
            name="paid_social", base_ctr=0.008, base_cvr=0.012,
            saturation_alpha=2.0, cost_per_impression=14.0, min_spend_for_signal=150.0,
            segment_affinity={"startup_founders": 1.3, "smb_owners": 0.8, "enterprise": 0.5},
        ),
        ChannelConfig(
            name="organic_content", base_ctr=0.006, base_cvr=0.030,
            saturation_alpha=0.8, cost_per_impression=8.0, min_spend_for_signal=300.0,
            segment_affinity={"startup_founders": 1.1, "smb_owners": 1.2, "enterprise": 1.3},
        ),
        ChannelConfig(
            name="email_lifecycle", base_ctr=0.025, base_cvr=0.030,
            saturation_alpha=1.0, cost_per_impression=5.0, min_spend_for_signal=100.0,
            segment_affinity={"startup_founders": 0.9, "smb_owners": 1.5, "enterprise": 1.1},
        ),
        ChannelConfig(
            name="outbound_sales", base_ctr=0.003, base_cvr=0.045,
            saturation_alpha=1.2, cost_per_impression=50.0, min_spend_for_signal=500.0,
            segment_affinity={"startup_founders": 0.5, "smb_owners": 0.9, "enterprise": 1.8},
        ),
    ],
    segments=[
        SegmentConfig(
            name="startup_founders", size=0.4, price_sensitivity=0.7,
            message_preference={
                "cost_savings": 0.1, "performance": 0.3, "reliability": 0.1,
                "innovation": 0.3, "ease_of_use": 0.15, "security": 0.05,
            },
            base_churn=0.08,
        ),
        SegmentConfig(
            name="smb_owners", size=0.35, price_sensitivity=0.5,
            message_preference={
                "cost_savings": 0.25, "performance": 0.15, "reliability": 0.25,
                "innovation": 0.05, "ease_of_use": 0.2, "security": 0.1,
            },
            base_churn=0.05,
        ),
        SegmentConfig(
            name="enterprise", size=0.25, price_sensitivity=0.2,
            message_preference={
                "cost_savings": 0.05, "performance": 0.15, "reliability": 0.3,
                "innovation": 0.1, "ease_of_use": 0.1, "security": 0.3,
            },
            base_churn=0.03,
        ),
    ],
    product=ProductConfig(base_price=149.0, differentiation=0.65, complexity=0.5),
    noise_level=0.15,
    enable_competitor=False,
    enable_regime_shifts=False,
    revenue_target=375000.0,
    available_experiments=EXPERIMENT_TYPES,
    available_pricing_actions=PRICING_ACTIONS,
    grader=grade_growth_strategist,
)

TASK_MARKET_DOMINATOR = TaskDefinition(
    task_id="market_dominator",
    name="Market Dominator",
    difficulty="hard",
    description=(
        "Maximize long-term revenue under adversarial conditions. "
        "Manage 7 channels, 4 segments with an active competitor and "
        "market regime shifts. Avoid compliance traps. 36 weeks, high noise."
    ),
    total_weeks=36,
    total_budget=300000.0,
    channels=[
        ChannelConfig(
            name="paid_search", base_ctr=0.010, base_cvr=0.018,
            saturation_alpha=1.8, cost_per_impression=22.0, min_spend_for_signal=250.0,
            segment_affinity={
                "startup_founders": 1.3, "smb_owners": 1.0, "enterprise": 0.7, "developer": 1.1,
            },
        ),
        ChannelConfig(
            name="paid_social", base_ctr=0.007, base_cvr=0.010,
            saturation_alpha=2.2, cost_per_impression=16.0, min_spend_for_signal=200.0,
            segment_affinity={
                "startup_founders": 1.3, "smb_owners": 0.7, "enterprise": 0.4, "developer": 1.0,
            },
        ),
        ChannelConfig(
            name="organic_content", base_ctr=0.005, base_cvr=0.025,
            saturation_alpha=0.8, cost_per_impression=10.0, min_spend_for_signal=350.0,
            segment_affinity={
                "startup_founders": 1.1, "smb_owners": 1.1, "enterprise": 1.2, "developer": 1.5,
            },
        ),
        ChannelConfig(
            name="email_lifecycle", base_ctr=0.020, base_cvr=0.025,
            saturation_alpha=1.0, cost_per_impression=6.0, min_spend_for_signal=100.0,
            segment_affinity={
                "startup_founders": 0.9, "smb_owners": 1.4, "enterprise": 1.0, "developer": 0.8,
            },
        ),
        ChannelConfig(
            name="outbound_sales", base_ctr=0.003, base_cvr=0.040,
            saturation_alpha=1.5, cost_per_impression=55.0, min_spend_for_signal=600.0,
            segment_affinity={
                "startup_founders": 0.4, "smb_owners": 0.8, "enterprise": 1.9, "developer": 0.3,
            },
        ),
        ChannelConfig(
            name="partnerships", base_ctr=0.004, base_cvr=0.035,
            saturation_alpha=1.0, cost_per_impression=35.0, min_spend_for_signal=400.0,
            segment_affinity={
                "startup_founders": 1.0, "smb_owners": 1.2, "enterprise": 1.5, "developer": 1.1,
            },
        ),
        ChannelConfig(
            name="influencer_marketing", base_ctr=0.009, base_cvr=0.015,
            saturation_alpha=2.5, cost_per_impression=25.0, min_spend_for_signal=300.0,
            segment_affinity={
                "startup_founders": 1.5, "smb_owners": 0.6, "enterprise": 0.3, "developer": 1.4,
            },
        ),
    ],
    segments=[
        SegmentConfig(
            name="startup_founders", size=0.3, price_sensitivity=0.7,
            message_preference={
                "cost_savings": 0.1, "performance": 0.3, "reliability": 0.1,
                "innovation": 0.3, "ease_of_use": 0.15, "security": 0.05,
            },
            base_churn=0.08,
        ),
        SegmentConfig(
            name="smb_owners", size=0.25, price_sensitivity=0.5,
            message_preference={
                "cost_savings": 0.25, "performance": 0.15, "reliability": 0.25,
                "innovation": 0.05, "ease_of_use": 0.2, "security": 0.1,
            },
            base_churn=0.05,
        ),
        SegmentConfig(
            name="enterprise", size=0.2, price_sensitivity=0.15,
            message_preference={
                "cost_savings": 0.05, "performance": 0.15, "reliability": 0.3,
                "innovation": 0.1, "ease_of_use": 0.1, "security": 0.3,
            },
            base_churn=0.02,
        ),
        SegmentConfig(
            name="developer", size=0.25, price_sensitivity=0.6,
            message_preference={
                "cost_savings": 0.05, "performance": 0.35, "reliability": 0.1,
                "innovation": 0.25, "ease_of_use": 0.2, "security": 0.05,
            },
            base_churn=0.1,
        ),
    ],
    product=ProductConfig(base_price=199.0, differentiation=0.6, complexity=0.6),
    noise_level=0.25,
    enable_competitor=True,
    enable_regime_shifts=True,
    revenue_target=400000.0,
    available_experiments=EXPERIMENT_TYPES,
    available_pricing_actions=PRICING_ACTIONS,
    grader=grade_market_dominator,
)


# ── Registry ───────────────────────────────────────────────────────────────

TASKS: Dict[str, TaskDefinition] = {
    "channel_optimizer": TASK_CHANNEL_OPTIMIZER,
    "growth_strategist": TASK_GROWTH_STRATEGIST,
    "market_dominator": TASK_MARKET_DOMINATOR,
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def create_simulator(task_id: str, seed: int | None = None) -> MarketSimulator:
    """Create a MarketSimulator configured for the given task."""
    t = get_task(task_id)
    return MarketSimulator(
        channels=t.channels,
        segments=t.segments,
        product=t.product,
        total_weeks=t.total_weeks,
        total_budget=t.total_budget,
        noise_level=t.noise_level,
        enable_competitor=t.enable_competitor,
        enable_regime_shifts=t.enable_regime_shifts,
        seed=seed,
    )
