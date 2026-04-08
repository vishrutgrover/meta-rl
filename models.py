"""Pydantic models for the GTM Strategy Optimizer environment."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator
import json

from openenv.core.env_server import Action, Observation, State


# ── Sub-models for structured metrics ──────────────────────────────────────


class ChannelMetrics(BaseModel):
    """Performance metrics for a single marketing channel."""

    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend: float = 0.0
    ctr: float = 0.0
    cvr: float = 0.0
    roi: float = 0.0


class FunnelMetrics(BaseModel):
    """Funnel-level metrics across all channels."""

    visitors: int = 0
    signups: int = 0
    activations: int = 0
    retained_users: int = 0
    signup_rate: float = 0.0
    activation_rate: float = 0.0
    retention_rate: float = 0.0


class SegmentMetrics(BaseModel):
    """Performance metrics for a customer segment."""

    conversion_rate: float = 0.0
    engagement_score: float = 0.0
    churn_rate: float = 0.0
    revenue: float = 0.0


class ExperimentResult(BaseModel):
    """Result of a completed experiment."""

    experiment_type: str
    uplift_estimate: float
    confidence: float
    recommendation: str


# ── Action ─────────────────────────────────────────────────────────────────


class GTMAction(Action):
    """Agent's weekly GTM decisions.

    All allocation dicts map names to fractions (0.0-1.0).
    Fractions in budget_allocation should sum to <= 1.0.
    Fractions in segment_targeting and messaging should each sum to ~1.0.
    """

    budget_allocation: Dict[str, float] = Field(
        default_factory=dict,
        description="Channel name -> fraction of weekly budget to allocate",
    )
    segment_targeting: Dict[str, float] = Field(
        default_factory=dict,
        description="Segment name -> targeting weight (should sum to ~1.0)",
    )
    messaging: Dict[str, float] = Field(
        default_factory=dict,
        description="Messaging dimension -> emphasis weight. Dimensions: cost_savings, performance, reliability, innovation, ease_of_use, security",
    )
    experiment: Optional[str] = Field(
        default=None,
        description="Experiment to launch: 'ab_test_landing', 'ab_test_pricing', 'ab_test_creative', 'run_survey', 'competitor_analysis', or null",
    )


    @model_validator(mode="before")
    @classmethod
    def parse_stringified_json(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field in ["budget_allocation", "segment_targeting", "messaging"]:
                if field in data and isinstance(data[field], str):
                    try:
                        data[field] = json.loads(data[field])
                    except json.JSONDecodeError:
                        pass
        return data

    pricing_action: Optional[str] = Field(
        default=None,
        description="Pricing change: 'discount_10', 'discount_20', 'raise_5', 'add_free_trial', or null",
    )


# ── Observation ────────────────────────────────────────────────────────────


class GTMObservation(Observation):
    """What the agent observes after each week of GTM activity."""

    week: int = 0
    total_weeks: int = 12
    budget_remaining: float = 0.0
    weekly_budget: float = 0.0

    channel_metrics: Dict[str, ChannelMetrics] = Field(default_factory=dict)
    funnel: FunnelMetrics = Field(default_factory=FunnelMetrics)
    segment_performance: Dict[str, SegmentMetrics] = Field(default_factory=dict)

    experiment_result: Optional[ExperimentResult] = None

    brand_score: float = 50.0

    total_revenue: float = 0.0
    total_conversions: int = 0
    average_cac: float = 0.0

    available_channels: List[str] = Field(default_factory=list)
    available_segments: List[str] = Field(default_factory=list)
    available_experiments: List[str] = Field(default_factory=list)
    available_pricing_actions: List[str] = Field(default_factory=list)
    messaging_dimensions: List[str] = Field(default_factory=list)

    message: str = ""


# ── State ──────────────────────────────────────────────────────────────────


class GTMState(State):
    """Internal environment state (includes hidden ground truth)."""

    task_id: str = "channel_optimizer"
    difficulty: str = "easy"
    true_brand_strength: float = 50.0
    true_market_demand: float = 1.0
    total_revenue: float = 0.0
    total_spend: float = 0.0
    total_conversions: int = 0
    compliance_violations: int = 0
    experiments_run: int = 0
    useful_experiments: int = 0
