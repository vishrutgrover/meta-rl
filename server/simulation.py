"""Market dynamics simulation engine for the GTM environment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Channel configuration ──────────────────────────────────────────────────


@dataclass
class ChannelConfig:
    """Static properties of a marketing channel."""

    name: str
    base_ctr: float  # base click-through rate
    base_cvr: float  # base conversion rate
    saturation_alpha: float  # diminishing returns steepness
    cost_per_impression: float  # cost per 1k impressions
    min_spend_for_signal: float  # minimum spend to get any data
    # affinity per segment (segment_name -> multiplier 0-2)
    segment_affinity: Dict[str, float] = field(default_factory=dict)


@dataclass
class SegmentConfig:
    """Static properties of a customer segment."""

    name: str
    size: float  # relative market size
    price_sensitivity: float  # 0-1, higher = more price sensitive
    # preferred messaging dimensions (dim -> ideal weight)
    message_preference: Dict[str, float] = field(default_factory=dict)
    base_churn: float = 0.05


@dataclass
class ProductConfig:
    """Product being marketed."""

    base_price: float = 99.0
    differentiation: float = 0.7  # 0-1
    complexity: float = 0.4  # 0-1


# ── Simulation state ───────────────────────────────────────────────────────


@dataclass
class SimState:
    """Mutable simulation state tracking all dynamics."""

    week: int = 0
    total_weeks: int = 12
    budget_remaining: float = 50000.0
    weekly_budget: float = 5000.0

    # true latent variables
    brand_strength: float = 50.0  # 0-100
    market_demand: float = 1.0  # multiplier
    competitor_aggression: float = 0.0  # 0-1

    # cumulative metrics
    total_revenue: float = 0.0
    total_spend: float = 0.0
    total_conversions: int = 0
    total_impressions: int = 0

    # channel cumulative spend (for diminishing returns)
    channel_cumulative_spend: Dict[str, float] = field(default_factory=dict)

    # messaging history (for consistency tracking)
    messaging_history: List[Dict[str, float]] = field(default_factory=list)

    # experiment state
    pending_experiment: Optional[Tuple[str, int]] = None  # (type, completion_week)
    experiments_run: int = 0
    useful_experiments: int = 0

    # pricing state
    current_discount: float = 0.0
    has_free_trial: bool = False

    # compliance
    compliance_violations: int = 0

    # per-week tracking for grading
    weekly_revenues: List[float] = field(default_factory=list)
    weekly_brand_scores: List[float] = field(default_factory=list)


MESSAGING_DIMS = [
    "cost_savings",
    "performance",
    "reliability",
    "innovation",
    "ease_of_use",
    "security",
]

EXPERIMENT_TYPES = [
    "ab_test_landing",
    "ab_test_pricing",
    "ab_test_creative",
    "run_survey",
    "competitor_analysis",
]

PRICING_ACTIONS = [
    "discount_10",
    "discount_20",
    "raise_5",
    "add_free_trial",
]


# ── Market Simulator ───────────────────────────────────────────────────────


class MarketSimulator:
    """Simulates market response to GTM actions for one episode."""

    def __init__(
        self,
        channels: List[ChannelConfig],
        segments: List[SegmentConfig],
        product: ProductConfig,
        total_weeks: int = 12,
        total_budget: float = 50000.0,
        noise_level: float = 0.1,
        enable_competitor: bool = False,
        enable_regime_shifts: bool = False,
        seed: Optional[int] = None,
    ):
        self.channels = {c.name: c for c in channels}
        self.segments = {s.name: s for s in segments}
        self.product = product
        self.noise_level = noise_level
        self.enable_competitor = enable_competitor
        self.enable_regime_shifts = enable_regime_shifts
        self.rng = random.Random(seed)

        weekly_budget = total_budget / total_weeks
        self.state = SimState(
            total_weeks=total_weeks,
            budget_remaining=total_budget,
            weekly_budget=weekly_budget,
            channel_cumulative_spend={c.name: 0.0 for c in channels},
        )

    def reset(self, seed: Optional[int] = None) -> SimState:
        """Reset to initial state."""
        if seed is not None:
            self.rng = random.Random(seed)
        total_budget = self.state.weekly_budget * self.state.total_weeks
        self.state = SimState(
            total_weeks=self.state.total_weeks,
            budget_remaining=total_budget,
            weekly_budget=total_budget / self.state.total_weeks,
            channel_cumulative_spend={c: 0.0 for c in self.channels},
        )
        return self.state

    def step(
        self,
        budget_allocation: Dict[str, float],
        segment_targeting: Dict[str, float],
        messaging: Dict[str, float],
        experiment: Optional[str] = None,
        pricing_action: Optional[str] = None,
    ) -> Dict:
        """Advance one week and return metrics.

        Returns dict with keys:
            channel_metrics, funnel, segment_performance,
            experiment_result, brand_score_observed, weekly_revenue
        """
        s = self.state
        s.week += 1

        # ── Apply pricing action ───────────────────────────────────
        self._apply_pricing(pricing_action)

        # ── Budget spend ───────────────────────────────────────────
        total_alloc = sum(budget_allocation.values())
        if total_alloc > 1.0:
            # normalize
            factor = 1.0 / total_alloc
            budget_allocation = {k: v * factor for k, v in budget_allocation.items()}

        weekly_spend = min(s.weekly_budget, s.budget_remaining)
        channel_spends = {}
        for ch_name, frac in budget_allocation.items():
            if ch_name in self.channels:
                channel_spends[ch_name] = frac * weekly_spend

        actual_total_spend = sum(channel_spends.values())
        s.budget_remaining -= actual_total_spend
        s.total_spend += actual_total_spend

        # ── Normalize targeting & messaging ────────────────────────
        segment_targeting = self._normalize_weights(
            segment_targeting, list(self.segments.keys())
        )
        messaging = self._normalize_weights(messaging, MESSAGING_DIMS)
        s.messaging_history.append(messaging.copy())

        # ── Compute channel performance ────────────────────────────
        channel_metrics = {}
        total_visitors = 0
        total_signups = 0
        total_activations = 0
        segment_conversions: Dict[str, float] = {seg: 0.0 for seg in self.segments}
        segment_revenue: Dict[str, float] = {seg: 0.0 for seg in self.segments}
        segment_engagement: Dict[str, float] = {seg: 0.0 for seg in self.segments}
        weekly_revenue = 0.0

        for ch_name, ch_cfg in self.channels.items():
            spend = channel_spends.get(ch_name, 0.0)
            s.channel_cumulative_spend[ch_name] += spend

            if spend < ch_cfg.min_spend_for_signal:
                channel_metrics[ch_name] = {
                    "impressions": 0, "clicks": 0, "conversions": 0,
                    "spend": spend, "ctr": 0.0, "cvr": 0.0, "roi": 0.0,
                }
                continue

            # impressions from spend (cost_per_impression is CPM)
            # Apply diminishing returns: more spend -> higher effective CPM
            cumulative = s.channel_cumulative_spend[ch_name]
            diminishing = math.exp(-ch_cfg.saturation_alpha * cumulative / 100000)
            # Weekly spend also has diminishing returns (audience saturation)
            weekly_diminishing = 1.0 / (1.0 + spend / 2000.0)
            effective_impressions = spend / ch_cfg.cost_per_impression * 1000 * weekly_diminishing * diminishing
            impressions = int(max(0, effective_impressions))

            # compute per-segment clicks and conversions
            ch_clicks = 0
            ch_conversions = 0
            ch_revenue = 0.0
            for seg_name, seg_cfg in self.segments.items():
                seg_weight = segment_targeting.get(seg_name, 0.0)
                if seg_weight < 0.01:
                    continue

                seg_impressions = int(impressions * seg_weight)
                affinity = ch_cfg.segment_affinity.get(seg_name, 1.0)
                msg_alignment = self._message_alignment(messaging, seg_cfg)
                brand_mult = s.brand_strength / 100.0

                eff_ctr = (
                    ch_cfg.base_ctr
                    * affinity
                    * brand_mult
                    * s.market_demand
                    * (1.0 + self._noise(0.1))
                )
                eff_cvr = (
                    ch_cfg.base_cvr
                    * msg_alignment
                    * self.product.differentiation
                    * (1.0 + self._noise(0.1))
                )

                clicks = int(seg_impressions * min(eff_ctr, 0.5))
                convs = int(clicks * min(eff_cvr, 0.8))

                # revenue per conversion
                price = self.product.base_price * (1.0 - s.current_discount)
                price_mult = 1.0 - seg_cfg.price_sensitivity * s.current_discount * 0.5
                rev = convs * price * max(price_mult, 0.3)

                ch_clicks += clicks
                ch_conversions += convs
                ch_revenue += rev
                segment_conversions[seg_name] += convs
                segment_revenue[seg_name] += rev
                segment_engagement[seg_name] += clicks * 0.01

            ctr = ch_clicks / max(impressions, 1)
            cvr = ch_conversions / max(ch_clicks, 1)
            roi = (ch_revenue - spend) / max(spend, 1.0)

            channel_metrics[ch_name] = {
                "impressions": impressions,
                "clicks": ch_clicks,
                "conversions": ch_conversions,
                "spend": round(spend, 2),
                "ctr": round(ctr, 4),
                "cvr": round(cvr, 4),
                "roi": round(roi, 4),
            }

            total_visitors += ch_clicks
            total_signups += ch_conversions
            weekly_revenue += ch_revenue
            s.total_conversions += ch_conversions

        # ── Funnel metrics ─────────────────────────────────────────
        total_activations = int(total_signups * 0.6 * (1 + self._noise(0.05)))
        retained = int(total_activations * 0.7 * (1 + self._noise(0.05)))
        funnel = {
            "visitors": total_visitors,
            "signups": total_signups,
            "activations": total_activations,
            "retained_users": retained,
            "signup_rate": round(total_signups / max(total_visitors, 1), 4),
            "activation_rate": round(total_activations / max(total_signups, 1), 4),
            "retention_rate": round(retained / max(total_activations, 1), 4),
        }

        # ── Segment performance ────────────────────────────────────
        segment_performance = {}
        for seg_name in self.segments:
            total_seg_imp = max(
                sum(
                    channel_metrics.get(ch, {}).get("impressions", 0)
                    * segment_targeting.get(seg_name, 0.0)
                    for ch in self.channels
                ),
                1,
            )
            conv_rate = segment_conversions[seg_name] / total_seg_imp
            segment_performance[seg_name] = {
                "conversion_rate": round(conv_rate, 6),
                "engagement_score": round(min(segment_engagement[seg_name], 100.0), 2),
                "churn_rate": round(self.segments[seg_name].base_churn * (1 + self._noise(0.1)), 4),
                "revenue": round(segment_revenue[seg_name], 2),
            }

        # ── Brand evolution ────────────────────────────────────────
        consistency = self._messaging_consistency()
        organic_boost = sum(
            channel_spends.get(ch, 0.0)
            for ch in self.channels
            if "organic" in ch or "content" in ch
        ) / max(weekly_spend, 1.0)
        s.brand_strength = min(100.0, max(0.0,
            s.brand_strength
            + 0.5 * consistency
            + 0.3 * organic_boost
            - 0.2 * (1.0 - consistency)
            + self._noise(0.3)
        ))
        brand_observed = s.brand_strength + self._noise(5.0) * self.noise_level * 10
        brand_observed = max(0.0, min(100.0, brand_observed))

        # ── Competitor response (hard mode) ────────────────────────
        if self.enable_competitor and s.week > 4:
            if weekly_revenue > s.total_revenue / max(s.week - 1, 1) * 1.2:
                s.competitor_aggression = min(1.0, s.competitor_aggression + 0.1)
                s.market_demand *= max(0.9, 1.0 - s.competitor_aggression * 0.05)

        # ── Market regime shifts (hard mode) ───────────────────────
        if self.enable_regime_shifts:
            if s.week in (12, 24):
                shift = self.rng.uniform(-0.3, 0.3)
                s.market_demand = max(0.5, min(1.5, s.market_demand + shift))

        # ── Experiment processing ──────────────────────────────────
        experiment_result = None
        if experiment and experiment in EXPERIMENT_TYPES:
            exp_cost = weekly_spend * 0.1
            s.budget_remaining -= exp_cost
            s.total_spend += exp_cost
            s.experiments_run += 1
            s.pending_experiment = (experiment, s.week + 2)

        if s.pending_experiment and s.week >= s.pending_experiment[1]:
            exp_type = s.pending_experiment[0]
            uplift = self.rng.uniform(-0.05, 0.15)
            confidence = self.rng.uniform(0.6, 0.95)
            useful = uplift > 0.02 and confidence > 0.75
            if useful:
                s.useful_experiments += 1
            experiment_result = {
                "experiment_type": exp_type,
                "uplift_estimate": round(uplift, 4),
                "confidence": round(confidence, 4),
                "recommendation": (
                    f"Adopt variant — {uplift:.1%} uplift at {confidence:.0%} confidence"
                    if useful
                    else f"No significant uplift detected ({uplift:.1%} at {confidence:.0%} confidence)"
                ),
            }
            s.pending_experiment = None

        # ── Update cumulative ──────────────────────────────────────
        s.total_revenue += weekly_revenue
        s.weekly_revenues.append(weekly_revenue)
        s.weekly_brand_scores.append(s.brand_strength)

        return {
            "channel_metrics": channel_metrics,
            "funnel": funnel,
            "segment_performance": segment_performance,
            "experiment_result": experiment_result,
            "brand_score_observed": round(brand_observed, 1),
            "weekly_revenue": round(weekly_revenue, 2),
        }

    # ── Helpers ────────────────────────────────────────────────────

    def _noise(self, scale: float) -> float:
        return self.rng.gauss(0, scale * self.noise_level)

    def _normalize_weights(
        self, weights: Dict[str, float], valid_keys: List[str]
    ) -> Dict[str, float]:
        filtered = {k: max(v, 0.0) for k, v in weights.items() if k in valid_keys}
        total = sum(filtered.values())
        if total < 0.01:
            # equal distribution
            n = len(valid_keys)
            return {k: 1.0 / n for k in valid_keys}
        return {k: v / total for k, v in filtered.items()}

    def _message_alignment(
        self, messaging: Dict[str, float], segment: SegmentConfig
    ) -> float:
        """Cosine-like alignment between messaging and segment preference."""
        dot = 0.0
        mag_m = 0.0
        mag_s = 0.0
        for dim in MESSAGING_DIMS:
            m = messaging.get(dim, 0.0)
            s = segment.message_preference.get(dim, 1.0 / len(MESSAGING_DIMS))
            dot += m * s
            mag_m += m * m
            mag_s += s * s
        if mag_m < 1e-9 or mag_s < 1e-9:
            return 0.5
        return dot / (math.sqrt(mag_m) * math.sqrt(mag_s))

    def _messaging_consistency(self) -> float:
        """How consistent messaging has been over recent weeks."""
        history = self.state.messaging_history
        if len(history) < 2:
            return 1.0
        recent = history[-min(4, len(history)):]
        # compute variance across dimensions
        total_var = 0.0
        for dim in MESSAGING_DIMS:
            vals = [m.get(dim, 0.0) for m in recent]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            total_var += var
        # low variance = high consistency
        return max(0.0, 1.0 - total_var * 10)

    def _apply_pricing(self, pricing_action: Optional[str]) -> None:
        s = self.state
        if pricing_action == "discount_10":
            s.current_discount = 0.10
        elif pricing_action == "discount_20":
            s.current_discount = 0.20
        elif pricing_action == "raise_5":
            s.current_discount = max(0.0, s.current_discount - 0.05)
        elif pricing_action == "add_free_trial":
            s.has_free_trial = True
            # free trial boosts conversions via brand
            s.brand_strength = min(100.0, s.brand_strength + 1.0)

    @property
    def is_done(self) -> bool:
        return (
            self.state.week >= self.state.total_weeks
            or self.state.budget_remaining <= 0
        )
