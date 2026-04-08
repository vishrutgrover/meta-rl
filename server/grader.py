"""Grader functions for the GTM Strategy Optimizer tasks.

Each grader takes a completed SimState and returns a normalized score in [0, 1].
"""

from __future__ import annotations

from .simulation import SimState


def grade_channel_optimizer(s: SimState) -> float:
    """Easy task: pure revenue vs target with partial credit."""
    revenue_target = 120000.0
    score = min(1.0, s.total_revenue / revenue_target)
    return round(max(0.0, score), 4)


def grade_growth_strategist(s: SimState) -> float:
    """Medium task: weighted score across revenue, efficiency, brand, experiments."""
    revenue_target = 375000.0
    rev_score = min(1.0, s.total_revenue / revenue_target)

    efficiency = s.total_revenue / max(s.total_spend, 1.0)
    eff_score = min(1.0, efficiency / 3.0)  # 3x ROI = perfect

    brand_score = s.brand_strength / 100.0

    exp_score = 0.0
    if s.experiments_run > 0:
        exp_score = min(1.0, s.useful_experiments / max(s.experiments_run * 0.5, 1.0))

    score = 0.40 * rev_score + 0.30 * eff_score + 0.20 * brand_score + 0.10 * exp_score
    return round(max(0.0, min(1.0, score)), 4)


def grade_market_dominator(s: SimState) -> float:
    """Hard task: revenue, ROI, brand trajectory, adaptability, compliance."""
    revenue_target = 400000.0
    rev_score = min(1.0, s.total_revenue / revenue_target)

    # risk-adjusted ROI
    roi = s.total_revenue / max(s.total_spend, 1.0)
    roi_score = min(1.0, roi / 4.0)

    # brand trajectory (improving over time)
    brand_scores = s.weekly_brand_scores
    if len(brand_scores) >= 4:
        first_quarter = sum(brand_scores[: len(brand_scores) // 4]) / max(len(brand_scores) // 4, 1)
        last_quarter = sum(brand_scores[-len(brand_scores) // 4 :]) / max(len(brand_scores) // 4, 1)
        trajectory = min(1.0, max(0.0, (last_quarter - first_quarter + 10) / 20.0))
    else:
        trajectory = 0.5

    # adaptability: performance recovery after regime shifts
    revenues = s.weekly_revenues
    if len(revenues) >= 18:
        pre_shift = sum(revenues[8:12]) / 4 if len(revenues) > 12 else 1.0
        post_shift = sum(revenues[13:17]) / 4 if len(revenues) > 17 else 0.0
        adapt_score = min(1.0, post_shift / max(pre_shift, 1.0))
    else:
        adapt_score = 0.5

    # compliance
    compliance_score = max(0.0, 1.0 - s.compliance_violations * 0.03)

    score = (
        0.35 * rev_score
        + 0.25 * roi_score
        + 0.20 * trajectory
        + 0.10 * adapt_score
        + 0.10 * compliance_score
    )
    return round(max(0.0, min(1.0, score)), 4)
