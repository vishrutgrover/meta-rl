"""Custom Gradio dashboard for the GTM Strategy Optimizer.

Mounted directly via gr.mount_gradio_app in server/app.py — bypasses OpenEnv's
default Playground UI so users only see this polished interface.

Three tabs:
  • Compare Strategies — RL vs heuristic vs random head-to-head
  • Interactive Playground — sliders for budget/segment/messaging, no JSON
  • Episode Replay — week-by-week walkthrough of a trained RL run
"""

from __future__ import annotations

import os
import random as _random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

# Make repo root importable when this module is loaded by uvicorn from /app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GTMAction
from rl.infer import run_inference
from server.environment import GTMEnvironment
from server.simulation import MESSAGING_DIMS
from server.tasks import TASKS, get_task

# ── Theme colors ───────────────────────────────────────────────────────────

RL_COLOR = "#7c3aed"        # purple
HEUR_COLOR = "#3b82f6"      # blue
RAND_COLOR = "#94a3b8"      # slate
ACCENT_GRADIENT = "linear-gradient(135deg,#7c3aed 0%,#3b82f6 100%)"

CHANNEL_PALETTE = [
    "#7c3aed", "#3b82f6", "#10b981", "#f59e0b",
    "#ef4444", "#06b6d4", "#ec4899",
]

# ── Custom CSS injected at mount time ──────────────────────────────────────

DASHBOARD_CSS = """
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
}
.hero-banner {
    background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
    color: white;
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 18px;
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.18);
}
.hero-banner h1 {
    margin: 0 0 6px 0;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.hero-banner p {
    margin: 0;
    opacity: 0.92;
    font-size: 15px;
}
.score-card {
    border-radius: 12px;
    padding: 18px 22px;
    background: white;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
    border: 1px solid #e2e8f0;
}
.score-card .label {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
}
.score-card .score {
    font-size: 42px;
    font-weight: 800;
    color: #0f172a;
    margin: 4px 0 8px 0;
    line-height: 1;
    letter-spacing: -0.02em;
}
.score-card .stats {
    font-size: 13px;
    color: #475569;
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
}
.score-card.winner {
    background: linear-gradient(135deg, #faf5ff 0%, #eff6ff 100%);
    border: 2px solid #7c3aed;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.18);
}
.metric-tile {
    background: #f8fafc;
    border-left: 4px solid #7c3aed;
    padding: 14px 18px;
    border-radius: 8px;
    margin-bottom: 8px;
}
.metric-tile .k {
    font-size: 12px;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-tile .v {
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
}
"""


# ── Strategy runners ───────────────────────────────────────────────────────


def _equal_action(task) -> GTMAction:
    return GTMAction(
        budget_allocation={c.name: 1.0 / len(task.channels) for c in task.channels},
        segment_targeting={s.name: 1.0 / len(task.segments) for s in task.segments},
        messaging={d: 1.0 / len(MESSAGING_DIMS) for d in MESSAGING_DIMS},
    )


def _random_action(task, rng: _random.Random) -> GTMAction:
    def _simplex(n: int) -> List[float]:
        xs = [rng.random() + 1e-6 for _ in range(n)]
        s = sum(xs)
        return [x / s for x in xs]

    bs = _simplex(len(task.channels))
    ss = _simplex(len(task.segments))
    ms = _simplex(len(MESSAGING_DIMS))
    return GTMAction(
        budget_allocation={c.name: bs[i] for i, c in enumerate(task.channels)},
        segment_targeting={s.name: ss[i] for i, s in enumerate(task.segments)},
        messaging={d: ms[i] for i, d in enumerate(MESSAGING_DIMS)},
    )


def _run_with_action_fn(task_id: str, action_fn: Callable, seed: int) -> Dict[str, Any]:
    env = GTMEnvironment()
    task = get_task(task_id)
    obs = env.reset(task_id=task_id, seed=seed)
    weeks: List[Dict[str, Any]] = []
    while not obs.done:
        action = action_fn(task)
        obs = env.step(action)
        weeks.append({
            "week": obs.week,
            "total_revenue": float(obs.total_revenue),
            "brand_score": float(obs.brand_score),
            "budget_allocation": dict(action.budget_allocation),
            "reward": float(obs.reward) if obs.reward is not None else 0.0,
        })
    grader_score = env.get_grader_score(env.state.episode_id)
    return {
        "weeks": weeks,
        "grader_score": grader_score,
        "total_revenue": float(obs.total_revenue),
        "total_conversions": int(obs.total_conversions),
        "brand_score": float(obs.brand_score),
    }


def run_heuristic(task_id: str, seed: int) -> Dict[str, Any]:
    return _run_with_action_fn(task_id, _equal_action, seed)


def run_random(task_id: str, seed: int) -> Dict[str, Any]:
    rng = _random.Random(seed)
    return _run_with_action_fn(task_id, lambda t: _random_action(t, rng), seed)


def run_trained_rl(task_id: str, seed: int) -> Dict[str, Any]:
    result = run_inference(task_id, seed=seed)
    weeks = [
        {
            "week": a["week"],
            "total_revenue": float(a["total_revenue"]),
            "brand_score": float(a["brand_score"]),
            "budget_allocation": a["budget_allocation"],
            "reward": float(a["weekly_reward"]) if a["weekly_reward"] is not None else 0.0,
        }
        for a in result["actions"]
    ]
    return {
        "weeks": weeks,
        "grader_score": result["grader_score"],
        "total_revenue": float(result["total_revenue"]),
        "total_conversions": int(result["total_conversions"]),
        "brand_score": float(result["brand_score"]),
        "checkpoint_loaded": result["checkpoint_loaded"],
    }


# ── Plot builders ──────────────────────────────────────────────────────────


def _layout(title: str, height: int = 400) -> Dict[str, Any]:
    return dict(
        title=dict(text=title, x=0.02, y=0.96, font=dict(size=15, color="#0f172a")),
        template="plotly_white",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif", color="#334155"),
    )


def build_revenue_plot(rl: Dict, heur: Dict, rand: Dict) -> go.Figure:
    fig = go.Figure()
    for label, data, color, dash, fill in [
        ("Trained RL", rl, RL_COLOR, "solid", "tozeroy"),
        ("Heuristic", heur, HEUR_COLOR, "dash", None),
        ("Random", rand, RAND_COLOR, "dot", None),
    ]:
        weeks = data["weeks"]
        fig.add_trace(go.Scatter(
            x=[w["week"] for w in weeks],
            y=[w["total_revenue"] for w in weeks],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=3.5 if label == "Trained RL" else 2.5, dash=dash),
            marker=dict(size=8 if label == "Trained RL" else 6),
            fill=fill,
            fillcolor=f"rgba(124,58,237,0.08)" if fill else None,
            hovertemplate=f"<b>{label}</b><br>Week %{{x}}<br>$%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Cumulative Revenue ($)",
        **_layout("Cumulative Revenue Over Time", height=440),
    )
    return fig


def build_budget_plot(rl: Dict) -> go.Figure:
    weeks_data = rl["weeks"]
    if not weeks_data:
        return go.Figure()
    weeks = [w["week"] for w in weeks_data]
    channels = list(weeks_data[0]["budget_allocation"].keys())
    fig = go.Figure()
    for i, ch in enumerate(channels):
        color = CHANNEL_PALETTE[i % len(CHANNEL_PALETTE)]
        fig.add_trace(go.Scatter(
            x=weeks,
            y=[w["budget_allocation"].get(ch, 0.0) for w in weeks_data],
            mode="lines",
            name=ch,
            stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=color,
            hovertemplate=f"<b>{ch}</b><br>Week %{{x}}<br>%{{y:.0%}}<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Budget Share",
        yaxis_tickformat=".0%",
        **_layout("RL Budget Allocation per Week", height=380),
    )
    return fig


def build_brand_plot(rl: Dict, heur: Dict, rand: Dict) -> go.Figure:
    fig = go.Figure()
    for label, data, color, dash in [
        ("Trained RL", rl, RL_COLOR, "solid"),
        ("Heuristic", heur, HEUR_COLOR, "dash"),
        ("Random", rand, RAND_COLOR, "dot"),
    ]:
        weeks = data["weeks"]
        fig.add_trace(go.Scatter(
            x=[w["week"] for w in weeks],
            y=[w["brand_score"] for w in weeks],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=3 if label == "Trained RL" else 2, dash=dash),
            marker=dict(size=7),
            hovertemplate=f"<b>{label}</b><br>Week %{{x}}<br>Brand %{{y:.0f}}/100<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Brand Health (0-100)",
        yaxis=dict(range=[0, 100]),
        **_layout("Brand Health Over Time", height=380),
    )
    return fig


def build_action_table(rl: Dict) -> pd.DataFrame:
    rows = []
    for w in rl["weeks"]:
        row: Dict[str, Any] = {"Week": w["week"]}
        for ch, frac in w["budget_allocation"].items():
            row[ch] = round(frac, 3)
        row["Brand"] = round(w["brand_score"], 1)
        row["Reward"] = round(w["reward"], 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ── HTML score cards ───────────────────────────────────────────────────────


def _score_card(label: str, emoji: str, color_hex: str, result: Dict, is_winner: bool = False) -> str:
    score = result.get("grader_score")
    score_str = f"{score:.3f}" if score is not None else "—"
    crown = " 👑" if is_winner else ""
    cls = "score-card winner" if is_winner else "score-card"
    return (
        f"<div class='{cls}' style='border-top:4px solid {color_hex}'>"
        f"<div class='label'>{emoji} {label}{crown}</div>"
        f"<div class='score'>{score_str}</div>"
        f"<div class='stats'>"
        f"<span>💵 ${result['total_revenue']:,.0f}</span>"
        f"<span>🎯 {result['total_conversions']} convs</span>"
        f"<span>⭐ {result['brand_score']:.0f}/100</span>"
        f"</div></div>"
    )


def _empty_result() -> Dict[str, Any]:
    return {
        "grader_score": None,
        "total_revenue": 0.0,
        "total_conversions": 0,
        "brand_score": 0.0,
    }


# ── Tab 1: Compare Strategies callback ─────────────────────────────────────


def run_comparison(task_id: str, seed_value: float):
    seed = int(seed_value) if seed_value is not None else 42
    rl = run_trained_rl(task_id, seed)
    heur = run_heuristic(task_id, seed)
    rand = run_random(task_id, seed)

    scores = {
        "rl": rl.get("grader_score") or -1,
        "heur": heur.get("grader_score") or -1,
        "rand": rand.get("grader_score") or -1,
    }
    winner = max(scores, key=scores.get)

    revenue_fig = build_revenue_plot(rl, heur, rand)
    budget_fig = build_budget_plot(rl)
    brand_fig = build_brand_plot(rl, heur, rand)
    action_df = build_action_table(rl)

    rl_card = _score_card("Trained RL", "🤖", RL_COLOR, rl, is_winner=(winner == "rl"))
    heur_card = _score_card("Heuristic", "📊", HEUR_COLOR, heur, is_winner=(winner == "heur"))
    rand_card = _score_card("Random", "🎲", RAND_COLOR, rand, is_winner=(winner == "rand"))

    ckpt_warning = ""
    if not rl.get("checkpoint_loaded"):
        ckpt_warning = (
            "<div style='padding:12px 16px;background:#fef3c7;border-left:4px solid #f59e0b;"
            "border-radius:8px;color:#78350f;font-size:13px;margin-bottom:8px'>"
            "⚠️ <b>No trained checkpoint for this task</b> — RL is using a random-init policy. "
            f"Run <code>python -m rl.train --task {task_id}</code> locally and commit "
            f"<code>checkpoints/{task_id}.pt</code> for real results."
            "</div>"
        )

    return revenue_fig, budget_fig, brand_fig, action_df, rl_card, heur_card, rand_card, ckpt_warning


# ── Tab 2: Interactive Playground (sliders, no JSON) ───────────────────────


# Simple in-memory state. Each tab session keeps its own dict via gr.State.


def playground_init(task_id: str):
    """Reset the env for the chosen task and return initial sliders + metrics."""
    task = get_task(task_id)
    env = GTMEnvironment()
    obs = env.reset(task_id=task_id, seed=0)

    state = {"task_id": task_id, "env": env, "history": [], "done": False}

    # Default slider values: equal allocation
    n_ch = len(task.channels)
    n_seg = len(task.segments)
    budget_defaults = {c.name: 100.0 / n_ch for c in task.channels}
    segment_defaults = {s.name: 100.0 / n_seg for s in task.segments}
    msg_defaults = {d: 100.0 / len(MESSAGING_DIMS) for d in MESSAGING_DIMS}

    # Build slider update payloads — one Slider component per channel etc.
    # Returns: state, status_html, weeks_chart, channel sliders (max 7), segment sliders (max 4), msg sliders (6)
    return _format_playground_state(state, task, budget_defaults, segment_defaults, msg_defaults, "Episode reset. Set your sliders and click Step Week.")


def _format_playground_state(state, task, budget_vals, segment_vals, msg_vals, status_msg):
    """Build the tuple returned to the playground callback outputs."""
    obs = state.get("env").state if state.get("env") else None  # not used directly
    history = state["history"]

    # Status banner
    if state.get("done"):
        status_html = f"<div style='padding:14px;background:#dcfce7;border-radius:8px;color:#166534;font-weight:600'>✅ Episode complete. {status_msg}</div>"
    else:
        status_html = f"<div style='padding:14px;background:#eff6ff;border-radius:8px;color:#1e3a8a;font-weight:500'>{status_msg}</div>"

    # Live revenue chart (single line)
    fig = go.Figure()
    if history:
        fig.add_trace(go.Scatter(
            x=[h["week"] for h in history],
            y=[h["total_revenue"] for h in history],
            mode="lines+markers",
            name="Revenue",
            line=dict(color=RL_COLOR, width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(124,58,237,0.10)",
        ))
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Cumulative Revenue ($)",
        **_layout("Your Episode So Far", height=320),
    )

    # Metric tiles
    if history:
        last = history[-1]
        metrics_html = (
            f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px'>"
            f"<div class='metric-tile'><div class='k'>Week</div><div class='v'>{last['week']}/{task.total_weeks}</div></div>"
            f"<div class='metric-tile'><div class='k'>Revenue</div><div class='v'>${last['total_revenue']:,.0f}</div></div>"
            f"<div class='metric-tile'><div class='k'>Brand</div><div class='v'>{last['brand_score']:.0f}/100</div></div>"
            f"<div class='metric-tile'><div class='k'>Reward</div><div class='v'>{last['reward']:.2f}</div></div>"
            f"</div>"
        )
    else:
        metrics_html = (
            f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px'>"
            f"<div class='metric-tile'><div class='k'>Week</div><div class='v'>0/{task.total_weeks}</div></div>"
            f"<div class='metric-tile'><div class='k'>Revenue</div><div class='v'>$0</div></div>"
            f"<div class='metric-tile'><div class='k'>Brand</div><div class='v'>50/100</div></div>"
            f"<div class='metric-tile'><div class='k'>Reward</div><div class='v'>—</div></div>"
            f"</div>"
        )

    # Slider updates: pad with hidden ones to fixed width
    MAX_CHANNELS = 7
    MAX_SEGMENTS = 4

    channel_updates = []
    for i in range(MAX_CHANNELS):
        if i < len(task.channels):
            ch = task.channels[i]
            channel_updates.append(gr.update(
                visible=True,
                label=f"💰 {ch.name}",
                value=budget_vals.get(ch.name, 0.0),
            ))
        else:
            channel_updates.append(gr.update(visible=False))

    segment_updates = []
    for i in range(MAX_SEGMENTS):
        if i < len(task.segments):
            seg = task.segments[i]
            segment_updates.append(gr.update(
                visible=True,
                label=f"👥 {seg.name}",
                value=segment_vals.get(seg.name, 0.0),
            ))
        else:
            segment_updates.append(gr.update(visible=False))

    msg_updates = [gr.update(value=msg_vals.get(d, 0.0)) for d in MESSAGING_DIMS]

    return (state, status_html, fig, metrics_html, *channel_updates, *segment_updates, *msg_updates)


def playground_step(state, task_id, *all_sliders):
    """Take one env step using the current slider values."""
    task = get_task(task_id)
    n_ch = len(task.channels)
    n_seg = len(task.segments)

    # Unpack sliders: 7 channel + 4 segment + 6 messaging (positionally fixed)
    channel_slider_values = list(all_sliders[:7])
    segment_slider_values = list(all_sliders[7:11])
    msg_slider_values = list(all_sliders[11:17])

    # Build budget alloc (only active channels) — normalize to 1.0
    raw_b = [max(0.0, v or 0.0) for v in channel_slider_values[:n_ch]]
    sb = sum(raw_b) or 1.0
    budget_alloc = {task.channels[i].name: raw_b[i] / sb for i in range(n_ch)}

    raw_s = [max(0.0, v or 0.0) for v in segment_slider_values[:n_seg]]
    ss = sum(raw_s) or 1.0
    segment_target = {task.segments[i].name: raw_s[i] / ss for i in range(n_seg)}

    raw_m = [max(0.0, v or 0.0) for v in msg_slider_values]
    sm = sum(raw_m) or 1.0
    messaging = {MESSAGING_DIMS[i]: raw_m[i] / sm for i in range(len(MESSAGING_DIMS))}

    if state is None or state.get("env") is None or state.get("done"):
        # Auto-reset if needed
        env = GTMEnvironment()
        env.reset(task_id=task_id, seed=0)
        state = {"task_id": task_id, "env": env, "history": [], "done": False}

    env = state["env"]
    action = GTMAction(
        budget_allocation=budget_alloc,
        segment_targeting=segment_target,
        messaging=messaging,
    )
    obs = env.step(action)
    state["history"].append({
        "week": obs.week,
        "total_revenue": float(obs.total_revenue),
        "brand_score": float(obs.brand_score),
        "reward": float(obs.reward) if obs.reward is not None else 0.0,
    })
    state["done"] = bool(obs.done)

    if state["done"]:
        grader = env.get_grader_score(env.state.episode_id)
        msg = f"Final grader score: {grader:.3f}" if grader is not None else "Episode complete."
    else:
        msg = f"Stepped to week {obs.week}/{obs.total_weeks}. Revenue this week: ${state['history'][-1]['total_revenue'] - (state['history'][-2]['total_revenue'] if len(state['history']) > 1 else 0):,.0f}"

    # Recompose slider current values (display them as percentages: scale to ~100)
    budget_vals = {task.channels[i].name: raw_b[i] / sb * 100 for i in range(n_ch)}
    segment_vals = {task.segments[i].name: raw_s[i] / ss * 100 for i in range(n_seg)}
    msg_vals = {MESSAGING_DIMS[i]: raw_m[i] / sm * 100 for i in range(len(MESSAGING_DIMS))}

    return _format_playground_state(state, task, budget_vals, segment_vals, msg_vals, msg)


# ── Tab 3: Episode Replay ──────────────────────────────────────────────────


def replay_episode(task_id: str, seed_value: float):
    """Build a single-trace cumulative revenue chart for a trained RL run."""
    seed = int(seed_value) if seed_value is not None else 42
    rl = run_trained_rl(task_id, seed)
    weeks = rl["weeks"]
    if not weeks:
        return go.Figure(), pd.DataFrame(), "<div>No data</div>"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[w["week"] for w in weeks],
        y=[w["total_revenue"] for w in weeks],
        mode="lines+markers",
        name="Cumulative revenue",
        line=dict(color=RL_COLOR, width=4),
        marker=dict(size=10),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=[w["week"] for w in weeks],
        y=[w["brand_score"] * (max(w["total_revenue"] for w in weeks) / 100) for w in weeks],
        mode="lines",
        name="Brand health (scaled)",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        yaxis="y",
    ))
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Revenue ($)",
        **_layout("Episode Replay — Trained RL", height=440),
    )

    df = build_action_table(rl)
    summary = (
        f"<div class='metric-tile'><div class='k'>Final Grader Score</div>"
        f"<div class='v'>{rl['grader_score']:.3f}</div></div>"
        if rl.get("grader_score") is not None else ""
    )
    return fig, df, summary


# ── Main Blocks builder ────────────────────────────────────────────────────


def build_dashboard() -> gr.Blocks:
    with gr.Blocks(title="GTM Strategy Optimizer") as demo:

        # Hero header
        gr.HTML(
            "<div class='hero-banner'>"
            "<h1>📈 GTM Strategy Optimizer</h1>"
            "<p>A reinforcement learning agent that allocates marketing budget across channels, "
            "customer segments, and messaging dimensions to maximize long-term revenue. "
            "Trained with custom PPO on a Go-To-Market simulator.</p>"
            "</div>"
        )

        with gr.Tabs():

            # ─── Tab 1: Compare ─────────────────────────────────────
            with gr.Tab("🏁 Compare Strategies"):
                gr.Markdown(
                    "Run the **trained RL policy**, an **equal-allocation heuristic**, and a "
                    "**uniform random** agent on the same task and seed. The winner gets a 👑."
                )

                with gr.Row():
                    cmp_task = gr.Dropdown(
                        choices=list(TASKS.keys()),
                        value="channel_optimizer",
                        label="Task",
                        info="Increasing difficulty: more channels, segments, adversarial dynamics",
                        scale=3,
                    )
                    cmp_seed = gr.Number(value=42, label="Seed", precision=0, scale=1)
                    cmp_btn = gr.Button("▶ Run Comparison", variant="primary", scale=1, size="lg")

                cmp_warning = gr.HTML(value="")

                with gr.Row():
                    cmp_rl_card = gr.HTML(_score_card("Trained RL", "🤖", RL_COLOR, _empty_result()))
                    cmp_heur_card = gr.HTML(_score_card("Heuristic", "📊", HEUR_COLOR, _empty_result()))
                    cmp_rand_card = gr.HTML(_score_card("Random", "🎲", RAND_COLOR, _empty_result()))

                cmp_revenue = gr.Plot(show_label=False)

                with gr.Row():
                    cmp_budget = gr.Plot(show_label=False)
                    cmp_brand = gr.Plot(show_label=False)

                gr.Markdown("### 📋 Per-week actions (Trained RL)")
                cmp_table = gr.Dataframe(interactive=False, wrap=True)

                cmp_btn.click(
                    fn=run_comparison,
                    inputs=[cmp_task, cmp_seed],
                    outputs=[
                        cmp_revenue, cmp_budget, cmp_brand, cmp_table,
                        cmp_rl_card, cmp_heur_card, cmp_rand_card, cmp_warning,
                    ],
                )

            # ─── Tab 2: Interactive playground ──────────────────────
            with gr.Tab("🎛️ Interactive Playground"):
                gr.Markdown(
                    "Drive the simulator yourself. Drag the sliders to set your weekly budget mix, "
                    "then click **Step Week** to advance one week. No JSON required."
                )

                with gr.Row():
                    pg_task = gr.Dropdown(
                        choices=list(TASKS.keys()),
                        value="channel_optimizer",
                        label="Task",
                        scale=3,
                    )
                    pg_reset_btn = gr.Button("🔄 Reset Episode", variant="secondary", scale=1)
                    pg_step_btn = gr.Button("▶ Step Week", variant="primary", scale=1, size="lg")

                pg_status = gr.HTML(value="<div style='padding:14px;background:#eff6ff;border-radius:8px;color:#1e3a8a;font-weight:500'>Pick a task and click Step Week to begin.</div>")
                pg_metrics = gr.HTML(value="")
                pg_chart = gr.Plot(show_label=False)

                gr.Markdown("#### 💰 Budget allocation (per channel)")
                with gr.Row():
                    pg_ch_sliders = [
                        gr.Slider(minimum=0, maximum=100, value=33, step=1, label="channel", visible=False)
                        for _ in range(7)
                    ]

                gr.Markdown("#### 👥 Segment targeting")
                with gr.Row():
                    pg_seg_sliders = [
                        gr.Slider(minimum=0, maximum=100, value=50, step=1, label="segment", visible=False)
                        for _ in range(4)
                    ]

                gr.Markdown("#### 💬 Messaging emphasis")
                with gr.Row():
                    pg_msg_sliders = [
                        gr.Slider(minimum=0, maximum=100, value=17, step=1, label=dim)
                        for dim in MESSAGING_DIMS
                    ]

                pg_state = gr.State(value=None)

                pg_outputs = [
                    pg_state, pg_status, pg_chart, pg_metrics,
                    *pg_ch_sliders, *pg_seg_sliders, *pg_msg_sliders,
                ]

                pg_reset_btn.click(
                    fn=playground_init,
                    inputs=[pg_task],
                    outputs=pg_outputs,
                )
                pg_task.change(
                    fn=playground_init,
                    inputs=[pg_task],
                    outputs=pg_outputs,
                )
                pg_step_btn.click(
                    fn=playground_step,
                    inputs=[pg_state, pg_task, *pg_ch_sliders, *pg_seg_sliders, *pg_msg_sliders],
                    outputs=pg_outputs,
                )

                # Initialize on load
                demo.load(fn=playground_init, inputs=[pg_task], outputs=pg_outputs)

            # ─── Tab 3: Episode Replay ──────────────────────────────
            with gr.Tab("🎬 Episode Replay"):
                gr.Markdown(
                    "Step-by-step trajectory of the trained RL agent — see exactly what budget mix "
                    "it picked each week and how revenue / brand health evolved."
                )
                with gr.Row():
                    rp_task = gr.Dropdown(
                        choices=list(TASKS.keys()),
                        value="channel_optimizer",
                        label="Task",
                        scale=3,
                    )
                    rp_seed = gr.Number(value=42, label="Seed", precision=0, scale=1)
                    rp_btn = gr.Button("🎬 Replay", variant="primary", scale=1, size="lg")

                rp_summary = gr.HTML(value="")
                rp_chart = gr.Plot(show_label=False)
                rp_table = gr.Dataframe(interactive=False, wrap=True)

                rp_btn.click(
                    fn=replay_episode,
                    inputs=[rp_task, rp_seed],
                    outputs=[rp_chart, rp_table, rp_summary],
                )

        gr.HTML(
            "<div style='margin-top:20px;padding:14px;color:#64748b;font-size:13px;text-align:center'>"
            "Built with OpenEnv · Custom PPO trainer in <code>rl/train.py</code> · "
            "Reward function in <code>server/environment.py:_compute_reward</code>"
            "</div>"
        )

    return demo
