"""
reward_analysis.py
==================
1. Estrazione di mean_eval_reward da file events.out.tfevents
2. Plot comparativo tra run (Plotly)
3. Valutazione della baseline secondo la reward function

Dipendenze: tensorboard, plotly, pandas, numpy, kaleido (per salvataggio PNG)
    pip install tensorboard plotly pandas kaleido

Convenzioni (state / action)
-----------------------------
X : np.ndarray shape (N, 4)   →  [q1, q2, dq1, dq2]
U : np.ndarray shape (N, 2)   →  [tau1, tau2]
    Pendubot → tau1 attivo, tau2 = 0  (actuated_joint=0)
    Acrobot  → tau1 = 0, tau2 attivo  (actuated_joint=1)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# ESTRAZIONE DA TFEVENTS
# ─────────────────────────────────────────────────────────────────────────────

# Tag che SB3 / double_pendulum possono usare — li proviamo in ordine
_DEFAULT_TAGS = [
    "eval/mean_reward",
    "mean_eval_reward",
    "rollout/ep_rew_mean",
    "eval/mean_ep_length",
    "train/mean_reward",
]


def list_available_tags(log_path: str | Path) -> list[str]:
    """
    Elenca tutti i tag scalari disponibili in un file o directory tfevents.
    Utile per scoprire il nome esatto del tag nel proprio run.

    Example
    -------
    >>> print(list_available_tags("./logs/pendubot/sac/"))
    ['eval/mean_reward', 'eval/mean_ep_length', 'train/learning_rate', ...]
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(log_path))
    ea.Reload()
    return ea.Tags().get("scalars", [])


def extract_scalar(
    log_path: str | Path,
    tag: str | None = None,
    fallback_tags: list[str] | None = None,
) -> pd.DataFrame:
    """
    Estrae una serie scalare da un file o directory tfevents.

    Parameters
    ----------
    log_path : str | Path
        Path al file events.out.tfevents.* oppure alla sua directory.
    tag : str | None
        Tag da estrarre. Se None, prova automaticamente _DEFAULT_TAGS.
    fallback_tags : list[str] | None
        Tag alternativi aggiuntivi da provare dopo `tag`.

    Returns
    -------
    pd.DataFrame con colonne: step, wall_time, value

    Raises
    ------
    ValueError se nessun tag viene trovato.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(log_path))
    ea.Reload()
    available = ea.Tags().get("scalars", [])

    candidates = []
    if tag is not None:
        candidates.append(tag)
    candidates += (fallback_tags or [])
    candidates += _DEFAULT_TAGS

    chosen = next((t for t in candidates if t in available), None)

    if chosen is None:
        raise ValueError(
            f"Nessuno dei tag trovato in '{log_path}'.\n"
            f"Tag disponibili: {available}\n"
            f"Usa list_available_tags() per ispezionare il file."
        )

    events = ea.Scalars(chosen)
    df = pd.DataFrame(
        [(e.step, e.wall_time, e.value) for e in events],
        columns=["step", "wall_time", "value"],
    )
    return df


def extract_all_runs(
    runs: dict[str, str | Path],
    tag: str | None = None,
    fallback_tags: list[str] | None = None,
) -> pd.DataFrame:
    """
    Estrae mean_eval_reward da più run e le combina in un DataFrame.

    Parameters
    ----------
    runs : dict {nome_run: path_log}
        Es.:
            runs = {
                "pendubot_sac":      "./logs/pendubot/sac/",
                "pendubot_sac_snes": "./logs/pendubot/sac_snes/",
                "pendubot_sac_lqr":  "./logs/pendubot/sac_lqr/",
                "acrobot_sac":       "./logs/acrobot/sac/",
                "acrobot_sac_snes":  "./logs/acrobot/sac_snes/",
                "acrobot_sac_lqr":   "./logs/acrobot/sac_lqr/",
            }
    tag : str | None
        Tag da estrarre (None = auto-detect).

    Returns
    -------
    pd.DataFrame con colonne: run, step, value
    """
    dfs = []
    for name, path in runs.items():
        try:
            df = extract_scalar(path, tag=tag, fallback_tags=fallback_tags)
            df["run"] = name
            dfs.append(df[["run", "step", "value"]])
            print(f"  [OK]   {name:<30} {len(df):>5} punti  |  tag: _detected_")
        except Exception as e:
            print(f"  [WARN] {name:<30} {e}")

    if not dfs:
        raise RuntimeError("Nessun run caricato. Controlla i path e usa list_available_tags().")

    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_eval_reward(
    df: pd.DataFrame,
    title: str = "Mean Eval Reward during Training",
    x_label: str = "Timestep",
    y_label: str = "Mean Eval Reward",
    smooth_window: int = 1,
    save_to: str | None = None,
    show: bool = False,
) -> None:
    """
    Plotta la mean_eval_reward di più run in un unico grafico Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Output di extract_all_runs() — colonne [run, step, value].
    smooth_window : int
        Finestra rolling mean per smoothing (1 = nessuno smoothing).
    save_to : str | None
        Percorso PNG di output (es. "./results/eval_reward.png").
    show : bool
        Se True mostra il plot interattivo nel browser.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    # Palette coerente con Nexus design system
    palette = [
        "#01696f", "#da7101", "#006494", "#a12c7b",
        "#437a22", "#7a39bb", "#a13544", "#d19900",
    ]

    fig = go.Figure()
    runs = df["run"].unique()

    for i, run_name in enumerate(runs):
        sub = df[df["run"] == run_name].sort_values("step").copy()
        color = palette[i % len(palette)]

        # Valore grezzo (trasparente)
        if smooth_window > 1:
            fig.add_trace(go.Scatter(
                x=sub["step"], y=sub["value"],
                mode="lines", name=run_name,
                line=dict(color=color, width=1),
                opacity=0.25,
                showlegend=False,
            ))
            # Smoothed
            sub["smoothed"] = sub["value"].rolling(smooth_window, min_periods=1, center=True).mean()
            fig.add_trace(go.Scatter(
                x=sub["step"], y=sub["smoothed"],
                mode="lines", name=run_name,
                line=dict(color=color, width=2.5),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=sub["step"], y=sub["value"],
                mode="lines", name=run_name,
                line=dict(color=color, width=2),
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Inter, sans-serif"), x=0.5),
        xaxis=dict(
            title=x_label, showgrid=True,
            gridcolor="#dcd9d5", gridwidth=1,
            zeroline=False,
        ),
        yaxis=dict(
            title=y_label, showgrid=True,
            gridcolor="#dcd9d5", gridwidth=1,
        ),
        plot_bgcolor="#f9f8f5",
        paper_bgcolor="#f7f6f2",
        font=dict(family="Inter, sans-serif", size=13, color="#28251d"),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            bgcolor="rgba(249,248,245,0.9)",
            bordercolor="#dcd9d5", borderwidth=1,
        ),
        hovermode="x unified",
        margin=dict(l=70, r=40, t=80, b=70),
        width=1200, height=550,
    )

    if save_to:
        os.makedirs(os.path.dirname(os.path.abspath(save_to)) or ".", exist_ok=True)
        pio.write_image(fig, save_to, scale=2)
        print(f"[plot] Salvato → {save_to}")
    if show:
        fig.show()


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT REWARD FUNCTION (double pendulum — upright goal)
# ─────────────────────────────────────────────────────────────────────────────

def make_default_reward_fn(
    l1: float,
    l2: float,
    Q_pos: float = 1.0,
    Q_vel: float = 0.01,
    R_tau: float = 0.001,
    actuated_joint: int = 0,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Costruisce una reward function di default per il double pendulum.

    La reward incoraggia la posizione eretta, penalizza velocità e coppia:
        r(x, u) = (cos(q1) + cos(q1+q2))    ← end-effector height proxy
                  - Q_vel * (dq1² + dq2²)    ← penalty velocità
                  - R_tau * tau²              ← penalty coppia

    I pesi Q_pos, Q_vel, R_tau sono personalizzabili per replicare
    la reward usata durante il training SAC.

    Parameters
    ----------
    Q_pos : float   Peso reward di posizione (default 1.0)
    Q_vel : float   Peso penalty velocità    (default 0.01)
    R_tau : float   Peso penalty coppia      (default 0.001)

    Returns
    -------
    Callable: (obs: np.ndarray[4], action: np.ndarray[1]) -> float
    """
    def reward_fn(obs: np.ndarray, action: np.ndarray) -> float:
        q1, q2, dq1, dq2 = obs[0], obs[1], obs[2], obs[3]
        tau = action[0]

        # Proxy di altezza EE normalizzato in [-2, 2], massimo a (pi, 0)
        pos_reward = Q_pos * (np.cos(q1) + np.cos(q1 + q2))
        vel_cost   = Q_vel * (dq1 ** 2 + dq2 ** 2)
        tau_cost   = R_tau * tau ** 2

        return float(pos_reward - vel_cost - tau_cost)

    return reward_fn


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE REWARD EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_baseline_reward(
    T,
    X,
    U,
    reward_fn: Callable[[np.ndarray, np.ndarray], float],
    actuated_joint: int = 0,
) -> dict:
    """
    Valuta la reward function sulla traiettoria della baseline.

    Parameters
    ----------
    T, X, U : array-like
        Output di simulate_and_animate() (liste o np.ndarray).
    reward_fn : callable
        Signature: (obs: np.ndarray[4], action: np.ndarray[1]) -> float
        Passare make_default_reward_fn(...) oppure la propria reward function.
        Per usare la reward dell'env gym:
            reward_fn = lambda obs, act: env.unwrapped._get_reward(obs, act)
    actuated_joint : int
        0 → Pendubot, 1 → Acrobot

    Returns
    -------
    dict:
        rewards      : np.ndarray (N,)  reward ad ogni step
        mean_reward  : float            media episodio (= eval metric in SB3)
        total_reward : float            return grezzo Σ r_t
        cumulative   : np.ndarray (N,)  return cumulativo
    """
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[:, np.newaxis]

    n = min(len(T), len(X), len(U))
    T, X, U = T[:n], X[:n], U[:n]

    rewards = np.array([
        reward_fn(X[i], U[i, actuated_joint: actuated_joint + 1])
        for i in range(n)
    ])

    print(f"  mean_reward  : {rewards.mean():.4f}")
    print(f"  total_reward : {rewards.sum():.4f}")
    print(f"  min / max    : {rewards.min():.4f} / {rewards.max():.4f}")

    return {
        "rewards":     rewards,
        "mean_reward": float(rewards.mean()),
        "total_reward": float(rewards.sum()),
        "cumulative":  np.cumsum(rewards),
    }


def plot_baseline_reward(
    T,
    baseline_results: dict,
    label: str = "Baseline",
    reference_lines: dict[str, float] | None = None,
    title: str = "Step-wise Reward — Baseline",
    save_to: str | None = None,
    show: bool = False,
) -> None:
    """
    Plotta la reward step-by-step della baseline.

    Parameters
    ----------
    T                : timestamps (output di simulate_and_animate)
    baseline_results : output di evaluate_baseline_reward()
    reference_lines  : dict {nome: valore} di linee orizzontali di confronto
                       Es. {"SAC mean": 45.3, "SAC+LQR mean": 67.1}
    save_to          : path PNG di output
    show             : mostra plot interattivo
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    T = np.asarray(T, dtype=float)
    rewards = baseline_results["rewards"]
    n = min(len(T), len(rewards))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=T[:n], y=rewards[:n],
        mode="lines", name=label,
        line=dict(color="#01696f", width=2),
        fill="tozeroy", fillcolor="rgba(1,105,111,0.07)",
    ))

    # Linea media baseline
    mean_r = baseline_results["mean_reward"]
    fig.add_hline(
        y=mean_r,
        line_dash="dash", line_color="#01696f", line_width=1.5,
        annotation_text=f"{label} mean = {mean_r:.3f}",
        annotation_position="top right",
        annotation_font=dict(color="#01696f"),
    )

    # Linee di riferimento (es. mean reward degli agenti SAC)
    ref_colors = ["#da7101", "#006494", "#a12c7b", "#437a22"]
    if reference_lines:
        for j, (ref_name, ref_val) in enumerate(reference_lines.items()):
            fig.add_hline(
                y=ref_val,
                line_dash="dot",
                line_color=ref_colors[j % len(ref_colors)],
                line_width=1.5,
                annotation_text=f"{ref_name} = {ref_val:.3f}",
                annotation_position="bottom right",
                annotation_font=dict(color=ref_colors[j % len(ref_colors)]),
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Inter, sans-serif"), x=0.5),
        xaxis=dict(title="Time [s]", showgrid=True, gridcolor="#dcd9d5"),
        yaxis=dict(title="Reward", showgrid=True, gridcolor="#dcd9d5"),
        plot_bgcolor="#f9f8f5",
        paper_bgcolor="#f7f6f2",
        font=dict(family="Inter, sans-serif", size=13, color="#28251d"),
        legend=dict(bgcolor="rgba(249,248,245,0.9)", bordercolor="#dcd9d5", borderwidth=1),
        hovermode="x unified",
        margin=dict(l=70, r=40, t=80, b=70),
        width=1200, height=480,
    )

    if save_to:
        os.makedirs(os.path.dirname(os.path.abspath(save_to)) or ".", exist_ok=True)
        pio.write_image(fig, save_to, scale=2)
        print(f"[plot] Salvato → {save_to}")
    if show:
        fig.show()
