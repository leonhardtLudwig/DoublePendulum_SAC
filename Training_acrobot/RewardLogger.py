import numpy as np
import matplotlib.pyplot as plt
import os
from config import RUN_NAME, LOG_DIR_BASE


log_dir = os.path.join(LOG_DIR_BASE, RUN_NAME)
data = np.load(os.path.join(log_dir, "evaluations.npz"))

# contenuto di evaluations.npz
timesteps = data["timesteps"]        # step a cui è stata fatta la valutazione
results   = data["results"]          # shape (n_eval, n_eval_episodes) — reward per episodio
ep_lengths = data["ep_lengths"]      # shape (n_eval, n_eval_episodes) — lunghezza episodi

mean_rewards = results.mean(axis=1)
std_rewards  = results.std(axis=1)
mean_lengths = ep_lengths.mean(axis=1)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# ── reward media di valutazione ────────────────────────────────────────────────
axes[0].plot(timesteps, mean_rewards, color="steelblue", linewidth=2, label="Mean eval reward")
axes[0].fill_between(timesteps,
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     alpha=0.2, color="steelblue", label="±1 std")
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[0].set_ylabel("Mean Episode Return")
axes[0].set_xlabel("Timesteps")
axes[0].set_title("Evaluation Reward during Training")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── lunghezza media degli episodi ──────────────────────────────────────────────
axes[1].plot(timesteps, mean_lengths, color="darkorange", linewidth=2)
axes[1].set_ylabel("Mean Episode Length (steps)")
axes[1].set_xlabel("Timesteps")
axes[1].set_title("Mean Episode Length during Training")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("eval_curve.png", dpi=150)

# ==============================================================================
# FIGURA 2: Visione d'Insieme (Asse X Lineare)
# - Sopra: Reward (Y in SymLog)
# - Sotto: Lunghezza episodi (Normale / Lineare)
# ==============================================================================
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 10))

# ── 1. Reward (Solo Y Logaritmico) ─────────────────────────────────────────────
axes2[0].plot(timesteps, mean_rewards, color="seagreen", linewidth=2, label="Mean eval reward")
axes2[0].fill_between(timesteps,
                      mean_rewards - std_rewards,
                      mean_rewards + std_rewards,
                      alpha=0.2, color="seagreen", label="±1 std")
axes2[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes2[0].set_yscale('symlog') 
axes2[0].set_ylabel("Mean Episode Return (SymLog)")
axes2[0].set_xlabel("Timesteps")
axes2[0].set_title("Evaluation Reward (Visione d'Insieme)")
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

# ── 2. Lunghezza Episodi (Normale) ─────────────────────────────────────────────
axes2[1].plot(timesteps, mean_lengths, color="darkorange", linewidth=2, label="Mean length")
axes2[1].set_ylabel("Mean Episode Length (steps)")
axes2[1].set_xlabel("Timesteps")
axes2[1].set_title("Episode Length (Visione d'Insieme)")
axes2[1].grid(True, alpha=0.3)
axes2[1].legend()

fig2.tight_layout()
fig2.savefig("eval_overview.png", dpi=150)


# ==============================================================================
# FIGURA 3: Zoom sulle Fasi Iniziali (Asse X Logaritmico)
# - Sopra: Reward (X Log, Y SymLog)
# - Sotto: Lunghezza episodi (X Log, Y Normale)
# ==============================================================================
fig3, axes3 = plt.subplots(2, 1, figsize=(12, 10))

# ── 1. Reward (Log su X e Y) ───────────────────────────────────────────────────
axes3[0].plot(timesteps, mean_rewards, color="purple", linewidth=2, label="Mean eval reward")
axes3[0].fill_between(timesteps,
                      mean_rewards - std_rewards,
                      mean_rewards + std_rewards,
                      alpha=0.2, color="purple", label="±1 std")
axes3[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

axes3[0].set_yscale('symlog') 
axes3[0].set_xscale('log')
axes3[0].set_xlim(left=max(1, timesteps[0]), right=timesteps[-1])

axes3[0].set_ylabel("Mean Episode Return (SymLog)")
axes3[0].set_xlabel("Timesteps (Log Scale)")
axes3[0].set_title("Evaluation Reward (Zoom Inizio Addestramento)")
axes3[0].legend()
axes3[0].grid(True, alpha=0.3)

# ── 2. Lunghezza Episodi (Log su X) ────────────────────────────────────────────
axes3[1].plot(timesteps, mean_lengths, color="chocolate", linewidth=2, label="Mean length")

axes3[1].set_xscale('log')
axes3[1].set_xlim(left=max(1, timesteps[0]), right=timesteps[-1])

axes3[1].set_ylabel("Mean Episode Length (steps)")
axes3[1].set_xlabel("Timesteps (Log Scale)")
axes3[1].set_title("Episode Length (Zoom Inizio Addestramento)")
axes3[1].grid(True, alpha=0.3)
axes3[1].legend()

fig3.tight_layout()
fig3.savefig("eval_early_stages_zoom.png", dpi=150)

# plt.show()