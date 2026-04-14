# TrainingMonitor.py - versione aggiornata

import numpy as np
import csv
from stable_baselines3.common.callbacks import BaseCallback
from RewardConfiguration_V6 import _in_goal
from config import STATE_REPR

class TrainingMonitorCallback(BaseCallback):

    def __init__(self, log_path, dynamics_func, mpar, L1, L2, max_velocity,
                 goal_pos_thr=0.2, goal_vel_thr=2.0, verbose=0):
        super().__init__(verbose)
        self.log_path      = log_path
        self.dynamics_func = dynamics_func
        self.mpar          = mpar
        self.L1            = L1
        self.L2            = L2
        self.max_velocity  = max_velocity

        # E_ref calcolata una volta sola
        g  = mpar.g
        self.E_ref = (mpar.m[0]*g*mpar.r[0] + mpar.m[1]*g*(mpar.l[0]+mpar.r[1])) * 2

        self._reset_ep()

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep", "episode_return", "ep_length",
                "h_max", "h_mean",
                "E_mean", "E_std",        # energia media e varianza → alta std = spinning
                "E_error_mean",           # |E - E_ref| medio → 0 = sul manifold iso-energetico
                "E_at_top_mean",          # E media quando h_norm > 0.8 → vicino all'upright
                "p1_mean_top",            # p1 medio quando h_norm > 0.8 → π = goal, π/2 = orizzontale
                "v_when_top_mean",        # velocità media quando h_norm > 0.8 → alta = spinning
                "goal_reached",
            ])

    def _reset_ep(self):
        self._ep_reward    = 0.0
        self._ep_h_max     = -999.0
        self._ep_h_sum     = 0.0
        self._ep_steps     = 0
        self._ep_goal      = False
        self._ep_E         = []   # energia ad ogni step
        self._ep_E_top     = []   # energia quando h_norm > 0.8
        self._ep_p1_top    = []   # p1 quando h_norm > 0.8
        self._ep_v_top     = []   # velocità quando h_norm > 0.8

    def _compute_energy(self, p1, p2, v1, v2):
        mp = self.mpar
        g  = mp.g
        I1 = mp.I[0]
        I2 = mp.I[1]
        m2 = mp.m[1]
        l1 = mp.l[0]
        r2 = mp.r[1]
        Ir = mp.Ir
        gr = mp.gr

        M = np.array([
            [I1 + I2 + m2*l1**2 + 2*l1*m2*r2*np.cos(p2) + Ir*(1 + gr**2),
             I2 + l1*m2*r2*np.cos(p2) - gr*Ir],
            [I2 + l1*m2*r2*np.cos(p2) - gr*Ir,
             I2 + Ir*gr**2]
        ])
        qdot = np.array([v1, v2])
        T = 0.5 * qdot @ M @ qdot

        V = (-mp.m[0]*g*mp.r[0]*np.cos(p1)
             - mp.m[1]*g*(l1*np.cos(p1) + r2*np.cos(p1+p2)))

        return T + V

    def _on_step(self) -> bool:
        obs    = self.locals["new_obs"][0]
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]



        if STATE_REPR == 2:
            p1 = obs[0] * np.pi + np.pi   # → [0, 2π]
            p2 = obs[1] * np.pi + np.pi   # → [0, 2π]
            v1 = obs[2] * self.max_velocity
            v2 = obs[3] * self.max_velocity
        elif STATE_REPR == 3:
            p1 = np.arctan2(obs[1], obs[0])
            p2 = np.arctan2(obs[3], obs[2])
            v1 = obs[4] * self.max_velocity
            v2 = obs[5] * self.max_velocity
        h     = -self.L1*np.cos(p1) - self.L2*np.cos(p1+p2)
        h_max = self.L1 + self.L2
        h_norm = h / h_max

        E = self._compute_energy(p1, p2, v1, v2)
        v_tot = np.sqrt(v1**2 + v2**2)

        self._ep_reward += reward
        self._ep_h_max   = max(self._ep_h_max, h)
        self._ep_h_sum  += h
        self._ep_steps  += 1
        self._ep_E.append(E)

        # metriche "zona alta" — il cuore della diagnosi
        if h_norm > 0.8:
            self._ep_E_top.append(E)
            self._ep_p1_top.append(abs(p1))          # abs perché ±π = stesso angolo
            self._ep_v_top.append(v_tot)

        if _in_goal(p1, p2, v1, v2):
            self._ep_goal = True

        if done:
            h_mean      = self._ep_h_sum / max(self._ep_steps, 1)
            E_arr       = np.array(self._ep_E)
            E_mean      = E_arr.mean()
            E_std       = E_arr.std()
            E_err_mean  = np.abs(E_arr - self.E_ref).mean()

            # metriche zona alta (se il pendolo non ci è mai arrivato → nan)
            E_top_mean  = np.mean(self._ep_E_top)  if self._ep_E_top  else float('nan')
            p1_top_mean = np.mean(self._ep_p1_top) if self._ep_p1_top else float('nan')
            v_top_mean  = np.mean(self._ep_v_top)  if self._ep_v_top  else float('nan')

            if self.verbose > 0:
                print(
                    f"[{self.num_timesteps:>8d}] "
                    f"ret={self._ep_reward:>8.1f} | "
                    f"E_err={E_err_mean:.3f} | "
                    f"E_std={E_std:.3f} | "
                    f"p1@top={np.degrees(p1_top_mean):.1f}° | "  # ← 180° = goal, 90° = spinning
                    f"v@top={v_top_mean:.2f} | "
                    f"goal={'YES' if self._ep_goal else 'no'}"
                )

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    round(self._ep_reward, 3),
                    self._ep_steps,
                    round(self._ep_h_max, 4),
                    round(h_mean, 4),
                    round(E_mean, 4),
                    round(E_std, 4),
                    round(E_err_mean, 4),
                    round(E_top_mean, 4) if not np.isnan(E_top_mean) else '',
                    round(np.degrees(p1_top_mean), 2) if not np.isnan(p1_top_mean) else '',
                    round(v_top_mean, 3) if not np.isnan(v_top_mean) else '',
                    int(self._ep_goal),
                ])

            self._reset_ep()

        return True
