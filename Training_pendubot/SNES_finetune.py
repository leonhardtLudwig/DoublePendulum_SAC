# SNES_pendubot_finetune.py
#
# Fine-tuning di una policy SAC pre-addestrata tramite SNES (EvoTorch).
# Segue la struttura e lo stile di SAC_pendubot_train.py.
#
# Uso:
#   python SNES_pendubot_finetune.py
#
# Dipendenze aggiuntive rispetto a SAC_pendubot_train.py:
#   pip install evotorch

import os
import copy
import tempfile
import numpy as np
import torch

from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.utils import set_random_seed

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import CustomEnv, double_pendulum_dynamics_func

from evotorch import Problem
from evotorch.algorithms.distributed.gaussian import SNES

from RewardConfiguration_V6 import make_reward_func, make_terminated_func, make_noisy_reset_func

from config import (
    SEED, DT, INTEGRATOR, ROBOT, STATE_REPR, MAX_VELOCITY,
    MAX_STEPS, LEARNING_RATE, MODEL_PAR_PATH, OBS_DIM,
    TORQUE_LIMIT, SCALING,
)

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETRI SNES  (modifica qui prima di lanciare)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH    = "./models/original/best_model.zip"  # policy SAC di partenza
POPSIZE       = 40        # dimensione della popolazione (e num. workers Ray)
N_GENERATIONS = 1500       # generazioni totali di SNES
STDEV_INIT    = 0.01    # std iniziale della distribuzione di ricerca
N_EVAL_EPS    = 10        # episodi per stimare il fitness di ogni individuo
 
SAVE_DIR      = "./snes_savings4_std0075_noisyreset_popsize40_1500gen"

# ══════════════════════════════════════════════════════════════════════════════
# SEED
# ══════════════════════════════════════════════════════════════════════════════

set_random_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# MODELLO FISICO  — identico a SAC_pendubot_train.py
# ══════════════════════════════════════════════════════════════════════════════

mpar      = model_parameters(filepath=MODEL_PAR_PATH)
plant     = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)


class DynamicsFuncWrapper:
    def __init__(self, dynamics_func):
        self.dynamics_func = dynamics_func
        self.max_velocity  = dynamics_func.max_velocity

    def __call__(self, state, action, scaling=True):
        return self.dynamics_func(state, action, scaling=scaling).astype(np.float32)

    def normalize_state(self, state):
        return self.dynamics_func.normalize_state(state).astype(np.float32)

    def unscale_state(self, state):
        return self.dynamics_func.unscale_state(state)

    def unscale_action(self, action):
        return self.dynamics_func.unscale_action(action)


dynamics_func = DynamicsFuncWrapper(
    double_pendulum_dynamics_func(
        simulator=simulator,
        dt=DT,
        integrator=INTEGRATOR,
        robot=ROBOT,
        state_representation=STATE_REPR,
        max_velocity=MAX_VELOCITY,
        torque_limit=TORQUE_LIMIT,
        scaling=SCALING,
    )
)

obs_space = spaces.Box(np.array([-1.0] * OBS_DIM), np.array([1.0] * OBS_DIM), dtype=np.float32)
act_space = spaces.Box(np.array([-1.0]), np.array([1.0]), dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# CARICAMENTO POLICY DI RIFERIMENTO
# ══════════════════════════════════════════════════════════════════════════════

# Serve un env dummy solo per costruire la struttura di SAC
_dummy_terminated = make_terminated_func()
_dummy_env = CustomEnv(
    dynamics_func=dynamics_func,
    reward_func=make_reward_func(mpar),
    terminated_func=_dummy_terminated,
    reset_func=lambda: dynamics_func.normalize_state(np.array([-1.0, -1.0, 0.0, 0.0])),
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=MAX_STEPS,
)

REFERENCE_AGENT = SAC(MlpPolicy, _dummy_env, verbose=0, learning_rate=LEARNING_RATE)
REFERENCE_AGENT.set_parameters(MODEL_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: deepcopy del modello  (stessa logica di magic.py)
# ══════════════════════════════════════════════════════════════════════════════

def _deepcopy_model(model: SAC) -> SAC:
    """Serializza e ricarica il modello per ottenere una copia indipendente."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tmp_model")
        model.save(path)
        return SAC.load(path)

# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE DI VALUTAZIONE
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_agent(agent: SAC, n_episodes: int = N_EVAL_EPS) -> float:
    """
    Esegue n_episodes episodi con la policy deterministica dell'agente
    e restituisce il reward totale medio.
    Ogni episodio usa istanze fresche di reward_func e terminated_func
    per evitare contaminazione di stato (prev_action, flag interni).
    """
    rewards = []
    for _ in range(n_episodes):
        r_func = make_reward_func(mpar)
        t_func = make_terminated_func()
        _noisy_reset = make_noisy_reset_func(dynamics_func)


        def reset_func():
            t_func.reset()
            r_func.reset()
            return _noisy_reset()
            #return dynamics_func.normalize_state(np.array([-1.0, -1.0, 0.0, 0.0]))

        env = CustomEnv(
            dynamics_func=dynamics_func,
            reward_func=r_func,
            terminated_func=t_func,
            reset_func=reset_func,
            obs_space=obs_space,
            act_space=act_space,
            max_episode_steps=MAX_STEPS,
        )

        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)

    return float(np.mean(rewards))

# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE SIMULATE  — interfaccia richiesta da EvoTorch
# ══════════════════════════════════════════════════════════════════════════════

def simulate(policy_params: torch.Tensor) -> float:
    """
    Riceve i parametri di latent_pi come vettore flat (da EvoTorch),
    li inietta in una copia dell'agente e ne valuta il fitness.
    Segue la stessa logica di main.py (tutor).
    """
    agent = _deepcopy_model(REFERENCE_AGENT)

    with torch.no_grad():
        state_dict  = agent.policy.actor.latent_pi.state_dict()
        keys        = list(state_dict.keys())
        split_sizes = [torch.numel(state_dict[k]) for k in keys]
        params_split = torch.split(policy_params.clone().detach(), split_sizes)
        state_dict.update({
            k: p.reshape(state_dict[k].shape)
            for k, p in zip(keys, params_split)
        })
        agent.policy.actor.latent_pi.load_state_dict(state_dict)

    score = _evaluate_agent(agent)
    return score if not np.isnan(score) else 0.0

# ══════════════════════════════════════════════════════════════════════════════
# SNES SETUP  — segue main.py (tutor)
# ══════════════════════════════════════════════════════════════════════════════

# vettore iniziale = parametri correnti di latent_pi
initial_solution = np.concatenate([
    p.data.cpu().numpy().flatten()
    for p in REFERENCE_AGENT.policy.actor.latent_pi.parameters()
])

problem = Problem(
    "max",
    simulate,
    solution_length=len(initial_solution),
    num_actors=POPSIZE,      # Ray parallelizza gli individui
)

optimizer = SNES(
    problem,
    popsize=POPSIZE,
    center_init=initial_solution,
    stdev_init=STDEV_INIT,
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOOP DI OTTIMIZZAZIONE
# ══════════════════════════════════════════════════════════════════════════════

best_score_ever = -np.inf

for generation in range(N_GENERATIONS):
    optimizer.step()

    best_score   = float(optimizer.status["best_eval"])
    center_score = float(optimizer.status.get("center_eval", float("nan")))

    print(
        f"[Gen {generation:04d}]  "
        f"best={best_score:.4f}  "
        f"center={center_score:.4f}"
    )

    # ── aggiorna REFERENCE_AGENT con il miglior individuo trovato ──────────
    best_params = optimizer.status["best"].values

    with torch.no_grad():
        state_dict  = REFERENCE_AGENT.policy.actor.latent_pi.state_dict()
        keys        = list(state_dict.keys())
        split_sizes = [torch.numel(state_dict[k]) for k in keys]
        params_split = torch.split(best_params.clone().detach(), split_sizes)
        state_dict.update({
            k: p.reshape(state_dict[k].shape)
            for k, p in zip(keys, params_split)
        })
        REFERENCE_AGENT.policy.actor.latent_pi.load_state_dict(state_dict)

    # ── salvataggio se nuovo best globale ─────────────────────────────────
    if best_score > best_score_ever:
        best_score_ever = best_score
        gen_dir = os.path.join(SAVE_DIR, f"gen_{generation:04d}_score_{best_score:.4f}")
        os.makedirs(gen_dir, exist_ok=True)

        REFERENCE_AGENT.save(os.path.join(gen_dir, "best_model.zip"))
        torch.save(
            REFERENCE_AGENT.policy.actor.latent_pi.state_dict(),
            os.path.join(gen_dir, "latent_pi.pth"),
        )
        print(f"  → nuovo best salvato in {gen_dir}")

print(f"\nOttimizzazione completata. Best score: {best_score_ever:.4f}")
