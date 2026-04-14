import os
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import CustomEnv, double_pendulum_dynamics_func
from stable_baselines3.common.noise import NormalActionNoise



#from RewardConfiguration_Hybrid import make_reward_func, make_terminated_func, make_noisy_reset_func
from RewardConfiguration_V6 import make_reward_func, make_terminated_func, make_noisy_reset_func
#from RewardConfiguration_V5 import make_reward_func, make_terminated_func, make_noisy_reset_func




from TrainingMonitor import TrainingMonitorCallback


# SAC_pendubot_train.py
from config import (
    SEED, DT, INTEGRATOR, ROBOT, STATE_REPR, MAX_VELOCITY,
    MAX_STEPS, TOTAL_TIMESTEPS, LEARNING_RATE, GAMMA, TAU,
    BATCH_SIZE, ENT_COEF, TARGET_ENTROPY, NET_ARCH, N_CRITICS,
    EVAL_FREQ, N_EVAL_EPISODES, RUN_NAME, LOG_DIR_BASE,
    ROA_S_PATH, ROA_RHO_PATH, N_ENVS, LEARNING_STARTS,MODEL_PAR_PATH, OBS_DIM, BUFFER_SIZE, EP_SECONDS, TORQUE_LIMIT, SCALING
)



# ══════════════════════════════════════════════════════════════════════════════
# SEED — fisso ovunque
# ══════════════════════════════════════════════════════════════════════════════

set_random_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# MODELLO
# ══════════════════════════════════════════════════════════════════════════════

mpar = model_parameters(filepath=MODEL_PAR_PATH)
plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)


# ── wrapper float32 ────────────────────────────────────────────────────────────
class DynamicsFuncWrapper:
    def __init__(self, dynamics_func):
        self.dynamics_func = dynamics_func
        self.max_velocity = dynamics_func.max_velocity

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

obs_space = spaces.Box(np.array([-1.0]*OBS_DIM), np.array([1.0]*OBS_DIM), dtype=np.float32)
act_space = spaces.Box(np.array([-1.0]),          np.array([1.0]),          dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# REWARD / TERMINATED / RESET
# ══════════════════════════════════════════════════════════════════════════════


### HYBRID

reward_func     = make_reward_func(mpar)
terminated_func = make_terminated_func()
terminated_func_eval = make_terminated_func()

noisy_reset_func     = make_noisy_reset_func(dynamics_func)

# def eval_reset_func():
#     terminated_func_eval.reset()
#     p1 = np.random.uniform(-np.pi + 0.3, np.pi - 0.8)
#     p2 = np.random.uniform(-0.1, 0.1)
#     v1, v2 = 0.0, 0.0
#     return dynamics_func.normalize_state(np.array([p1, p2, v1, v2]))
def eval_reset_func():
    terminated_func_eval.reset()
    # sempre stesso stato iniziale — bottom fermo
    return dynamics_func.normalize_state(np.array([-1.0, -1.0, 0.0, 0.0]))


env_kwargs = dict(
    dynamics_func=dynamics_func,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=noisy_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=MAX_STEPS,
)

train_env = make_vec_env(CustomEnv, n_envs=N_ENVS, env_kwargs=env_kwargs, seed=SEED)

eval_env = CustomEnv(**{
    **env_kwargs,
    "reset_func":      eval_reset_func,
    "terminated_func": terminated_func_eval,   # ← istanza eval separata
})

# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS E LOGGING
# ══════════════════════════════════════════════════════════════════════════════

# ── path completo per questa run ─────────────────────────────────────
LOG_DIR = os.path.join(LOG_DIR_BASE, RUN_NAME)   # ← unica riga da aggiungere
os.makedirs(LOG_DIR, exist_ok=True)


eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(LOG_DIR, "best_model"),
    log_path=LOG_DIR,
    eval_freq=max(EVAL_FREQ//N_ENVS, 1),  # scala con n_envs
    n_eval_episodes=N_EVAL_EPISODES,
    verbose=1,
)


# ══════════════════════════════════════════════════════════════════════════════
# AGENTE SAC
# ══════════════════════════════════════════════════════════════════════════════


agent = SAC(
    MlpPolicy,
    train_env,
    verbose=1,
    tensorboard_log=os.path.join(LOG_DIR, "tb_logs"),
    learning_rate=LEARNING_RATE    

)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"RUN: {RUN_NAME}")
    print(f"Obs dim: {OBS_DIM} | Episode: {EP_SECONDS:.1f}s ({MAX_STEPS} steps)")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,} | Buffer: {BUFFER_SIZE:,}")
    print(f"Batch: {BATCH_SIZE} | LR: {LEARNING_RATE} | Seed: {SEED}")
    print(f"{'='*60}\n")

    monitor = TrainingMonitorCallback(
        log_path=os.path.join(LOG_DIR, "training_monitor.csv"),
        dynamics_func=dynamics_func,
        mpar = mpar,
        L1=mpar.l[0],
        L2=mpar.l[1],
        max_velocity=MAX_VELOCITY,
        verbose=1,
    )



    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback,monitor],
        progress_bar=True,
    )
    agent.save(os.path.join(LOG_DIR, "sac_pendubot_final"))
