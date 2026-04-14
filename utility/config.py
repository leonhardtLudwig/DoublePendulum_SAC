import numpy as np
# ══════════════════════════════════════════════════════
# FISICA
# ══════════════════════════════════════════════════════
INTEGRATOR      = "runge_kutta"
MAX_VELOCITY    = 20.0          # rad/s
GOAL            = np.array([np.pi, 0.0, 0.0, 0.0])


# ══════════════════════════════════════════════════════
# TRAINING SAC
# ══════════════════════════════════════════════════════
SEED            = 0
STATE_REPR      = 2
N_ENVS = 50
SCALING = True
N_EVAL_EPISODES = 10
OBS_DIM         = 6 if STATE_REPR == 3 else 4
DT              = 0.01

MAX_STEPS = 1/DT #per pendubot
#MAX_STEPS = 1/DT*5 #per acrobot 
EP_SECONDS      = MAX_STEPS * DT  
LEARNING_RATE = 0.001
TORQUE_LIMIT_PENDUBOT = [10.0,0.0] #per pendubot
TORQUE_LIMIT_ACROBOT = [0.0,4.0] #per acrobot

TOTAL_TIMESTEPS = 1_000_000_000


