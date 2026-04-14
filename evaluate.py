import os

def setup_directories():
    base_dir = "results"
    robots = ["acrobot", "pendubot"]
    algorithms = ["baseline", "sac", "sac_lqr", "sac_snes"]

    print("Verifica delle directory di output in corso...")
    for robot in robots:
        for algo in algorithms:
            # Crea il percorso completo, es: results/acrobot/sac_lqr
            path = os.path.join(base_dir, robot, algo)
            
            # exist_ok=True evita errori se la cartella esiste già,
            # mentre crea automaticamente anche le cartelle "padre" se mancano.
            os.makedirs(path, exist_ok=True)
            
    print("Directory pronte!")

setup_directories()

import numpy as np
from gymnasium import spaces

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
import torch


from utility.config import DT, INTEGRATOR, STATE_REPR, TORQUE_LIMIT_PENDUBOT, TORQUE_LIMIT_ACROBOT, SCALING, GOAL, MAX_VELOCITY, SEED

set_random_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from utility.metrics_evaluation import evaluate_and_save

from utility.RewardConfiguration_pendubot import _in_goal as in_goal_pendubot
from utility.RewardConfiguration_acrobot import _in_goal as in_goal_acrobot

import utility.baseline_pendubot
import utility.baseline_acrobot

TF = 5.0
dt_SAC         = DT 
dt_CONTROLLER  = 0.001
integrator = INTEGRATOR

pendubot_par_path = "./parameters/pendubot_parameters.yml"
acrobot_par_path = "./parameters/acrobot_parameters.yml"

pendubot_mpar = model_parameters(filepath=pendubot_par_path)
acrobot_mpar = model_parameters(filepath=acrobot_par_path)

pendubot_plant = SymbolicDoublePendulum(model_pars=pendubot_mpar)
pendubot_simulator = Simulator(plant=pendubot_plant)

acrobot_plant = SymbolicDoublePendulum(model_pars=acrobot_mpar)
acrobot_simulator = Simulator(plant=acrobot_plant)

goal = GOAL

integrator = INTEGRATOR
state_representation = STATE_REPR
max_velocity = MAX_VELOCITY

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

pendubot_SAC_dynamics_func = DynamicsFuncWrapper(
    double_pendulum_dynamics_func(
        simulator=pendubot_simulator,
        dt=dt_SAC,
        integrator=integrator,
        robot="pendubot",
        state_representation=state_representation,
        max_velocity=max_velocity,
        torque_limit=TORQUE_LIMIT_PENDUBOT,
        scaling=SCALING,
    )
)

pendubot_Controller_dynamics_func = DynamicsFuncWrapper(
    double_pendulum_dynamics_func(
        simulator=pendubot_simulator,
        dt=dt_CONTROLLER,
        integrator=integrator,
        robot="pendubot",
        state_representation=state_representation,
        max_velocity=max_velocity,
        torque_limit=TORQUE_LIMIT_PENDUBOT,
        scaling=SCALING,
    )
)

acrobot_SAC_dynamics_func = DynamicsFuncWrapper(
    double_pendulum_dynamics_func(
        simulator=acrobot_simulator,
        dt=dt_SAC,
        integrator=integrator,
        robot="acrobot",
        state_representation=state_representation,
        max_velocity=max_velocity,
        torque_limit=TORQUE_LIMIT_ACROBOT,
        scaling=SCALING,
    )
)

acrobot_Controller_dynamics_func = DynamicsFuncWrapper(
    double_pendulum_dynamics_func(
        simulator=acrobot_simulator,
        dt=dt_CONTROLLER,
        integrator=integrator,
        robot="acrobot",
        state_representation=state_representation,
        max_velocity=max_velocity,
        torque_limit=TORQUE_LIMIT_ACROBOT,
        scaling=SCALING,
    )
)


class FixedSACController(SACController):
    def __init__(self, model_path, dynamics_func, dt, scaling=True):
        AbstractController.__init__(self)

        self.model = SAC.load(model_path)
        self.dynamics_func = dynamics_func
        self.dt = dt
        self.scaling = scaling

        obs_dim = self.model.observation_space.shape[0]
        self.model.predict(np.zeros(obs_dim))

    def get_control_output_(self, x, t=None):
        obs = self.dynamics_func.normalize_state(x)
        action, _ = self.model.predict(obs, deterministic=True)
        return self.dynamics_func.unscale_action(action)

def condition2_pendubot(t, x):
    # attiva LQR: vicino al goal, q2 allineato, velocità contenuta
    p1_err = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    p2     = (x[1] + np.pi) % (2*np.pi) - np.pi
    v1, v2 = x[2], x[3]
    return in_goal_pendubot(p1_err,p2,v1,v2)

def condition1_pendubot(t, x):
    # torna al SAC: LQR ha perso il controllo
    p1_err = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    p2     = (x[1] + np.pi) % (2*np.pi) - np.pi
    return (abs(p1_err) > 0.6 or    # troppo lontano da ±π
            abs(p2)     > 0.3)      # q2 completamente fuori

def condition2_acrobot(t, x):
    # attiva LQR: vicino al goal, q2 allineato, velocità contenuta
    p1_err = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    p2     = (x[1] + np.pi) % (2*np.pi) - np.pi
    v1, v2 = x[2], x[3]
    return in_goal_acrobot(p1_err,p2,v1,v2)

def condition1_acrobot(t, x):
    # torna al SAC: LQR ha perso il controllo
    p1_err = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    p2     = (x[1] + np.pi) % (2*np.pi) - np.pi
    return (abs(p1_err) > 0.6 or    # troppo lontano da ±π
            abs(p2)     > 0.3)      # q2 completamente fuori


pendubot_SAC_controller = FixedSACController(
     model_path="./pendubot_models/pendu_sac_and_snes/pendu_sac",
     dynamics_func=pendubot_SAC_dynamics_func,
     dt=dt_SAC,
)
pendubot_SAC_SNES_controller = FixedSACController(
     model_path="./pendubot_models/pendu_sac_and_snes/pendu_sac_snes",
     dynamics_func=pendubot_SAC_dynamics_func,
     dt=dt_SAC,
)

pendubot_SAC_lacking_LQR_controller = FixedSACController(
     model_path="./pendubot_models/pendu_sac_lqr/pendu_sac_lqr",
     dynamics_func=pendubot_Controller_dynamics_func,
     dt=dt_CONTROLLER,
)

acrobot_SAC_controller = FixedSACController(
     model_path="./acrobot_models/acro_sac_and_snes/acro_sac",
     dynamics_func=acrobot_SAC_dynamics_func,
     dt=dt_SAC,
)

acrobot_SAC_SNES_controller = FixedSACController(
     model_path="./acrobot_models/acro_sac_and_snes/acro_sac_snes",
     dynamics_func=acrobot_SAC_dynamics_func,
     dt=dt_SAC,
)

acrobot_SAC_lacking_LQR_controller = FixedSACController(
     model_path="./acrobot_models/acro_sac_lqr/acro_sac_lqr",
     dynamics_func=acrobot_Controller_dynamics_func,
     dt=dt_CONTROLLER,
)

Q= np.diag([2.0, 200.0, 0.3,10])  
R = np.eye(2)*0.0001

LQR_pendubot= LQRController(model_pars=pendubot_mpar)
LQR_pendubot.set_goal(goal)
LQR_pendubot.set_cost_matrices(Q=Q, R=R)
LQR_pendubot.set_parameters(failure_value=0.0,
                          cost_to_go_cut=100000)

LQR_acrobot= LQRController(model_pars=acrobot_mpar)
LQR_acrobot.set_goal(goal)
LQR_acrobot.set_cost_matrices(Q=Q, R=R)
LQR_acrobot.set_parameters(failure_value=0.0,
                          cost_to_go_cut=100000)

pendubot_SAC_LQR_controller = CombinedController(
    controller1=pendubot_SAC_lacking_LQR_controller,
    controller2=LQR_pendubot,
    condition1=condition1_pendubot,
    condition2=condition2_pendubot,
    compute_both=False,
    verbose = True,
)

acrobot_SAC_LQR_controller = CombinedController(
    controller1=acrobot_SAC_lacking_LQR_controller,
    controller2=LQR_acrobot,
    condition1=condition1_acrobot,
    condition2=condition2_acrobot,
    compute_both=False,
    verbose = True,
)

pendubot_SAC_controller.init()
pendubot_SAC_LQR_controller.init()
pendubot_SAC_SNES_controller.init()

acrobot_SAC_controller.init()
acrobot_SAC_LQR_controller.init()
acrobot_SAC_SNES_controller.init()

T, X, U = pendubot_simulator.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=TF,
    dt=dt_SAC,
    controller=pendubot_SAC_controller,
    integrator=integrator,
    save_video=True,
    video_name="./results/pendubot/sac/pendubot_sac.mp4",
    scale=0.3,
)

plot_timeseries(
    T, X, U,
    pos_y_lines=[np.pi,0,-np.pi],
    tau_y_lines=[-TORQUE_LIMIT_PENDUBOT[0],TORQUE_LIMIT_PENDUBOT[0]],
    save_to="./results/pendubot/sac/pendubot_sac.png",
    show=False,
)

evaluate_and_save(
    T=T, X=X, U=U, 
    robot_type="pendubot", 
    in_goal_func=in_goal_pendubot, 
    experiment_name="SAC",
    filename="./results/pendubot/sac/metrics_report.csv" 
)


T, X, U = pendubot_simulator.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=TF,
    dt=dt_SAC,
    controller=pendubot_SAC_SNES_controller,
    integrator=integrator,
    save_video=True,
    video_name="./results/pendubot/sac_snes/pendubot_sac_snes.mp4",
    scale=0.3,
)

plot_timeseries(
    T, X, U,
    pos_y_lines=[np.pi,0,-np.pi],
    tau_y_lines=[-TORQUE_LIMIT_PENDUBOT[0],TORQUE_LIMIT_PENDUBOT[0]],
    save_to="./results/pendubot/sac_snes/pendubot_sac_snes.png",
    show=False,
)

evaluate_and_save(
    T=T, X=X, U=U, 
    robot_type="pendubot", 
    in_goal_func=in_goal_pendubot, 
    experiment_name="SAC_SNES",
    filename="./results/pendubot/sac_snes/metrics_report.csv" 
)

T, X, U = pendubot_simulator.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=TF,
    dt=dt_CONTROLLER,
    controller=pendubot_SAC_LQR_controller,
    integrator=integrator,
    save_video=True,
    video_name="./results/pendubot/sac_lqr/pendubot_sac_lqr.mp4",
    scale=0.3,
)

plot_timeseries(
    T, X, U,
    pos_y_lines=[np.pi,0,-np.pi],
    tau_y_lines=[-TORQUE_LIMIT_PENDUBOT[0],TORQUE_LIMIT_PENDUBOT[0]],
    save_to="./results/pendubot/sac_lqr/pendubot_sac_lqr.png",
    show=False,
)

evaluate_and_save(
    T=T, X=X, U=U, 
    robot_type="pendubot", 
    in_goal_func=in_goal_pendubot, 
    experiment_name="SAC_LQR",
    filename="./results/pendubot/sac_lqr/metrics_report.csv" 
)

T, X, U = acrobot_simulator.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=TF*2,
    dt=dt_SAC,
    controller=acrobot_SAC_controller,
    integrator=integrator,
    save_video=True,
    video_name="./results/acrobot/sac/acrobot_sac.mp4",
    scale=0.3,
)

plot_timeseries(
    T, X, U,
    pos_y_lines=[np.pi, 0, -np.pi],
    tau_y_lines=[-TORQUE_LIMIT_ACROBOT[0], TORQUE_LIMIT_ACROBOT[0]],
    save_to="./results/acrobot/sac/acrobot_sac.png",
    show=False,
)

evaluate_and_save(
    T=T, X=X, U=U, 
    robot_type="acrobot", 
    in_goal_func=in_goal_acrobot, 
    experiment_name="SAC",
    filename="./results/acrobot/sac/metrics_report.csv" 
)

T, X, U = acrobot_simulator.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=TF,
    dt=dt_SAC,
    controller=acrobot_SAC_SNES_controller,
    integrator=integrator,
    save_video=True,
    video_name="./results/acrobot/sac_snes/acrobot_sac_snes.mp4",
    scale=0.3,
)

plot_timeseries(
    T, X, U,
    pos_y_lines=[np.pi, 0, -np.pi],
    tau_y_lines=[-TORQUE_LIMIT_ACROBOT[0], TORQUE_LIMIT_ACROBOT[0]],
    save_to="./results/acrobot/sac_snes/acrobot_sac_snes.png",
    show=False,
)
evaluate_and_save(
    T=T, X=X, U=U, 
    robot_type="acrobot", 
    in_goal_func=in_goal_acrobot, 
    experiment_name="SAC_SNES",
    filename="./results/acrobot/sac_snes/metrics_report.csv" 
)

T, X, U = acrobot_simulator.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=TF,
    dt=dt_CONTROLLER,
    controller=acrobot_SAC_LQR_controller,
    integrator=integrator,
    save_video=True,
    video_name="./results/acrobot/sac_lqr/acrobot_sac_lqr.mp4",
    scale=0.3,
)

plot_timeseries(
    T, X, U,
    pos_y_lines=[np.pi, 0, -np.pi],
    tau_y_lines=[-TORQUE_LIMIT_ACROBOT[0], TORQUE_LIMIT_ACROBOT[0]],
    save_to="./results/acrobot/sac_lqr/acrobot_sac_lqr.png",
    show=False,
)
evaluate_and_save(
    T=T, X=X, U=U, 
    robot_type="acrobot", 
    in_goal_func=in_goal_acrobot, 
    experiment_name="SAC_LQR",
    filename="./results/acrobot/sac_lqr/metrics_report.csv" 
)


# nota importante: sac e snes originano dallo stesso training, lqr da un training diverso con meno step totali. Per lqr stampo direttamente la versione finale senza fronzoli
#TODO: STAMPARE LE REWARD
#TODO: INSERIRE BASELINE
#TODO: VALUTARE BASELINE SU REWARD