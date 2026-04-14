import os
import numpy as np
from gymnasium import spaces

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.plotting import plot_timeseries
from stable_baselines3 import SAC

# sostituisci i valori hardcoded con quelli da config
from config import RUN_NAME, DT, INTEGRATOR, STATE_REPR, MAX_VELOCITY, LOG_DIR_BASE, TORQUE_LIMIT, SCALING


# poi usa
model_path = os.path.join(LOG_DIR_BASE, RUN_NAME, "best_model", "best_model")
#model_path = "./models/200M_SNES"
dt         = DT
integrator = INTEGRATOR


import RewardLogger

# ── parametri modello ──────────────────────────────────────────────────────────
model_par_path = "acrobot_parameters.yml"
mpar = model_parameters(filepath=model_par_path)


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


# ── dynamics func ──────────────────────────────────────────────────────────────
#dt = 0.01
integrator = "runge_kutta"
state_representation = STATE_REPR
max_velocity = 20.0

# dynamics_func = DynamicsFuncWrapper(
#     double_pendulum_dynamics_func(
#         simulator=simulator,
#         dt=dt,
#         integrator=integrator,
#         robot="pendubot",
#         state_representation=state_representation,
#         max_velocity=max_velocity,
#         torque_limit=mpar.tl,
#         scaling=True,
#     )
# )

dynamics_func = DynamicsFuncWrapper(
    double_pendulum_dynamics_func(
        simulator=simulator,
        dt=dt,
        integrator=integrator,
        robot="acrobot",
        state_representation=state_representation,
        max_velocity=max_velocity,
        torque_limit=TORQUE_LIMIT,
        scaling=SCALING,
    )
)


# ── controller SAC corretto ────────────────────────────────────────────────────
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


# ── istanza controller ─────────────────────────────────────────────────────────
# controller = FixedSACController(
#     model_path="./log_data/SAC_pendubot/" + RUN_NAME + "/best_model/best_model",
#     dynamics_func=dynamics_func,
#     dt=dt,
# )
# controller.init()

#S_lqr, rho = compute_and_save_roa(model_par_path=model_par_path, verbose=False)
# ROA_S_PATH   = os.path.join("./", "roa_S.npy")
# ROA_RHO_PATH = os.path.join("./", "roa_rho.npy")

# if os.path.exists(ROA_S_PATH) and os.path.exists(ROA_RHO_PATH):
#     print("RoA già calcolata, carico da file...")
#     S_lqr = np.load(ROA_S_PATH)
#     rho   = float(np.load(ROA_RHO_PATH)[0])
# else:
#     print("Calcolo RoA...")
#     S_lqr, rho = compute_and_save_roa(
#         model_par_path=model_par_path,
#         save_S_path=ROA_S_PATH,
#         save_rho_path=ROA_RHO_PATH,
#         verbose=True,
#     )

# Q_lqr_online = np.diag([1.92, 1.92, 0.3, 0.3])
# R_lqr_online = np.diag([0.82, 0.82])
# S_online = np.array([
#     [7.857934201124567153e01, 5.653751913776947191e01, 1.789996146741196981e01, 8.073612858295813766e00],
#     [5.653751913776947191e01, 4.362786774581156379e01, 1.306971194928728330e01, 6.041705515910111401e00],
#     [1.789996146741196981e01, 1.306971194928728330e01, 4.125964000971944046e00, 1.864116086667296113e00],
#     [8.073612858295813766e00, 6.041705515910111401e00, 1.864116086667296113e00, 8.609202333737846491e-01]
# ])
# rho_online = 8.690673829091186575e-01  # 1.690673829091186575e-01

# controller = HybridController(
#     sac_model_path="./log_data/SAC_pendubot/" + RUN_NAME + "/best_model/best_model",
#     dynamics_func=dynamics_func,
#     S_lqr=S_online, rho=rho_online,
#     model_par_path=model_par_path,
#     dt=dt)
# controller.init()
SAC_controller = FixedSACController(
     #model_path = "./models/200M_SNES/best_model.zip",
     model_path="./log_data/SAC_acrobot/" + RUN_NAME + "/best_model/best_model",
     dynamics_func=dynamics_func,
     dt=dt,
)
SAC_controller.init()


# ── simulazione ────────────────────────────────────────────────────────────────
T, X, U = simulator.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=5.0,
    dt=dt,
    controller=SAC_controller,
    integrator=integrator,
    save_video=True,
    video_name="sac_acrobot_swing.mp4",
    scale=0.3,
)


plot_timeseries(
    T, X, U,
    pos_y_lines=[np.pi,0,-np.pi],
    tau_y_lines=[-TORQUE_LIMIT[0],TORQUE_LIMIT[0]],
    save_to="ts_sac_acrobot_swing.png",
    show=False,
)

T, X, U = simulator.simulate_and_animate(
    t0=0.0,
    x0=[np.pi, 0.0, 0.0, 0.0],
    tf=5.0,
    dt=dt,
    controller=SAC_controller,
    integrator=integrator,
    save_video=True,
    video_name="sac_acrobot_stabilize.mp4",
    scale=0.3,
)

plot_timeseries(
    T, X, U,
    pos_y_lines=[np.pi,0,-np.pi],
    tau_y_lines=[-TORQUE_LIMIT[0],TORQUE_LIMIT[0]],
    save_to="ts_sac_acrobot_stabilize.png",
    show=False,
)