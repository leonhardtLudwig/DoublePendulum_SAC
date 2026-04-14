import os
from datetime import datetime

import matplotlib.pyplot
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.controller.random_exploration.random_exploration_controller import Controller_Random_exploration
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func

from stable_baselines3.common.utils import set_random_seed
import torch
from utility.config import SEED
set_random_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


from  utility.TripleController import TripleController

from utility.metrics_evaluation import evaluate_and_save
from utility.RewardConfiguration_acrobot import _in_goal as in_goal_acrobot

from utility.RewardConfiguration_acrobot import make_reward_func


from double_pendulum.analysis.leaderboard import get_swingup_time



def wrap_angle(angle):
    """Mantiene l'angolo nel range [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def condition1(t, x):
    return False

def condition2(t, x):
    err_q1 = abs(wrap_angle(x[0] - np.pi))
    err_q2 = abs(wrap_angle(x[1]))
    
    vel_norm = np.linalg.norm(x[2:])

    angle_err = 1
    vel_err = 20 
    is_close = err_q1 < angle_err and err_q2 < angle_err
    is_slow = vel_norm < vel_err
    
    return is_close and is_slow




def condition_goal(t, x):
    """LOCK """
    err_q1 = wrap_angle(x[0] - np.pi)
    err_q2 = wrap_angle(x[1])
    
    pos_ok = abs(err_q1) < 0.15 and abs(err_q2) < 0.15
    vel_ok = abs(x[2]) < 1.0 and abs(x[3]) < 1.0
    
    return pos_ok and vel_ok 


model_par_path = './parameters/acrobot_parameters.yml'
torque_limit = [0.0, 10.0]
active_act = 1

mpar = model_parameters(filepath=model_par_path)

mpar.set_torque_limit(torque_limit)

dt = 0.001
t_final = 10.0
integrator = "runge_kutta"
goal = [np.pi, 0.0, 0.0, 0.0]
plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller1 = Controller_Random_exploration(ctrl_rate=100,
                                            filt_freq=50,
                                            seed=0,
                                            system_freq=1/dt,
                                            u_max=10,
                                            controlled_dof=[1],
                                            plot_profile=False)

controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
Q =  np.diag([10, 10, 60, 60])
R = np.eye(2) * 2
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0,
                          cost_to_go_cut=1000)


# Stabilizing controller (robust LQR)

Q_stable = np.diag([5.0, 5.0, 120.0, 120.0]) 
R_stable = np.diag([2, 2])*1.5 

controller3 = LQRController(model_pars=mpar)
controller3.set_goal(goal)
controller3.set_cost_matrices(Q=Q_stable, R=R_stable)
controller3.set_parameters(failure_value=0, cost_to_go_cut=20)



controller = TripleController(
    controller1=controller1,
    controller2=controller2,
    controller3=controller3,
    condition1=condition1,
    condition2=condition2,
    condition_goal=condition_goal,
    compute_both=False,
    #verbose=True 
)

controller.init()

# start simulation
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0.0]*4,
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=True,
    video_name="./results/acrobot/baseline/acrobot_baseline.mp4",
    scale=0.3

)

print('Swingup time: '+str(get_swingup_time(T, np.array(X), mpar=mpar)))

# plot timeseries
plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
    save_to="./results/acrobot/baseline/acrobot_baseline.png",
    show=False,
)

evaluate_and_save(
    T=T, X=X, U=U, 
    robot_type="acrobot", 
    in_goal_func=in_goal_acrobot, 
    experiment_name="Baseline",
    filename="./results/acrobot/baseline/metrics_report.csv" 
)

t_final = 5.0 
dt = 0.01

T, X, U = sim.simulate(
    t0=0.0,
    x0=[0, 0, 0, 0],
    tf=t_final,
    dt=dt,
    controller=controller,
)

dynamics_func = double_pendulum_dynamics_func(
    simulator=sim,
    dt=dt,        # stesso dt usato nel training SAC
    integrator=integrator,
    robot="acrobot",
    state_representation=2,
    max_velocity=20,
    torque_limit=[0, 4.0],
    scaling=True,
)

reward_func = make_reward_func(mpar)
reward_func.reset()

rewards = []
for i in range(len(T) - 1):
    obs = dynamics_func.normalize_state(X[i])
    act = act = float(U[i][1]) / 4.0
    r = reward_func(obs, [act])
    rewards.append(r)

total_reward = sum(rewards)   # ← Episode Return, non media
print(f"Episode Return: {total_reward:.2f}")
