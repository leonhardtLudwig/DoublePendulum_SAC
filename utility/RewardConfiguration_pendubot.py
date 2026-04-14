import numpy as np
from utility.config import MAX_VELOCITY,STATE_REPR
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.utils.wrap_angles import wrap_angles_diff

L1 = 0.4
L2 = 0.1

#got it from https://github.com/AlbertoSinigaglia/double_pendulum/blob/main/leaderboard/pendubot/simulation_v1/con_sac_lqr.py
rho = 1.690673829091186575e-01
S = np.array(
        [
            [
                7.857934201124567153e01,
                5.653751913776947191e01,
                1.789996146741196981e01,
                8.073612858295813766e00,
            ],
            [
                5.653751913776947191e01,
                4.362786774581156379e01,
                1.306971194928728330e01,
                6.041705515910111401e00,
            ],
            [
                1.789996146741196981e01,
                1.306971194928728330e01,
                4.125964000971944046e00,
                1.864116086667296113e00,
            ],
            [
                8.073612858295813766e00,
                6.041705515910111401e00,
                1.864116086667296113e00,
                8.609202333737846491e-01,
            ],
        ]
    )

Q = np.diag([1.92, 1.92, 0.3,0.3])


R = np.eye(2)*0.82


# ── helpers ───────────────────────────────────────────────────────────────────
def _decode_obs(obs):
    if STATE_REPR == 2:
        p1 = obs[0] * np.pi + np.pi   # → [0, 2π]
        p2 = (obs[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi   # → [-π, π]
        v1 = obs[2] * MAX_VELOCITY
        v2 = obs[3] * MAX_VELOCITY
    elif STATE_REPR == 3:
        p1 = np.arctan2(obs[1], obs[0])
        p2 = np.arctan2(obs[3], obs[2])
        v1 = obs[4] * MAX_VELOCITY
        v2 = obs[5] * MAX_VELOCITY
    
    return p1, p2, v1, v2

def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def _end_effector_height(p1, p2):
    y1 = -L1 * np.cos(p1)
    y2 = y1 - L2 * np.cos(p1 + p2)
    return y1, y2


def _goal_error(p1, p2, v1, v2):
    dp1 = _wrap(p1 - np.pi)
    dp2 = _wrap(p2)
    return np.array([dp1, dp2, v1, v2])

def _in_goal(p1, p2, v1, v2):

    #roa based

    x = _goal_error(p1, p2, v1, v2)


    V = x.T @ S @ x


    return V < rho*1.0

# ── reward ────────────────────────────────────────────────────────────────────



def make_reward_func(mpar):
    g  = mpar.g
    I1 = mpar.I[0]
    I2 = mpar.I[1]
    l1 = mpar.l[0]
    l2 = mpar.l[1]
    gr = mpar.gr
    Ir = mpar.Ir
    m1 = mpar.m[0]
    m2 = mpar.m[1]
    r1 = mpar.r[0]
    r2 = mpar.r[1]
    tl = mpar.tl[0]


    # soglia altezza 
    y_th = 0.30

    # ── iperparametri  ──────────────────────────
    beta = 1.0
    rho2  = 0.05    # penalità azione fase bassa
    phi2  = 0.15   # penalità variazione azione fase bassa
    eta   = 0.02   # penalità velocità fase bassa

    

    # ── stato interno: azione precedente per calcolare Δa ─────────────────
    prev_action = [0.0]

    def reward_func(observation, action):
        p1, p2, v1, v2 = _decode_obs(observation)
        u = tl*float(action[0])   # azione normalizzata ∈ [-1, 1]

        # Δa = differenza con azione precedente
        delta_a = abs(u - prev_action[0])
        prev_action[0] = u

        # ── energia cinetica ──────────────────────────────────────────────
        M = np.array([
            [I1 + I2 + m2*l1**2 + 2*l1*m2*r2*np.cos(p2) + Ir*(1 + gr**2),
             I2 + l1*m2*r2*np.cos(p2) - gr*Ir],
            [I2 + l1*m2*r2*np.cos(p2) - gr*Ir,
             I2 + Ir*gr**2]
        ])
        qdot = np.array([v1, v2])
        T = 0.5 * qdot @ M @ qdot

        # ── energia potenziale ────────────────────────────────────────────
        V = (-m1*g*r1*np.cos(p1)
             - m2*g*(l1*np.cos(p1) + r2*np.cos(p1 + p2)))

        # ── altezza end-effector ──────────────────────────────────────────
        s = np.array([p1,p2,v1,v2])
        y = wrap_angles_diff(s)
        p1 = y[0]
        p2 = y[1]
        v1 = y[2]
        v2 = y[3]
        _, y = _end_effector_height(p1, p2)

        s_norm_sq = v1**2 + v2**2

        # ── reward per fase ───────────────────────────────────────────────
        if y >= y_th:
            reward = (V - beta  * T)       
        else:
            reward = (
                    V  
                    - eta * np.square(u)
                    - rho2 * (s_norm_sq)
                    - phi2* delta_a
                )

        # ── bonus terminale ───────────────────────────────────────────────
        if _in_goal(p1, p2, v1, v2):
            x = _goal_error(p1, p2, v1, v2)
            #print(f"goal_error = {x}  (atteso: [0, 0, 0, 0])")
            bonus = -np.cos(p1)+np.cos(p2)+x[0]+x[1]
            #print(f"goal_reward = {bonus*1500.0} ")

            reward += bonus*1500.0  

        return float(reward)

    def reset():
        prev_action[0] = 0.0

    reward_func.reset = reset
    return reward_func


# ── terminated ────────────────────────────────────────────────────────────────


def make_terminated_func():
    def terminated_func(observation):
        p1, p2, v1, v2 = _decode_obs(observation)
        s = np.array([p1,p2,v1,v2])
        y = wrap_angles_diff(s)
        p1 = y[0]
        p2 = y[1]
        v1 = y[2]
        v2 = y[3]

        if _in_goal(p1, p2, v1, v2):           # successo
            print("successo")
            return True

        return False

    terminated_func.reset = lambda: None
    return terminated_func



# ── reset ─────────────────────────────────────────────────────────────────────

def make_noisy_reset_func(dynamics_func):

    def noisy_reset_func():

        mode = np.random.choice(
            ['bottom','mid','near_top', 'exact'],
            p=[0.5,0.24,0.24, 0.02]
        )

        if mode == 'bottom':

            p1 = np.random.uniform(-0.15, 0.15)
            p2 = np.random.uniform(-0.15, 0.15)

            v1 = np.random.uniform(-0.3, 0.3)
            v2 = np.random.uniform(-0.3, 0.3)


        elif mode == 'mid':

            sign = np.random.choice([-1,1])

            # metà swing
            p1 = sign * np.random.uniform(0.8, 2.0)
            p2 = np.random.uniform(-0.4, 0.4)

            # velocità coerente con lo swing
            v1 = sign * np.random.uniform(0.5, 2.0)
            v2 = np.random.uniform(-1.5, 1.5)


        elif mode == 'near_top':   # near_top

            sign = np.random.choice([-1,1])

            # vicino alla cima
            p1 = sign * np.random.uniform(np.pi-0.15, np.pi+0.15)
            # 20% delle volte: p2 sbagliato (il caso che devi imparare a correggere)
            if np.random.uniform() < 0.20:
                p2_sign = np.random.choice([-1, 1])
                p2 = p2_sign * np.random.uniform(np.pi - 0.4, np.pi+0.4)
            else:
                p2 = np.random.uniform(-0.1, 0.1)

            # velocità piccole
            v1 = np.random.uniform(-1, 1)
            v2 = np.random.uniform(-1, 1)

            p1 += np.random.normal(0,0.03)
            p2 += np.random.normal(0,0.03)
        else: 
            p1 = np.pi
            p2 = 0
            v1 = 0
            v2 = 0


        state = np.array([p1,p2,v1,v2])
        #print(f"stato resettato a {state}")
        return dynamics_func.normalize_state(state)

    return noisy_reset_func





    

