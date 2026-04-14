import numpy as np
from utility.config import MAX_VELOCITY,STATE_REPR
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.utils.wrap_angles import wrap_angles_diff

L1 = 0.1
L2 = 0.4

#got it from https://github.com/AlbertoSinigaglia/double_pendulum/blob/main/leaderboard/acrobot/simulation_v1/con_sac_lqr.py
rho = 2.349853516578003232e-01
S = np.array(
    [
        [
            9.770536750948697318e02,
            4.412387317512778395e02,
            1.990562043567418016e02,
            1.018948893750672369e02,
        ],
        [
            4.412387317512778395e02,
            1.999223464452055055e02,
            8.995900469226445750e01,
            4.605280324531641156e01,
        ],
        [
            1.990562043567418016e02,
            8.995900469226445750e01,
            4.059381113966859544e01,
            2.077912430021438439e01,
        ],
        [
            1.018948893750672369e02,
            4.605280324531641156e01,
            2.077912430021438439e01,
            1.063793947790017036e01,
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
    y_th = 0.35

    # ── iperparametri ──────────────────────────
   
    beta = 0.9 #penalità energia cinetica
    rho2  = 0.015    # penalità azione fase bassa
    phi2  = 0.15   # penalità variazione azione fase bassa
    eta   = 0.05   # penalità velocità fase bassa

    

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
        y1, y = _end_effector_height(p1, p2)


        x_err  = np.array([_wrap(p1-np.pi), _wrap(p2), v1, v2])
        r_quad =(x_err @ Q @ x_err)  # Q piccola, solo per forma locale
        align = (1.0 - np.cos(p1+p2))**2
        #alignment_bonus = np.cos(p2)
        #disalign = (1.0+np.cos(2*p1+p2))**2
        disalign = (1+np.cos(p2))**2
        #disalign = _wrap(p2)**2
        s_norm_sq = v1**2 + v2**2



        # ── reward per fase ───────────────────────────────────────────────
        if y >= y_th:
            reward = (V - beta *T)       
        else:
            reward = (
                    V  
                    - rho2* np.square(u)
                    - eta * (s_norm_sq)
                    - phi2 * delta_a
                    - 1/(1e-4 +s_norm_sq)
                    - np.exp(-y/y_th)
                )
            

        x = _goal_error(p1, p2, v1, v2)    
        bonus = -np.cos(p1)+np.cos(p2)-np.abs(x[0])-np.abs(x[1]) + 1.0 - (v1**2 / 100.0) - (v2**2 / 100.0)

        # ── bonus terminale ───────────────────────────────────────────────
        if _in_goal(p1, p2, v1, v2):
        
            base_reward = 4.0
            #print(f"goal_error = {x}  (atteso: [0, 0, 0, 0])")
            #print(f"goal_reward = {bonus*base_reward} ")
            reward += bonus*base_reward   # scala con il range di V (~2-3 J)

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
        p1_wrap = y[0]
        p2_wrap = y[1]
        v1_wrap = y[2]
        v2_wrap = y[3]

        if _in_goal(p1_wrap, p2_wrap, v1_wrap, v2_wrap):           # successo
            print("successo")
            return True

        if abs(p1) > np.pi * 4:                 # angolo illimitato (fix encoding)
            print("angolo illimitato")
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
        return dynamics_func.normalize_state(state)

    return noisy_reset_func





    

