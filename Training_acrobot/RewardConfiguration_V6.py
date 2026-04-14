import numpy as np
from config import MAX_VELOCITY, L1, L2, STATE_REPR
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.utils.wrap_angles import wrap_angles_diff

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

#Q = 3.0*np.diag([0.64, 0.64, 0.1, 0.1])
Q = np.diag([1.92, 1.92, 0.3,0.3])
# Q = np.diag([15, 15, 0.1, 0.1])
# Q = np.diag([10,10,0.1,0.1])

R = np.eye(2)*0.82
#R = np.eye(2)*0.82
# R = np.eye(2)*0.0001

# ── helpers ───────────────────────────────────────────────────────────────────
def _decode_obs(obs):
    if STATE_REPR == 2:
        
        p1 = obs[0] * np.pi + np.pi   # → [0, 2π]
        #p2 = obs[1] * np.pi + np.pi   # → [0, 2π]
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
    y1_th = 0.1
    y_th = 0.35

    #energia al gol
    E_goal = m1*g*r1 + m2*g*(l1 + r2)   # V_phys(p1=π, p2=0)


    # ── iperparametri (dal paper / da calibrare) ──────────────────────────
    alpha_align = 3.0    # bonus allineamento link2 vicino alla cima
    alpha_disalign = 1.0
    # beta  = 1.0   # penalità energia cinetica vicino alla cima (vuoi fermarti)
    # rho1  = 0.15   # penalità azione fase alta
    # phi1  = 0.15   # penalità variazione azione fase alta
    beta = 1.0
    rho1 = 0.15
    phi1 = 0.15
    rho2  = 0.02    # penalità azione fase bassa
    phi2  = 0.15   # penalità variazione azione fase bassa
    eta   = 0.02   # penalità velocità fase bassa

    gamma_q = 1.0 #modulare parte quadratica
    gamma_v = 1.0 #modulare parte energetica in alto
    gamma_e = 0.5 #modulare parte energetica in basso
    gamma_end = 3.0 #modulare end effector 2
    alpha_mis = 3.0
    

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

        #LEGGERE:
        #CAMBI RADICALI
        # y_th -> 0.35 da 0.3
        # scala goal reward da 1500 a 3, poi da 3 a 4
        # max torque da 3 a 4
        # aggiunto bonus/maulus velocità vicino al goal
        # episodio da 3 secondi a 5 secondi
        # energia cinetica parte alta da 1 a 0.9
        # penalità azione parte bassa da 0.02 a 0.015
        # aggiungo termine parte bassa per penalizzare end effector in posizione <0


        # ── reward per fase ───────────────────────────────────────────────
        if y >= y_th:
            # fase alta: vuoi V alto, link2 allineato (p2→0), T basso
            
            reward = (V
                      #+ alpha_align * align
                      #- alpha_disalign *disalign
                      - 0.9 *T
                      #- rho1  * u**2
                      #- phi1  * delta_a)
                    )       
            
            
        else:
            # fase bassa: vuoi V alto (swing-up), velocità contenuta
            # ||ṡ||² = v1² + v2²
            # reward = (V- gamma_q * r_quad
            #           - rho2 * u**2
            #           - phi2 * delta_a
            #           - eta  * s_norm_sq
            #           )
            reward = (
                    V  
                    - 0.015 * np.square(u)
                    - 0.05 * (v1**2 + v2**2)
                    - 0.15 * delta_a
                    - 1/(1e-4 + v1**2 + v2**2)
                    - np.exp(-y/y_th)
                )
            

        x = _goal_error(p1, p2, v1, v2)    
        #posso pensare di aumentare il bonus in base alla velocità
        #un'idea potrebbe essere +1/(1+v1+v2), così quando velocità sono alte non aggiunge nulla, quando sono a zero il bonus aumenta di 1
        #oppure potrei pensare 1.0 - (v1**2 / 100.0) - (v2**2 / 100.0) così da velocità oltre 10 penalizzo
        bonus = -np.cos(p1)+np.cos(p2)-np.abs(x[0])-np.abs(x[1]) + 1.0 - (v1**2 / 100.0) - (v2**2 / 100.0)

        # ── bonus terminale ───────────────────────────────────────────────
        if _in_goal(p1, p2, v1, v2):
        
            base_reward = 4.0
            print(f"goal_error = {x}  (atteso: [0, 0, 0, 0])")
            print(f"goal_reward = {bonus*base_reward} ")

            #reward += bonus*1500.0   # scala con il range di V (~2-3 J), non 1e4
            reward += bonus*base_reward   # scala con il range di V (~2-3 J), non 1e4


        

        #reward += +r_end_effector + r_misalign

        # if np.cos(p1)<-0.9 and -np.cos(p1+p2)<0.9:
        #     bonus = 1/(abs(np.cos(p1+p2)))
        #     reward+= bonus*500
        # if np.cos(p1) < -0.9:
        #     alignment = np.cos(p2)  # 1 a p2=0, 0 a p2=π/2, tagliato
        #     reward += alignment * 500
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

        
        # if abs(p1) < 0.05 and abs(p2) < 0.05 and abs(v1) < 0.02 and abs(v2) < 0.02:
        #     print("in basso")
        #     return True

        # if abs(v1) > 12.0 or abs(v2) > 12.0:   # instabilità
        #     print("instabilità")
        #     return True

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
            #p2 = np.random.uniform(-0.25, 0.25)
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





    

