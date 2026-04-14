import numpy as np
import os
import csv

# ==========================================
# 1. FUNZIONI PER LE SINGOLE METRICHE
# ==========================================

def wrap_angle(angle):
    """Riporta un angolo qualsiasi nell'intervallo [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calc_max_torque(U):
    """Calcola il picco massimo della coppia."""
    return np.max(np.abs(U))

def calc_integrated_torque(U, dt):
    """Calcola lo sforzo totale (Norma L1)."""
    return np.sum(np.abs(U)) * dt

def calc_torque_cost(U, dt):
    """Calcola il costo della coppia (Norma L2)."""
    return np.sum(U**2) * dt

def calc_torque_smoothness(U):
    """Calcola la fluidità (variazione) del segnale di controllo."""
    # Gestisce sia array (N,) che (N, 1)
    U_flat = U.flatten()
    return np.sqrt(np.mean(np.diff(U_flat)**2))

def calc_velocity_cost(X, dt):
    """Calcola il costo basato sull'energia cinetica (velocità dei giunti).
    Assumiamo X = [q1, q2, q1_dot, q2_dot]
    """
    q1_dot = X[:, 2]
    q2_dot = X[:, 3]
    return np.sum(q1_dot**2 + q2_dot**2) * dt

def calc_energy(U, X, dt, robot_type):
    """
    Calcola il lavoro meccanico speso.
    - Pendubot: motore al giunto 1 -> velocità è q1_dot (X[:, 2])
    - Acrobot: motore al giunto 2 -> velocità è q2_dot (X[:, 3])
    """
    U_flat = U.flatten()
    
    if robot_type.lower() == "pendubot":
        v_actuated = X[:, 2]
    elif robot_type.lower() == "acrobot":
        v_actuated = X[:, 3]
    else:
        raise ValueError("robot_type deve essere 'pendubot' o 'acrobot'")
        
    return np.sum(np.abs(U_flat * v_actuated)) * dt

def calc_precision_error(T, X):
    """
    Calcola l'errore di precisione (RMSE) nell'ultimo mezzo secondo di simulazione.
    Ritorna:
    - rmse_deg: L'errore medio di posizione in GRADI (molto più leggibile per noi umani)
    - rmse_vel: L'errore medio di velocità (il "Jitter" o vibrazione)
    """
    T = np.array(T)
    X = np.array(X)
    
    # Prendiamo solo la "coda" della simulazione (l'ultimo 0.5 secondi)
    dt = np.mean(np.diff(T))
    steps_in_half_second = int(0.5 / dt)
    window_size = min(steps_in_half_second, len(X) // 2)
    
    X_tail = X[-window_size:]
    
    # 1. ERRORE DI POSIZIONE (distanza dal goal q1=pi, q2=0)
    # Usiamo wrap_angle per gestire i giri completi.
    # Se q1 è 3*pi, 3*pi - pi = 2*pi -> wrap(2*pi) = 0 (Errore zero!)
    error_q1 = wrap_angle(X_tail[:, 0] - np.pi) 
    error_q2 = wrap_angle(X_tail[:, 1] - 0.0)
    
    # Calcoliamo la distanza quadratica media totale (RMSE) in radianti
    rmse_rad = np.sqrt(np.mean(error_q1**2 + error_q2**2))
    
    # Convertiamo in gradi per avere un numero facile da leggere nel CSV (es. 2.5 gradi di errore)
    rmse_deg = np.degrees(rmse_rad)
    
    # 2. ERRORE DI VELOCITÀ (Jitter)
    # Idealmente in cima le velocità devono essere 0. 
    # Questa metrica penalizza i controller che vibrano molto.
    rmse_vel = np.sqrt(np.mean(X_tail[:, 2]**2 + X_tail[:, 3]**2))
    
    return rmse_deg, rmse_vel

def calc_swingup_metrics(T, X, in_goal_func=None):
    """
    Analisi tollerante con DEBUG integrato.
    """
    T = np.array(T)
    X = np.array(X)
    
    # SOGLIE ALLARGATE (le ho rese molto più permissive)
    COS_Q1_THRESHOLD = -0.8   # Circa 36 gradi di tolleranza dalla verticale
    COS_Q2_THRESHOLD = 0.8    # Circa 36 gradi di tolleranza sull'allineamento
    VELOCITY_MAX = 5.0        # Margine enorme per le micro-vibrazioni del SAC
    
    def is_physically_in_goal(state):
        q1, q2, v1, v2 = state
        return (np.cos(q1) < COS_Q1_THRESHOLD) and \
               (np.cos(q2) > COS_Q2_THRESHOLD) and \
               (abs(v1) < VELOCITY_MAX) and \
               (abs(v2) < VELOCITY_MAX)

    is_physically_goal_state = np.array([is_physically_in_goal(state) for state in X])
    
    goal_indices = np.where(is_physically_goal_state)[0]
    
    if len(goal_indices) == 0:
        # ---- DIAGNOSTICA ----
        print("\n[DEBUG] FALLIMENTO: Mai entrato nel goal! Valori all'ultimo step:")
        q1, q2, v1, v2 = X[-1]
        print(f"q1: {q1:.2f} (cos={np.cos(q1):.2f}) -> OK? {np.cos(q1) < COS_Q1_THRESHOLD}")
        print(f"q2: {q2:.2f} (cos={np.cos(q2):.2f}) -> OK? {np.cos(q2) > COS_Q2_THRESHOLD}")
        print(f"v1: {v1:.2f} rad/s -> OK? {abs(v1) < VELOCITY_MAX}")
        print(f"v2: {v2:.2f} rad/s -> OK? {abs(v2) < VELOCITY_MAX}\n")
        return False, np.nan 
        
    swingup_time = T[goal_indices[0]]
    
    dt = np.mean(np.diff(T))
    steps_in_half_second = int(0.5 / dt)
    window_size = min(steps_in_half_second, len(X) // 2)
    final_window = is_physically_goal_state[-window_size:]
    
    # Abbassato al 50%: basta che la metà dei frame finali sia dentro i limiti
    success_rate = np.mean(final_window)
    success = success_rate >= 0.50
    
    if not success:
        # ---- DIAGNOSTICA ----
        print(f"\n[DEBUG] FALLIMENTO: Uscito dal goal alla fine. Stabilità: {success_rate*100}% (Richiesto > 50%)")
        q1, q2, v1, v2 = X[-1]
        print(f"Ultimo stato -> cos(q1)={np.cos(q1):.2f}, cos(q2)={np.cos(q2):.2f}, v1={v1:.2f}, v2={v2:.2f}\n")
    
    return success, swingup_time
# ==========================================
# 2. METODO AGGREGATORE
# ==========================================

def evaluate_all_metrics(T, X, U, robot_type, in_goal_func, experiment_name=""):
    """
    Calcola tutte le metriche e restituisce un dizionario.
    Il parametro dt è calcolato automaticamente dal vettore T.
    """
    # 1. Converti in numpy array
    T = np.array(T)
    X = np.array(X)
    U = np.array(U)
    
    # 2. ALLINEAMENTO LUNGHEZZE (501 vs 500)
    # Se abbiamo uno stato in più rispetto alle azioni, togliamo l'ultimo stato
    if len(X) == len(U) + 1:
        X = X[:-1]
        T = T[:-1]
    elif len(X) != len(U):
        # Fallback di sicurezza: taglia alla lunghezza minore
        min_len = min(len(X), len(U))
        X = X[:min_len]
        T = T[:min_len]
        U = U[:min_len]

    # 3. ESTRAZIONE DELLA COPPIA ATTIVA (Il problema del "1000")
    # Se U ha due colonne (es. [tau_giunto_1, tau_giunto_2]), prendiamo solo quella del motore
    if U.ndim > 1 and U.shape[1] > 1:
        if robot_type.lower() == "pendubot":
            U_single = U[:, 0]  # Il motore del Pendubot è al giunto 1
        elif robot_type.lower() == "acrobot":
            U_single = U[:, 1]  # Il motore dell'Acrobot è al giunto 2
        else:
            U_single = U[:, 0]
    else:
        U_single = U.flatten()
        
    # Calcola dt medio
    dt = np.mean(np.diff(T))
    
    # 4. Calcolo metriche
    success, swingup_time = calc_swingup_metrics(T, X, in_goal_func)
    pos_error_deg, vel_jitter = calc_precision_error(T, X)
    
    # NOTA: Passiamo 'U_single' a tutte le funzioni per evitare altri errori
    metrics = {
        "Experiment": experiment_name,
        "Robot": robot_type.capitalize(),
        "Success": "Yes" if success else "No",
        "Swingup Time [s]": round(swingup_time, 3),
        "Final Pos Error [deg]": round(pos_error_deg, 2) if success else np.nan,
        "Final Jitter [rad/s]": round(vel_jitter, 3) if success else np.nan,
        "Energy [J]": round(calc_energy(U_single, X, dt, robot_type), 2),
        "Max Torque [Nm]": round(calc_max_torque(U_single), 2),
        "Int. Torque [Nm]": round(calc_integrated_torque(U_single, dt), 2),
        "Torque Cost [N^2m^2]": round(calc_torque_cost(U_single, dt), 2),
        "Torque Smoothness [Nm]": round(calc_torque_smoothness(U_single), 3),
        "Velocity Cost [m^2/s^2]": round(calc_velocity_cost(X, dt), 2)
    }
    
    return metrics

# ==========================================
# 3. METODO PER SALVARE SU FILE (CSV)
# ==========================================

def save_metrics_to_csv(metrics_dict, filename="./results/all_metrics.csv"):
    """
    Salva il dizionario delle metriche in un file CSV.
    Se il file non esiste, scrive anche l'intestazione (i nomi delle colonne).
    Se esiste, aggiunge semplicemente una nuova riga alla fine (modalità append).
    """
    # Assicurati che la cartella esista
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    file_exists = os.path.isfile(filename)
    
    # Apre in modalità 'a' (append)
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics_dict.keys())
        
        writer.writeheader()
            
        writer.writerow(metrics_dict)
    
    print(f"Metriche salvate correttamente in: {filename}")


# ==========================================
# 4. METODO "ALL-IN-ONE" (CALCOLA E SALVA)
# ==========================================

def evaluate_and_save(T, X, U, robot_type, in_goal_func, experiment_name="", filename="./results/all_metrics.csv"):
    """
    Funzione wrapper che calcola tutte le metriche e le salva direttamente 
    nel file CSV in un'unica chiamata.
    Restituisce anche il dizionario delle metriche nel caso serva stamparlo.
    """
    # 1. Calcola le metriche usando la funzione sicura (che gestisce numpy, flatten e allineamento)
    metrics = evaluate_all_metrics(T, X, U, robot_type, in_goal_func, experiment_name)
    
    # 2. Salva i risultati nel CSV
    save_metrics_to_csv(metrics, filename)
    
    # 3. Stampa un riepilogo a schermo (opzionale ma comodo)
    print(f"[{experiment_name} - {robot_type.capitalize()}] Elaborato e salvato!")
    
    return metrics