import numpy as np
from double_pendulum.controller.abstract_controller import AbstractController

class ProgressiveSwingController(AbstractController):
    def __init__(self, torque_limit):
        # Inizializza la classe genitore (fondamentale per il framework!)
        super().__init__()
        
        self.max_tau = torque_limit[0] if isinstance(torque_limit, (list, np.ndarray)) else torque_limit
        
        # Sequenza progressiva di angoli
        # base_targets = [
        #     np.pi/32, 
        #     np.pi/16, 
        #     np.pi/8, 
        #     np.pi/4, 
        #     np.pi/2, 
        #     3*np.pi/4, 
        #     np.pi
        # ]

        base_targets = [
            np.pi/32, 
            np.pi/14
        ]
        
        self.targets = []
        for t in base_targets:
            self.targets.append(t)   # Spinta a destra
            self.targets.append(-t)  # Spinta a sinistra
            
        self.current_target_idx = 0
        self.mode = "PUSHING"
        
    def init_(self):
        # METODO CON TRATTINO BASSO
        self.current_target_idx = 0
        self.mode = "PUSHING"

    def get_control_output_(self, x, t=None):
        # METODO CON TRATTINO BASSO
        q1 = (x[0] + np.pi) % (2*np.pi) - np.pi
        
        if self.current_target_idx >= len(self.targets):
            target_q1 = self.targets[-1]
        else:
            target_q1 = self.targets[self.current_target_idx]
            
        tau = 0.0
        
        if self.mode == "PUSHING":
            tau = self.max_tau if target_q1 > 0 else -self.max_tau
            
            # Controlla se abbiamo raggiunto l'obiettivo (con una piccola tolleranza o superamento)
            reached = (target_q1 > 0 and q1 >= target_q1) or (target_q1 < 0 and q1 <= target_q1)
            
            if reached:
                self.mode = "COASTING" # Smetti di spingere
                if self.current_target_idx < len(self.targets) - 1:
                    self.current_target_idx += 1
                    
        elif self.mode == "COASTING":
            tau = 0.0
            # Quando il pendolo torna al centro, riprendi a spingere per il prossimo target
            if abs(q1) < 0.1: 
                self.mode = "PUSHING"
                
        return np.array([tau, 0.0])
    



class ProgressiveFeedbackSwingController(AbstractController):
    # NOTA: Kp abbassato drasticamente, inserito 'torque_fraction'
    def __init__(self, torque_limit, kp=15.0, kd=1.85, torque_fraction=1.0):
        super().__init__()
        
        real_limit = torque_limit[0] if isinstance(torque_limit, (list, np.ndarray)) else torque_limit
        
        # TRUCCO: Limitiamo l'azione massima di swing-up a una frazione del limite reale.
        # Questo costringe il sistema a fare oscillazioni lente e aggraziate.
        self.max_tau = real_limit * torque_fraction
        
        # Guadagni molto più "morbidi"
        self.kp = kp  
        self.kd = kd  
        
        # Sequenza progressiva con gradini più fitti per una transizione fluida
        base_targets = [
            np.pi/16, 
            np.pi/8, 
            np.pi/6,
            np.pi/4, 
            np.pi/3,
            np.pi/2, 
            2*np.pi/3, 
            3*np.pi/4, 
            np.pi
        ]
        
        self.targets = []
        for t in base_targets:
            self.targets.append(t)   # Destra
            self.targets.append(-t)  # Sinistra
            
        self.current_target_idx = 0
        
    def init_(self):
        self.current_target_idx = 0

    def get_control_output_(self, x, t=None):
        q1 = (x[0] + np.pi) % (2*np.pi) - np.pi
        q1_dot = x[2]
        
        if self.current_target_idx >= len(self.targets):
            target_q1 = self.targets[-1]
        else:
            target_q1 = self.targets[self.current_target_idx]
            
        # 1. Calcolo dell'errore
        error = target_q1 - q1
        
        # 2. Controllo PD morbido
        tau = (self.kp * error) - (self.kd * q1_dot)
        
        # 3. Saturazione castrata (es. max 2 Nm invece di 5 Nm)
        tau = np.clip(tau, -self.max_tau, self.max_tau)
        
        # 4. Condizione di avanzamento resa più "rilassata" per assecondare la lentezza
        # Avanziamo quando è vicino e la velocità sta scendendo
        if abs(error) < 0.2 and abs(q1_dot) < 2.0:
            if self.current_target_idx < len(self.targets) - 1:
                self.current_target_idx += 1
                
        return np.array([tau, 0.0])