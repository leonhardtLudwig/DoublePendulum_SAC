import numpy as np
from double_pendulum.controller.abstract_controller import AbstractController


class TripleController(AbstractController):
    """
    Controller that alternates between swing-up and LQR until the desired
    position is reached, then activates a third controller (stabilizer)
    and remains locked there with no further switching.

    Parameters
    ----------
    controller1 : Controller
        Swing-up controller (e.g., ProgressiveFeedbackSwingController)
    controller2 : Controller
        LQR controller used to maintain the position
    controller3 : Controller
        Final stabilizing controller (e.g., robust LQR)
    condition1 : function(t, x)
        Condition for switching to controller1
    condition2 : function(t, x)
        Condition for switching to controller2
    condition_goal : function(t, x)
        Condition for activating the lock on controller3.
        Once True, no further switching will occur.
    verbose : bool
        If True, prints controller transitions
    """

    def __init__(self, controller1, controller2, controller3,
                 condition1, condition2, condition_goal,
                 compute_both=False, verbose=False):
        super().__init__()
        
        self.controllers = [controller1, controller2, controller3]
        self.active = 0
        self.conditions = [condition1, condition2]
        self.condition_goal = condition_goal
        self.compute_both = compute_both
        self.verbose = verbose
        
        self.locked = False 
        self.lock_time = None

    def init_(self):
        """Inizializza tutti i tre controllori"""
        for c in self.controllers:
            c.init_()
        self.active = 0
        self.locked = False
        self.lock_time = None

    def set_parameters(self, controller1_pars, controller2_pars, controller3_pars):
        """Setta i parametri dei tre controllori"""
        self.controllers[0].set_parameters(*controller1_pars)
        self.controllers[1].set_parameters(*controller2_pars)
        self.controllers[2].set_parameters(*controller3_pars)

    def set_start(self, x):
        """Setta lo stato iniziale per i tre controllori"""
        for c in self.controllers:
            c.set_start(x)

    def set_goal(self, x):
        """Setta l'obiettivo per i tre controllori"""
        for c in self.controllers:
            c.set_goal(x)

    def save_(self, save_dir):
        """Salva i parametri di tutti e tre i controllori"""
        self.controllers[0].save_(save_dir)
        self.controllers[1].save_(save_dir)
        self.controllers[2].save_(save_dir)

    def reset_(self):
        """Resetta tutti e tre i controllori"""
        for c in self.controllers:
            c.reset_()

    def get_control_output_(self, x, t):
        """
        Switching logic:
        1. If locked = True, use controller3 only (NO SWITCHING).
        2. If condition_goal is True, activate the lock on controller3.
        3. Otherwise, alternate between controller1 and controller2.
        """
        
        # Are we in final regime?
        if not self.locked and self.condition_goal(t, x): 
            self.locked = True
            self.lock_time = t
            self.active = 2  
            if self.verbose:
                print(f"🎯 LOCKED! Attivato controllore stabilizzatore a t={t:.2f}s")
            return self.controllers[2].get_control_output_(x, t)
        
        if self.locked:
            return self.controllers[2].get_control_output_(x, t)
        
        inactive = 1 - self.active
        
        if self.conditions[inactive](t, x): 
            self.active = 1 - self.active
            if self.verbose:
                print(f"↔️  Switch a Controller {self.active + 1} a t={t:.2f}s")
        
        if self.compute_both:
            _ = self.controllers[inactive].get_control_output_(x, t)
        
        return self.controllers[self.active].get_control_output_(x, t)

    def get_forecast(self):
        return self.controllers[self.active].get_forecast()

    def get_init_trajectory(self):
        return self.controllers[self.active].get_init_trajectory()