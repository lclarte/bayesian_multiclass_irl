# fichier pour comparer les resultats de l'inference avec les vrais parametres

import numpy as np
import core.policy as policy

def quadratic_ws(true_ws : np.ndarray, infered_ws : np.ndarray):
    return np.linalg.norm(true_ws - infered_ws, ord = 2)

def quadratic_q(true_ws : np.ndarray, infered_ws : np.ndarray, features : np.ndarray, transition : np.ndarray, gamma : float):
    true_linear, infered_linear = policy.linear_reward_function(true_ws, features), policy.linear_reward_function(infered_ws, features)
    # shape des q : (s, a)
    true_q, infered_q = policy.q_function(true_linear, transition, gamma), policy.q_function(infered_linear, transition, gamma)
    return np.linalg.norm(true_q - infered_q, ord = np.inf)