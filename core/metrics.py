# fichier pour comparer les resultats de l'inference avec les vrais parametres

import numpy as np
import core.policy as policy

def quadratic_ws(true_ws : np.ndarray, infered_ws : np.ndarray):
    return np.linalg.norm(true_ws - infered_ws, ord = 2)

def quadratic_reward(true_ws : np.ndarray, infered_ws : np.ndarray, features : np.ndarray):
    """
    # TODO : Ne pas utiliser pour l'instant, car la taille des matrices ne correspond pas
    """
    true_linear, infered_linear = policy.linear_reward_function(true_ws, features), policy.linear_reward_function(infered_ws, features)
    return np.linalg.norm(true_linear - infered_linear, ord = 2)