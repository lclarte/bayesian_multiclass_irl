from scipy import stats, optimize
import numpy as np
from model import *

def linear_reward_function(w, basis):
    """
    Returns a S x A matrix representing the reward function 
    parameters:
        - basis : base de fonctions : tenseur de taille S x A x n 
        - w : vecteur de taille n 
    """
    return np.dot(basis, w)

def q_function(reward_function, transition, gamma):
    """
    Calcule la q_function optimale par value iteration
    transition : array of size S * A * S normalized on axis 2 
    """
    S, A = reward_function.shape
    q = np.empty_like(reward_function)
    nb_iter = 50
    for n in range(nb_iter):
        for s in range(S):
            for a in range(A):
                q[s, a] = reward_function[s, a] + gamma * transition[s, a, :].dot(np.amax(q, axis=1))
    return q

def softmax(q_function, eta):
    """
    Retourne la matrice np.exp(\eta * Q[s, a]) / (Normalisation_sur_a). Cette matrice represente la policy avec une 
    certaine q_function 
    """
    e = np.exp(eta * q_function)
    return e / e.sum(axis=1)[:, None]