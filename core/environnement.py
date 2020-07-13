import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from typing import NamedTuple

class Environment(NamedTuple):
    """
    Classe contenant les probas de transition et d'observation et les features pour la reward 
    Attention a ce que les matrices soient by normalisees
    Taille des matrices : 
        - trans_matx : S x A x S
        - obsvn_matx : S x A x O
        - features   : S x A x n
        - init_dist  : S
    """
    trans_matx : np.ndarray
    obsvn_matx : np.ndarray
    features   : np.ndarray
    init_dist  : np.ndarray
    gamma      : float = 0.9

    def check_compatible_sizes(self):
        S1, A1, S2 = self.trans_matx.shape
        S3, A2, O  = self.obsvn_matx.shape
        S4, A3, _  = self.features.shape
        S5         = self.init_dist.shape

        return (S1 == S2 == S3 == S4 == S5) and (A1 == A2 == A3)


# FONCTIONS POUR SAMPLER DES MATRICES POUR L'ENVIRONNEMENT

def random_transition_matrix(S, A):
    """
    sample random transition matrix 
    parameters : 
        -S : nb d'etats
        -A : nb d'actions
    """
    trans = np.random.rand(S, A, S) # s, a, s'
    for s in range(S):
        trans[s] = trans[s] / np.sum(trans[s], axis = 1)[:, np.newaxis]
    return trans

def random_observation_matrix(S, A, O):
    obs = np.random.rand(S, A, O)
    for s in range(S):
        obs[s] = obs[s] / np.sum(obs[s], axis = 1)[:, np.newaxis]
    return obs

def noisy_id_observation_matrix(S, A, O,  eps=0.1):
    """
    Construit une matrice d'observations proche de l'identite legerement perturbee
    arguments:
        -eps : perturbation de la matrice identite
    """
    assert S == O
    obs = np.zeros((S, A, O))
    for a in range(A):
        obs[:, a, :] = (1. - eps)*np.eye(S) + (eps / S) * np.ones((S, O))
    return obs

def get_observations_from_states_actions(states, actions, obsvn_matx) -> np.ndarray:
    assert len(states) == len(actions) + 1
    T = len(actions)
    S, A, O = obsvn_matx.shape
    # o_t = f(a_t, s_{t+1})
    return np.array([np.random.choice(O, p=obsvn_matx[states[i+1], actions[i]]) for i in range(T)])