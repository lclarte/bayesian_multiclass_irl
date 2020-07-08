import numpy as np
import scipy.stats as stats

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

def sample_trajectory(rho_0, policy, transition, T):
    """
    Rmque : renvoie un etat de plus que d'actions (car il n'y a pas d'action pour le dernier step de la traj)
    parameters:
        rho_0 : vecteur de taille S, distribution sur les etats
        policy : matrice de taille S x A, normalise sur le deuxieme axe
        transition : matrice S x A x S
    """
    S, A = policy.shape
    states, actions = [np.random.choice(S, p=rho_0)], []
    for t in range(T):
        state = states[-1]
        action = np.random.choice(A, p=policy[state])
        next_state = np.random.choice(S, p=transition[state, action])
        states.append(next_state)
        actions.append(action)
    return states, actions

def sample_norm_inv_wish(mu_0, k_0, Sigma_0, nu_0, size):
    """
    Retourne un tuple (mu, Sigma) tires de la loi Normale x inverse Wishart. 
    parameters:
        - size : tuple (d0, ..., dn)
    returns : 
        - mu : np.array de taille (d0, ..., dn, N)
        - Sigma : np.array de taille (d0, ..., dn, N, N)
    """
    Sigma = stats.invwishart.rvs(nu_0, Sigma_0, size)
    mu = np.zeros(size + mu_0.shape)
    # iterate over all the covariance matrices
    it = np.nditer(np.zeros(size), flags=['multi_index'])
    for _ in it:
        indx = it.multi_index
        mu[indx] = stats.multivariate_normal.rvs(mean = mu_0, cov = Sigma[indx] / k_0)
    return mu, Sigma