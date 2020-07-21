from scipy import stats, optimize
from scipy.special import logsumexp
import numpy as np

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
    certaine q_function.
    Fais les calculs dans "le log" pour eviter les inf et nan dus aux exponentielles trop grosses (exp(1000) par ex.)
    """
    # log du numerateur du softmax
    log_num = eta * q_function
    return_array = log_num - logsumexp(log_num, axis=1)[:, None] 
    return np.exp(return_array)

def sample_trajectory(rho_0, policy, transition, T) -> (np.ndarray, np.ndarray):
    """
    Rmque : renvoie un etat de plus que d'actions (car il n'y a pas d'action pour le dernier step de la traj)
    parameters:
        rho_0 : vecteur de taille S, distribution sur les etats
        policy : matrice de taille S x A, normalise sur le deuxieme axe
        transition : matrice S x A x S
    returns : 
        - states : liste de longueur (T+1), valeurs dans [0, S-1]
        - actions : liste de longueur T, valeurs dans [0, A-1]
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

def sample_trajectory_from_w(rho_0, w, basis, transition, gamma, eta, T):
    policy = softmax(q_function(linear_reward_function(w, basis), transition, gamma), eta)
    return sample_trajectory(rho_0, policy, transition, T)