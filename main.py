# TODO : Faire l'algo. pour le decoding du HMM (par exemple avec Viterbi)
# TODO : Code pour un env. de RL avec matrice de transition / distribution d'observation custom 
# RMK : Dans un premier temps, on suppose que S et A sont finis, dans un ensemble [0, |S| - 1 ] et [0, |A| - 1]
# On peut donc tout stocker sous forme matricielle (reward_function et q_function)
# RMK : Dans le IRL pour un POMDP, on a une observation w = f(s_t+1, a_t) et les actions a_t

import pymc3 as pm 
from scipy import stats, optimize
import numpy as np
import matplotlib.pyplot as plt
import itertools

def mh_transition_trajectories(current_traj, candidate_traj, observations, policy, trans_matx, obs_matx, initial_distribution):
    """
    A partir de deux trajectoires, accepte l'une ou l'autre selon la formule de transition de Metropolis Hastings
    parameters :   
        - current_traj et candidate_traj : dictionnaires avec des champs states et actions
        i.e current_traj['actions'] et current_traj['states']
    returns:
        - booleen si on accepte (ou non) de changer de trajectoire
    """

    current_states, current_actions = current_traj['states'], current_traj['actions']
    candidate_states, candidate_actions = candidate_traj['states'], candidate_traj['actions']

    like_current = trajectory_conditional_likelihood(current_states, current_actions, observations, policy, trans_matx, obs_matx, initial_distribution)
    like_candidate = trajectory_conditional_likelihood(candidate_states, candidate_actions, observations, policy, trans_matx, obs_matx, initial_distribution)

    return np.random.binomial(n=1, p = min(1., like_candidate / like_current))
    
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

def trajectory_conditional_likelihood(states, actions, observations, policy, trans_matx, obs_matx, initial_distribution):
    """
    Calcule la vraisemblance (non normalisee par p(o_t | w) !!!!) d'une trajectoire conditionne aux observations et au vecteur w qui nous donne 
    la reward function
    parameters : 
        - states, actions : tableaux (T+1, S) et (T, A)
        - observations    : tableau (T, O)
        - policy          : matrice S x A 
        - trans_matx      : matrice de transition (S x A x S)
        - obs_matx        : matrice des observations (S x A x O)
    """
    assert len(states) == len(actions) + 1
    T = len(actions)

    # calculer p(\tau | policy )
    p_tau_policy = initial_distribution[states[0]]
    for t in range(T):
        s, a = states[t], actions[t]
        p_tau_policy *= policy[s, a] * trans_matx[s, a, states[t+1]]

    # calculer p( p_{o_t} | \tau )
    p_obs_tau = 1.
    for t in range(T):
        s, a, o = states[t+1], actions[t], observations[t]
        p_obs_tau *= obs_matx[s, a, o]

    return p_tau_policy * p_obs_tau

def trajectory_likelihood_policy(states, actions, policy):
    """
    Retourne la vraisemblance de la trajectoire donnee en fonction de la policy, c'est a dire le produit 
    des probas \pi( a_t | s_t ), ne prend pas en compte la distribution initiale ni la transition de l'
    environnement. 
    """
    likelihood = 1.
    T = len(actions)
    for t in range(T):
        likelihood *= policy[states[t], actions[t]]
    return likelihood

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

def get_trajectory_reward(states, actions, reward_function):
    """
    parameters: 
        - reward_function : S x A matrix
    """
    # le tableau states contient l'etat final pour lequel aucune action a ete prise
    assert len(states) == len(actions) + 1
    return sum(reward_function[states[i], actions[i]] for i in range(len(actions)))

def get_belief_from_observations(observations, actions, transition, obs_matrix, initial_distribution):
    """
    retourne une distributions sur les trajectoires a partir des observations de cette derniere
    parameters :
        - observations : matrice T x O
        - actions      : matrice      T x A
        - transition   : matrice de transition S x A x S   
        - obs_matrix   : matrice S x A x O (s_t+1, a_t) -> o_t
        - initial_distribution : vecteur de taille S, proba sur l'instant initial s_0
    returns :
        - trajectory   : matrice T x S des belief a chaque instant 
    """
    # Rque : meme longueur observations et actions, car le premier etat de la trajectoire 
    # n'est associe a aucune action
    T = len(observations)
    S, A, O = obs_matrix.shape
    trajectory = np.zeros(shape=(T+1, S))
    # taille T+1 car le premier etat est connu

    # rappelons qu'on ne peut pas connaitre exactement le premier etat
    trajectory[0] = initial_distribution # pour l'instant, un dirac
    # exemple : la premiere observation / action (indice 0) sert a estimer le second etat (indice 1) 
    for t in range(1, T+1):
        a, w = actions[t-1], observations[t-1]
        
        for s2 in range(S):
            trajectory[t, s2] = obs_matrix[s2, a, w] * transition[:, a, s2].dot(trajectory[t-1])
    return trajectory

def trajectories_to_state_occupation(trajectories, S):
    """
    arguments:
        - trajectories : Array de longueur N (nb de trajectoires), chaque element est un array de longueur T etats
    returns : 
        - empirical_occupation : moyenne de l'occupation de l'etat s au temps T sur l'ensemble des trajectoires 
    """
    T = len(trajectories[0])
    avg_occ = np.zeros((T, S))
    for traj in trajectories:
        for t in range(T):
            avg_occ[t, traj[t]] += 1.
    return avg_occ / float(len(trajectories))

def map_w_from_map_trajectory(states, actions, mu, Sigma, basis, transition, gamma, eta):
    n = basis.shape[-1]
    def minus_penalized_likelihood(w):
        policy = softmax(q_function(linear_reward_function(w , basis), transition, gamma), eta)
        retour = trajectory_likelihood_policy(states, actions, policy) * stats.multivariate_normal.pdf(w, mean=mu, cov=Sigma) 
        return -retour 
    res = optimize.minimize(minus_penalized_likelihood, x0=np.zeros(n))
    return res.x