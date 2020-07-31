from typing import List

import numpy as np
from scipy import stats, optimize, special

from core.environnement import *
from core.bp import *
from core.policy import *
from core.trajectory import *
from core.niw import *

def trajectory_conditional_likelihood(states, obstraj : ObservedTrajectory, policy, env : Environment):
    """
    Calcule la vraisemblance (non normalisee par p(o_t | w) !!!!) d'une trajectoire conditionne aux observations et au vecteur w qui nous donne 
    la reward function
    parameters : 
        - states, actions : tableaux (T+1, S) et (T, A)
        - observations    : tableau (T, O)
        - policy          : matrice S x A 
        - env 
    """
    T = len(states) - 1

    initial_distribution, trans_matx, obs_matx = env.init_dist, env.trans_matx, env.obsvn_matx
    actions, observations = obstraj.actoins, obstraj.observations

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

def complete_trajectory_log_likelihood(traj : CompleteTrajectory, w : np.ndarray, env : Environment, eta : float) -> float:
    # retourne au format logarithmique car sinon pb d'echelle 
    assert traj.check_valid() == True, "Dimensions of trajectory is invalid"
    
    policy = softmax(q_function(linear_reward_function(w, env.features), env.trans_matx, env.gamma), eta)
    actions, states, observations = traj.actions, traj.states, traj.observations
    T = len(actions)
    
    log_p = 0.
    log_p += np.log(env.init_dist[states[0]])
    for t in range(T):
        log_p += np.log(policy[states[t], actions[t]])
        log_p += np.log(env.trans_matx[states[t], actions[t], states[t+1]])
        log_p += np.log(env.obsvn_matx[states[t+1], actions[t], observations[t]])
    return log_p

def observed_trajectory_log_likelihood(otraj : ObservedTrajectory, w : np.ndarray, env : Environment, eta : float) -> float:
    """
    log likelihood d'une trajectoire observation = actions & observations en fonction de w
    """
    assert otraj.check_valid() == True

    S, A, O = env.obsvn_matx.shape
    T = len(otraj.actions)

    total_log_p = 0.
    size = (S, )* (T+1)
    # iterate over all possible tuples
    it = np.nditer(np.zeros(size), flags=['multi_index'])
    for _ in it:
        indx = it.multi_index
        states = np.array(indx)
        traj = CompleteTrajectory(actions = otraj.actions, observations = otraj.observations, states = states)
        total_log_p += complete_trajectory_log_likelihood(traj, w, env, eta)
    return total_log_p

def get_trajectory_reward(states, actions, reward_function):
    """
    parameters: 
        - reward_function : S x A matrix
    """
    # le tableau states contient l'etat final pour lequel aucune action a ete prise
    assert len(states) == len(actions) + 1
    return sum(reward_function[states[i], actions[i]] for i in range(len(actions)))

def get_belief_from_observations(observations, actions, env : Environment):
    """
    retourne une distributions sur les trajectoires a partir des observations de cette derniere
    parameters :
        - observations : matrice T x O
        - actions      : matrice      T x A
        - env          : environnement
    returns :
        - trajectory   : matrice T x S des belief a chaque instant 
    """
    # Rque : meme longueur observations et actions, car le premier etat de la trajectoire 
    # n'est associe a aucune action
    obs_matrix, trans_matx, initial_distribution = env.obsvn_matx, env.trans_matx, env.init_dist

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
            trajectory[t, s2] = obs_matrix[s2, a, w] * trans_matx[:, a, s2].dot(trajectory[t-1])
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

"""
Ci-dessous, inference du vecteur de poids w en fonction des trajectoires observees / etats inferes 
"""

def mle_w_from_complete_trajectory(ctraj : CompleteTrajectory, eta : float, env : Environment):
    """ 
    Methode non bayesienne pour estimer les parametres 
    """
    features = env.features
    n = features.shape[-1]

    def minus_log_complete_likelihood(w):
        retour = complete_trajectory_log_likelihood(ctraj, w, env, eta)
        return - retour
    
    res = optimize.minimize(minus_log_complete_likelihood, x0 = np.zeros(n))
    return res.x

def mle_w_belief_propagation(traj : ObservedTrajectory, eta : float, env : Environment):
    """
    Retourne le MLE (en utilisant de la gradient descent) de w  \sum_{trajs} p(obs | traj) * p(traj | w)
    Ne necessite pas de deternminer le MLE des etats
    """
    features, trans_matx, gamma = env.features, env.trans_matx, env.gamma
    _, _, n = features.shape
    
    def w_exact_posterior(w):
        policy = softmax(q_function(linear_reward_function(w , features), trans_matx, gamma), eta)
        unary, binary = get_chain_potentials(traj, policy, env)
        log_posterior_proba = compute_chain_normalization(np.log(unary), np.log(binary))
        return - log_posterior_proba
    
    res = optimize.minimize(w_exact_posterior, x0 = np.zeros(n))
    
    if not res.success:
        raise Exception()
    return res.x
