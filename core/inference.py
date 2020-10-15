# inference.py
# all functions that infer a parameter from observation of an agent's trajectory 

from typing import List

import numpy as np
from scipy import stats, optimize, special
from sklearn import mixture
from matplotlib import cm

from core.environnement import *
from core.bp import *
from core.policy import *
from core.trajectory import *
from core.niw import *

# ================== HELPER FUNCTIONS =================

def get_class(x, mus, Sigmas) -> int:
    """
    retoune la classe associee au x
    """
    C = len(mus)
    return np.argmax([stats.multivariate_normal.pdf(x, mean=mus[c], cov=Sigmas[c], allow_singular=True)  for c in range(C)])

# ======================= COMPUTATION OF LIKELIHOODS ===========================

def complete_trajectory_log_likelihood(traj : CompleteTrajectory, w : np.ndarray, env : Environment, eta : float) -> float:
    """
    Pour une trajectoire complete, seule la quantite p(a_t | w) depend du vecteur de poids (le reste est un facteur constant de la traj.)
    """
    # retourne au format logarithmique car sinon pb d'echelle 
    assert traj.check_valid() == True, "Dimensions of trajectory is invalid"
    
    policy = softmax(q_function(linear_reward_function(w, env.features), env.trans_matx, env.gamma), eta)
    actions, states = traj.actions, traj.states
    T = len(actions)
    
    log_p = 0.
    for t in range(T):
        log_p += np.log(policy[states[t], actions[t]])
    return log_p

def observed_trajectory_log_likelihood(otraj : ObservedTrajectory, w : np.ndarray, env : Environment, eta : float) -> float:
    """
    Computes the log likelihood of a parameter w given the partially observed trajectory = states, actions & observations of an agent :
    log (p(states, observations | w )) 
    """
    assert otraj.check_valid() == True

    act, obs = otraj.actions, otraj.observations
    log_trans, log_obs = np.log(env.trans_matx), np.log(env.obsvn_matx)

    log_policy = np.log(softmax(q_function(linear_reward_function(w, env.features), env.trans_matx, env.gamma), eta))

    S = env.obsvn_matx.shape[0]
    # indices des states : 0, ..., T (T+1 etats dans la trajectoire)
    T = len(otraj.actions)

    # on fait tout en echelle logarithmique
    try:
        message = np.zeros(shape=(T, S))

        # faire le cas (T-1) -> T
        for s in range(S):
            message[T-1, s] = special.logsumexp(log_trans[s, act[T-1], :] + log_obs[:, act[T-1], obs[T-1]])

        # message passing de la fin au debut
        for t in range(T-2, -1, -1):
            for s in range(S):
                # mu_{t -> t+1}
                message[t, s] = special.logsumexp(log_trans[s, act[t], :] + log_policy[:, act[t+1]] + log_obs[:, act[t], obs[t]] + message[t + 1])

        # traiter le cas t == 0 (distribution initiale + p(a_0 | s_0))
        proba_0 = np.log(env.init_dist) + log_policy[:, act[0]] + message[0]
    except Exception as e:
        print('Error : ', e)
    return special.logsumexp(proba_0)

# ============================= ESTIMATION OF PARAMETER FROM TRAJECTORY  ===========================

def generic_map_w_from_trajectory(traj, params : MultivariateParams, eta : float, env : Environment, likelihood_func : callable):
    """
    Maximum  A Posteriori estimation of weight vector w from data (traj) and prior term (multivariate normal w/ params)
    """
    mu, Sigma = params.mu, params.Sigma
    features, _, _ = env.features, env.trans_matx, env.gamma
    _, _, n = features.shape
    
    def map_w_observed_aux(w):
        # Essai : sommer sur les trajectoires (coute cher)
        log_posterior_proba = likelihood_func(traj, w, env, eta)

        log_prior_proba = np.log(stats.multivariate_normal.pdf(w, mu, Sigma, allow_singular=True))
        return - log_posterior_proba - log_prior_proba
    
    x_initial = mu
    res = optimize.minimize(map_w_observed_aux, x0 = x_initial)

    return res.x

def map_w_from_observed_trajectory(traj : ObservedTrajectory, params : MultivariateParams, eta : float, env : Environment):
    return generic_map_w_from_trajectory(traj, params, eta, env, observed_trajectory_log_likelihood)

def map_w_from_complete_trajectory(traj : CompleteTrajectory, params : MultivariateParams, eta : float, env : Environment):
    return generic_map_w_from_trajectory(traj, params, eta, env, complete_trajectory_log_likelihood)