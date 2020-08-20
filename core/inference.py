import pprint
from typing import List

import numpy as np
from scipy import stats, optimize, special
from sklearn import mixture

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
    Methode non bayesienne pour estimer les poids w a partir d'une trajectoire complete (states + actions + observation) 
    """
    features = env.features
    n = features.shape[-1]

    def minus_log_complete_likelihood(w):
        # terme de regularisation 
        retour = complete_trajectory_log_likelihood(ctraj, w, env, eta) 
        return - retour
    
    res = optimize.minimize(minus_log_complete_likelihood, x0 = np.zeros(n))
    return res.x

def mle_w_from_observed_trajectory(traj : ObservedTrajectory, eta : float, env : Environment):
    """
    Methode non bayesienne pour estimer les poids w a partir d'observations (pour l'instant obs + actions)
    """
    features, trans_matx, gamma = env.features, env.trans_matx, env.gamma
    _, _, n = features.shape
    
    def mle_w_observed_aux(w):
        # Essai : sommer sur les trajectoires (coute cher)
        log_posterior_proba = observed_trajectory_log_likelihood(traj, w, env, eta)
        return -log_posterior_proba
    
    res = optimize.minimize(mle_w_observed_aux, x0 = np.zeros(n))
    return res.x

def map_w_from_observed_trajectory(traj : ObservedTrajectory, params : MultivariateParams, eta : float, env : Environment):
    mu, Sigma = params.mu, params.Sigma
    features, _, _ = env.features, env.trans_matx, env.gamma
    _, _, n = features.shape
    
    def map_w_observed_aux(w):
        # Essai : sommer sur les trajectoires (coute cher)
        log_posterior_proba = observed_trajectory_log_likelihood(traj, w, env, eta)
        log_prior_proba = np.log(stats.multivariate_normal.pdf(w, mu, Sigma))
        return - log_posterior_proba - log_prior_proba
    
    res = optimize.minimize(map_w_observed_aux, x0 = mu)

    if not res.success:
        pass 
        # print('Optimization failed :', res.message)
    return res.x

# FONCTIONS PRINCIPALES 

def get_class(x, mus, Sigmas) -> int:
    """
    retoune la classe associee au x
    """
    C = len(mus)
    return np.argmax([stats.multivariate_normal.pdf(x, mean=mus[c], cov=Sigmas[c])  for c in range(C)])

def bayesian_pomdp(trajectories : ObservedTrajectory, niw_prior : NIWParams, dp_tau : float, eta : float, env : Environment, n_iter : int, *, verbose : bool = False):
    """
    Variante 1 de l'algorithme : 
    Methode bayesienne
    arguments : 
        - dp_tau : prior sur le nombre de classes pour le process de Dirichlet
    """
    M = len(trajectories)
    _, _, n = env.features.shape
    means_prior = niw_prior.mu_mean
    precisions_prior = niw_prior.mu_scale
    covariances_prior = niw_prior.Sigma_mean
    # dof = degrees of freedom pour le sampling de la matrice de covariance
    dofs_prior = niw_prior.Sigma_scale
    gaussianmixture = mixture.BayesianGaussianMixture(weight_concentration_prior=dp_tau, mean_prior=means_prior, mean_precision_prior=precisions_prior,
                                                      covariance_prior=covariances_prior, degrees_of_freedom_prior=dofs_prior)

    infered_ws = np.zeros(shape=(M, n))
    infered_classes = np.zeros(shape=(M, ), dtype=int)

    # initialement, une seule classe et un seul parametre
    infered_mus = np.zeros(shape=(1, n))
    infered_Sigmas = np.array([covariances_prior])

    for k in range(n_iter):
        if verbose:
            print('Iteration ', k)
            print('infered_mus : ', infered_mus)
            print('infered_Sigmas : ', infered_Sigmas)

        # step 1 : compute all ws 
        for m in range(M):
            c = infered_classes[m]
            params = MultivariateParams(mu = infered_mus[c], Sigma = infered_Sigmas[c])  
            infered_ws[m] = map_w_from_observed_trajectory(trajectories[m], params, eta, env)
        
        # step 2 : update all class assignements + parameters
        gaussianmixture.fit(infered_ws)

        infered_mus = gaussianmixture.means_
        # infered_Sigmas will be the average value of the covariance on inverse wishart distribution : 
        # E(Sigma) = Sigma_0 / (nu - dim- 1)
        infered_Sigmas = gaussianmixture.covariances_

    return infered_mus, infered_Sigmas, infered_classes, infered_ws


def em_pomdp(trajectories : List[ObservedTrajectory], n_classes : int, eta : float, env : Environment, n_iter : int, *, verbose : float = False):
    """
    Variante 2 de l'algorithme :
    Methode non bayesienne, basee sur l'EM
    """
    _, _, n = env.features.shape
    gaussianmixture = mixture.GaussianMixture(n_components=n_classes)
    M = len(trajectories)

    infered_ws = np.zeros(shape=(M, n))
    infered_classes = np.zeros(shape=(M, ), dtype=int)

    infered_mus = np.zeros(shape=(n_classes, n))
    infered_Sigmas = np.array([100*np.eye(n) for _ in range(n_classes)])

    for k in range(n_iter):
        if verbose:
            print('Iteration ' + str(k))
            print('infered_mus = ' + str(infered_mus))
            print('infered_Sigmas = ' + str(infered_Sigmas))

        # step 1 : compute all ws 
        for m in range(M):
            c = infered_classes[m]
            params = MultivariateParams(mu = infered_mus[c], Sigma = infered_Sigmas[c])  
            infered_ws[m] = map_w_from_observed_trajectory(trajectories[m], params, eta, env)

        # step 2 : update all class assignements + parameters
        gaussianmixture.fit(infered_ws)
        infered_mus = gaussianmixture.means_
        infered_Sigmas = gaussianmixture.covariances_

        infered_classes = list(map(lambda x : get_class(x, infered_mus, infered_Sigmas), infered_ws))

    return infered_mus, infered_Sigmas, infered_classes, infered_ws