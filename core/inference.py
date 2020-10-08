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

def map_w_from_observed_trajectory(traj : ObservedTrajectory, params : MultivariateParams, eta : float, env : Environment):
    """
    Maximum  A Posteriori estimation of weight vector w from data (traj) and prior term (multivariate normal w/ params)
    """
    mu, Sigma = params.mu, params.Sigma
    features, _, _ = env.features, env.trans_matx, env.gamma
    _, _, n = features.shape
    
    def map_w_observed_aux(w):
        # Essai : sommer sur les trajectoires (coute cher)
        log_posterior_proba = observed_trajectory_log_likelihood(traj, w, env, eta)

        log_prior_proba = np.log(stats.multivariate_normal.pdf(w, mu, Sigma, allow_singular=True))
        return - log_posterior_proba - log_prior_proba
    
    x_initial = mu
    res = optimize.minimize(map_w_observed_aux, x0 = x_initial)

    return res.x

# ========================== MAIN FUNCTIONS ===============================

# TODO : Faire une classe "HBPOMDP"

def bayesian_pomdp(trajectories : ObservedTrajectory, niw_prior : NIWParams, dp_tau : float, eta : float, env : Environment, n_iter : int, *, verbose : bool = False):
    """
    Hierarchical Bayesian algorithm, makes use of the normal inv. wishart prior for the distribution of the weight vector. 
    NIW -> Multivariate Gaussian -> weight vector w 
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


# TODO : Faire une classe "EMPOMDP" dont la fonction fit() change les attribus (infered_mus, infered_Sigmas, infered_classes, infered_ws)

def em_pomdp(trajectories : List[ObservedTrajectory], n_classes : int, eta : float, env : Environment, n_iter : int, *, verbose : float = False, Sigmas_norms : list = None):
    """
    Non hierarchical bayesian version of the algorithm : components of the gaussian mixture are identified with EM
    """
    if Sigmas_norms is None:
        Sigmas_norms = []
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
        Sigmas_norms.append(min(np.linalg.norm(infered_Sigmas[i], ord=2) for i in range(2)))

        infered_classes = list(map(lambda x : get_class(x, infered_mus, infered_Sigmas), infered_ws))

    return infered_mus, infered_Sigmas, infered_classes, infered_ws