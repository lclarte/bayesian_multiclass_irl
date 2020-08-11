# Essai du traitement non-Bayesien et avec l'algo d'EM

from datetime import date
import itertools
import sys
import time
import warnings
sys.path.append("..")

import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib
import sacred
import scipy.stats as stats

import core.niw as niw
import core.inference as inference
import core.gibbs_class as gibbs_class
import core.dirichletprocess as dp
import core.environnement as environnement
import core.gibbs_class as gibbs_class
import core.trajectory as trajectory
import core.policy as policy
import core.metrics as metrics
import core.logs as logs
import core.maxentirl as maxentirl
from . import visualization

import envs.chain as chain

exp = sacred.Experiment("experiment_1")

def compute_trajectories_from_ws(ws : np.ndarray, env : environnement.Environment, eta : float, T : int):
    M = len(ws)
    
    states = np.zeros(shape = (M, T+1), dtype=int)
    actions = np.zeros(shape = (M, T), dtype=int)
    observations = np.zeros(shape = (M, T), dtype=int)

    for m in range(M):
        states[m], actions[m] = policy.sample_trajectory_from_w(env.init_dist, ws[m], env.features, env.trans_matx, env.gamma, eta, T)
        observations[m] = environnement.get_observations_from_states_actions(states[m], actions[m], env.obsvn_matx)

    return states, actions, observations

def get_class(x, mus, Sigmas) -> int:
    """
    retoune la classe associee au x
    """
    C = len(mus)
    return np.argmax([stats.multivariate_normal.pdf(x, mean=mus[c], cov=Sigmas[c])  for c in range(C)])

def main_aux(M : int, mus : np.ndarray, Sigmas : np.ndarray, env : environnement.Environment, eta : float, T : int, fixed_params : bool):
    n_classes, n = mus.shape
    niw_params = niw.default_niw_prior(n)

    true_classes = [np.random.choice(n_classes) for _ in range(M)]
    # current estimation of classes 
    infered_classes = [np.random.choice(n_classes) for _ in range(M)]

    ws = np.zeros(shape=(M, n))
    for m in range(M):
        c = true_classes[m]
        ws[m] = np.random.multivariate_normal(mus[c], Sigmas[c])

    _, actions, observations = compute_trajectories_from_ws(ws, env, eta, T)
    infered_mus, infered_Sigmas = np.zeros(shape=(n_classes, n)), [ np.eye(n) for _ in range(n_classes) ]

    infered_map_ws = np.zeros(shape=(M, n))

    K = 10
 
    gaussianmixture = mixture.GaussianMixture(n_components=2)

    if not fixed_params:
        gaussianmixture = mixture.BayesianGaussianMixture(n_components=2, mean_prior=niw_params.mu_mean, mean_precision_prior=niw_params.mu_scale, degrees_of_freedom_prior=niw_params.Sigma_scale,
                                                                    covariance_prior=niw_params.Sigma_mean)
                                                            
    for k in range(K):

        print('iteration #', k)
        print('infered_mus : ', infered_mus)
        print('infered_Sigmas : ', infered_Sigmas)

        # Etape 1 : Estimer les ws
        for m in range(M):
            otraj = trajectory.ObservedTrajectory(actions = actions[m], observations = observations[m])
        
            # Methode bayesienne (dans un premier temps, on fait l'hypothese qu'on connait les classes)
            params = niw.MultivariateParams(mu = infered_mus[true_classes[m]], Sigma = infered_Sigmas[true_classes[m]])
            infered_map_ws[m] = inference.map_w_from_observed_trajectory(otraj, params, eta, env)

        # infer parameters of gaussian mixture
        gaussianmixture.fit(infered_map_ws)
        infered_mus, infered_Sigmas = gaussianmixture.means_, (gaussianmixture.covariances_ / (gaussianmixture.degrees_of_freedom_ - n - 1))
        
    # on calcule les classe a la fin car la fonction gaussianmixture.fit fait son propre calcul des probas d'appartenance lors de l'EM
    infered_classes = list(map(lambda x : get_class(x, infered_mus, infered_Sigmas), infered_map_ws))

    # a commenter eventuellement 
    if n == 2:
        colors = ['r', 'b', 'g', 'k', 'c', 'm']
        plt.scatter(infered_map_ws[:, 0], infered_map_ws[:, 1], c=infered_classes, cmap=matplotlib.colors.ListedColormap(colors))
        plt.show()

    return ws , infered_map_ws, gaussianmixture.means_

@exp.config
def config():
    mus = [[1., 0.], [0., 1.]]
    N_trials = 10
    M = 50
    T = 50
    save_file = 'experiments/logs/exp_chain_default.npy'
    alpha = beta = 1.
    delta = gamm = 0.
    fixed_params = True

@exp.automain
def main(N_trials : int, M : int, save_file : str, alpha : float, beta : float, delta : float, gamm : float, T : int, mus : list, fixed_params : bool):
    n_classes, n = len(mus), len(mus[0])
    env = chain.get_chain_env(S = 5, alpha = alpha, beta = beta, delta = delta, gamma = gamm, eps=.1)
    eta = 1.0

    mus = np.array(mus)
    Sigmas = np.array([.1*np.eye(n) for _ in range(n_classes)])

    niw_params = niw.default_niw_prior(n)
    # sample from normal inverse wishart
    if not fixed_params:
        mus, Sigmas = niw.sample_niw(niw_params, size=(2, ))
    
    # multiplier par deux car deux moyenne par essai
    infered_mus_trials = np.zeros(shape=(n_classes*N_trials, n))
    
    for i in range(N_trials):
        debut = time.time()
        ws, infered_ws, infered_mus  = main_aux(M, mus, Sigmas, env, eta, T, fixed_params)
        infered_mus_trials[n_classes*i:n_classes*(i+1)] = infered_mus
        print('Trial #', i, ': ', time.time() - debut, ' seconds')

    mus_mixture = mixture.GaussianMixture(n_components=n_classes)
    mus_mixture.fit(infered_mus_trials)

    params = {}
    params['alpha'] = alpha
    params['beta'] = beta
    params['env'] = 'exp_chain.py'
    params['mus'] = mus.tolist()
    params['date'] = str(date.today())
    
    logs.dump_results(save_file, infered_mus_trials, params)

    print("==== INFERED MUS (AVERAGE )====")
    print(mus_mixture.means_)
    print("==== TRUE MUS ====")
    print(mus)
    