# Essai du traitement non-Bayesien et avec l'algo d'EM

from datetime import date
import itertools
import sys
import time
import warnings
sys.path.append("..")

import numpy as np
from sklearn import mixture
import sklearn
import matplotlib.pyplot as plt
import matplotlib
import sacred
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

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
    return np.argmax([stats.multivariate_normal.pdf(x, mean=mus[c], cov=Sigmas[c], allow_singular=True)  for c in range(C)])

def main_aux(M : int, mus : np.ndarray, Sigmas : np.ndarray, env : environnement.Environment, eta : float, T : int, bayesian : bool):
    n_classes, n = mus.shape

    true_classes = [np.random.choice(n_classes) for _ in range(M)]
    ws = np.zeros(shape=(M, n))
    for m in range(M):
        c = true_classes[m]
        ws[m] = stats.multivariate_normal.rvs(mus[c], Sigmas[c] + 1e-5*np.eye(n))
    _, actions, observations = compute_trajectories_from_ws(ws, env, eta, T)
    trajectories = [trajectory.ObservedTrajectory(actions = actions[m], observations = observations[m]) for m in range(M)]

    K = 10
    
    # use em
    if not bayesian:
        infered_mus, infered_Sigmas, infered_classes, infered_ws = inference.em_pomdp(trajectories, n_classes, eta, env, n_iter=K, verbose=True)

    else:
        niw_prior = niw.default_niw_prior(n)
        dp_tau = 1. / n_classes
        infered_mus, infered_Sigmas, infered_classes, infered_ws = inference.bayesian_pomdp(trajectories, niw_prior, dp_tau, eta, env, n_iter=K, verbose=True)

    # on calcule les classe a la fin car la fonction gaussianmixture.fit fait son propre calcul des probas d'appartenance lors de l'EM
    infered_classes = list(map(lambda x : get_class(x, infered_mus, infered_Sigmas), infered_ws))

    # afficher l'accuracy 
    acc = sklearn.metrics.accuracy_score(true_classes, infered_classes)
    print('Accuracy is : ', max(acc, 1 - acc))

    # Affichage des resultats 
    if n == 3:
        colors = ['r', 'g', 'b', 'k', 'c', 'm']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(infered_ws[:, 0], infered_ws[:, 1], infered_ws[:, 2], c=infered_classes, cmap=matplotlib.colors.ListedColormap(colors))
        ax.scatter(ws[:, 0], ws[:, 1], ws[:, 2], marker='*', c=true_classes, cmap=matplotlib.colors.ListedColormap(colors))
        algo_name = 'EM-based'
        if bayesian:
            algo_name = 'hierarchical bayesian'
        plt.title('Inference of weights for environment with random features. ' + algo_name + ' algorithm')
        plt.show()

    return ws , infered_ws, infered_mus

@exp.config
def config():
    mus = [[1., 0., 0.], [0., 1., 0.]]
    N_trials = 10
    M = 50
    T = 50
    S = 5
    A = 2
    save_file = 'experiments/logs/exp_3_random_default.npy'
    # Use EM version of hierarchical bayesian version of algorithm
    bayesian = True

@exp.automain
def main(N_trials : int, M : int, save_file : str, T : int, mus : list, bayesian : bool):
    n_classes, n = len(mus), len(mus[0])
    
    env = environnement.get_observable_random_environment(S = 5, A = 2, O = 5, n = n)
    eta = 1.0

    mus = np.array(mus)
    Sigmas = np.array([0.1*np.eye(n) for _ in range(n_classes)])
        
    # multiplier par deux car deux moyenne par essai
    infered_mus_trials = np.zeros(shape=(n_classes*N_trials, n))
    
    for i in range(N_trials):
        debut = time.time()
        ws, infered_ws, infered_mus  = main_aux(M, mus, Sigmas, env, eta, T, bayesian)
        infered_mus_trials[n_classes*i:n_classes*(i+1)] = infered_mus
        print('Trial #', i, ': ', time.time() - debut, ' seconds')

    mus_mixture = mixture.GaussianMixture(n_components=n_classes)
    mus_mixture.fit(infered_mus_trials)

    params = {}
    params['env'] = 'exp_3_random_features.py'
    params['mus'] = mus.tolist()
    params['date'] = str(date.today())
    
    logs.dump_results(save_file, infered_mus_trials, params)

    print("==== INFERED MUS (AVERAGE )====")
    print(mus_mixture.means_)
    print("==== TRUE MUS ====")
    print(mus)
    