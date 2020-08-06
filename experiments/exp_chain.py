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

def main_aux(M : int, mus : np.ndarray, Sigmas : np.ndarray, env : environnement.Environment, eta : float, T : int):
    n_classes, n = mus.shape

    # current estimation of classes 
    true_classes = [np.random.choice(n_classes) for _ in range(M)]
    
    ws = np.zeros(shape=(M, n))
    for m in range(M):
        c = true_classes[m]
        ws[m] = np.random.multivariate_normal(mus[c], Sigmas[c])

    states, actions, observations = compute_trajectories_from_ws(ws, env, eta, T)
    
    infered_ws = np.zeros(shape=(M, n))

    # Etape 1 : Estimer les ws
    for m in range(M):

        obs_traj = trajectory.ObservedTrajectory(actions = actions[m], observations = observations[m])
        belief = inference.get_belief_from_observations(observations[m], actions[m], env)
        mle_traj = np.argmax(belief, axis=1)
        ctraj = trajectory.CompleteTrajectory(states = mle_traj, actions = actions[m], observations = observations[m])
        otraj = trajectory.ObservedTrajectory(actions = actions[m], observations = observations[m])

        try:
            # Methode non bayesienne
            # infered_ws[m] = inference.mle_w_from_observed_trajectory(otraj, eta, env)
            # Methode bayesienne 
            params = niw.MultivariateParams(mu = mus[true_classes[m]], Sigma = 100*Sigmas[true_classes[m]])
            infered_ws[m] = inference.map_w_from_observed_trajectory(ctraj, params, eta, env)
        except Exception as e:
            print('Optimization failed !', m)
    
    # a commenter eventuellement 
    if n == 2:
        colors = ['r', 'b', 'g', 'k', 'c', 'm']
        plt.scatter(infered_ws[:, 0], infered_ws[:, 1], c=true_classes, cmap=matplotlib.colors.ListedColormap(colors))
        plt.scatter(ws[:, 0], ws[:, 1], marker='+', c=true_classes, cmap=matplotlib.colors.ListedColormap(colors))
        plt.show()

    # classe pour estimer les parametres 
    # https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
    gaussianmixture = mixture.GaussianMixture(n_components=n_classes)
    gaussianmixture.fit(infered_ws)

    return ws, infered_ws, gaussianmixture.means_

@exp.config
def config():
    mus = [[1., 0.], [0., 100.]]
    N_trials = 10
    M = 50
    T = 50
    save_file = 'experiments/logs/exp_chain_default.npy'
    alpha = beta = 1.

@exp.automain
def main(N_trials : int, M : int, save_file : str, alpha : float, beta : float, T : int, mus : list):
    n_classes, n = len(mus), len(mus[0])
    # multiplier par deux car deux moyenne par essai
    infered_mus_trials = np.zeros(shape=(n_classes*N_trials, n))

    M = 50
    eta = 1.0
    
    env = chain.get_chain_env(S = 5, alpha = alpha, beta = beta, eps=.1)

    mus = np.array(mus)
    Sigmas = np.array([np.eye(n) for _ in range(n_classes)])

    for i in range(N_trials):
        debut = time.time()
        ws, infered_ws, infered_mus  = main_aux(M, mus, Sigmas, env, eta, T)
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
    