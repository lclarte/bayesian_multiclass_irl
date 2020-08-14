# Essai du traitement non-Bayesien et avec l'algo d'EM

import itertools
import sys
import time
import warnings
sys.path.append("..")

import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
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

import envs.chain as chain

exp = sacred.Experiment("experiment_1")

def compute_trajectories_from_ws(ws : np.ndarray, env : environnement.Environment, eta : float, T : int):
    M, n = ws.shape
    S, A, O = env.obsvn_matx.shape

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
    true_classes = [0]*int(M/2) + [1]*int(M/2)
    n = mus.shape[-1]
    
    ws = np.zeros(shape=(M, n))
    for m in range(M):
        c = true_classes[m]
        ws[m] = np.random.multivariate_normal(mus[c], Sigmas[c])

    _, actions, observations = compute_trajectories_from_ws(ws, env, eta, T)
    
    infered_ws = np.zeros(shape=(M, n))

    # classe pour estimer les parametres 
    # https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
    gaussianmixture = mixture.GaussianMixture(n_components=2)

    # Etape 1 : Estimer les ws
    for m in range(M):
        obs_traj = trajectory.ObservedTrajectory(actions = actions[m], observations = observations[m])
        infered_ws[m] = inference.mle_w(obs_traj, eta, env)
        
    gaussianmixture.fit(infered_ws)

    return ws, infered_ws, gaussianmixture.means_

@exp.config
def config():
    N_trials = 10
    M = 50
    save_mus_file = 'experiments/logs/exp_3_random_features.npy'
    T = 20

@exp.automain
def main(N_trials : int, M : int, save_mus_file : str, T : int):
    n = 3
    # multiplier par deux car deux moyenne par essai
    infered_mus_trials = np.zeros(shape=(2*N_trials, n))

    eta = 1.0
    
    env = environnement.get_observable_random_environment(S = 5, A = 2, O = 5, n = n)

    mus = np.array([[10., 10., 10.], [40., 0., 0.]])
    Sigmas = np.array([np.eye(n), np.eye(n)])

    for i in range(N_trials):
        debut = time.time()
        ws, infered_ws, infered_mus  = main_aux(M, mus, Sigmas, env, eta, T)
        infered_mus_trials[2*i:2*(i+1)] = infered_mus
        print('Trial #', i, ': ', time.time() - debut)

    # fichier pour sauvegarder les parametres

    mus_mixture = mixture.GaussianMixture(n_components=2)
    mus_mixture.fit(infered_mus_trials)

    np.save(save_mus_file, infered_mus_trials)

    print("==== INFERED MUS (AVERAGE )====")
    print(mus_mixture.means_)
    print("==== TRUE MUS ====")
    print(mus)