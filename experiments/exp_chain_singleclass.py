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
    n = mus.shape[-1]

    ws = np.zeros(shape=(M, n))
    for m in range(M):
        ws[m] = np.random.multivariate_normal(mus, Sigmas)

    states, actions, observations = compute_trajectories_from_ws(ws, env, eta, T)
    
    infered_ws = np.zeros(shape=(M, n))
    gaussianmixture = mixture.GaussianMixture()

    # Etape 1 : Estimer les ws
    for m in range(M):
        obs_traj = trajectory.ObservedTrajectory(actions = actions[m], observations = observations[m])
        infered_ws[m] = inference.mle_w_belief_propagation(obs_traj, eta, env)
        
    gaussianmixture.fit(infered_ws)

    return ws, infered_ws, gaussianmixture.means_

@exp.config
def config():
    N_trials = 10
    M = 50
    save_mus_file = 'experiments/logs/exp_chain.npy'
    alpha = beta = 1.
    mus = [0., 1.]

@exp.automain
def main(N_trials : int, M : int, save_file : str, alpha : float, beta : float, mus : list):
    n = 2
    # multiplier par deux car deux moyenne par essai
    infered_mus_trials = np.zeros(shape=(N_trials, 2))

    M, T = 50, 50
    eta = 1.0
    S = 5
    
    env = chain.get_chain_env(S = S, alpha = alpha, beta = beta, eps=.1)

    mus = np.array(mus)
    Sigmas = np.eye(2)

    for i in range(N_trials):
        debut = time.time()
        ws, infered_ws, infered_mus  = main_aux(M, mus, Sigmas, env, eta, T)
        infered_mus_trials[i] = infered_mus
        print('Trial #', i, ': ', time.time() - debut)

    mus_mixture = mixture.GaussianMixture()
    mus_mixture.fit(infered_mus_trials)

    params = {}
    params['alpha'] = alpha
    params['beta'] = beta
    params['env'] = 'exp_chain_singleclass.py'
    params['mus'] = mus.tolist()
    params['date'] = str(date.today())
    params['description'] = 'Experiment on chain with ' + str(S) + ' states and one class of MDP. Result is the estimations of the mus for ' + str(N_trials) + ' trials'

    logs.dump_results(save_file, infered_mus_trials, params)

    print("==== INFERED MUS (AVERAGE )====")
    print(mus_mixture.means_)
    print("==== TRUE MUS ====")
    print(mus)
    