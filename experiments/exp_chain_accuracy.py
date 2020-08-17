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
    
    states = np.zeros(shape=(M, T+1), dtype=int)
    actions = np.zeros(shape=(M, T), dtype=int)
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

def main_aux(M : int, mus : np.ndarray, Sigmas : np.ndarray, env : environnement.Environment, eta : float, Ts):
    n_classes, n = mus.shape
    em_accuracies = []

    for t in Ts:
        true_classes = [np.random.choice(n_classes) for _ in range(M)]
        ws = np.zeros(shape=(M, n))
        for m in range(M):
            c = true_classes[m]
            ws[m] = np.random.multivariate_normal(mus[c], Sigmas[c])
        _, actions, observations = compute_trajectories_from_ws(ws, env, eta, t)
        trajectories = [trajectory.ObservedTrajectory(actions = actions[m], observations = observations[m]) for m in range(M)]

        K = 10

        _, _, em_classes, _ = inference.em_pomdp(trajectories, n_classes, eta, env, n_iter=K, verbose=False)
        em_acc = sklearn.metrics.accuracy_score(true_classes, em_classes)
        em_accuracies.append(max(em_acc, 1-em_acc))
        print('With T = ', t, ', accuracy for EM is ', em_accuracies[-1])

    # remplacer le None par les precisions de la methode bayesienne
    return em_accuracies, None

@exp.config
def config():
    mus = [[1., 0.], [0., 1.]]
    N_trials = 10
    M = 50
    Ts = list(range(10, 110, 10))
    alpha = beta = 1.
    delta = gamm = 0.
    save_file = None

@exp.automain
def main(N_trials : int, M : int, alpha : float, beta : float, delta : float, gamm : float, Ts, mus : list, save_file: str):
    n_classes, n = len(mus), len(mus[0])
    env = chain.get_chain_env(S = 5, alpha = alpha, beta = beta, delta = delta, gamma = gamm, eps=.1)
    eta = 1.0

    mus = np.array(mus)
    Sigmas = np.array([np.eye(n) for _ in range(n_classes)])
        
    em_accuracies = np.zeros(shape=(N_trials, len(Ts)))
   
    for i in range(N_trials):
        em_accuracies[i, :], _ = main_aux(M, mus, Sigmas, env, eta, Ts)
    
    avg_em_accuracies = np.mean(em_accuracies, axis=0)

    plt.plot(Ts, avg_em_accuracies, color='b', label='EM version')
    plt.title('Accuracy of MDP clustering as function of T with M=' + str(M) + ' MDPs')
    plt.legend()
    if not save_file is None:
        plt.savefig(save_file)
    else:
        plt.show()