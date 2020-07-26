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

@exp.automain
def main():
    n = 2

    prior_niw  = niw.NIWParams(mu_mean = np.zeros(shape=(2, )), mu_scale=1., Sigma_mean = 0.1*np.eye(2), Sigma_scale=1.)
    mu_0, k_0, Sigma_0, nu_0 = prior_niw.mu_mean, prior_niw.mu_scale, prior_niw.Sigma_mean, prior_niw.Sigma_scale
    
    num_classes, M, T =  2, 50, 50
    tau, eta = 2., 1.0
    
    env = environnement.get_observable_random_environment(S = 5, A = 2, O = 5, n = n)

    mus, Sigmas = niw.sample_niw(prior_niw)
    true_classes = [0]*int(M/2) + [1]*int(M/2)
    
    # current estimation of classes 
    current_classes = [np.random.choice(2) for _ in range(M)]
    
    ws = np.zeros(shape=(M, n))
    for m in range(M):
        c = true_classes[m]
        ws[m] = np.random.multivariate_normal(mus[c], 0.1*Sigmas[c])

    plt.scatter(mus[:, 0], mus[:, 1], marker='*')
    plt.scatter(ws[:, 0], ws[:, 1])
    plt.title("True ws")
    plt.plot()

    states, actions, observations = compute_trajectories_from_ws(ws, env, eta, T)
    
    infered_ws = np.zeros(shape=(M, n))

    # classe pour estimer les parametres 
    # https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
    bgm = mixture.GaussianMixture(n_components=num_classes)

    infered_mus, infered_Sigmas = np.array([[0.5, 0.], [0., 0.5]]), np.array([Sigma_0, Sigma_0])

    # Certain nombre d'iterations de l'algo 
    for i in range(10):
    
        # Etape 1 : Estimer les ws
        for m in range(M):
            c = current_classes[m]
            infered_ws[m] = inference.map_w_from_observations(trajectory.ObservedTrajectory(actions = actions[m], observations = observations[m]), mu_0, Sigma_0, eta, env)

        plt.scatter(infered_ws[:, 0], infered_ws[:, 1]) ; plt.show()

        bgm.fit(infered_ws)
        
        infered_mus = bgm.means_
        infered_Sigmas = bgm.covariances_
        
        # Mettre a jour les classes ? 
        for m in range(M):
            current_classes[m] = get_class(infered_ws[m], mus, Sigmas)

    print(current_classes)
    print("=============")
    print(infered_mus)
    print("=============")
    print(mus)
    