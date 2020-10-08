# Essai du traitement non-Bayesien et avec l'algo d'EM

from datetime import date
import itertools
import sys
import time
import timeit
import warnings
sys.path.append("..")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sacred
import scipy.stats as stats
from sklearn import mixture
import sklearn

import core.niw as niw
import core.inference as inference
import core.dirichletprocess as dp
import core.environnement as environnement
import core.trajectory as trajectory
import core.policy as policy
import core.metrics as metrics
import core.logs as logs
import core.hbpomdp as hbpomdp
import core.empomdp as empomdp

import envs.chain as chain

from . import visualization
from . import generation

exp = sacred.Experiment("experiment_1")

def get_class(x, mus, Sigmas) -> int:
    """
    retoune la classe associee au x
    """
    C = len(mus)
    return np.argmax([stats.multivariate_normal.pdf(x, mean=mus[c], cov=Sigmas[c])  for c in range(C)])

def main_aux(M : int, mus : np.ndarray, Sigmas : np.ndarray, env : environnement.Environment, eta : float, T : int, bayesian : bool):
    n_classes, n = mus.shape

    # initialize true values 
    true_classes = [np.random.choice(n_classes) for _ in range(M)]
    ws = np.zeros(shape=(M, n))
    for m in range(M):
        c = true_classes[m]
        ws[m] = np.random.multivariate_normal(mus[c], Sigmas[c])

    trajectories = generation.compute_observed_trajectories_from_ws(ws, env, eta, T)

    # number of iteration for inference (arbitrary)
    n_iter = 10
    verbose = False
    
    # use em
    if not bayesian:
        model = empomdp.EMPOMDP(n_classes=n_classes, verbose=verbose, n_iter=n_iter)
        model.infer(env, trajectories, eta)

    else:
        niw_prior = niw.default_niw_prior(n)
        # prior on tau = true number of classes here (should be 1 ?)
        dp_tau = float(n_classes)
        model = hbpomdp.HBPOMDP(verbose=verbose, n_iter=n_iter, niw_prior=niw_prior, dp_tau=dp_tau)
        model.infer(env, trajectories, eta)
    
    infered_ws, infered_mus, infered_Sigmas, infered_classes = model.inf_ws, model.inf_mus, model.inf_Sigmas, model.inf_classes

    visualization.plot_inference_2d(infered_ws, infered_classes)
    plt.show()

    # afficher l'accuracy 
    accuracy = sklearn.metrics.accuracy_score(true_classes, infered_classes)
    print('Accuracy is : ', max(accuracy, 1 - accuracy))

    return ws , infered_ws, infered_mus

@exp.config
def config():
    M = 50
    T = 50
    # Use EM version of hierarchical bayesian version of algorithm
    bayesian = True

@exp.automain
def main(M : int, T : int, bayesian : bool):
    
    # define environment
    n_classes, n = 2, 2
    env = chain.get_chain_env(S = 5, eps=.1)
    eta = 1.
    niw_prior = niw.default_niw_prior(2)
    mus, Sigmas = niw.sample_niw(niw_prior, size=(2,))

    # run inference
    debut = time.time()
    ws, infered_ws, infered_mus  = main_aux(M, mus, Sigmas, env, eta, T, bayesian)
    fin = time.time()