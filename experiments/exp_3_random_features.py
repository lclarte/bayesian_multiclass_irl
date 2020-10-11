# Essai du traitement non-Bayesien et avec l'algo d'EM

from .exp_base import *

exp = sacred.Experiment("exp_3_random_features")

@exp.config
def config():
    M = 10
    T = 10
    # Use EM version of hierarchical bayesian version of algorithm
    bayesian = True

@exp.automain
def main(M : int, T : int, bayesian : bool):
    
    # define environment
    n_classes, n = 2, 2
    env = environnement.get_observable_random_environment(S = 5, A = 2, O = 5, n = n)
    eta = 100.
    niw_prior = niw.default_niw_prior(2)
    mus, Sigmas = niw.sample_niw(niw_prior, size=(2,))

    # run inference
    debut = time.time()
    ws, infered_ws, infered_mus  = run_experiment(M, mus, Sigmas, env, eta, T, bayesian)
    fin = time.time()

