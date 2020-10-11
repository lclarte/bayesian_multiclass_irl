# Essai du traitement non-Bayesien et avec l'algo d'EM

from .exp_base import *

exp = sacred.Experiment("exp_chain")

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
    env = chain.get_chain_env(S = 5, eps=.1)
    eta = 100.
    niw_prior = niw.default_niw_prior(2)
    mus, Sigmas = niw.sample_niw(niw_prior, size=(2,))

    # run inference
    debut = time.time()
    ws, infered_ws, infered_mus  = run_experiment(M, mus, Sigmas, env, eta, T, bayesian)
    fin = time.time()