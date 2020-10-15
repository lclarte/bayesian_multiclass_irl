# Essai du traitement non-Bayesien et avec l'algo d'EM

from .exp_base import *

exp = sacred.Experiment("exp_chain")

@exp.config
def config():
    M = 10
    T = 10
    # Use EM version of hierarchical bayesian version of algorithm
    bayesian = True
    POMDP = True

@exp.automain
def main(M : int, T : int, bayesian : bool, POMDP : bool):
    
    # define environment
    n_classes, n = 2, 1
    env = chain.get_chain_env(S = 5, eps=.1)
    eta = 100.
    niw_prior = niw.default_niw_prior(1)
    mus_1d, Sigmas_1d = niw.sample_niw(niw_prior, size=(2,))
    
    mus = np.zeros((2, 2))
    Sigmas = np.zeros((2, 2, 2))
    mus[:, 0] = mus_1d.squeeze()
    mus[:, 1] = -mus_1d.squeeze()

    Sigmas[0] = Sigmas[1] = [[1., -1.], [-1., 1.]]
    # scale back sigmas
    Sigmas = 0.1 * Sigmas

    # run inference
    debut = time.time()
    if POMDP:
        ws, infered_ws, infered_mus  = run_experiment_pomdp(M, mus, Sigmas, env, eta, T, bayesian)
    else:
        ws, infered_ws, infered_mus  = run_experiment_mdp(M, mus, Sigmas, env, eta, T, bayesian)
    fin = time.time()