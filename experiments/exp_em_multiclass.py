# Essai du traitement non-Bayesien et avec l'algo d'EM

import itertools
import sys
import time
import warnings
sys.path.append("..")

import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt

import core.niw as niw
import core.inference as inference
import core.gibbs_class as gibbs_class
import core.dirichletprocess as dp
import core.environnement as environnement
import core.gibbs_class as gibbs_class
import core.trajectory as trajectory
import core.policy as policy
import core.metrics as metrics

def sample_multiclass_ws(prior_niw : niw.NIWParams, num_classes : int, M : int):
    n = len(prior_niw.mu_mean)
    """
    arguments:
        - prior_niw : prior pour sampler les (mu, Sigma)
        - c : nombre de classes (fixe ici pour EM algo)
        - M : nombre de MDP
    """
    classes = np.random.choice(num_classes, size= (M, ))

    # sample as many NIW as necessary 
    mus, Sigmas = niw.sample_niw(prior_niw, size=(num_classes, ))

    ws = np.zeros(shape=(M, n))
    for m in range(M):
        c = classes[m]
        mu, Sigma = mus[c], Sigmas[c]
        ws[m] = np.random.multivariate_normal(mu, Sigma)

    return mus, Sigmas, classes, ws

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

def main():
    prior_niw  = niw.default_niw_prior(2)
    mu_0, k_0, Sigma_0, nu_0 = prior_niw.mu_mean, prior_niw.mu_scale, prior_niw.Sigma_mean, prior_niw.Sigma_scale
    
    num_classes, M, T =  4, 25, 500
    tau, n, eta = 2., 2, 1.0
    

    env = environnement.get_observable_random_environment(S = 5, A = 2, O = 5, n = n)

    mus, Sigmas, true_classes, ws = sample_multiclass_ws(prior_niw, num_classes, M)
    
    states, actions, observations = compute_trajectories_from_ws(ws, env, eta, T)
    ws_mle = np.zeros(shape=(M, n))

    # classe pour estimer les parametres 
    # https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
    bgm = mixture.BayesianGaussianMixture(n_components=num_classes, mean_prior=mu_0, mean_precision_prior=k_0, degrees_of_freedom_prior=nu_0, covariance_prior=Sigma_0, 
    weight_concentration_prior=tau)

    ts = [25 * k for k in range(1, 7)]
    distances_w, distances_reward = [], []
    # On etudie la precision au cours du temps
    for t in ts:
        begin = time.time()
        # Etape 1 : Estimer les ws : dans 1 premier cas, on utilise le MLE
        for m in range(M):
            ws_mle[m]  = inference.mle_w(trajectory.ObservedTrajectory(actions = actions[m, :t], observations = observations[m, :t]), eta, env)
        # Etape 2 : Estimer les parametres 
        bgm.fit(ws_mle)
        infered_mus = bgm.means_
    
        # Etape 3 : calculer la distance. Fait a la main pour l'instant 
        # Pour ce faire, on prend la permutation qui minimise la distance

        distances_w_t = [metrics.quadratic_ws(mus, infered_mus[list(p)]) for p in itertools.permutations(range(num_classes))]
        distances_w.append(min(distances_w_t))
        print(distances_w[-1])

        print('Time for ' + str(t) + ' steps of the MDP : ' + str(time.time() - begin))

    plt.plot(ts, distances_w)
    plt.show()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()