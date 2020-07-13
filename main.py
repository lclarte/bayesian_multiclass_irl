import numpy as np
from scipy.stats import stats

from core.niw import *
from core.environnement import *
from core.dirichletprocess import *
from core.policy import *
from core.inference import *

parameters = {
'n' :  2,
'tau' : 5.,
'mu_0' : np.zeros(2),
'k_0'  : 1.0,
'Sigma_0' : np.identity(2),
'nu_0' : 2,

'rho_0' : np.array([1.] + [0]*4),

'S' : 5,
'A' : 2,
'O' : 5,
'T' : 50, # S x T = 50 trajectoires possibles (pas trop gros pour faire des tests)

'gamma' : 0.1,
'eta'   : 1.0 # param for softmax policy. Plus c'est haut, plus la distribution sera piquee 
}

def test_posterior_sampling(p):
    """
    Sample un vecteur w sur une loi (mu, Sigma) donnee et infere ce vecteur w en ayant observe (mu, Sigma) ainsi 
    que les trajectoires (actions, observations). Il faut aussi inferer en meme temps les etats w
    """
    S, A, O, T = p['S'], p['A'], p['O'], p['T']
    n = p['n'] # size of latent space
    gamma = parameters['gamma']
    eta = parameters['eta']
    rho_0 = parameters['rho_0']

    trans_matx = random_transition_matrix(S, A)
    obs_matx   = noisy_id_observation_matrix(S, A, O, eps=0.1)
    basis = np.random.rand(S, A, n)

    env = Environment(gamma = gamma, trans_matx = trans_matx, obsvn_matx = obs_matx, features = basis, init_dist = rho_0)

    mu, Sigma = np.array([1.0, 1.0]), np.eye(2)
    w = stats.multivariate_normal.rvs(mean=mu, cov=Sigma)

    # compute the agent's policy 
    reward_function = linear_reward_function(w, basis)
    qstar = q_function(reward_function, trans_matx, gamma)
    policy = softmax(qstar, eta)
    
    # sampler une trajectoire 
    states, actions = sample_trajectory(rho_0, policy, trans_matx, T)
    observations = [np.random.choice(O, p=obs_matx[states[i+1], actions[i], :]) for i in range(len(actions))]
    traj = ObservedTrajectory(actions = actions, observations = observations)

    w_map = map_w_from_observations(traj, mu, Sigma, eta, env)
    # remplace le map des etats par les etats reels (pour simplifier)
    w_map_2 = map_w_from_map_trajectory(states, actions, mu, Sigma, eta, env)
    w_mle = mle_w(traj, eta, env)
    print('True w : ', w)
    print('MAP of w computed with true posterior : ', w_map)
    print('MAP of w computed with MAP of states : ', w_map_2)
    print('MLE of w', w_mle)

def test_posterior_niw(p):
    M = 10
    S, A, O, T = p['S'], p['A'], p['O'], p['T']
    n = p['n'] # size of latent space
    gamma = parameters['gamma']
    eta = parameters['eta']
    rho_0 = parameters['rho_0']
    
    trans_matx = random_transition_matrix(S, A)
    obs_matx   = noisy_id_observation_matrix(S, A, O, eps=0.1)
    basis = np.random.rand(S, A, n)
    env = Environment(gamma = gamma, trans_matx = trans_matx, obsvn_matx = obs_matx, features = basis, init_dist = rho_0)

    mu_0, k_0, Sigma_0, nu_0 = p['mu_0'], p['k_0'], p['Sigma_0'], p['nu_0']

    mu, Sigma = sample_niw(NIWParams(mu_0, k_0, Sigma_0, nu_0), size=(1, ))
    mu, Sigma = mu[0], Sigma[0]

    # on sample un certain nombre de MDP a partir de mu, Sigma
    ws = np.random.multivariate_normal(mean=mu, cov=Sigma, size=(M, ))
    states = np.zeros(shape=(M, T+1), dtype=int)
    map_states = np.zeros_like(states)
    actions = np.zeros(shape=(M, T), dtype=int)
    observations = np.zeros(shape=(M, T), dtype=int)

    for m in range(M):
        w = ws[m]
        reward_function = linear_reward_function(w, basis)
        qstar = q_function(reward_function, trans_matx, gamma)
        policy = softmax(qstar, eta)
        states[m], actions[m] = sample_trajectory(rho_0, policy, trans_matx, T)
        observations[m] = [np.random.choice(O, p = obs_matx[states[m, i+1], actions[m, i], :]) for i in range(len(actions[m]))]
        map_states[m] = np.argmax(get_belief_from_observations(observations[m], actions[m], env) , axis=1) 

    K = 10
    mu_curr, k_curr, Sigma_curr, nu_curr = mu_0, k_0, Sigma_0, nu_0
    mu_curr_2, k_curr_2, Sigma_curr_2, nu_curr_2 = mu_0, k_0, Sigma_0, nu_0
    w_map = np.zeros(shape=(M, n))
    w_map_2 = np.zeros(shape=(M, n))
    for k in range(K):
        print('k = ', k)
        for m in range(M):
            w_map[m] = map_w_from_observations(ObservedTrajectory(actions = actions[m], observations = observations[m]), mu_curr, Sigma_curr / k_curr, eta, env)
            w_map_2[m] = map_w_from_map_trajectory(map_states[m], actions[m], mu_curr_2, Sigma_curr_2, eta, env)
        mu_curr, k_curr, Sigma_curr, nu_curr = posterior_normal_inverse_wishart(w_map, mu_0, k_0, Sigma_0, nu_0)
        mu_curr_2, k_curr_2, Sigma_curr_2, nu_curr_2 = posterior_normal_inverse_wishart(w_map_2, mu_0, k_0, Sigma_0, nu_0)
    print('Infered mu, Sigma : ')
    print('mu = ', mu_curr)
    print('Infered mu, Sigma from MAP of trajectories: ')
    print('mu = ', mu_curr_2)
    print('True parameters : ')
    print('mu = ', mu)

def test_niw_posterior_sampling(p):
    mu_0 = p['mu_0']
    Sigma_0 = p['Sigma_0']
    k_0 = p['k_0']
    nu_0 = p['nu_0']

    prior = NIWParams(mu_0, k_0, Sigma_0, nu_0)

    k = 1000

    mu, Sigma = sample_niw(prior, size=(1,))
    print('Sampled mean : ', mu)
    print('Covariance matrix : ', Sigma)

    ws = stats.multivariate_normal.rvs(mean=mu[0], cov=Sigma[0], size=(k,))

    mu_p, k_p, Sigma_p, nu_p = niw_posterior_params(prior, ws)
    print('Infered mean : ', mu_p)
    print('Infered covariance :', Sigma_p)
    print('k_post = ', k_p, ' & nu_p', nu_p)

if __name__ == '__main__':
    test_posterior_niw(parameters)
    test_posterior_sampling(parameters)
    test_niw_posterior_sampling(parameters)