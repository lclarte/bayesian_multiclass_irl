import numpy as np
import unittest
from environnement import DirichletProcess
from main import *

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
'T' : 100, # S x T = 50 trajectoires possibles (pas trop gros pour faire des tests)

'gamma' : 0.1,
'eta'   : 1.0 # param for softmax policy. Plus c'est haut, plus la distribution sera piquee 
}

# Test 1 : Posterior sampling des trajectoires etant donne un vecteur w et des obervations

def test_posterior_sampling(p):
    S, A, O, T = p['S'], p['A'], p['O'], p['T']
    n = p['n'] # size of latent space
    gamma = parameters['gamma']
    eta = parameters['eta']
    rho_0 = parameters['rho_0']
    
    trans_matx = random_transition_matrix(S, A)
    obs_matx   = noisy_id_observation_matrix(S, A, O, eps=0.1)

    mu, Sigma = np.zeros((2, )), np.eye(2)
    w = stats.multivariate_normal.rvs(mean=mu, cov=Sigma)

    print('True w : ', w)

    basis = np.random.rand(S, A, n)

    reward_function = linear_reward_function(w, basis)

    print('reward function : ', reward_function)

    qstar = q_function(reward_function, trans_matx, gamma)
    policy = softmax(qstar, eta)

    print('policy : ', policy)
    
    # rque ; dans le POMDP, les etats pas sont observes, mais les actions oui
    states, actions = sample_trajectory(rho_0, policy, trans_matx, T)
    observations = [np.random.choice(O, p=obs_matx[states[i+1], actions[i], :]) for i in range(len(actions))]
    
    
    """
    # ici, on va faire une suite de trajectoires et etudier leur distribution, comparee
    K = 1000
    current_states = np.random.choice(S, size=T+1)
    states_history = [current_states]

    for k in range(K):
        # take random trajectory
        candidate_states = np.random.choice(S, size=T+1)
        cand_traj = {'states' : candidate_states, 'actions' : actions}
        curr_traj = {'states' : current_states, 'actions' : actions}
        # Metropolis-Hastings to reassign new states
        if mh_transition_trajectories(curr_traj, cand_traj, observations, policy, trans_matx, obs_matx, rho_0) == 1:
            current_states = candidate_states
        states_history.append(current_states)
    """
        
    print('Ground truth : ', states)
    belief = get_belief_from_observations(observations, actions, trans_matx, obs_matx, rho_0)
    print('MAP of states : ', np.argmax(belief, axis=1))

    w_map = map_w_from_map_trajectory(states, actions, mu, Sigma, basis, trans_matx, gamma, eta)
    print('MAP of w : ', w_map)

test_posterior_sampling(parameters)

class TestDirichletProcess(unittest.TestCase):
    def test_sample_class(self):
        dp = DirichletProcess()
        dp.tau = 1.0
        samples = [ dp.sample_class() for _ in range(100) ]
        np.histogram(samples, bins=[-.5, .5, 1.5, 2.5, 3.5, 4.5])
    

    def test_add_betas_probas(self):
        dp = DirichletProcess()
        dp.tau = 1.0
        is_empty = (dp.betas.shape == (0, ) and dp.probas.shape == (0,))
        dp.add_betas_probas()
        is_correct_shape = (dp.betas.shape == (dp.num_betas,) and dp.probas.shape == (dp.num_betas, ))
        self.assertTrue(is_empty and is_correct_shape)

    def test_sample_norm_inv_wish(self):
        dp = DirichletProcess()
        dp.mu_0 = np.zeros((2, ))
        dp.Sigma_0 = np.eye(2)
        dp.nu_0    = 2.
        dp.k_0     = 1.
        n = 10
        mu, Sigma = dp.sample_norm_inv_wish(size=(n,))
        self.assertTrue(mu.shape == (n, 2) and Sigma.shape == (n, 2, 2))