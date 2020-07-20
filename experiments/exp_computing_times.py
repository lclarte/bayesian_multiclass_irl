# exp_computing_times.py
# sert a evaluer la vitesse d'execution 

import functools
import sys
sys.path.append("..")
import time
import unittest

import numpy as np
import matplotlib.pyplot as plt

import core.inference as inference
import core.environnement as environnement
import core.niw as niw
import core.policy as policy
import core.trajectory as trajectory

class ExpComputingTimes:
    def wrapper(function):
        @functools.wraps(function)
        def wrapper(self):
            self.setUp()

            begin_time = time.time()
            result = function(self)
            end_time = time.time()
            delta_time = end_time - begin_time
            print('Elapsed time for function' + function.__name__ + ' : ' + str(delta_time))
            
            self.tearDown()
            return result
        return wrapper
    
    def __init__(self):
        pass

    def setUp(self):
        self.S, self.A, self.O, self.n, self.T = 20, 5, 10, 5, 50
        print('Parameters of environment (S, A, O, features_dim, T): ', self.S, self.A,self.O, self.n, self.T)
        self.env = environnement.get_random_environment(self.S, self.A, self.O, self.n)
        self.prior_niw = niw.default_niw_prior(self.n)
        self.nb_mdp = 10
        
        self.map_iterations = 5

        mu, Sigma = niw.sample_niw(self.prior_niw, size=(1, ))
        mu, Sigma = mu[0], Sigma[0]
        self.true_mu = mu

        ws = np.random.multivariate_normal(mu, Sigma, size=(self.nb_mdp))
        states = np.zeros(shape=(self.nb_mdp, self.T + 1), dtype=int)
        actions = np.zeros(shape=(self.nb_mdp, self.T), dtype=int)
        observations = np.zeros(shape=(self.nb_mdp, self.T), dtype=int)

        for i in range(self.nb_mdp):
            pol = policy.softmax(policy.q_function(policy.linear_reward_function(ws[i], self.env.features), self.env.trans_matx, self.env.gamma), eta=1.)
            states[i], actions[i] = policy.sample_trajectory(self.env.init_dist, pol, self.env.trans_matx, self.T)
            observations[i] = environnement.get_observations_from_states_actions(states[i], actions[i], self.env.obsvn_matx)
        self.states = states
        self.actions = actions
        self.observations = observations
        self.ws = ws

    def tearDown(self):
        pass
    
    @wrapper
    def duration_mu_map_from_trajectory_map(self):
        """
        Calcule le MAP de mu de maniere iterative, a partir des MAPs des ws et du prior (mu_0, Sigma_0). 
        Ici, les MAPs des ws sont calcules a partir du MAP des etats s_1, ..., s_T (c'est donc approximatif) 
        # TODO
        """
        mu_0, Sigma_0 = self.prior_niw.mu_mean, self.prior_niw.Sigma_mean
        mu_curr, Sigma_curr = mu_0, Sigma_0
        a = 2 # ou a = self.nb_mdp

        states_map = np.zeros_like(self.states)
        ws_map = np.zeros_like(self.ws)
        
        for i in range(a):
            states_map[i] = np.argmax(inference.get_belief_from_observations(self.observations[i], self.actions[i], self.env), axis=1)

        for k in range(self.map_iterations):
            for i in range(a):
                ws_map[i] = inference.map_w_from_map_trajectory(states_map[i], self.actions[i], mu_curr, Sigma_curr, 1., self.env)

            niw_posterior = niw.niw_posterior_params(self.prior_niw, ws_map[:a])
            mu_curr, Sigma_curr = niw_posterior.mu_mean, niw_posterior.Sigma_mean / niw_posterior.mu_scale

        print('True mu : ', self.true_mu)
        print('Posterior mu : ', mu_curr)

    @wrapper
    def duration_mu_map_from_belief(self):
        """
        Calcule le MAP de mu de maniere iterative, a partir des MAPs des ws et du prior (mu_0, Sigma_0). 
        Ici, les MAPS des ws sont calcules a partir du belief = distribution posterieure exacte des etats {s_t}_t
        # TODO
        """
        pass

if __name__ == "__main__":
    exp = ExpComputingTimes()
    result = exp.duration_mu_map_from_trajectory_map()