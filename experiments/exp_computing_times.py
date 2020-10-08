# exp_computing_times.py
# sert a evaluer la vitesse d'execution 

import functools
import sys
sys.path.append("..")
import time
import unittest

import numpy as np
import matplotlib.pyplot as plt
from sacred import Experiment

import core.inference as inference
import core.environnement as environnement
import core.niw as niw
import core.policy as policy
import core.trajectory as trajectory
import core.metrics as metrics

ex = Experiment('experiment1')

# renvoie un tuple : resultat + temps de calcul
def measure_time(func):
    def wrapped(*args, **kwargs):
        begin_time = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - begin_time
    return wrapped

timed_map_w_from_map_trajectory = measure_time(inference.map_w_from_mle_trajectory)
timed_map_w_from_observations = measure_time(inference.map_w_from_observations)

@ex.config
def compare_config():
    S, A, O, n = 10, 2, 10, 2
    num_ws, T  = 30, 100
    T_step = 25

@ex.automain
def compare_states_mle_states_belief(S, A, O, n, num_ws, T, T_step):
    """
    Compare le calcule posterieur de w avec le MLE des etats avec le posterieur exact base sur les beliefs
    """
    # dans le cas ou l'environnement est observable, les deux methodes sont equivalentes 
    env = environnement.get_random_environment(S, A, O, n)
    mu, Sigma = np.ones(shape=(n,)), np.eye(N=n)
    eta = 1.0

    # compare for timesteps w/ increments of 10
    ts = list(range(T_step, T, T_step))

    distances_mle, distances_belief = np.zeros(shape=(len(ts), num_ws)), np.zeros(shape=(len(ts), num_ws))

    tps_mle, tps_belief, tps_new_mle = np.zeros(shape=(len(ts), num_ws)), np.zeros(shape=(len(ts), num_ws)), np.zeros(shape=(len(ts), num_ws))

    for i in range(num_ws):
        w = np.random.multivariate_normal(mean=mu, cov=Sigma)
        # sample trajectory from this w 
        states, actions = policy.sample_trajectory_from_w(env.init_dist, w, env.features, env.trans_matx, env.gamma, eta, T)
        observations = environnement.get_observations_from_states_actions(states, actions, env.obsvn_matx)
            
        for j in range(len(ts)):
            t = ts[j]
            # states_belief = inference.get_belief_from_observations(observations[:t], actions[:t], env)

            # method 1 : MLE of states
            states_mle = np.argmax(states_belief, axis=1)

            w_from_mle, tps_mle[j, i] = timed_map_w_from_map_trajectory(states_mle, actions[:t], mu, Sigma, eta, env) 
            w_from_belief, tps_belief[j, i] = timed_map_w_from_observations(trajectory.ObservedTrajectory(actions = actions[:t], observations = observations[:t]), mu, Sigma, eta, env)
            
            # difference for current sample of w
            dist_mle = metrics.quadratic_ws(w, w_from_mle)
            dist_belief = metrics.quadratic_ws(w, w_from_belief)
            
            distances_mle[j, i] = dist_mle
            distances_belief[j, i] = dist_belief
            
    average_dist_mle = np.mean(distances_mle, axis=1)
    average_dist_belief = np.mean(distances_belief, axis=1)

    std_dist_mle = np.std(distances_mle, axis=1)
    std_dist_belief = np.std(distances_belief, axis=1)
    
    average_tps_mle = np.mean(tps_mle, axis=1)
    average_tps_belief = np.mean(tps_belief, axis=1)
    
    std_tps_mle = np.std(tps_mle, axis=1)
    std_tps_belief = np.std(tps_belief, axis=1)

    plt.plot(ts, average_dist_mle, color='b', label='MLE of states')
    # confidence interval
    plt.fill_between(ts, average_dist_mle - std_dist_mle, average_dist_mle + std_dist_mle, color='b', alpha=.1)
    plt.plot(ts, average_dist_belief, color='r', label='Belief of states')
    plt.fill_between(ts, average_dist_belief - std_dist_belief, average_dist_belief + std_dist_belief, color='r', alpha=.1)
    plt.title("Average square distance between true and infered parameters")
    plt.legend()
    plt.show()

    plt.plot(ts, average_tps_belief, color='r', label='Belief of states')
    plt.fill_between(ts, average_tps_belief - std_tps_belief, average_tps_belief + std_tps_belief, color='r', alpha=.1)
    plt.plot(ts, average_tps_mle, color='b', label='MLE of states')
    plt.fill_between(ts, average_tps_mle - std_tps_mle, average_tps_mle + std_tps_mle, color='b', alpha=.1)
    plt.title("Computing times")
    plt.legend()
    plt.show()