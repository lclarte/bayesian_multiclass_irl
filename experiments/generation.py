# generation.py
# script to generate data (trajectories, environments, etc.)

import numpy as np

import core.environnement as environment
import core.policy as policy
import core.trajectory as trajectory

def compute_trajectories_from_ws(ws : np.ndarray, env : environment.Environment, eta : float, T : int):
    """
    returns trajectories given environment + weights 
    """
    M = len(ws)
    
    states = np.zeros(shape = (M, T+1), dtype=int)
    actions = np.zeros(shape = (M, T), dtype=int)
    observations = np.zeros(shape = (M, T), dtype=int)

    for m in range(M):
        states[m], actions[m] = policy.sample_trajectory_from_w(env.init_dist, ws[m], env.features, env.trans_matx, env.gamma, eta, T)
        observations[m] = environment.get_observations_from_states_actions(states[m], actions[m], env.obsvn_matx)

    return states, actions, observations

def compute_observed_trajectories_from_ws(ws : np.ndarray, env : environment.Environment, eta : float, T : int):
    _, actions, observations = compute_trajectories_from_ws(ws, env, eta, T)
    return [ trajectory.ObservedTrajectory(actions = actions[i], observations = observations[i]) for i in range(len(actions))]

def compute_complete_trajectories_from_ws(ws : np.ndarray, env : environment.Environment, eta : float, T : int):
    states, actions, _ = compute_trajectories_from_ws(ws, env, eta, T)
    return [ trajectory.CompleteTrajectory(actions = actions[i], states = states[i]) for i in range(len(actions))]