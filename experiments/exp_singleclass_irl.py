# exp_singleclass_irl.py
# infere le parametre pour une seule classe 

import sys
sys.path.append("..")

import numpy as np

import core.niw as niw 
import core.inference as inference
import core.environnement as environnement
import core.policy as policy
import core.trajectory as trajectory

parameters = {
'n' :  2,

'mu_0' : np.zeros(2),
'k_0'  : 1.0,
'Sigma_0' : np.identity(2),
'nu_0' : 2,

# number of tasks
'k' : 50,

# parametres des trajectoires$

# longueur 
'T' : 100,
'S' : 10,
'A' : 2,
'O' : 10,

'eta' : 10.,

}

def get_environnement(p):
    S, A, O, T = p['S'], p['A'], p['O'], p['T']
    n = p['n']
    
    transition_matrix = environnement.random_transition_matrix(S, A)
    # take a slightly perturbated identity matrix
    observation_matrix = environnement.noisy_id_observation_matrix(S, A, O, eps=0.01)
    basis = np.random.rand(S, A, n)
    rho_0 = np.zeros(shape=(S))
    rho_0[0] = 1.

    return environnement.Environment(trans_matx = transition_matrix, 
                            obsvn_matx = observation_matrix,
                            features = basis,
                            init_dist = rho_0,
                            # take default gamma 0.9
                            )


def scmtirl(p):
    """
    Single class multitask IRL (SCMTIRL)
    """
    mu_0 = p['mu_0']
    k_0  = p['k_0']
    Sigma_0 = p['Sigma_0']
    nu_0 = p['nu_0']
    k = p['k']
    T = p['T']
    eta = p['eta']


    prior = niw.NIWParams(mu_0, k_0, Sigma_0, nu_0)
    # ws : pas observé
    mu, Sigma = niw.sample_niw(prior, size=(1, ))
    mu, Sigma = mu[0], Sigma[0]

    # ws : pas observé 
    ws = np.random.multivariate_normal(mean=mu, cov=Sigma, size=(k, ))
    infered_ws = np.zeros_like(ws)
    env = get_environnement(p)

    states = np.zeros(shape=(k, T+1), dtype=int)
    actions = np.zeros(shape=(k, T), dtype=int)
    observations = np.zeros_like(actions)

    # On sample des trajectoires + observations 
    for i in range(k):
        reward_function = env.features.dot(ws[i])
        pol = policy.softmax(policy.q_function(reward_function, env.trans_matx, env.gamma), eta)
        states[i], actions[i] = policy.sample_trajectory(env.init_dist, pol, env.trans_matx, T)
        observations[i] = environnement.get_observations_from_states_actions(states[i], actions[i], env.trans_matx)
    #
    # Essai 1 : on va simplement inferer le MAP des ws a partir du prior, puis faire du posterior de Normal Inverse Wishart a partir de ces ws
    #
    for i in range(k):
        ws[i] = inference.map_w_from_observations(trajectory.ObservedTrajectory(states, actions), mu_0, Sigma_0, eta, env)
    posterior_params = niw.niw_posterior_params(prior, ws)

    print('Posterior of mu : ', posterior_params.mu_mean)



if __name__ == "__main__":
    scmtirl(parameters)