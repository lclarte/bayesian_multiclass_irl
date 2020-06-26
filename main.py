# TODO : Faire l'algo. pour le decoding du HMM (par exemple avec Viterbi)
# TODO : Code pour un env. de RL avec matrice de transition / distribution d'observation custom 
# RMK : Dans un premier temps, on suppose que S et A sont finis, dans un ensemble [0, |S| - 1 ] et [0, |A| - 1]
# On peut donc tout stocker sous forme matricielle (reward_function et q_function)

# import pymc3 as pm 
from scipy import stats 
import numpy as np
import matplotlib.pyplot as plt

"""
Classe contenant  l'environnement, pour que l'agent interagisse avec l'env. 
CHAMPS DE LA CLASSE : 
    - Matrice de transition T(. | s, a)
    - Matrice des observations W(S, A, O) (ensemble des observations finis)
    - Distribution sur l'etat initial
1) Pour le sampling:
    - Fonction .step() comme dans Gym, retourne l'observation et la recompense
    - Fonction reset : donne l'observation initiale
2) Pour l'inference:
    -> Etant donne une liste d'observations, calculer le MAP des (s_t, a_t)
    -> Etant donne une liste d'observations et une liste de (s_t, a_t), calculer la proba (Forward / Backward)
"""

def random_transition_matrix(S, A):
    """
    sample random transition matrix 
    parameters : 
        -S : nb d'etats
        -A : nb d'actions
    """
    trans = np.random.rand(S, A, S) # s, a, s'
    for s in range(S):
        trans[s] = trans[s] / np.sum(trans[s], axis = 1)[:, np.newaxis]
    return trans

def sample_norm_inv_wish(mu_0, k_0, Sigma_0, nu_0, size):
    """
    Sample mu, Sigma from Normal Inverse Wishart distribution 
    parameters : 
        - mu_0 : center of normal distribution
        - k_0 : scaling of normal distribution
        - Sigma_0 : scale parameter of inverse wishart
        - nu_0    : degrees of parameters of inverse wishart 
    """
    Sigma = stats.invwishart.rvs(nu_0, Sigma_0, size)
    mu = np.zeros(size + mu_0.shape)
    # iterate over all the covariance matrices 
    it = np.nditer(mu, flags=['multi_index'])
    for x in it:
        indx = it.multi_index[:-1]
        mu[indx] = stats.multivariate_normal.rvs(mean = mu_0, cov = Sigma[indx] / k_0)
    return mu, Sigma

def linear_reward_function(w, basis):
    """
    Returns a S x A matrix representing the reward function 
    parameters:
        - basis : base de fonctions : tenseur de taille S x A x n 
        - w : vecteur de taille n 
    """
    return np.dot(basis, w)

def q_function(reward_function, transition, gamma):
    """
    Calcule la q_function optimale par value iteration
    transition : array of size S * A * S normalized on axis 2 
    """
    q = np.empty_like(reward_function)
    nb_iter = 50
    for n in range(nb_iter):
        q = reward_function + gamma * np.sum(transition * np.amax(q, axis=1), axis=2)
    return q

def softmax(q_function, eta):
    """
    Retourne la matrice np.exp(\eta * Q[s, a]) / (Normalisation_sur_a). Cette matrice represente la policy avec une 
    certaine q_function 
    """
    e = np.exp(eta * q_function)
    return e / e.sum(axis=1)[:, None]

def trajectory_likelihood(trajectory, w, eta):
    """
    Calcule la vraisemblance d'une trajectoire 
    parameters : 
        trajectory : sequence [s_1, a_1, ... , s_T]
        w : parametre de la fonction de recompense w^T Phi(s, a)
        eta : exploration parameter
    """
    # step 1 : calculer le produit des softmax pour le couples (s, a)
    # step 2 : calculer la vraisemblance du vecteur w choisi 
    raise NotImplementedError

def sample_trajectory(rho_0, policy, transition, T):
    """
    parameters:
        rho_0 : vecteur de taille S, distribution sur les etats
        policy : matrice de taille S x A, normalise sur le deuxieme axe
        transition : matrice S x A x S
    """
    S, A = policy.shape
    states, actions = [np.random.choice(S, p=rho_0)], []
    for t in range(T):
        state = states[-1]
        action = np.random.choice(A, p=policy[state])
        next_state = np.random.choice(S, p=transition[state, action])
        states.append(next_state)
        actions.append(action)
    return states, actions

# placeholder
# Espace latent est de dimension 2
n = 2
tau = 5.
mu_0 = np.zeros(n)
k_0  = 1.0
Sigma_0 = np.identity(n)
nu_0 = len(Sigma_0)

S = 10
A = 2

gamma = 0.9
eta   = 1.0

# distribution sur les etats initiaux : dirac sur 0 ici
rho_0 = np.array([1.] + [0]*(S - 1))

def main():
    transition = random_transition_matrix(S, A)
    basis = np.random.rand(S, A, n)

    # Code pour echantilloner une trajectoire from scratch 
    # N : nombre de samples = nb de MDPs 
    N = 1000
    # T : duree d'une trajectoire
    T = 50
    # K : nombre de classe de MDPs. Est infini en theorie, ici on prend juste un grand nombre 
    K = 10
    beta = stats.beta.rvs(1, tau, size=(K)) 
    p    = np.empty_like(beta)
    p[0] = beta[0] #1er element : proba de la classe = beta_0
    p[1:] = beta[1:] * (1 - beta[1:]).cumprod(axis=0)
    # normalisation parce qu'on a pas un nb illimite de betas
    p /= np.sum(p)
    # liste des parametres pour chaque classe
    omega = sample_norm_inv_wish(mu_0, k_0, Sigma_0, nu_0, size=(K, ))
    # assignation de la classe pour chaque MDP
    classes = np.random.choice(a=K, p=p, size=(N,))
    ws = np.zeros(shape=(N, n))
    qs = np.zeros(shape=(N, S, A))

    for x in range(N):
        mu, Sigma = omega[0][classes[x]], omega[1][classes[x]]
        ws[x] = np.random.multivariate_normal(mu, Sigma)
        reward_function = linear_reward_function(ws[x], basis)
        qs[x] = q_function(reward_function, transition, gamma)
        policy = softmax(qs[x], eta)
        # generer une trajectoire 
        states, actions = sample_trajectory(rho_0, policy, transition, T)

main()
