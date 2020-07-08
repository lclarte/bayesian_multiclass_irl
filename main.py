# TODO : Faire l'algo. pour le decoding du HMM (par exemple avec Viterbi)
# TODO : Code pour un env. de RL avec matrice de transition / distribution d'observation custom 
# RMK : Dans un premier temps, on suppose que S et A sont finis, dans un ensemble [0, |S| - 1 ] et [0, |A| - 1]
# On peut donc tout stocker sous forme matricielle (reward_function et q_function)
# RMK : Dans le IRL pour un POMDP, on a une observation w = f(s_t+1, a_t) et les actions a_t


import matplotlib.pyplot as plt
import itertools
from sampling import *
from inference import *

def plot_w_posterior_likelihood(w_grid, mu, Sigma, states, actions, basis, trans_matx, gamma, eta):
    """
    Ne marche qu'en deux dimensions ! 
    Plot la probabilite posterieure du vecteur de poids w sur un produit d'intervalles donné    
    """
    def penalized_likelihood(w):
        policy = softmax(q_function(linear_reward_function(w , basis), trans_matx, gamma), eta)
        retour = trajectory_likelihood_policy(states, actions, policy) * stats.multivariate_normal.pdf(w, mean=mu, cov=Sigma) 
        return np.log(retour)

    X, Y = w_grid
    likelihood = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            x, y = X[i], Y[j]
            w = np.array([x, y])
            likelihood[i, j] = penalized_likelihood(w)
    # plot results
    X, Y = np.meshgrid(X, Y)
    fig, ax = plt.subplots()
    cont = plt.contour(X, Y, likelihood)
    fig.colorbar(cont, shrink=0.8, extend='both')
    plt.show()