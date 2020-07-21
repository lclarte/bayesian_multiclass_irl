# Essai du traitement Bayesien pour le cas multiclasse 

import sys
sys.path.append("..")

import numpy as np

import core.niw as niw
import core.inference as inference
import core.gibbs_class as gibbs_class
import core.dirichletprocess as dp
import core.environnement as environnement
import core.gibbs_class as gibbs_class=

def sample_multiclass_ws(prior_niw : niw.NIWParams, tau : float, M : int):
    n = len(prior_niw.mu_mean)
    
    dp_sampler = dp.DirichletProcess(tau)

    env = environnement.get_random_environment(S = 5, A = 2, O = 5, n = n)

    classes = dp_sampler.sample_classes(size=(M, ))
    
    max_class = np.amax(classes)
    num_classes = max_class + 1
    # sample as many NIW as necessary 
    mus, Sigmas = niw.sample_niw(prior_niw, size=(num_classes, ))

    ws = np.zeros(shape=(M, n))
    for m in range(M):
        c = classes[m]
        mu, Sigma = mus[c], Sigmas[c]
        ws[m] = np.random.multivariate_normal(mu, Sigma)

    return mus, Sigmas, classes, ws

if __name__ == "__main__":
    prior_niw, tau, M = niw.default_niw_prior(2), 2., 50

    mus, Sigmas, true_classes, ws = sample_multiclass_ws(prior_niw, tau, M)
    max_class = np.amax(true_classes)

    # connaissant les ws (pas de IRL pour l'instant) on estime les classes
    curr_c = np.random.randint(0, max_class, size=(M, ))
    for k in range(100):
        for _ in range(M):
            m = np.random.randint(0, M)
            curr_c[m] = gibbs_class.metropolis_hastings_class_assignement(m, ws, curr_c, prior_niw, tau)
    print("Estimated classes : ", curr_c)
    print("True classes : ", true_classes)