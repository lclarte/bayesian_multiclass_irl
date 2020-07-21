# gibbs_class.py
# Fichier pour le Gibbs sampling des classes 

import numpy as np
from typing import List

from scipy.special import gamma

import core.niw as niw
import core.inference as inference 

def posterior_class_gibbs_sampling(m : int, ws : np.ndarray, curr_c : List, cand_c : int, prior_niw : niw.NIWParams, tau : float):
    """
    Calcule une proba (non normalisée) postérieure de la classe d'un MDP donne en fonction 
    des autres classes + des parametres prior.
    arguments : 
        - m : indice du MDP
        - ws : liste des poids
        - curr_c : current class assignement of MDP
        - cand_c : candidate class assignement of MDP
        - prior_niw : prior parameters of the NIW law
        - tau : prior du Dirichlet Process  
    """
    c_list = np.argwhere(curr_c == cand_c)

    # integrale sur les parametres, conditionne par les w_m de la classe + prior
    proba_class = posterior_class_integral(m, ws, curr_c, cand_c, prior_niw)
    
    # prior sur les classes cf. process de Dirichlet
    factor = tau
    if len(c_list) > 1:
        factor = len(c_list - 1)
    
    # manque le facteur de normalisation => faire du MH
    return factor * proba_class

def posterior_prior_ratio(prior_niw : niw.NIWParams, posterior_niw : niw.NIWParams):
    """
    Valeur explicite de l'integrale dans la 1ere equa° la section 8.3 dans https://hal.inria.fr/inria-00475214/document
    a partir des prior et des posteriors de NIW
    """
    mu_0, k_0, Sigma_0, nu_0 = prior_niw
    mu_p, k_p, Sigma_p, nu_p = posterior_niw
    
    d = len(mu_0)
    det_0, det_p = np.linalg.det(Sigma_0), np.linalg.det(Sigma_p)
    # dans l'equation de 8.3 : numerateur et denominateur avec la fonction Gamma
    gamma_num, gamma_den = gamma(nu_p / 2.), gamma((nu_p - d) / 2.) 


    return (k_0 / (np.pi * k_p))**(d / 2.) * det_0**(nu_0 / 2.) / (det_p**(nu_p / 2.)) \
            * gamma_num / gamma_den


def posterior_class_integral(m : int, ws : np.ndarray, curr_c : List, cand_c : int, prior_niw : niw.NIWParams):
    """
    Calcule l'integrale (sur les params mu, Sigma) de p(w_m | \theta_c ) * p( \theta_c | {c_m}, \Psi_0)
    qui est la proba posterieure de la classe c_m p( c_m | {c_m'}, \Psi_0 )
    Calcule a partir des observations des tirages gaussiens w_m
    arguments : 
        - m : indice du MDP
        - ws : liste des poids
        - curr_c : current class assignement of MDP
        - cand_c : candidate class assignement of MDP
        - prior_niw : prior parameters of the NIW law 
    """

    # step 1 : Get indices of MDPs that are candidate class
    c_indices = np.argwhere( curr_c == cand_c )[:, 0]
    if (not m in c_indices):
        c_indices = np.concatenate((c_indices, [m]))
    c_ws = ws[c_indices]

    # step 2 : compute posterior params 
    posterior_niw = niw.niw_posterior_params(prior_niw, c_ws)

    proba = posterior_prior_ratio(prior_niw, posterior_niw)

    return proba

def metropolis_hastings_class_assignement(m : int, ws : np.ndarray, curr_c : List, prior_niw : niw.NIWParams, tau : float):
    """
    Assigne une classe suivant MH a un MDP donne en parametre
    """
    # Si une classe est videe, alors on diminue toutes les classes qui viennent apres d'une unité
    max_class = np.amax(curr_c)
    # add 1 class for new class
    cand_c = np.random.randint(low=0, high=max_class + 1)
    cand_likelihood = posterior_class_gibbs_sampling(m, ws, curr_c, cand_c, prior_niw, tau)
    curr_likelihood = posterior_class_gibbs_sampling(m, ws, curr_c, curr_c[m], prior_niw, tau)

    proba_accept_cand = min(1., cand_likelihood / curr_likelihood)
    accept_cand = np.random.binomial(1, proba_accept_cand)

    if accept_cand:
        return cand_c
    else:
        return curr_c[m]