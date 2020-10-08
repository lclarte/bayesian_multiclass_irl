# logging.py
# Utile pour sauvegarder les resultats des experiences avec le script utilise, les hyperparametres, etc. 

import json

import numpy as np
import matplotlib.pyplot as plt

def dump_results(file_name : str, result : np.ndarray, params : dict = None):
    """
    Ecrit dans le fichier file_name le resultat des experiences et un np array
    """
    params = params or {}
    data = params.copy()
    data['result'] = result.tolist() 
    with open(file_name, 'w') as f:
        json.dump(data, f)

def display_dumped_results(file_name : str):
    with open(file_name, 'rb') as f:
        loaded = json.load(f)
    
    points = np.array(loaded['result'])

    # pour l'instant, on fait que le cas 2D
    if points.shape[-1] != 2:
        print('Only 2D case is treated pour l\'instant')
        return
    
    plt.scatter(points[:, 0], points[:, 1])
    title = 'file = ' + file_name
    if 'title' in loaded:
        title += ';' + str(loaded['title'])
    plt.title(title)
    plt.show()