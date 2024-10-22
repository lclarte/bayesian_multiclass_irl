from mpl_toolkits import mplot3d 
from matplotlib import cm
import numpy as np 
import matplotlib.pyplot as plt

def plot_3d_function(function, x_bounds, y_bounds, num = 20, ws = None):
    """
    arguments:
        - function : fonction (x, y) \in R^2 -> z \in R
        - x_bounds
        - y_bounds
    """
    x = np.linspace(x_bounds[0], x_bounds[1], num=num)
    y = np.linspace(y_bounds[0], y_bounds[1], num=num)

    z = np.array([[function(a, b) for b in y] for a in x])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, z,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    if not ws is None:
        z_ws = [function(a[0], a[1]) for a in ws]
        ax.scatter(ws[:, 0], ws[:, 1], z_ws)

    plt.show()