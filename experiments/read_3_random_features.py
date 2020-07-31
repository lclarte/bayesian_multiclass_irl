from mpl_toolkits import mplot3d 
import numpy as np 
import matplotlib.pyplot as plt 

dots = np.load('experiments\logs\exp_3_random_features_gcp.npy')
  
x, y, z = dots[:, 0], dots[:, 1], dots[:, 2]
  
# Creating figure 
fig = plt.figure(figsize = (10, 7)) 
ax = plt.axes(projection ="3d") 
  
# Creating plot 
ax.scatter3D(x, y, z, color = "green"); 
plt.title("simple 3D scatter plot") 
  
# show plot 
plt.show() 