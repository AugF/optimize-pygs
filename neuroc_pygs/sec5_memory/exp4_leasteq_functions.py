import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import leastsq
from neuroc_pygs.configs import PROJECT_PATH

def err(p, x, y):
    return p[0] * x + p[1] - y


# http://liao.cpython.org/scipytutorial07/
tab_data = np.load(os.path.join(PROJECT_PATH, 'sec5_memory', 'log', 'gat_memory_curve_data.npy'))
tab_data = np.array(tab_data)[:,3]
tab_data = list(map(lambda x: int(x), tab_data))

# nodes
p0 = [10, 20]
Xi=np.array([1] + np.arange(2.5, 50, 2.5).tolist())
Yi=np.array(tab_data[:20])
ret = leastsq(err, p0, args = (Xi, Yi))
k, b = ret[0]
plt.title('GAT Nodes')
plt.figure(figsize=(8,6))
plt.scatter(Xi,Yi,color="red",label="Sample Point",linewidth=3)
x = np.linspace(0,57.5,1000)
y = k * x + b
plt.plot(x,y,color="orange",label="Fitting Line",linewidth=2)
plt.legend()
plt.savefig(os.path.join(PROJECT_PATH, 'sec5_memory', 'log', 'gat_nodes.png'))  
print('nodes', k, b)

# edges
p0 = [10, 20]
Xi=np.array(np.arange(1, 10).tolist() + np.arange(10, 71, 5).tolist())
Yi=np.array(tab_data[20:])
ret = leastsq(err, p0, args = (Xi, Yi))
k, b = ret[0]
plt.figure(figsize=(8,6))
plt.title('GAT Edges')
plt.scatter(Xi,Yi,color="red",label="Sample Point",linewidth=3)
x = np.linspace(0,57.5,1000)
y = k * x + b
plt.plot(x,y,color="orange",label="Fitting Line",linewidth=2)
plt.legend()
plt.savefig(os.path.join(PROJECT_PATH, 'sec5_memory', 'log', 'gat_edges.png'))  
print('edges', k, b)

"""
nodes 5572789.330298206 742098468.3389027
14830572.177004842 57483866.17193353
"""