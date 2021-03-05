import os
import numpy as np
from sklearn.linear_model import LinearRegression
from neuroc_pygs.configs import PROJECT_PATH


tab_data = np.load(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_res', 'gat_memory_2dims_curve_data.npy'))
nodes = list(map(lambda x: int(x), tab_data[:, 1]))
edges = list(map(lambda x: int(x), tab_data[:, 2]))
memory = list(map(lambda x: int(x) / (1024*1024), tab_data[:, 3]))

X = np.array([nodes, edges]).T
y = np.array(memory)

X_train, y_train = X[:300], y[:300]
X_test, y_test = X[300:], y[300:]

reg = LinearRegression().fit(X_train, y_train)

print(reg.coef_)
print(reg.predict(X_test), y_test)