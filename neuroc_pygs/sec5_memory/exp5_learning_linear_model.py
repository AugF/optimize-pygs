import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from neuroc_pygs.configs import PROJECT_PATH


def get_X_y(model='gat'):
    tab_data = np.load(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_res', 'gat_memory_2dims_curve_data.npy'))
    nodes = list(map(lambda x: int(x), tab_data[:, 1]))
    edges = list(map(lambda x: int(x), tab_data[:, 2]))
    memory = list(map(lambda x: int(x) / (1024*1024), tab_data[:, 3]))

    X = np.array([nodes, edges]).T
    y = np.array(memory)
    return X, y


def get_other_metrics(y_pred, y_true):
    num = len(y_pred)
    bad_exps = 0
    max_bias = 0
    for i in range(num):
        if y_pred[i] > y_true[i]:
            max_bias = max(max_bias, (y_pred[i] - y_true[i]) / y_true[i])
            bad_exps += 1
    return bad_exps / num, max_bias


# 随机10轮，给出一个百分比; 
def experiment():
    np.random.seed(1)
    mask = np.arange(400)
    np.random.shuffle(mask)
    X, y = get_X_y(model='gat')
    X_train, y_train = X[mask[:390]], y[mask[:390]]
    X_test, y_test = X[mask[390:]], y[mask[390:]]

    reg = LinearRegression().fit(X_train, y_train)
    print(reg.coef_)
    y_pred = reg.predict(X_test)
    print(mean_squared_error(y_pred, y_test), get_other_metrics(y_pred, y_test))


# 做一些扩展性实验