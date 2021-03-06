import os
import numpy as np
from sklearn.linear_model import LinearRegression
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.sec5_memory.utils import get_metrics
from tabulate import tabulate


def get_X_y(model='gat'):
    tab_data = np.load(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_res', f'{model}_memory_2dims_curve_data_final.npy'))
    nodes = list(map(lambda x: int(x), tab_data[:, 1]))
    edges = list(map(lambda x: int(x), tab_data[:, 2]))
    memory = list(map(lambda x: int(x) / (1024*1024), tab_data[:, 3]))

    X = np.array([nodes, edges]).T
    y = np.array(memory)
    return X, y


# 随机10轮，给出一个百分比: 参考from sklearn.model_selection import train_test_split, 默认划分比例为test_size=int(0.25);
def experiment(X_train, y_train, X_test, y_test):
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return get_metrics(y_pred, y_test)


def linear_model_exp():
    # 做一些扩展性实验; 实验总数扩展, 表示可以进行运行时更新
    for model in ['gcn', 'gat']:
        X, y = get_X_y(model)
        # np.random.seed(1)
        # mask = np.arange(len(X))
        # np.random.shuffle(mask)
        # X, y = X[mask], y[mask]
        tab_data = []
        tab_data.append(['num_nodes', 'mse', 'high', 'bias', 'bias_per'])
        X_test, y_test = X[-500:], y[-500:]
        for end_step in range(500, 2001, 500):
            tab_data.append([end_step] + list(experiment(X[:end_step], y[:end_step], X_test, y_test)))
        print(tabulate(tab_data[1:], headers=tab_data[0], tablefmt="github"))
        np.save(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_res', f'{model}_linear_model_res.npy'), np.array(tab_data))
    # f"peak: {r_peak.mean():.4f} ± {r_peak.std():.4f} "
    # gcn: len(X)=2604
    # gat: len(X)=2604


if __name__ == '__main__':
    linear_model_exp()