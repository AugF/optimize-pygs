import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.sec5_memory.utils import get_other_metrics


def get_X_y(model='gat'):
    tab_data = np.load(os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_res', f'{model}_memory_2dims_curve_data_final.npy'))
    nodes = list(map(lambda x: int(x), tab_data[:, 1]))
    edges = list(map(lambda x: int(x), tab_data[:, 2]))
    memory = list(map(lambda x: int(x) / (1024*1024), tab_data[:, 3]))

    X = np.array([nodes, edges]).T
    y = np.array(memory)
    return X, y


# 随机10轮，给出一个百分比: 参考from sklearn.model_selection import train_test_split, 默认划分比例为test_size=int(0.25);
def experiment(X, y):
    r_mse, r_high, r_peak, r_peak_percent = [], [], [], []
    num_dataset = len(X)
    split = int(np.ceil(num_dataset * 0.25 + 0.5))
    for i in range(10):
        np.random.seed(1)
        mask = np.arange(num_dataset)
        np.random.shuffle(mask)
        X_train, y_train = X[mask[:split]], y[mask[:split]]
        X_test, y_test = X[mask[split:]], y[mask[split:]]

        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        r_mse.append(mean_squared_error(y_pred, y_test))
        res = get_other_metrics(y_pred, y_test)
        r_high.append(res[0])
        r_peak.append(res[1])
        r_peak_percent.append(res[2])
    r_mse, r_high, r_peak, r_peak_percent = np.array(r_mse), np.array(r_high), np.array(r_peak), np.array(r_peak_percent)
    print(f"num_dataset: {num_dataset}, "
            f"mse: {r_mse.mean():.4f} ± {r_mse.std():.4f} "
            f"high: {r_high.mean():.4f} ± {r_high.std():.4f} "
            f"peak: {r_peak.mean():.4f} ± {r_peak.std():.4f} "
            f"peak_percent: {r_peak_percent.mean():.4f} ± {r_peak_percent.std():.4f} ") # 1e-17


# 做一些扩展性实验; 实验总数扩展, 表示可以进行运行时更新
for model in ['gcn', 'gat']:
    X, y = get_X_y(model)
    print(y) # MB
    exit(0)
    np.random.seed(1)
    mask = np.arange(2500)
    np.random.shuffle(mask)
    X, y = X[mask], y[mask]
    for end_step in range(500, 2501, 500):
        experiment(X[:end_step], y[:end_step])