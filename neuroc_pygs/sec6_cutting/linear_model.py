import os, sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from neuroc_pygs.configs import PROJECT_PATH


dir_path = os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_res')

def run_exp():
    for model in ['reddit_sage']:
        X, y = None, None
        for bs in [1024, 2048, 4096, 8192, 16384]:
            real_path = dir_path + f'/{model}_{bs}_v1.csv'
            df = pd.read_csv(real_path, index_col=0)
            X1, y1 = df[['nodes', 'edges']].values.tolist(), (df['memory'].values / (1024 * 1024)).tolist()
            if X is None:
                X, y = X1, y1
            else:
                X.extend(X1)
                y.extend(y1)
        X, y = np.array(X), np.array(y)
        np.random.seed(1)
        mask = np.arange(len(y))
        np.random.shuffle(mask)
        X, y = X[mask], y[mask]
        X_train, y_train, X_test, y_test = X[:-100], y[:-100], X[-100:], y[-100:]
        t1 = time.time()
        reg = LinearRegression().fit(X_train, y_train)
        t2 = time.time()
        dump(reg, dir_path + f'/{model}_{bs}_v1.pth')
        # reg = load(dir_path + f'/{model}_linear_model_v0.pth')
        y_pred = reg.predict(X_test)
        t3 = time.time()
        mse = mean_squared_error(y_pred, y_test)
        max_bias, max_bias_per = 0, 0
        for i in range(100):
            max_bias = max(max_bias, abs(y_pred[i] - y_test[i]))
            max_bias_per = max(max_bias_per, abs(y_pred[i] - y_test[i]) / y_pred[i])
        print(model, t2 - t1, (t3 - t2) / 100)
        print(model, mse, max_bias, max_bias_per)


if __name__ == '__main__':
    run_exp()