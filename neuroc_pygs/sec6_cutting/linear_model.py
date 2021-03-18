import os, sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
from joblib import dump, load
from neuroc_pygs.configs import PROJECT_PATH

# 计算overhead
dir_path = os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_res')

def run_exp():
    for model in ['reddit_sage', 'cluster_gcn']:
        X, y = [], []
        for bs in [1024, 2048, 4096, 8192, 16384]:
            real_path = dir_path + f'/{model}_{bs}_v1.csv'
            df = pd.read_csv(real_path, index_col=0).values
            X.append(df[:,:-1]);  y.append(df[:,-1])

        X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)
        t1 = time.time()
        reg = LinearRegression().fit(X_train, y_train)
        t2 = time.time()
        dump(reg, dir_path + f'/{model}_linear_model_v2.pth')
        # reg = load(dir_path + f'/{model}_linear_model_v1.pth')
        y_pred = reg.predict(X_test)
        print(y_pred)
        t3 = time.time()
        r2 = r2_score(y_test, y_pred)
        mae, mape = mean_absolute_error(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred)
        print(f'{model}, r2={r2}, mae={mae}, mape={mape}')


if __name__ == '__main__':
    run_exp()