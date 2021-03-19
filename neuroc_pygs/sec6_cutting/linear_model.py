import os, sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
from joblib import dump
from neuroc_pygs.configs import PROJECT_PATH

# 计算overhead
dir_path = os.path.join(PROJECT_PATH, 'sec6_cutting', 'exp_diff_res')


df_data = {}
for model in ['reddit_sage', 'cluster_gcn']:
    X, y = [], []
    if model == 'reddit_sage':
        batch_sizes = [9000, 9100, 9200]
    else:
        batch_sizes = [2048, 4096, 8192]
        
    for bs in batch_sizes:
        real_path = dir_path + f'/{model}_{bs}_v0.csv'
        df = pd.read_csv(real_path, index_col=0).values
        X.append(df[:, :2]);  y.append(df[:,-1])

    X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    t1 = time.time()
    reg = LinearRegression().fit(X_train, y_train)
    t2 = time.time()
    dump(reg, dir_path + f'/{model}_linear_model_v0.pth')
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae, mape = mean_absolute_error(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred)
    print(f'{model}, r2={r2}, mae={mae}, mape={mape}')
    df_data[model] = [mape, t2 - t1]
pd.DataFrame(df_data, index=['mape', 'fit time']).to_csv(dir_path + '/regression_mape_res.csv')

