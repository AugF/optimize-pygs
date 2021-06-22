import os, sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from joblib import dump


for model in ['reddit_sage', 'cluster_gcn']:
    X, y = [], []
        
    for bs in [1024, 2048, 3096]:
        real_path = f'out_linear_model_datasets/{model}_{bs}.csv'
        df = pd.read_csv(real_path, index_col=0).values
        X.append(df[:, :2]);  y.append(df[:,-1])

    X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    t1 = time.time()
    reg = LinearRegression().fit(X_train, y_train)
    t2 = time.time()

    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mape =  mean_absolute_percentage_error(y_test, y_pred)
    print(f'{model}, r2={r2}, mape={mape}')

    dump(reg, f'out_linear_model_pth/{model}.pth')
    with open(f'out_linear_model_pth/{model}.txt', 'w') as f:
        f.write(str(mape))
    

