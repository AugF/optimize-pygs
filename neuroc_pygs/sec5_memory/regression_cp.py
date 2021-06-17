import os, time
import numpy as np 
import pandas as pd
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from joblib import dump
from collections import defaultdict
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.sec5_memory.configs import MODEL_PARAS
from sklearn import svm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
from matplotlib.font_manager import _rebuild
_rebuild() 

base_size = 14
plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = base_size

# 构造数据集
dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_automl_datasets_diff')


def run_linear_model(model='gcn', data='reddit'):
    real_path = dir_path + f'/{model}_{data}_automl_model_diff_v2.csv'
    df = pd.read_csv(real_path, index_col=0).values
    X, y = np.array(df[:, :2], dtype=np.float32), np.array(df[:, -1], dtype=np.float32)
    reg = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test) 
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f'r2={r2}, mape={mape}')
    dump(reg, dir_path + f'/{model}_{data}_linear_model_diff_v2.pth')
    return mape, r2


def save_model(files, model, file_type):
    X, y = [], []
    for file in files:
        real_path = dir_path + f'/{model}_{file}_automl_model_diff_v2.csv'
        df = pd.read_csv(real_path, index_col=0).values
        X.append(df[:,:-2]);  y.append(df[:,-1])

    X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    if file_type == 'linear_model':
        X = X[:, :2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)
    # X_train, y_train = X, y
    if file_type == 'automl':
        reg = RandomForestRegressor(random_state=1) # Random Forest
        reg.fit(X_train, y_train)
    elif file_type == 'linear_model':
        reg = LinearRegression()  # Random Forest
        reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # dump(reg, dir_path + f'/{model}_{file_type}_diff_v2.pth')
    return mape, r2


def get_train_time(files, model, file_type):
    X, y = [], []
    for file in files:
        real_path = dir_path + f'/{model}_{file}_automl_model_diff_v2.csv'
        df = pd.read_csv(real_path, index_col=0).values
        X.append(df[:,:-2]);  y.append(df[:,-1])

    X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    if file_type == 'linear_model':
        X = X[:, :2]
    print(len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)
    # X_train, y_train = X, y
    if file_type == 'automl':
        reg = RandomForestRegressor(random_state=1) # Random Forest
    elif file_type == 'linear_model':
        reg = LinearRegression()  # Random Forest

    # for i in range(50, len(X_train), 50):  
    t1 = time.time()
    # reg.fit(X_train[:i], y_train[:i])
    reg.fit(X_train[:120], y_train[:120])
    t2 = time.time()
    print(t2 - t1)
        # print(i, t2-t1)
    return


if __name__ == '__main__':
    # run automl
    df = defaultdict(list)
    files = ['classes', 'nodes_edges', 'features', 'reddit', 'yelp',  'paras']
    for model in ['gat']:
        # get_train_time(files, model, file_type='automl')
        get_train_time(files, model, file_type='linear_model')

        # run_automl(files, model, file_type='automl') 
    #     mape, r2 = save_model(files, model=model, file_type='automl')
    #     print(f'model: {model}, automl mape: {mape}, r2: {r2}')
    #     df[model].append(mape)
    # pd.DataFrame(df, index=['automl']).to_csv(dir_path + f'/regression_mape_res.csv') 
    
    # run linear
    # df = defaultdict(list)
    # for model in ['gcn', 'gat']:
    #     for data in ['reddit', 'yelp']:
    #         mape, r2 = run_linear_model(model, data)
    #         print(f'model: {model}, data: {data}, mape: {mape}, r2: {r2}')
    #         df[model + '_' + data].extend([mape, r2])
    # pd.DataFrame(df, index=['mape', 'r2']).to_csv(dir_path + f'/regression_linear_model_metrics.csv') 