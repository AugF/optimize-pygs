import os
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

plt.style.use("grayscale")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = 16

dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_automl_datasets_diff')

def mean_percentage_error(y_true, y_pred, sample_weight=None,
                                   multioutput='uniform_average'):
    # https://github.com/scikit-learn/scikit-learn/blob/04f84c6d0/sklearn/metrics/_regression.py#L280
    epsilon = np.finfo(np.float64).eps
    mape = (y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape,
                               weights=sample_weight, axis=0)
    output_errors = np.average(mape,
                               weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def train_model(X, y):
    filenames = ['Random Forest', 'GBDT', 'Linear Regreesion', 'AdaBoost', 'DecisionTree']
    reg1 = RandomForestRegressor(random_state=1) # Random Forest
    reg2 = GradientBoostingRegressor(random_state=1) # GBDT
    reg3 = LinearRegression() # Linear Regression
    reg4 = AdaBoostRegressor(random_state=0) # AdaBoost, random_state=0, n_estimators=100
    reg5 = DecisionTreeRegressor() # DecisionTree, max_depth=5
    
    df_r2, df_mae, df_mape, df_mpe = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=0)
    for i, reg in enumerate([reg1, reg2, reg3, reg4, reg5]):
        for step in range(50, len(y_train) + 1, 50):
            reg.fit(X_train[:step], y_train[:step])
            y_pred = reg.predict(X_test) 
            r2 = r2_score(y_test, y_pred)
            mae, mape = mean_absolute_error(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred)
            mpe = mean_percentage_error(y_test, y_pred)
            # print(f'{filenames[i]}, r2={r2}, mae={mae}, mape={mape}, mpe={mpe}')
            df_r2[filenames[i]].append(r2)
            df_mae[filenames[i]].append(mae)
            df_mape[filenames[i]].append(mape)
            df_mpe[filenames[i]].append(mpe)
    return df_r2, df_mae, df_mape, df_mpe
            

def run_automl(files, model='gcn', file_type='automl'):
    X, y = [], []
    for file in files:
        real_path = dir_path + f'/{model}_{file}_automl_model_diff_v2.csv'
        df = pd.read_csv(real_path, index_col=0).values
        X.append(df[:,:-2]);  y.append(df[:,-1])

    X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    if file_type == 'linear_model':
        X = X[:, :2]

    df_r2, df_mae, df_mape, df_mpe = train_model(X, y)
    df_r2, df_mae, df_mape, df_mpe = pd.DataFrame(df_r2), pd.DataFrame(df_mae), pd.DataFrame(df_mape), pd.DataFrame(df_mpe)
    titles = ['决定系数 (R2)', '平均绝对误差 (MAE)', '平均绝对百分比误差 (MAPE)', '平均百分比误差 (MPE)']
    names = ['r2', 'mae', 'mape', 'mpe']
    markers = 'oD^sdp'
    colors = plt.get_cmap('Dark2')(np.linspace(0.15, 0.85, 5))
    # RdYlGn, Greys, Dark2
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 10, 1, 10))]
    # https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
    xs = list(range(50, len(y) - 49, 50))
    x = np.arange(len(xs))
    for i, df in enumerate([df_r2, df_mae, df_mape, df_mpe]):
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        ax.set_xlabel('训练集规模', fontsize=18)
        ax.set_ylabel(titles[i], fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(xs, fontsize=18)
        
        x_smooth = np.linspace(df.index.min(), df.index.max(), 300)
        # x_smooth = df.index
        for j, c in enumerate(df.columns):
            y_smooth = make_interp_spline(df.index, df[c])(x_smooth)
            # y_smooth = df[c]
            ax.plot(x_smooth, y_smooth, label=c, color=colors[j], linestyle=linestyles[j], linewidth=2)

        ax.legend(fontsize=16)
        fig.savefig(os.path.join(PROJECT_PATH, 'sec5_memory') + f'/exp_figs/exp_{file_type}_{model}_{names[i]}_diff.png')


def run_linear_model(model='gcn', data='reddit'):
    real_path = dir_path + f'/{model}_{data}_automl_model_diff_v2.csv'
    df = pd.read_csv(real_path, index_col=0).values
    X, y = np.array(df[:, :2], dtype=np.float32), np.array(df[:, -1], dtype=np.float32)
    reg = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test) 
    r2 = r2_score(y_test, y_pred)
    mae, mape = mean_absolute_error(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred)
    mpe = mean_percentage_error(y_test, y_pred)
    print(f'r2={r2}, mae={mae}, mape={mape}, mpe={mpe}')
    dump(reg, dir_path + f'/{model}_{data}_linear_model_diff_v2.pth')
    return mape, mpe


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
    mpe = mean_percentage_error(y_test, y_pred)
    dump(reg, dir_path + f'/{model}_{file_type}_diff_v2.pth')
    return mape, mpe


if __name__ == '__main__':
    # run automl
    df = defaultdict(list)
    files = ['classes', 'nodes_edges', 'features', 'reddit', 'yelp',  'paras']
    for model in ['gcn', 'gat']:
        run_automl(files, model, file_type='automl') 
        mape, mpe = save_model(files, model=model, file_type='automl')
        print(f'model: {model}, automl mape: {mape}, mpe: {mape}')
        df[model].append(mape)
    pd.DataFrame(df, index=['automl']).to_csv(dir_path + f'/regression_mape_res.csv') 
    
    # run linear
    df = defaultdict(list)
    for model in ['gcn', 'gat']:
        for data in ['yelp', 'reddit']:
            mape, _ = run_linear_model(model, data)
            df[model].append(mape)
    pd.DataFrame(df, index=['reddit', 'yelp']).to_csv(dir_path + f'/regression_linear_mape_res.csv') 