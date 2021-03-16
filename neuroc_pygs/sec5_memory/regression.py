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

dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_automl_datasets')

def train_model(X, y):
    filenames = ['GBDT', 'Random Forest', 'Linear Regreesion', 'AdaBoost', 'DecisionTree']
    reg1 = GradientBoostingRegressor(random_state=1) # GBDT
    reg2 = RandomForestRegressor(random_state=1) # Random Forest
    reg3 = LinearRegression() # Linear Regression
    reg4 = AdaBoostRegressor(random_state=0) # AdaBoost, random_state=0, n_estimators=100
    reg5 = DecisionTreeRegressor() # DecisionTree, max_depth=5
    
    df_r2, df_mae, df_mape = defaultdict(list), defaultdict(list), defaultdict(list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)
    for i, reg in enumerate([reg1, reg2, reg3, reg4, reg5]):
        for step in range(50, 451, 50):
            reg.fit(X_train[:step], y_train[:step])
            y_pred = reg.predict(X_test) 
            r2 = r2_score(y_test, y_pred)
            mae, mape = mean_absolute_error(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred)
            print(f'{filenames[i]}, r2={r2}, mae={mae}, mape={mape}')
            df_r2[filenames[i]].append(r2)
            df_mae[filenames[i]].append(mae)
            df_mape[filenames[i]].append(mape)
    return df_r2, df_mae, df_mape
            

def run_automl(files, model='gcn', file_type='automl'):
    X, y = [], []
    for file in files:
        real_path = dir_path + f'/{model}_{file}_automl_model_v2.csv'
        df = pd.read_csv(real_path, index_col=0).values
        X.append(df[:,:-1]);  y.append(df[:,-1])

    X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    df_r2, df_mae, df_mape = train_model(X, y)
    df_r2, df_mae, df_mape = pd.DataFrame(df_r2), pd.DataFrame(df_mae), pd.DataFrame(df_mape)
    titles = ['决定系数 (R2)', '平均绝对误差 (MAE)', '平均绝对百分比误差 (MAPE)']
    names = ['r2', 'mae', 'mape']
    markers = 'oD^sdp'
    xs = list(range(50, 451, 50))
    x = np.arange(len(xs))
    for i, df in enumerate([df_r2, df_mae, df_mape]):
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        ax.set_xlabel('训练集规模', fontsize=18)
        ax.set_ylabel(titles[i], fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(xs, fontsize=18)
        
        # x_smooth = np.linspace(df.index.min(), df.index.max(), 300)
        x_smooth = df.index
        for j, c in enumerate(df.columns):
            # y_smooth = make_interp_spline(df.index, df[c])(x_smooth)
            y_smooth = df[c]
            ax.plot(x_smooth, y_smooth, marker=markers[j], markersize=8, label=c)
        ax.legend(fontsize=16)
        fig.savefig(os.path.join(PROJECT_PATH, 'sec5_memory') + f'/exp_figs/exp_{file_type}_{model}_{names[i]}.png')
    
           
def save_model(files, model, file_type):
    X, y = [], []
    for file in files:
        real_path = dir_path + f'/{model}_{file}_automl_model_v2.csv'
        df = pd.read_csv(real_path, index_col=0).values
        X.append(df[:,:-1]);  y.append(df[:,-1])

    X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)
    if file_type == 'automl':
        reg = RandomForestRegressor(random_state=1) # Random Forest
        reg.fit(X_train, y_train)
    elif file_type == 'linear_model':
        reg = LinearRegression()  # Random Forest
        reg.fit(X_train, y_train)
        
    dump(reg, dir_path + f'/{model}_{file_type}_v2.pth')


if __name__ == '__main__':
    files = ['classes', 'nodes_edges', 'features', 'paras', 'paras_v3', 'paras_v4']
    # files = ['nodes_edges']
    # for model in ['gcn', 'gat']:
        # run_automl(files, model, file_type='automl') 

    # for model in ['gcn', 'gat']:
    #     run_automl(files, model, file_type='linear_model')
    for model in ['gcn', 'gat']:
        save_model(files, model=model, file_type='automl')
        # save_model(files=['nodes_edges'], model=model, file_type='linear_model')
