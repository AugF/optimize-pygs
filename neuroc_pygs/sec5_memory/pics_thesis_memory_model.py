import os, time
import numpy as np 
import pandas as pd
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from collections import defaultdict
from neuroc_pygs.configs import PROJECT_PATH
from sklearn import svm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from matplotlib.font_manager import _rebuild
_rebuild()
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)

base_size = 12
plt.style.use("grayscale")
plt.rcParams['axes.unicode_minus']=False

dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'out_random_forest_csv')


def make_contrast(X, y):
    filenames = ['Random Forest', 'GBDT', 'Linear Regreesion', 'AdaBoost', 'DecisionTree']
    reg1 = RandomForestRegressor(random_state=1) # Random Forest
    reg2 = GradientBoostingRegressor(random_state=1) # GBDT
    reg3 = LinearRegression() # Linear Regression
    reg4 = AdaBoostRegressor(random_state=0) # AdaBoost, random_state=0, n_estimators=100
    reg5 = DecisionTreeRegressor() # DecisionTree, max_depth=5
    
    df_r2, df_mape = defaultdict(list), defaultdict(list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=0)
    for i, reg in enumerate([reg1, reg2, reg3, reg4, reg5]):
        for step in range(50, 651, 50):
            reg.fit(X_train[:step], y_train[:step])
            y_pred = reg.predict(X_test) 
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            df_r2[filenames[i]].append(r2)
            df_mape[filenames[i]].append(mape)
    return df_r2, df_mape


def run_linear_model(model='gcn', data='reddit'):
    real_path = dir_path + f'/{model}_{data}_automl_model.csv'
    df = pd.read_csv(real_path, index_col=0).values
    X, y = np.array(df[:, :2], dtype=np.float32), np.array(df[:, -1], dtype=np.float32)
    reg = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test) 
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f'r2={r2}, mape={mape}')
    dump(reg, dir_path + f'/{model}_{data}_linear_model_diff_v2.pth')
    return mape, r2


def run_automl(files, model='gcn', file_type='automl'):
    X, y = [], []
    for file in files:
        real_path = dir_path + f'/{model}_{file}_automl_model.csv'
        df = pd.read_csv(real_path, index_col=0).values
        X.append(df[:,:-2]);  y.append(df[:,-1])

    X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    if file_type == 'linear_model':
        X = X[:, :2]

    df_r2, df_mape = train_model(X, y)
    df_r2, df_mape = pd.DataFrame(df_r2), pd.DataFrame(df_mape)
    titles = ['决定系数 (R2)', '平均绝对百分比误差 (MAPE)']
    names = ['r2', 'mape', 'mae', 'mpe']
    markers = 'oD^sdp'
    colors = plt.get_cmap('Dark2')(np.linspace(0.15, 0.85, 5))
    # RdYlGn, Greys, Dark2
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 10, 1, 10))]
    xs = list(range(50, 651, 100))
    x = np.arange(len(xs) * 2)
    xticklabels = []
    for t in xs:
        xticklabels.extend([t, None])

    for i, df in enumerate([df_r2, df_mape]):
        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        ax.set_xlabel('训练集规模', fontsize=base_size + 2)
        ax.set_ylabel(titles[i], fontsize=base_size + 2)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=base_size + 2)
        
        x_smooth = np.linspace(df.index.min(), df.index.max(), 300)
        for j, c in enumerate(df.columns):
            y_smooth = make_interp_spline(df.index, df[c])(x_smooth)
            ax.plot(x_smooth, y_smooth, label=c, color=colors[j], linestyle=linestyles[j], linewidth=2)

        ax.legend()
        fig.savefig(f'exp5_thesis_figs/memory_model/exp_memory_training_{model}_{file_type}_{names[i]}_diff.png', dpi=400)
        

def get_automl():
    files = ['classes', 'nodes_edges', 'features', 'reddit', 'yelp',  'paras']
    for model in ['gcn', 'gat']:
        run_automl(files, model)


# for model in ['gcn', 'gat']:
#     for data in ['reddit', 'yelp']:
#         for bs in [1]