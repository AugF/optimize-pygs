import os
import copy
import pandas as pd

from neuroc_pygs.configs import PROJECT_PATH

# 合并文件
def get_datasets(model_name):
    dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'sec5_2_memory_log')
    df = pd.read_csv(dir_path + '/{model_name}_datasets.csv', index_col=0)
    y = df['memory'].values
    del df['memory']
    X = df.values
    return X, y

