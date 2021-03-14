import time
import os, shutil
import sklearn.metrics
import autosklearn.regression
import numpy as np

from joblib import dump, load
from neuroc_pygs.sec5_memory.utils import get_automl_datasets, get_metrics
from neuroc_pygs.configs import PROJECT_PATH
from tabulate import tabulate

tmp_dir = os.path.join(PROJECT_PATH, 'sec5_memory', 'tmp')
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)

class AutoML(object):
    # https://github.com/automl/auto-sklearn/blob/58e36bea83c30a21872890c7b4b235fd198dcf7b/autosklearn/estimators.py#L16
    def __init__(self, time_left_for_this_task=120, per_run_time_limit=30, memory_limit=3072):
        self.automl = autosklearn.regression.AutoSklearnRegressor(
            # time_left_for_this_task=time_left_for_this_task,
            # per_run_time_limit=per_run_time_limit,
            # memory_limit=memory_limit,
            tmp_folder=tmp_dir + '/regression_tmp',
            output_folder=tmp_dir + '/regression_out'
        )
    
    def fit(self, x_train, y_train, dataset_name):
        self.automl.fit(x_train, y_train, dataset_name=dataset_name)
    
    def predict(self, x_test):
        return self.automl.predict(x_test)
    
    def show_models(self):
        return self.automl.show_models()
    
    def get_r2_score(self, y_pred, y_real):
        return sklearn.metrics.r2_score(y_real, y_pred)

    def save_model(self, file_name):
        dump(self.automl, file_name)
    
    def load_model(self, file_name):
        self.automl = load(file_name)



def automl_exp():
    for model in ['gcn', 'gat']:
        x_train, y_train, x_test, y_test = get_automl_datasets(model)

        # np.random.seed(1)
        # mask = np.arange(len(x_train))
        # np.random.shuffle(mask)
        # x_train, y_train = np.array(x_train)[mask], np.array(y_train)[mask]
        x_test, y_test = x_test[:1000], y_test[:1000]

        tab_data = []
        tab_data.append(['mse', 'high', 'bias', 'bias_per'])
        automl = AutoML()
        automl.fit(x_train, y_train, dataset_name=f'{model}')
        y_pred = automl.predict(x_test)
        print(y_pred)
        print(y_test)
        res = get_metrics(y_pred, y_test)
        tab_data.append(list(res))    

        print(tabulate(tab_data[1:], headers=tab_data[0], tablefmt="github"))

if __name__ == '__main__':
    automl_exp()
