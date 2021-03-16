import time
import os, shutil
import sklearn.metrics
import autosklearn.regression
import numpy as np

from joblib import dump, load
from neuroc_pygs.sec5_memory.utils import get_automl_datasets, get_metrics
from neuroc_pygs.configs import PROJECT_PATH
from tabulate import tabulate
from sklearn.metrics import mean_squared_error

tmp_dir = os.path.join(PROJECT_PATH, 'sec5_memory', 'tmp')
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)

dir_path = os.path.join(PROJECT_PATH, 'sec5_memory', 'exp_motivation_datasets')

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

        automl = AutoML()
        automl.fit(x_train, y_train, dataset_name=f'{model}')
        y_pred = automl.predict(x_test)
        automl.save_model(dir_path + f'/{model}_automl_v1.pth')
        mse = mean_squared_error(y_pred, y_test)
        max_bias, max_bias_per = 0, 0
        for i in range(1000):
            max_bias = max(max_bias, abs(y_pred[i] - y_test[i]))
            max_bias_per = max(max_bias_per, abs(y_pred[i] - y_test[i]) / y_pred[i])   
        print([model, mse, max_bias, max_bias_per])    


if __name__ == '__main__':
    automl_exp()
