import sklearn.datasets
import sklearn.metrics
import autosklearn.regression
from neuroc_pygs.sec5_memory.utils import get_datasets

X, y = get_datasets()

X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec5_memory/tmp/autosklearn_regression_example_tmp',
    output_folder='/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec5_memory/tmp/autosklearn_regression_example_out',
)
automl.fit(X_train, y_train, dataset_name='memory_model')

print(automl.show_models())
predictions = automl.predict(X_test)
print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))

from joblib import dump, load
dump(automl, 'filename.joblib') 

# clf = load('filename.joblib') 