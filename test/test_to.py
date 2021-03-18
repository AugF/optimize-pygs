from sklearn.neighbors import (NeighborhoodComponentsAnalysis,
KNeighborsClassifier)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,
stratify=y, test_size=0.7, random_state=42)

# clf = RandomForestRegressor()
# clf = LinearRegression()
clf = SGDRegressor(random_state=1)

clf.partial_fit(X_train[:20], y_train[:20])
clf.partial_fit(X_train[20:], y_train[20:])
# clf.fit(X, y)
print(clf.score(X_test, y_test))