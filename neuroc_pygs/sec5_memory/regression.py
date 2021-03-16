from sklearn import svm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor

X, y = make_regression(random_state=0)
ss = StandardScaler()
# ss = MinMaxScaler()
X_std, y_std = ss.fit_transform(X), ss.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
reg1 = GradientBoostingRegressor(random_state=1) # GBDT
reg2 = RandomForestRegressor(random_state=1) # Random Forest
reg3 = LinearRegression() # Linear Regression
reg4 = AdaBoostRegressor(random_state=0, n_estimators=100) # AdaBoost
reg5 = DecisionTreeRegressor(max_depth=5) # DecisionTree
reg6 = svm.SVR() # SVM

for reg in [reg1, reg2, reg3, reg4, reg5, reg6]:
    reg.fit(X_train, y_train)
    reg.predict(X_test[1:2]) # [[]]
    reg.score(X_test, y_test)
