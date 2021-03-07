
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,classification_report
from sklearn.cluster import KMeans
 
 
def linearmodel():
    """
    线性回归对波士顿数据集处理
    :return: None
    """
    ld = load_boston()
    x_train,x_test,y_train,y_test = train_test_split(ld.data,ld.target.reshape(-1, 1))
    print(x_train.shape, x_test.shape)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    std_y  = StandardScaler()
    y_train = std_y.fit_transform(y_train)
    y_test = std_y.transform(y_test)

    # LinearRegression
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    y_lr_predict = lr.predict(x_test)
    y_lr_predict = std_y.inverse_transform(y_lr_predict)

    # SGDRegressor
    sgd = SGDRegressor()
    sgd.fit(x_train,y_train.ravel())

    y_sgd_predict = sgd.predict(x_test)

    y_sgd_predict = std_y.inverse_transform(y_sgd_predict)


    # print("SGD预测值：",y_sgd_predict)

    # 带有正则化的岭回归

    rd = Ridge(alpha=0.01)

    rd.fit(x_train,y_train.ravel())

    y_rd_predict = rd.predict(x_test)

    y_rd_predict = std_y.inverse_transform(y_rd_predict)

    # print(rd.coef_)

    # 两种模型评估结果

    print("lr的均方误差为：",mean_squared_error(std_y.inverse_transform(y_test),y_lr_predict))

    print("SGD的均方误差为：",mean_squared_error(std_y.inverse_transform(y_test),y_sgd_predict))

    print("Ridge的均方误差为：",mean_squared_error(std_y.inverse_transform(y_test),y_rd_predict))


if __name__ == '__main__':
    linearmodel()