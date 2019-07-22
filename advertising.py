import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np


def predict_sales():

    path = './doc/advertising.csv'
    data = pd.read_csv(path)
    x = data[['TV', 'radio', 'newspaper']]
    # print(x.head(6))
    y = data['sales']
    # print(y.head(6))

    # plt.figure(figsize=(9, 12))
    # plt.subplot(311)
    # plt.plot(data['TV'], y, 'ro')
    # plt.title('TV')
    # plt.grid()
    # plt.subplot(312)
    # plt.plot(data['radio'], y, 'g^')
    # plt.title('radio')
    # plt.grid()
    # plt.subplot(313)
    # plt.plot(data['newspaper'], y, 'b*')
    # plt.title('newspaper')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    line_reg = LinearRegression()
    model = line_reg.fit(x_train, y_train)
    # print(model)
    # print(model.intercept_)
    # print(model.coef_)
    # print(line_reg.intercept_)
    # print(line_reg.coef_)

    y_predit = line_reg.predict(x_test)
    # print(x_test)
    # print(y_predit)
    # print(type(y_predit))

    sum_mean = 0
    for i in range(len(y_predit)):
        sum_mean += (y_predit[i]-y_test.values[i])**2

    print("RMSE by brand 1:", np.sqrt(sum_mean/len(y_predit)))

    plt.figure(1)
    plt.plot(range(len(y_predit)),y_predit,'b',label='predict')
    plt.plot(range(len(y_predit)),y_test,'r',label='test')
    plt.legend(loc='upper right')
    plt.xlabel('the number of sales')
    plt.ylabel('the value of sales')
    plt.show()

    # 去除newspaper参数
    x = data[['TV','radio']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    line_reg = LinearRegression()
    model = line_reg.fit(x_train, y_train)
    y_predit = line_reg.predict(x_test)
    sum_mean = 0
    for i in range(len(y_predit)):
        sum_mean += (y_predit[i]-y_test.values[i])**2

    print("RMSE by brand 2:", np.sqrt(sum_mean/len(y_predit)))
    plt.figure(2)
    plt.plot(range(len(y_predit)),y_predit,'b',label='predict')
    plt.plot(range(len(y_predit)),y_test,'r',label='test')
    plt.legend(loc='upper right')
    plt.xlabel('the number of sales')
    plt.ylabel('the value of sales')
    plt.show()

if __name__ == '__main__':
    predict_sales()
