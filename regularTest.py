# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import leastsq

# 采样
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([6, 15, 30, 46, 56, 80, 130, 180, 220, 287])
plt.scatter(x, y, marker='x')
x = x.reshape(-1, 1)
plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(x, y)
x_train_std = ss.transform(x_train)
x_test_std = ss.transform(x_test)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train_std, y_train)
print("Coefficients:%s, intercept %s" % (lr.coef_, lr.intercept_))
print('Score: %.2f' % lr.score(x_test_std, y_test))

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

lr_featurizer = PolynomialFeatures(degree=6)
X_pf_train = lr_featurizer.fit_transform(x_train)
X_pf_test = lr_featurizer.transform(x_test)

# pf_scores = cross_val_score(lr, X_pf_train, y_train, cv=10, scoring='neg_mean_squared_error')
# print pf_scores.mean()

lr.fit(X_pf_train, y_train)
print lr.score(X_pf_test, y_test)
print lr.score(X_pf_train, y_train)
print("Coefficients:%s, intercept %s" % (lr.coef_, lr.intercept_))
# print('Score: %.2f' % lr.score(x_test_std, y_test))
