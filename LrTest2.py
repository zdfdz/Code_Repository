# -*- coding:UTF-8 -*-
import time
from __builtin__ import classmethod

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.cross_validation import train_test_split

df = pd.read_csv('data/ex4x.dat', names=['Exam 1', 'Exam 2', 'Admin'])
# print df
positive = df[df['Admin'] == 1]
negative = df[df['Admin'] == 0]
plt.scatter(positive['Exam 1'], positive['Exam 2'], marker='o', c='b', linewidths=2)
plt.scatter(negative['Exam 1'], negative['Exam 2'], marker='x', c='r', linewidths=2)

# df.insert(0,"ones",1)
df_std = df.as_matrix()
col = df_std.shape[1]
# print col
X = df_std[:, 0:col - 1]
y = df_std[:, col - 1:col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 特征值缩放
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 训练感知机模型
from sklearn.linear_model import Perceptron, LogisticRegression

# n_iter：可以理解成梯度下降中迭代的次数
# eta0：可以理解成梯度下降中的学习率
# random_state：设置随机种子的，为了每次迭代都有相同的训练集顺序
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
# 进行预测
y_pred = ppn.predict(X_test_std)
print "匹配度为" + str(accuracy_score(y_test, y_pred))


import matplotlib.pyplot as plt
cls = LogisticRegression()
# 把数据交给模型训练
cls.fit(X_train_std, y_train)
print("Coefficients:%s, intercept %s" % (cls.coef_, cls.intercept_))
print("Residual sum of squares: %.2f" % np.mean((cls.predict(X_test_std) - y_test) ** 2))
print('Score: %.2f' % cls.score(X_test_std, y_test))

