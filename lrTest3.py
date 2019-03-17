# -*-coding:utf-8-*-
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/LRtestData.txt', names=['x1', 'x2', 'y'])
# print df
positive = df[df['y'] == 1]
nagative = df[df['y'] == 0]
plt.scatter(positive['x1'],positive['x2'],c='r',marker='o')
plt.scatter(nagative['x1'],nagative['x2'],c = 'b',marker='x')
plt.show()


df_std = df.as_matrix()
col = df_std.shape[1]
X = df_std[:, 0:col - 1]
y = df_std[:, col - 1:col]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# n_iter：可以理解成梯度下降中迭代的次数
# eta0：可以理解成梯度下降中的学习率
# random_state：设置随机种子的，为了每次迭代都有相同的训练集顺序
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train, y_train)
# 进行预测
y_pred = ppn.predict(X_test)
print "匹配度为" + str(accuracy_score(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
cls = LogisticRegression()
cls.fit(X_train,y_train)

print cls.predict(X_test)
print("Coefficients:%s, intercept %s" % (cls.coef_, cls.intercept_))
print('Score: %.2f' % cls.score(X_test, y_test))