# -*-coding:utf-8-*-
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/ex4x.dat', names=['score1', 'score2', 'admit'])
df_admint = df[df['admit'] == 1]
df_noadmint = df[df['admit'] == 0]
plt.scatter(df_admint['score1'], df_admint['score2'], marker='o', c='r')
plt.scatter(df_noadmint['score1'], df_noadmint['score2'], marker='x', c='b')
plt.show()

df = df.as_matrix()
# print df
col = df.shape[1]
X = df[:, 0:col - 1]
y = df[:, col - 1:col]
# print X,y
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(x_train, x_test)

x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

LR = LogisticRegression()
scores = cross_val_score(LR, x_train_std, y_train, cv=10, scoring='neg_mean_squared_error')  # 计算均方误差
print scores.mean()
LR.fit(x_train_std, y_train)
# predict = LR.predict(x_test_std)
# print predict
# print(LR.coef_, LR.intercept_)
print LR.score(x_test_std, y_test)
print LR.score(x_train_std, y_train)

df = pd.read_csv('data/ex4y.dat',names=['score1','score2','admit'])
df = df.as_matrix()
col = df.shape[1]
X_pre = df[:,0:col-1]
x_pre_std = sc.transform(X_pre)
# print x_pre_std
pre = LR.predict(x_pre_std)
print '预测%s'%pre
print '实际%s'%df[:,col-1:col].T
print "--" * 30


# 顺便用多项式玩一下。。
from sklearn.preprocessing import PolynomialFeatures
for i in range(1,10):
    print "i = %s"%i
    pf = PolynomialFeatures(degree=i)
    x_train_std_pf = pf.fit_transform(x_train_std)
    x_test_std_pf = pf.transform(x_test_std)
    pf_score = cross_val_score(LR, x_train_std_pf, y_train, cv=10)
    print "交叉验证方差%s" % pf_score.mean()

    LR.fit(x_train_std_pf, y_train)
    print("mode score = %s" % LR.score(x_train_std_pf, y_train))
    print("mode predict score = %s" % LR.score(x_test_std_pf, y_test))


