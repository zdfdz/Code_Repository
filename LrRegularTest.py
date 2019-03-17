# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/data2.txt', names=['1', '2', 'admit'])
df_positive = df[df['admit'] == 1]
df_negative = df[df['admit'] == 0]
plt.scatter(df_positive['1'], df_positive['2'], marker='o')
plt.scatter(df_negative['1'], df_negative['2'], marker='x')
plt.show()

df = df.as_matrix()
col = df.shape[1]
print col
X = df[:, 0:col - 1]
y = df[:, col - 1:col]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(x_train, x_test)
x_train_std = ss.transform(x_train)
x_test_std = ss.transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

lr = LogisticRegression()
scores = cross_val_score(lr, x_train_std, y_train, cv=10, scoring='neg_mean_squared_error')  # 计算均方误差
print 'var = %s' % scores.mean()
lr.fit(x_train_std, y_train)
print 'score = %s' % lr.score(x_train_std, y_train)
print 'predict = %s' % lr.score(x_test_std, y_test)

from sklearn.preprocessing import PolynomialFeatures

for i in range(1, 10):
    print 'i = %s' % i
    pf = PolynomialFeatures(degree=i)
    x_train_std_pf = pf.fit_transform(x_train_std)
    x_test_std_pf = pf.transform(x_test_std)
    score = cross_val_score(lr, x_train_std_pf, y_train, cv=10, scoring='neg_mean_squared_error')
    print 'var = %s' % score.mean()
    lr.fit(x_train_std_pf, y_train)
    print 'model score = %s' % lr.score(x_train_std_pf, y_train)
    print 'predict = %s' % lr.score(x_test_std_pf, y_test)


print '---'*20
pf = PolynomialFeatures(degree=3)
x_train_std_pf = pf.fit_transform(x_train_std)
x_test_std_pf = pf.transform(x_test_std)
score = cross_val_score(lr, x_train_std_pf, y_train, cv=10, scoring='neg_mean_squared_error')
print 'var = %s' % score.mean()
lr.fit(x_train_std_pf, y_train)
print 'model score = %s' % lr.score(x_train_std_pf, y_train)
print 'predict= %s' % lr.score(x_test_std_pf, y_test)
