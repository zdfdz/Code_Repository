# -*-coding:utf-8-*-
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/data2.txt', names=['S1', 'S2', 'target'])
df_0 = df[df['target'] == 0]
df_1 = df[df['target'] == 1]
plt.scatter(df_1['S1'], df_1['S2'], marker='o', c='r')
plt.scatter(df_0['S1'], df_0['S2'], marker='x', c='b')
plt.show()

X = df.drop('target',axis=1)
y = df['target']
print X.shape,y.shape

from sklearn.model_selection import train_test_split
X_tarin, X_test, y_tarin, y_test = train_test_split(X, y, test_size=0.3)
print X_tarin.shape,y_tarin.shape

from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
#
gn = GaussianNB()
gn.fit(X_tarin, y_tarin)
y_predict = gn.predict(X_test)
print "accuracy_score = %s" % accuracy_score(y_predict, y_test)
print "recall_score = %s" % recall_score(y_predict, y_test)
print "f1_score = %s" % f1_score(y_predict, y_test)
