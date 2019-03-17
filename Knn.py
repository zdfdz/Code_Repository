# -*-coding:utf-8-*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/ex4x.dat', names=['score1', 'score2', 'admit'])
df_admint = df[df['admit'] == 1]
df_noadmint = df[df['admit'] == 0]
plt.scatter(df_admint['score1'], df_admint['score2'], marker='o', c='r')
plt.scatter(df_noadmint['score1'], df_noadmint['score2'], marker='x', c='b')
plt.show()
y_train = df.pop('admit')
df = df.values

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=0)

for train_index, test_index in kf.split(df):
    df_tarin = df[train_index]
    y_tarin = y_train[train_index]

    df_test = df[test_index]
    y_test = y_train[test_index]

# NearestNeighbors 可以用来确定 这个一个附近的几个点
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(df_tarin,y_tarin)
y_predict = knn.predict(df_test)
print y_predict
from sklearn.metrics import accuracy_score,recall_score,f1_score
print "accuracy_score = %s"%accuracy_score(y_predict,y_test)
print "recall_score = %s"%recall_score(y_predict,y_test)
print "f1_score = %s"%f1_score(y_predict,y_test)
