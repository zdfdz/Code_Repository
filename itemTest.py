# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


X = np.array([1, 5, 7, 9, 13, 16])
Y = np.array([37, 66, 71, 79, 85, 99])
# 维度
print X.shape
plt.scatter(x=X, y=Y, marker='o', c='r')

def model_fun(theta0, theta1, x0):
    h = theta0 + theta1*x0
    return h

epsilon = 0.0001
alpha = 0.001
max_itor = 10000
diff =[0,0]

error_one = 0
error_two = 0


theta0 = 0
theta1 = 0
count = 0
while(count<max_itor):
    count+=1
    for i in range(0,len(X)):
        diff[0] = model_fun(theta0,theta1,X[i])- Y[i]
        theta0 -= alpha*diff[0]
        theta1 -= alpha*diff[0]*X[i]
    error_one = 0
    for i in range(0,len(X)):
        error_one+=((model_fun(theta0,theta1,X[i])- Y[i]) **2)/2*len(X)

    if abs(error_one-error_two)<epsilon:
        break
    error_two = error_one
    print theta0,theta1