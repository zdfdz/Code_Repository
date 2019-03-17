# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pdData = pd.read_csv("data/ex4x.dat",names = ['Exam 1','Exam 2','admin'])

positive = pdData[pdData['admin'] == 1]
negative = pdData[pdData['admin'] == 0]

plt.scatter(positive['Exam 1'],positive['Exam 2'],s = 30,c = 'b',marker = 'o',label = 'Admitted',linewidth=2)
plt.scatter(negative['Exam 1'],negative['Exam 2'],s = 30,c = 'r',marker = 'x',label = 'Not Admintted',linewidth=2)
# plt.legend()
# plt.set_xlabel('Exam 1 score')
# plt.set_ylabel('Exam 2 socre')
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model(theta0, theta1, theta2, x0, x1, x2):
    z = theta0 * x0 + theta1 * x1 + theta2 * x2
    return sigmoid(z)

def cost(theta0, theta1, theta2, x0, x1, x2,y):
  left = np.multiply(-y,np.log(model(theta0, theta1, theta2, x0, x1, x2)))
  right = np.multiply(1-y,np.log(1-model(theta0, theta1, theta2, x0, x1, x2)))
  return np.sum(left - right)/ (len(X))

pdData.insert(0,'Ones',1) # 每行的第零个元素插入值为1的数,列名为Ones
orig_data = pdData.as_matrix() # convert the frame to its Numpy-array representation
# shape[0] 总行数 shape[1]总列数
cols = orig_data.shape[1]
X = orig_data[:,0:cols-1] # 每行的最后一个数不取值
y = orig_data[:,cols-1:cols] # 获取每行的最后一个数
theta = np.zeros([1,3]) # [0,0,0]， # 没有具体的值，只是起到占位的作用

epsilon = 0.001
alpha = 0.001
max_itor = 100000
diff = [0, 0]
error_one = 0
error_two = 0
m = len(X)
print m
# 记录迭代次数
cont = 0

# 初始化参数
theta0 = 0
theta1 = 0
theta2 = 0
while cont < max_itor:
    cont += 1
    for i in range(m):
        diff[0] = model(theta0, theta1, theta2, X[i][0], X[i][1], X[i][2]) - y[i]

        theta0 -= alpha * diff[0] * X[i][0]
        theta1 -= alpha * diff[0] * X[i][1]
        theta2 -= alpha * diff[0] * X[i][2]

    error_one = 0
    for i in range(m):
        error_one += cost(theta0,theta1,theta2,X[i][0],X[i][1],X[i][2],y[i])

    if abs(error_one - error_two) < epsilon:
        break
    error_two = error_one

    print theta0, theta1, theta2
print theta0, theta1, theta2


y = model(theta0,theta1,theta2,1,50,80)
if y>0.5:
    print 1
else:
    print 0
print("迭代次数为%s"%cont)


