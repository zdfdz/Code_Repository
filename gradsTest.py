# -*-coding:utf-8-*-
# x[i]特征
x = [(1, 2.1, 5, 1, 45), (1, 1.4, 3, 2, 40), (1, 1.5, 3, 2, 30), (1, 8.5, 2, 1, 36), (1, 7.3, 2, 1, 40)]
# y[i] 样本点对应的输出
y = [4.6, 2.3, 3.1, 4.7, 1.5]

print len(x), len(y)

epsilon = 0.001

alpha = 0.001
max_itor = 100000
diff = [0]
error_one = 0
error_two = 0
m = len(x)
# 记录迭代次数
cont = 0

# 初始化参数
theta0 = 0
theta1 = 0
theta2 = 0
theta3 = 0
theta4 = 0


def model_fun(theta0, theta1, theta2, theta3, theta4, x0, x1, x2, x3, x4):
    h = theta0 * x0 + theta1 * x1 + theta2 * x2 + theta3 * x3 + theta4 * x4
    return h


while cont < max_itor:
    cont += 1
    for i in range(m):
        diff[0] = model_fun(theta0, theta1, theta2, theta3, theta4, x[i][0], x[i][1], x[i][2], x[i][3], x[i][4]) - y[i]

        theta0 -= alpha * diff[0] * x[i][0]
        theta1 -= alpha * diff[0] * x[i][1]
        theta2 -= alpha * diff[0] * x[i][2]
        theta3 -= alpha * diff[0] * x[i][3]
        theta4 -= alpha * diff[0] * x[i][4]

    error_one = 0
    for i in range(m):
        error_one += (model_fun(theta0, theta1, theta2, theta3, theta4, x[i][0], x[i][1], x[i][2], x[i][3], x[i][4]) -
                      y[i]) ** 2 / 2 * m

    if abs(error_one - error_two) < epsilon:
        break
    error_two = error_one

    print theta0, theta1, theta2, theta3, theta4

print "--"*20
print theta0, theta1, theta2, theta3, theta4

print("迭代次数为%s" % cont)
