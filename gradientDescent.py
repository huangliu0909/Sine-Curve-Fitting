import random
import math
import matplotlib.pyplot as plt
import numpy as np
num = 20
x = [random.uniform(0, 2 * math.pi) for _ in range(num)]
x = sorted(x)
y = [math.sin(x[i]) for i in range(num)]

# add voice
y = [math.sin(x[i]) + random.gauss(0, 0.12) for i in range(num)]
for i in range(num):
    while y[i] > 1:
        y[i] = y[i] - 1
    while y[i] < -1:
        y[i] = y[i] + 1

degree = 2
# 生成特征矩阵X
X = np.ones((num, degree+1))
for i in range(num):
    X[i, 0] = 1
    for j in range(1, degree + 1):
        X[i, j] = math.pow(x[i], j)

# 生成矩阵Y
Y = np.sin(x) + np.random.normal(scale=0.1, size=len(x))
for i in range(num):
    while Y[i] > 1:
        Y[i] = Y[i] - 1
    while Y[i] < -1:
        Y[i] = Y[i] + 1
Y = Y.reshape(num, 1)

theta = np.zeros((1, degree+1))

# 迭代阀值，当两次迭代损失函数之差小于该阀值时停止迭代
epsilon = 0.01
# 学习率
alpha = 0.01
diff = [0, 0]
max_itor = 1000
error1 = 0
error0 = 0
cnt = 0
while True:
    cnt += 1
    # 参数迭代计算
    for i in range(num):
         # 拟合函数为 y = theta0 * x[0] + theta1 * x[1] +theta2 * x[2]+····
         # 计算残差
         diff[0] += theta[0, 0]
         for j in range (1,degree+1):
             diff[0] += theta[0, j] * X[i][j]

         diff[0] = diff[0] - Y[i]
         #diff[0] = (theta0 + theta1 * x[i][1] + theta2 * x[i][2]+ ····) - y[i]
         # 梯度 = diff[0] * x[i][j]

         for j in range (degree+1):
            theta[0, j] -= alpha * diff[0] * X[i][j]

    # 计算损失函数
    error1 = 0
    for lp in range(num):
         s = Y[lp] - theta[0, 0]
         for k in range(1, degree+1):
            s -= theta[0, k] * X[lp][k]
         error1 += abs(s) ** 2 / 2
        #error1 += (Y[lp]-(theta0 + theta1 * x[lp][1] + theta2 * x[lp][2]+····))**2/2

    if abs(error1-error0) < epsilon:
        break
    else:
        error0 = error1

    print(theta)

print(theta)
print("迭代次数: " + str(cnt))
