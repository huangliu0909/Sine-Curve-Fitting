import random
import math
import matplotlib.pyplot as plt
import numpy as np

minLoss = 100000
minLossDegree = -1
mminLoss = 100000
mminLossDegree = -1
LLOSS = np.zeros(200)
A = [x/100 for x in range(100,300)]

for kk in range(200):
    # lamda = 2
    lamda = A[kk]
    degree = 9;
    # produce data
    x = [random.uniform(0, 2 * math.pi) for _ in range(50)]
    x = sorted(x)
    y = [math.sin(x[i]) for i in range(50)]

    # add voice
    y = [math.sin(x[i]) + random.gauss(0, 0.12) for i in range(50)]

    # 生成特征矩阵X
    X = np.ones((50, degree+1))
    for i in range(50):
        X[i, 0] = 1
        for j in range(1,degree+1):
            X[i, j] = math.pow(x[i], j)

    # 生成矩阵Y
    Y = np.sin(x) + np.random.normal(scale=0.1, size=len(x))
    Y = Y.reshape(50, 1)

    # 求有正则项系数B
    a = np.dot(X.T, X)
    b = np.ones((50, degree+1))
    for i in range(degree+1):
        for j in range(degree+1):
            if i == 0:
                b[i, j] = 0
            elif j == 0:
                b[i, j] = 0
            elif i == j:
                b[i, j] = 1
            else:
                b[i, j] = 0
    c = np.ones((degree+1, degree+1))
    for i in range(degree+1):
        for j in range(degree+1):
            c[i, j] = a[i, j] + b[i, j] * lamda
    B = np.linalg.inv(c).dot(X.T).dot(Y)
    print("有正则项系数矩阵为:")
    print(B)

    # 有正则项LOSS
    lloss = 0
    for i in range(50):
        lloss += (Y[i][0] - X.dot(B)[i][0]) * (Y[i][0] - X.dot(B)[i][0])
    lloss = lloss / 2
    LLOSS[kk] = lloss
    if lloss < mminLoss:
        mminLoss = lloss
        mminlamda = lamda

    # plt.plot(x, X.dot(A), 'r', label="predict data1")
    # plt.plot(x, X.dot(B), 'r', color="blue", label="predict data2")
    #  plt.plot(x, Y, 'g.', label="actual data")
    # plt.title("degree=" + str(degree) )
    # plt.show()

print("degree = 9 、num = 50 时，最小有正则loss= " + str(mminLoss) + "  lamda = "+str(mminlamda))
plt.plot(A, LLOSS, 'r')
plt.xlabel("lamda")
plt.ylabel("loss")
plt.show()


