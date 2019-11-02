import random
import math
import matplotlib.pyplot as plt
import numpy as np

minLoss = 100000
minLossDegree = -1
mminLoss = 100000
mminLossDegree = -1
kkk = 1
NUM = np.zeros(100)
LOSS = np.zeros(100)
LLOSS = np.zeros(100)
degree = 9
lamda = 2
for num in range (20,50):

    NUM[kkk] = num

    # produce data
    x = [random.uniform(0, 2 * math.pi) for _ in range(num)]
    x = sorted(x)
    y = [math.sin(x[i]) for i in range(num)]

    # add voice
    y = [math.sin(x[i]) + random.gauss(0, 0.12) for i in range(num)]

    # 生成特征矩阵X
    X = np.ones((num, degree+1))
    for i in range(num):
        X[i, 0] = 1
        for j in range(1,degree+1):
            X[i, j] = math.pow(x[i], j)

    # 生成矩阵Y
    Y = np.sin(x) + np.random.normal(scale=0.1, size=len(x))
    Y = Y.reshape(num, 1)

    # 求无正则项系数A
    A = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)
    print("无正则项系数矩阵为:")
    print(A)

    # 无正则项LOSS
    loss = 0
    for i in range (num):
        loss += (Y[i][0] - X.dot(A)[i][0]) * (Y[i][0] - X.dot(A)[i][0])
    loss = loss/2
    LOSS[i] = loss
    if loss < minLoss:
        minLoss = loss
        nnum = num

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
    for i in range(num):
        lloss += (Y[i][0] - X.dot(B)[i][0]) * (Y[i][0] - X.dot(B)[i][0])
    lloss = lloss / 2
    LLOSS[i] = lloss
    if lloss < mminLoss:
        mminLoss = lloss
        nnumm = num

    kkk = kkk + 1
    # plt.plot(x, X.dot(A), 'r', label="predict data1")
    # plt.plot(x, X.dot(B), 'r', color="blue", label="predict data2")
    # plt.plot(x, Y, 'g.', label="actual data")
    # plt.title("degree=" + str(degree) )
    # plt.show()

print("lamda = 2、degree = 9时，最小无正则loss= " + str(minLoss) + "  此时的num = "+str(nnum))
print("lamda = 2、degree = 9时，最小有正则loss= " + str(mminLoss) + "  此时的num = "+str(nnumm))
plt.plot(NUM[1:30], LOSS[1:30], 'r', label="without Regular term")
plt.plot(NUM[1:30], LLOSS[1:30], 'r',color="blue", label="with Regular term")
plt.title("LOSS-num" )
plt.xlabel("num")
plt.ylabel("loss")
plt.legend()
plt.show()