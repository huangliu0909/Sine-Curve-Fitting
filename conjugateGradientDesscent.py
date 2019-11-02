import random
import math
import matplotlib.pyplot as plt
import numpy as np
num = 100
aa = [random.uniform(0.01, 2 * math.pi) for _ in range(num)]
aa = sorted(aa)

# add voice
bb = [math.sin(aa[i]) + random.gauss(0, 0.12) for i in range(num)]

degree = 9

# 生成特征矩阵X
A = np.ones((num, degree+1))
for i in range(num):
    for j in range(degree+1):
        A[i, j] = math.pow(aa[i], j)
tezheng = A
A = A.T.dot(A)

# 生成矩阵Y
b = np.sin(aa) + np.random.normal(scale=0.1, size=len(aa))
b = b.reshape(num, 1)
b = tezheng.T .dot(b)

print("Conjugate Gradient x:")
x = np.zeros((degree+1, 1))  # 初始值x0
r = b - np.dot(A, x)  # 计算残量
p = r  # p0=r0
# #while np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b) >= 10 ** -6:
i = 0
while i < degree:
    r1 = r
    # if p.all() == 0:
     #    break
    # else:
    # if np.dot(p.T, np.dot(A, p)) != 0:
    a = np.dot(r.T, r) / np.dot(p.T, np.dot(A, p))
    x = x + a * p  # x(k+1)=x(k)+a(k)*p(k)
    print(x)
    # r = r - np.dot(A, x)  # r(k+1)=b-A*x(k+1)


    r = r - (a * A).dot(p)

    b = np.dot(r.T, r) / np.dot(r1.T, r1)

    p = r + b * p
    i = i + 1
    #r = rr



print(x)
print("done Conjugate Gradient!")
print(i)
plt.plot(aa, tezheng.dot(x), 'r', label="predict data1")
plt.plot(aa, bb, 'g.', label="actual data")
plt.title("degree=" + str(degree))
plt.show()