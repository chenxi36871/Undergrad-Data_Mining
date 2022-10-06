import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# 1. 写出或推导贝叶斯公式，分别包含集合情形，随机变量的情形（含离散和连续四类情形）

# 2. 对LDA模拟仿真生成的数据，使用QDA和朴素贝叶斯方法进行分析，画出决策边界
# QDA
n = 100
x1 = np.random.randn(n,2)
x2 = np.random.randn(n,2) + np.array([2,2])
X = np.vstack((x1,x2))
y1 = np.ones((n,1))
y2 = np.zeros((n,1))
y = np.vstack((y1,y2))
plt.figure()
plt.plot(x1[:,0],x1[:,1],'ro')
plt.plot(x2[:,0],x2[:,1],'bo')
plt.show()

p1 = 0.5
p2 = 0.5
mu1 = np.mean(x1,axis = 0)
mu2 = np.mean(x2,axis = 0)
s1 = np.cov(x1,rowvar=False)
s2 = np.cov(x2,rowvar=False)

delta1 = -0.5*np.log(la.det(s1))-0.5*(X-mu1) @ (la.inv(s1)) @ (X-mu1).T
delta2 = -0.5*np.log(la.det(s2))-0.5*(X-mu2).dot(la.inv(s2)).dot((X-mu2).T)
id  = delta1 > delta2

b0 = 0.5*mu1.dot(la.inv(S)).dot(mu1) - 0.5*mu2.dot(la.inv(S)).dot(mu2)
b = (la.inv(S)).dot(mu1-mu2)
u = np.linspace(-4,4,100)
fu = b0/b[1] - b[0]/b[1]*u

plt.figure()
plt.plot(X[id==True,0],X[id==True,1],'ro')
plt.plot(X[id==False,0],X[id==False,1],'bo')
plt.show()
plt.plot(u,fu,'k-')
plt.show()


# 朴素贝叶斯
n = 100
x1 = np.random.randn(n,2)
x2 = np.random.randn(n,2) + np.array([2,2])
X = np.vstack((x1,x2))
y1 = np.ones((n,1))
y2 = np.zeros((n,1))
y = np.vstack((y1,y2))
plt.figure()
plt.plot(x1[:,0],x1[:,1],'ro')
plt.plot(x2[:,0],x2[:,1],'bo')
plt.show()

p1 = 0.5
p2 = 0.5
mu1 = np.mean(x1,axis = 0)
mu2 = np.mean(x2,axis = 0)
s1 = np.cov(x1.T)
s2 = np.cov(x2.T)

delta1 = -np.log(s1[0,0]*s1[1,1])-(X[:,0]-mu1[0])**2/(2*(s1[0,0])**2)-(X[:,1]-mu1[1])**2/(2*(s1[1,1])**2)
delta2 = -np.log(s2[0,0]*s2[1,1])-(X[:,0]-mu2[0])**2/(2*(s2[0,0])**2)-(X[:,1]-mu2[1])**2/(2*(s2[1,1])**2)
id  = delta1 > delta2

u = np.linspace(-4,4,100)
b = np.log(s1[0,0]*s1[1,1]/s2[0,0]/s2[1,1])
bu = (u-mu1[0])**2/(2*(s1[0,0]**2))-(u-mu2[0])**2/(2*(s2[0,0]**2))
bu = np.array(bu)
import sympy as sy
fu = []
for i in range(100):
    y = sy.symbols('y')
    result = sy.solve(b+bu[i]-(y-mu2[1])**2/(2*s2[1,1]**2)+(y-mu1[1])**2/(2*s1[1,1]**2), y)
    fu.append(result)


plt.figure()
plt.plot(X[id==True,0],X[id==True,1],'ro')
plt.plot(X[id==False,0],X[id==False,1],'bo')
plt.show()
u1 = np.hstack([u,u])
fu = np.array(fu)
fu1 = np.hstack([fu[:,0],fu[:,1]])
plt.plot(u1,fu1,'k-')
plt.show()




# 3 对玩具数据$(X,Y):{(-3,0),(-2,0),(-1,0),(1,1),(2,1),(3,1)\}$，使用带有和不带有L2惩罚项的逻辑回归来估计参数，并观察结果。
x = np.asarray([-3,-2,-1,1,2,3]).reshape(6,1)
y = np.asarray([0,0,0,1,1,1]).reshape(6,1)
c = np.ones((6,1))
X = np.hstack((c,x))
beta = la.inv(X.T.dot(X)).dot(X.T).dot(y)

# 带有L2惩罚项的
from sklearn import linear_model
clf = linear_model.LogisticRegression(C=1e2,fit_intercept=False)
clf.fit(X,y)
betay = clf.coef_.T


# 不带有L2惩罚项的
def logistic_non(X,y,beta):
    diff = 1
    iter = 0
    while iter <1000 and diff >0.0001:
        like = np.sum(y*(X.dot(beta)) - np.log(1+np.exp(X.dot(beta))))
        p = np.exp(X.dot(beta))/(1+np.exp(X.dot(beta)))
        w = np.diag((p*(1-p)).ravel())
        z = X.dot(beta) + la.inv(w).dot(y-p)
        beta = la.pinv(X.T.dot(w).dot(X)).dot(X.T).dot(w).dot(z)
        like2 = np.sum(y*(X.dot(beta)) - np.log(1+np.exp(X.dot(beta))))
        diff = np.abs(like - like2)
        iter = iter + 1
    return beta

betan = logistic_non(X,y,beta)
phatn = 1/(1+np.exp(-X.dot(betan)))

sum(y*(X.dot(betan))- np.log(1+np.exp(X.dot(betan))))
sum(y*(X.dot(betay)) - np.log(1+np.exp(X.dot(betay))))


# 4. 写出带L2惩罚的逻辑回归的Python程序。（可选做）
def logistic_with(X,y,beta,ld):
    diff = 1
    iter = 0
    while iter <1000 and diff >0.0001:
        like = np.sum(y*(X.dot(beta)) - np.log(1+np.exp(X.dot(beta))))-sum(ld*beta**2)
        p = np.exp(X.dot(beta))/(1+np.exp(X.dot(beta)))
        w = np.diag((p*(1-p)).ravel())
        z = X.dot(beta) + la.inv(w).dot(y-p)
        beta = la.pinv(X.T.dot(w).dot(X)).dot(X.T).dot(w).dot(z)
        like2 = np.sum(y*(X.dot(beta)) - np.log(1+np.exp(X.dot(beta))))-sum(ld*beta**2)
        diff = np.abs(like - like2)
        iter = iter + 1
    return beta


n = 100
x1 = np.random.randn(n,2)
x2 = np.random.randn(n,2) + np.array([2,2])
X = np.vstack((x1,x2))
y1 = np.ones((n,1))
y2 = np.zeros((n,1))
y = np.vstack((y1,y2))
ld = 1
c = np.ones((2*n,1))
X2 = np.hstack((c,X))
beta = la.inv(X2.T.dot(X2)).dot(X2.T).dot(y)

betay = logistic_with(X2,y,beta,ld)
sum(y*(X2.dot(betay)) - np.log(1+np.exp(X2.dot(betay))))

#CSDN
class LogisticRegression:

    # 默认没有正则化，正则项参数默认为1，学习率默认为0.001，迭代次数为10001次
    def __init__(self, penalty=None, Lambda=1, a=0.001, epochs=10001):
        self.W = None
        self.penalty = penalty
        self.Lambda = Lambda
        self.a = a
        self.epochs = epochs
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def loss(self, x, y):
        m = x.shape[0]
        y_pred = self.sigmoid(x * self.W)
        return (-1 / m) * np.sum((np.multiply(y, np.log(y_pred)) + np.multiply((1 - y), np.log(1 - y_pred))))

    def fit(self, x, y):
        lossList = []
        # 计算总数据量
        m = x.shape[0]
        # 给x添加偏置项
        X = np.concatenate((np.ones((m, 1)), x), axis=1)
        # 计算总特征数
        n = X.shape[1]
        # 初始化W的值,要变成矩阵形式
        self.W = np.mat(np.ones((n, 1)))
        # X转为矩阵形式
        xMat = np.mat(X)
        # y转为矩阵形式，这步非常重要,且要是m x 1的维度格式
        yMat = np.mat(y.reshape(-1, 1))
        # 循环epochs次
        for i in range(self.epochs):
            # 预测值
            h = self.sigmoid(xMat * self.W)
            gradient = xMat.T * (h - yMat) / m

            # 加入l1和l2正则项，和之前的线性回归正则化一样
            if self.penalty == 'l2':
                gradient = gradient + self.Lambda * self.W
            elif self.penalty == 'l1':
                gradient = gradient + self.Lambda * np.sign(self.W)

            self.W = self.W - self.a * gradient
            if i % 50 == 0:
                lossList.append(self.loss(xMat, yMat))
            # 返回系数，和损失列表
        return self.W, lossList






