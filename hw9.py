# 参考局部常数回归，写出K邻近估计法的目标函数。
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
n = 100
x = np.random.rand(n,1)
error = np.random.randn(n,1)*0.3
y = np.sin(2*np.pi*x) + error

def knn1(y,x,k,u):
    fu = []
    for u0 in u:
        dist = np.abs(x-u0).ravel()
        id = dist.argsort()[:k]
        fu.append(np.mean(y[id]))
    return fu
#模拟仿真
u = np.linspace(0,1,100)
fu = knn1(y,x,15,u)

plt.plot(u,np.array(fu),'r-')
plt.plot(x,y,'bo')
plt.legend(['knn estimate, k=15'])
plt.show()

# (选做题)参考PPT6 局部似然估计部分，写出非参数逻辑回归的局部似然估计的牛顿算法并进行模拟仿真


# 编写模拟仿真程序，使用交叉验证法选择岭回归的惩罚参数
n = 100
x = np.random.rand(n,1)
error = np.random.randn(n,1)*0.3
y = np.sin(2*np.pi*x) + error
c = np.ones((n,1))
X = np.hstack((c,x))

def ridge_reg(X,y,ld,u):
        beta_ridge = la.inv(X.T.dot(X)+ld*np.eye(2)).dot(X.T).dot(y) #beta_ridge 2*1
        fu = u.dot(beta_ridge)
        return fu

from sklearn.model_selection import KFold
z = np.hstack((x,y))
kf = KFold(n_splits=10)
## cross-validation
cv_seq = []
ld_seq = np.linspace(1,100,100)
for ld in ld_seq:
    cv = 0
    for train, test in kf.split(z):
        train_X = X[train]
        train_y = y[train]
        test_X = X[test]
        test_y = y[test]
        yhat = ridge_reg(train_X,train_y,ld,test_X)
        cv = cv + np.mean(test_y - yhat)**2
    cv_seq.append(cv/10)
plt.plot(ld_seq,cv_seq,'-')
plt.show()

i=0
for i in range(len(cv_seq)):
    s = np.abs(np.abs(cv_seq[i+2]-cv_seq[i+1])-np.abs(cv_seq[i+1]-cv_seq[i]))
    if s < 0.000005:
        count = i+1
        break



