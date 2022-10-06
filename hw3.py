import os
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import numpy as np

# 1 编写程序，计算沪深300成份股每个股票收益率的峰度和偏度，看看是否有一些规律。
path = 'D:\\学习\\大三下\\数据挖掘\\GitHub dsai-book-master\\data\\hs300\\'
names = os.listdir(path)
result = []
for i in range(len(names)):
    data = pd.read_table(path + names[i], encoding='cp936', header=None)
    data = data[:-1]
    data.columns = ['date', 'o', 'h', 'l', 'c', 'v', 'to']
    data.index = data['date']
    data['ret'] = data['c'].pct_change().fillna(0)
    ret = data['ret']
    if ret.kurtosis()>=200:
        continue
    out = [names[i].rstrip('.txt'), ret.kurtosis(), ret.skew()]
    result.append(out)

result = DataFrame(result)
result.columns = ['name', 'k', 's']
plt.plot(result['k'], result['s'], '.')
plt.show()


# 2.对2维数据，编写核密度估计的代码并进行模拟。
def ourkde(x,h,u):
    fu = []
    for u0 in u:
        t = np.dot((x-u0),np.linalg.inv(h))  # 1000*2
        K = np.dot(np.exp((-0.5)*t**2), np.linalg.inv(h)/np.sqrt(2*np.pi))  # 1000*2
        fu.append(np.mean(K,axis=0))
    fu = np.asarray(fu)
    return fu

x = np.random.multivariate_normal((0, 0), [[0.8, 0.05], [0.05, 0.7]], 1000)  # x:1000*2
ux = np.linspace(-3,3,100)
uy = np.linspace(-3,3,100)
u = np.asarray([ux,uy]).T  # u:100*2
h = 1.06*np.cov(x, rowvar=False)*1000**(-0.2)  # h:2*2
fu = ourkde(x, h, u)
fu = fu.T

# #z, x1, y1 = np.histogram2d(x[:,0],x[:,1],30)
# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# ax2 = plt.axes(projection='3d')
# #ax1.scatter(x[:0],x[:,1],)
# ax2.plot_surface(u[:,0],u[:,1],fu,cmap='rainbow')
# plt.show()
X, Y= np.meshgrid(u[:, 0], u[:, 1])
fig = plt.figure()
plt.contour(X, Y, fu, camp='Blues')
plt.show()



#3.修改Kmeans程序，存储目标函数序列，并做图。
def ourkmean(x,k,mu,tol):
    n,p = x.shape
    dist_matx = np.zeros((n,k))
    id = []
    iter = 0
    max_it = 100
    diff = 100
    VAL = []
    VAL.append(500)
    while diff>tol and iter<max_it:
        for i in range(k):
            dist_matx[:,i] = np.sum((x - mu[i,:])**2, axis=1)
        id = np.argmin(dist_matx, axis=1)
        VAL.append(0)  #iter=0,VAL[1]
        for i in range(k):
            mu[i,:] = np.mean(x[id==i,:],axis = 0)
            VAL[iter+1] = VAL[iter+1] + np.sum((x[id==i,:] - mu[i,:])**2)
        diff = np.abs(VAL[iter+1] - VAL[iter])
        iter = iter +1
    return id, mu, VAL

n = 100
x1 = np.random.randn(n,2)
x2 = np.random.randn(n,2) + np.array([2,2])
x = np.vstack((x1,x2))
tol = 0.001
k = 2

mu = np.array([[-0.1,-0.1],[1.0,1.0]])
id, mu, VAL = ourkmean(x,k,mu,tol)
plt.plot(VAL,'b-')
plt.show()



