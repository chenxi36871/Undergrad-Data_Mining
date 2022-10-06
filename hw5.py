import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

# 1在回归模拟仿真中，编写计算R方的程序
# 生成数据
X = np.random.randn(100,3)
sigma = 0.6
error = np.random.randn(100,1)*sigma
beta = np.array([[1,-2,0.5]]).T
y = X.dot(beta) + error

plt.plot(X[:,0],y,'o')
plt.show()
np.corrcoef(X.T)
import statsmodels.stats.outliers_influence as infl
[infl.variance_inflation_factor(X,i) for i in range(3)]

beta_ols = la.inv(X.T.dot(X)).dot(X.T).dot(y)
y_pre = np.dot(X,beta_ols)

# 计算R方
rss = np.dot((y-y_pre).T,(y-y_pre))
tss = np.dot((y-np.mean(y)).T,(y-np.mean(y)))
rsquare = 1-(rss/tss)
rsquare


# 2编写向后选择法的程序并进行模拟仿真
def backward0(X,y,r2target):
    n,p = X.shape
    seq = list(range(p))
    dele = []
    inv_seq = list(range(p))
    for j in range(p):
        beta = la.pinv(X[:,seq].T.dot(X[:,seq])).dot(X[:,seq].T).dot(y)
        y_pre = X[:,seq].dot(beta)
        Z = y - y_pre
        tmp = np.hstack((Z,X[:,seq]))
        corr0 = np.corrcoef(tmp.T)
        id = np.abs(corr0[0,1:]).argmin()
        dele.append(seq[id])
        del seq[id]
        rss = np.dot((y - y_pre).T, (y - y_pre))
        tss = np.dot((y - np.mean(y)).T, (y - np.mean(y)))
        rsquare = 1 - (rss / tss)
        if rsquare > r2target:
            break
    return seq

n = 200
p = 10
x = np.random.rand(n,p)
beta_true = np.array(range(p))
error = np.random.randn(n,1)*0.3
y = x.dot(beta_true.reshape(p,1)) + error
back = backward0(x,y,0.999)

# 3在指数跟踪的案例中，使用移动窗口的方法回测跟踪误差
import pandas as pd
import os
index_path = r'D:\学习\大三下\数据挖掘\GitHub dsai-book-master\data\SZ399300.TXT'

index300 = pd.read_table(index_path, encoding = 'cp936',header = None)
idx = index300[:-1]
idx.columns = ['date','o','h','l','c','v','to']
idx.index = idx['date']

stock_path = r'D:\学习\大三下\数据挖掘\GitHub dsai-book-master\data\hs300'
names = os.listdir(stock_path)
close = []
for name in names:
    spath = stock_path + '\\' + name
    df0 = pd.read_table(spath,\
        encoding = 'cp936',header = None)
    df1 = df0[:-1]
    df1.columns = ['date','o','h','l','c','v','to']
    df1.index = df1['date']
    df2 = df1.reindex(idx.index,method = 'ffill')
    df3 = df2.fillna(method = 'bfill')
    close.append(df3['c'].values)

data = np.asarray(close).T

retx = (data[1:,:]-data[:-1,:])/data[:-1,:]

X = retx
y = np.mean(X,axis = 1).reshape(1339,1)

seq =  forward0(X[0:500,:],y[0:500,:])
X2 = X[0:500,:]
y2 = y[0:500,:]
id = seq[:50]
beta = la.pinv(X2[:,id].T.dot(X2[:,id])).dot(X2[:,id].T).dot(y2)
beta = beta/np.sum(beta)

ret_test = X[:,id].dot(beta)

# 计算跟踪误差，窗口为100
diff = y[0:100,:]-ret_test[0:100,:]
te = []
te.append(np.std(diff, ddof=1))
i = 1
j = 101
while j <= 1339:
    diff = y[i:j, :]-ret_test[i:j, :]
    te.append(np.std(diff, ddof=1))
    i = i+1
    j = j+1

max(te)
