
###################
# 使用主成分分析方法，分析手写数据集''zip.train''中任一其它数字（非数字3）的特征。
###################
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
path = r'D:\学习\大三下\数据挖掘\GitHub dsai-book-master\data\zip.train'
data = np.loadtxt(path)
id4 = data[:, 0] == 4
data4 = data[id4, 1:]

mean4 = np.mean(data4,axis = 0)
plt.imshow(mean4.reshape(16,16))
plt.show()

covx = np.cov(data4.T)
u,v = la.eig(covx)
# 第j个主成分方向
j = 1
plt.imshow(v[:,j-1].reshape(16,16))
plt.show()
# the j-th PC score
j = 4  # 1,2,3,4,5...
xi = (data4 -mean4).dot(v[:, j-1:j])  # 最小化重构误差得到xi的估计，xi是658个不同的3的第j个主成分的得分（在第j主成分方向投影后的n维向量）
id = xi.ravel().argsort()  # 将多维数组化为一维数组后从小到大提取排序索引, xi是(658*1),xi.ravel()是(658,)。argsort是将数组从小到大排列，返回索引
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(data4[id[-i-1]].reshape(16, 16))  # id[-i-1]表示id数组中的最后（最大）i个（即主成分j中得分最高的5个手写3）
for i in range(5):
    plt.subplot(2,5,5+i+1)
    plt.imshow(data4[id[i]].reshape(16, 16))  # data3[i]是(256,)的一维数组，是data3第id[i]行的数据（即主成分j中得分最低的5个手写3）
plt.show()

# 第一主成分代表了手写数字3的宽窄特征。如果一个数字3在该特征上的投影得分为正且数值较大，则该数字具有较宽的特征。反之，如果。。。

# variance explain（方差解释百分比） 该步用来计算前50个特征值在sum(特征值)的占比
np.sum(u[0:50])/np.sum(u)  # u是协方差阵的特征值
[np.sum(u[0:a])/np.sum(u) for a in range(50)]

# reconstruction
# 使用的主成分数量越多，重构约接近原始数据。如果使用所有的主成分，则重构得到的是原始数据。
k = 50  # k是取的主成分的个数（q）
xi = (data4 - mean4).dot(v[:, 0:k])  # a set of low dim vector
rec_data4 = mean4 + xi.dot(v[:, 0:k].T)
# 从原始数据和重构数据中取同样位置的一个对比一下（发现很类似）

j = 400
plt.subplot(1,2,1)
plt.imshow(data4[j,:].reshape(16,16))
plt.subplot(1,2,2)
plt.imshow(rec_data4[j,:].reshape(16,16))
plt.show()


###################
# 使用奇异值分解方法分析案例 股票收益率数据，并对比主成分分析结果。
###################
import numpy as np
import numpy.linalg as la
import pandas as pd
import os
index_path = 'D:\\学习\\大三下\\数据挖掘\\GitHub dsai-book-master\\data\\SZ399300.TXT'
index300 = pd.read_table(index_path, encoding='cp936', header=None)
idx = index300[:-1]
idx.columns = ['date', 'o', 'h', 'l', 'c', 'v', 'to']
idx.index = idx['date']

stock_path = r'D:\学习\大三下\数据挖掘\GitHub dsai-book-master\data\hs300'
names = os.listdir(stock_path)
close = []
for name in names:
    spath = stock_path + '\\' + name
    df0 = pd.read_table(spath, encoding='cp936', header=None)
    df1 = df0[:-1]
    df1.columns = ['date', 'o', 'h', 'l', 'c', 'v', 'to']
    df1.index = df1['date']
    df2 = df1.reindex(idx.index, method='ffill')
    df3 = df2.fillna(method='bfill')
    close.append(df3['c'].values)

data = np.asarray(close).T
retx = (data[1:, :]-data[:-1, :])/data[:-1, :]  # 算出300支股票的收益率
retx = retx.T

mean_retx = np.mean(retx, axis=0)

u, d, v = la.svd(retx - mean_retx)
xscore = np.dot((retx-mean_retx), v)

cname_path = 'D:\学习\大三下\数据挖掘\GitHub dsai-book-master\data/A_share_name.xlsx'
namesheet = pd.read_excel(cname_path, 'Sheet1')
cepair  = namesheet.values

id = xscore[:,2].argsort()[-10:]
sector1 = [names[a] for a in id]
cname  = []
for ecode in sector1:
    ecodex = ecode[2:-4]
    id = cepair[:,0] == int(ecodex)
    cname.append(cepair[id,1][0])
cname

id = xscore[:, 2].argsort()[:10]
sector2 = [names[a] for a in id]
cname  = []
for ecode in sector2:
    ecodex = ecode[2:-4]
    id = cepair[:,0] == int(ecodex)
    cname.append(cepair[id,1][0])
cname



















