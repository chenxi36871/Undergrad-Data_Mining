##1.1
import numpy as np

class process_stock_specific():
    code=''
    date=''
    quote={}
    result=[]

    def read(self):
        path='D:\\学习\\大三下\\数据挖掘\\GitHub dsai-book-master\\data\\hs300\\'+self.code+'.txt'
        f = open(path,'r')
        sdate=[]
        open_price=[]
        high_price=[]
        low_price=[]
        close_price=[]
        for line in f:
            st = line.split('\t')
            if len(st)>4:
              sdate.append(st[0])
              open_price.append(st[1])
              high_price.append(st[2])
              low_price.append(st[3])
              close_price.append(st[4])
        f.close()
        self.quote['sdate']=sdate
        self.quote['open']=open_price
        self.quote['high']=high_price
        self.quote['low']=low_price
        self.quote['close']=close_price

    def decide(self):
        a=np.asarray([self.quote['sdate'],self.quote['open'],self.quote['high'],self.quote['low'],self.quote['close']])
        a=a.T
        i=0
        while i<len(a):
            if a[i,0] == self.date:
                self.result=[a[i,1:5]]
                break
            i=i+1

    def print(self):
        print(np.asarray(self.result))


p1 = process_stock_specific()
p1.code = 'SH600000'
p1.date = '20100611'
p1.read()
p1.decide()
p1.print()

##1.2

import numpy as np
class process_stock_specific():
    code=''
    date=[]
    quote={}
    result=[]

    def read(self):
        path='D:\\学习\\大三下\\数据挖掘\\GitHub dsai-book-master\\data\\hs300\\'+self.code+'.txt'
        f = open(path,'r')
        sdate=[]
        open_price=[]
        high_price=[]
        low_price=[]
        close_price=[]
        for line in f:
            st = line.split('\t')
            if len(st)>4:
              sdate.append(st[0])
              open_price.append(st[1])
              high_price.append(st[2])
              low_price.append(st[3])
              close_price.append(st[4])
        f.close()
        self.quote['sdate']=sdate
        self.quote['open']=open_price
        self.quote['high']=high_price
        self.quote['low']=low_price
        self.quote['close']=close_price

    def decide(self):
        a=np.asarray([self.quote['sdate'],self.quote['open'],self.quote['high'],self.quote['low'],self.quote['close']])
        a=a.T
        i=0
        while i<len(a):
            j=0
            for j in range(len(self.date)):
              if a[i, 0] == self.date[j]:
                  self.result.append([a[i,0:5]])
            i = i+1

    def print(self):
        print(np.asarray(self.result))


p1 = process_stock_specific()
p1.code = 'SH600000'
p1.date = ['20100610', '20100628', '20100617']
p1.read()
p1.decide()
p1.print()

##2
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

class regression():
    n = 0
    x = []
    y = []
    error = []
    beta = []
    u=[]

    def est_beta(self):
        X=[]
        X = np.hstack((np.ones((self.n,1)),self.x))
        self.beta = la.inv(X.T.dot(X)).dot(X.T).dot(self.y)

    def predict_plt(self):
        fu = self.beta[0] + self.beta[1]*self.u
        plt.plot(self.x,self.y,'o')
        plt.plot(self.u,fu,'r-')
        plt.show()

r = regression()
r.n = 100
r.x = np.random.randn(r.n, 1)
r.error = np.random.randn(r.n, 1) * 0.4
r.y = 1 + 2 * r.x + r.error
r.u = np.linspace(-4,4,100)
r.est_beta()
r.predict_plt()

