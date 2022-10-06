# #模拟生成不平衡数据，使用逻辑回归进行分类，并比较欠采样，过采样和改变阈值三种方法的效果。
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# data generation
n = 100
x1 = np.random.randn(n,2)
x2 = np.random.randn(n,2) + np.array([2,2])
X = np.vstack((x1,x2))
y1 = np.ones((n,1))
y2 = np.zeros((n,1))
y = np.vstack((y1,y2))
c = np.ones((2*n,1))
X2 = np.hstack((c,X))
beta = la.inv(X2.T.dot(X2)).dot(X2.T).dot(y)
#logistic
def logistic(X,y,beta):
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
def conmatrix(y,beta_lg,X2):
    S = len(y)
    PO = np.sum(y)
    NO = S - PO
    prob = 1 / (1 + np.exp(-X2.dot(beta_lg)))
    ld = 0.5
    y2 = (prob > ld).astype(int)
    PX = np.sum(y2)
    NX = S - PX
    TP = np.sum(y2 * y)
    FP = PX - TP
    FN = PO - TP
    TN = NO - FP
    confusion_matrix=np.array([[TP, FP], [FN, TN]]).astype(int)
    return confusion_matrix
def ROC(y,beta_lg,X2):
    S = len(y)
    PO = np.sum(y)
    NO = S - PO
    prob = 1 / (1 + np.exp(-X2.dot(beta_lg)))
    ld = 0.5
    y2 = (prob > ld).astype(int)
    PX = np.sum(y2)
    NX = S - PX
    TP = np.sum(y2 * y)
    FP = PX - TP
    FN = PO - TP
    TN = NO - FP
    fpr_seq = []
    tpr_seq = []
    for ld in np.linspace(0.0, 0.99, 5000):
        y2 = (prob > ld).astype(int)
        PX = np.sum(y2)
        NX = S - PX
        TP = np.sum(y2 * y)
        FP = PX - TP
        FN = PO - TP
        TN = NX - FN
        TPR = TP / (TP + FN)
        FPR = FP / (TN + FP)
        fpr_seq.append(FPR.copy())
        tpr_seq.append(TPR.copy())
    fpr_seq2 = np.asarray(fpr_seq)[::-1]
    tpr_seq2 = np.asarray(tpr_seq)[::-1]

    fig = plt.figure(figsize=(10, 10))
    plt.plot(fpr_seq2, tpr_seq2)
    plt.xlim([-0.05, 1])
    plt.ylim([-0.01, 1.05])
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('ROC curve', fontsize=24)
    plt.show()
def AUC(y,beta_lg,X2):
    S = len(y)
    PO = np.sum(y)
    NO = S - PO
    prob = 1 / (1 + np.exp(-X2.dot(beta_lg)))
    ld = 0.5
    y2 = (prob > ld).astype(int)
    PX = np.sum(y2)
    TP = np.sum(y2 * y)
    FP = PX - TP
    FN = PO - TP
    TN = NO - FP
    np.array([[TP, FP], [FN, TN]]).astype(int)
    fpr_seq = []
    tpr_seq = []
    for ld in np.linspace(0.0, 0.99, 5000):
        y2 = (prob > ld).astype(int)
        PX = np.sum(y2)
        NX = S - PX
        TP = np.sum(y2 * y)
        FP = PX - TP
        FN = PO - TP
        TN = NX - FN
        TPR = TP / (TP + FN)
        FPR = FP / (TN + FP)
        fpr_seq.append(FPR.copy())
        tpr_seq.append(TPR.copy())
    fpr_seq2 = np.asarray(fpr_seq)[::-1]
    tpr_seq2 = np.asarray(tpr_seq)[::-1]
    fpr_seq3 = np.hstack((0, fpr_seq2))
    diff = np.diff(fpr_seq3)
    our_auc = np.sum(diff * tpr_seq2)
    return our_auc
beta_lg = logistic(X2,y,beta)
conmatrix(y,beta_lg,X2)
ROC(y,beta_lg,X2)
AUC(y,beta_lg,X2)
#欠采样
rus = RandomUnderSampler(random_state=0)
X_under, y_under = rus.fit_resample(X,y)
X2_under = np.hstack((c,X_under))
beta_under = la.inv(X2_under.T.dot(X2_under)).dot(X2_under.T).dot(y_under)
beta_lg_under = logistic(X2_under,y_under,beta_under)
conmatrix(y_under,beta_lg_under,X2_under)
ROC(y_under,beta_lg_under,X2_under)
AUC(y_under,beta_lg_under,X2_under)
#过采样
ros = RandomOverSampler(random_state=0)
X_over, y_over = ros.fit_resample(X,y)
X2_over = np.hstack((c,X_over))
beta_over = la.inv(X2_over.T.dot(X2_over)).dot(X2_over.T).dot(y_over)
beta_lg_over = logistic(X2_over,y_over,beta_over)
conmatrix(y_over,beta_lg_over,X2_over)
ROC(y_over,beta_lg_over,X2_over)
AUC(y_over,beta_lg_over,X2_over)
#改变阈值
from imblearn.under_sampling import EditedNearestNeighbours
enn = EditedNearestNeighbours()
X_change, y_change = enn.fit_resample(X, y)
cc=np.ones((X_change.shape[0],1))
X2_change = np.hstack((cc,X_change))
beta_change = la.inv(X2_change.T.dot(X2_change)).dot(X2_change.T).dot(y_change)
beta_lg_change = logistic(X2_change,y_change,beta_change)
conmatrix(y_change,beta_lg_change,X2_change)
ROC(y_change,beta_lg_change,X2_change)
AUC(y_change,beta_lg_change,X2_change)




# 编写自然三阶样条的模拟仿真程序
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
# data generation
n = 100
x = np.random.rand(n,1)
error = np.random.randn(n,1)*0.3
y = np.sin(2*np.pi*x) + error

from patsy import dmatrix
import statsmodels.api as sm
transformed_x3 = dmatrix("cr(train,df = 3)", {"train": x}, return_type='dataframe')
fit3 = sm.GLM(y, transformed_x3).fit()

pred3 = fit3.predict(dmatrix("cr(valid, df=3)", {"valid": x}, return_type='dataframe'))
plt.plot(x,y,'bo')
plt.plot(x, pred3,color='r', label='Natural spline')
plt.legend()
plt.show()




# 连续变化核技巧模拟仿真中的两个参数，观察最终拟合结果的变化
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
n = 100
x = np.random.rand(n,1)
error = np.random.randn(n,1)*0.3
y = np.sin(2*np.pi*x) + error

tau = 10
K = np.exp(-tau *(x - x.T)**2)
ld = 100
alpha = la.inv(K + ld* np.eye(n)).dot(y)
yhat2 = K.dot(alpha)
rk = x.ravel().argsort()
xi1 = 1/3
xi2 = 2/3
k1 = (x - xi1)**3*(x - xi1>0)
k2 = (x - xi2)**3*(x - xi2>0)

c = np.ones((n,1))
X = np.hstack((c,x,x**2,x**3,k1,k2))
beta = la.inv(X.T.dot(X)).dot(X.T).dot(y)
yhat = X.dot(beta)

plt.plot(x[rk],yhat[rk],'r-')
plt.plot(x[rk],yhat2[rk],'k--')
plt.plot(x,y,'bo')
plt.legend(['cubic spline','kernel regression'])
plt.show()



