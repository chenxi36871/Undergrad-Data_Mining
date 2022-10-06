import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cvxpy as cp


# 使用二次规划算法求解Lasso问题，并学习cvxpy模块
def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y)**2

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value

def generate_data(m=100, n=20, sigma=5, density=0.2):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    idxs = np.random.choice(range(n), int((1-density)*n), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    X = np.random.randn(m,n)
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y, beta_star

m = 100
n = 20
sigma = 5
density = 0.2

X, Y, _ = generate_data(m, n, sigma)
X_train = X[:50, :]
Y_train = Y[:50]
X_test = X[50:, :]
Y_test = Y[50:]

beta = cp.Variable(n)
lambd = cp.Parameter(nonneg=True)
problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))

lambd_values = np.logspace(-2, 3, 50)
train_errors = []
test_errors = []
beta_values = []
for v in lambd_values:
    lambd.value = v
    problem.solve()
    train_errors.append(mse(X_train, Y_train, beta))
    test_errors.append(mse(X_test, Y_test, beta))
    beta_values.append(beta.value)

def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    plt.show()

plot_train_test_errors(train_errors, test_errors, lambd_values)

def plot_regularization_path(lambd_values, beta_values):
    plt.figure()
    num_coeffs = len(beta_values[0])
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi[i] for wi in beta_values])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")
    plt.title("Regularization Path")
    plt.show()

plot_regularization_path(lambd_values, beta_values)


# 编写Lasso问题的局部二次近似算法的程序并进行模拟仿真
n = 100
beta_true = np.array([1,-2,3,-4,5,-6,7,-8])
p = len(beta_true)
X = np.random.randn(n,p)
error = np.random.randn(n,1)*0.3
y = X.dot(beta_true.reshape(p,1)) + error
beta0 = np.ravel(la.pinv(X.T.dot(X)).dot(X.T).dot(y))
lam = np.log(n)

def object_f(X,y,beta,beta0,lam):
    n,p = X.shape
    one = np.sum((y-X.dot(beta))**2)
    two = n*beta.T.dot(ld(beta0,lam)).dot(beta)
    return one+two

def ld(beta0,lam):
    return np.diag(beta0)*lam

def betan(X,y,beta0,lam):
    n,p = X.shape
    return la.inv((X.T).dot(X)+n*ld(beta0,lam)).dot(X.T).dot(y)

iter = 0
diff = 1
val = 10000
while iter<10000 and diff>0.0001:
    beta = betan(X,y,beta0,lam)
    a = object_f(X,y,beta,beta0,lam)
    val2 = a
    diff = np.abs(val2-val)
    val = val2
    iter = iter+1

beta
iter

# 理解奇异值分解在OLS，岭回归和主成分回归的运用







