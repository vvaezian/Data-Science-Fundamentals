#### Minimization
```py
# Approximating 1/x with polynomials
# i.e. finding the weights vector W that minimizes 
# 1/X - W * P(X)

from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

def objective_function(W, X, Y):
    error = mean_squared_error(Y, np.matmul(X, W))
    return(error)

X = np.array([[1, x, x**2, x**3] for x in range(1, 100)])
Y = 1/X[:, 1]  # true values
W_init = np.array([1] * len(X[0]))  # initial weights

result = minimize(objective_function, W_init, args=(X,Y), options={'maxiter': 5000})
W_hat = result.x  # learned weights
```

#### Condifence Interval
```py
from scipy import stats

a = range(10)
print(np.mean(a))
stats.t.interval(alpha=0.95,  # the probability that a drawn random sample will be inside the returned interval
                 df=len(a) - 1,  # degrees of freedom
                 loc=np.mean(a), 
                 scale=stats.sem(a)  # sem: standard error of the mean
                 )
```
