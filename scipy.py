# approximating 1/x with polynomials

from sklearn.metrics import mean_squared_error

def objective_function(W, X, Y):
    error = mean_squared_error(Y, np.matmul(X, W))
    return(error)
  
from scipy.optimize import minimize

X = np.array([[1, x, x**2, x**3] for x in range(1, 100)])
Y = 1/X[:, 1]
W_init = np.array([1] * len(X[0]))  # initial weights
result = minimize(objective_function, W_init, args=(X,Y), options={'maxiter': 5000})
W_hat = result.x  # learned weights
