# Gradient Descent polynomial regression (e.g. 1 dimentional independent variable, degree 2 polynomial: y(x, w) = w0 + w1*x + w2*x**2)
# we start with an initial guess for w
# and update it by taking steps in the direction of the gradient of error function (for maximization problem we take step toward the opposite of gradient).
# The error function we use here is sum of squared errors

from matplotlib import pyplot as plt
from IPython.display import clear_output
import random
import numpy as np



############################################
######## using matrix notation #############
############################################

import numpy as np

def least_squares(X, y, learning_rate=0.001, num_iterations=10000):
    # Initialize the parameters with zeros
    weights = np.zeros((X.shape[1], 1))

    # Perform gradient descent
    for _ in range(num_iterations):
        # Compute the predicted y values
        y_pred = np.dot(X, weights)

        # Compute the error (difference between predicted and actual y values)
        error = y_pred - y

        # Compute the gradient
        gradient = 2 * np.dot(X.T, error)

        # Update the parameters
        weights -= learning_rate * gradient

    return weights

# Example usage
# Assume X is a matrix of features (shape: m x n) and y is a column vector of target values (shape: m x 1)
X = np.array([[1], [4], [6]])
y = np.array([[5], [8], [15]])

# Add a column of ones to X for the bias term
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Call the least_squares function
weights = least_squares(X, y)
print(*weights)


def f(x):
    return weights[0] + weights[1] * x

plt.scatter(X[:, 1], y)
plt.plot((0, 6), (f(0), f(6)))


#################################################
########## without matrix notation ##############
#################################################



# with open('...') as f:
#   d = f.readlines()[1:]  # [1:] to exclude header
# data = [ eval(line.rstrip('\n')) for line in d ]
# X, Y = zip(*data)

data = [
        (5.0, 5582.42),
        (16.0, 3864.75),
        (29.5, 3588.7),
        (45.0, 2955.4),
        (63.5, 2512.09),
        (87.0, 1860.81),
        (120.0, 1288.25),
        (176.0, 688.24),
        (285.0, 346.5),
        (696.0, 74.21),
        (1000, 0.01),
        (1200, -10)
       ]

#data = [ (x, (x - 5) ** 2 + random.uniform(-2, 2)) for x in range(11) ]
data = [ (x, (x - 5) ** 2) for x in range(11) ]
n = len(data)

X, Y = zip(*data)

def step(w, direction, step_size):
  """move step_size in the direction from p"""
  return [w_i + step_size * direction_i
          for w_i, direction_i in zip(w, direction)]


def minimize_batch(loss_fn, gradient_loss_fn, w_0, tolerance=.000000001):
  """
  use gradient descent to find theta that minimizes target function.
  tolerance: to stop the program if the pace of the improvement is less than this number
  """
  
  w = w_0  # initial guess for weights
  ws = [w_0]  # list containing all weights
  step_size = -0.00007  # to minimize, move towards opposite of gradient
  errs = [loss_fn(w_0)]  # list containing all squared errors
  mae = calc_mae(w_0)  
  MAE_errs = [mae]  # list containing all MAE errors
  grads = []  # list containing all gradients
  
  plt.scatter(X, Y)
  plt.show()

  min_X = min(i[0] for i in data)
  max_X = max(i[0] for i in data)
  counter = 0
  while True:
    gradient = gradient_loss_fn(w)
    grads.append(gradient)
    next_w = step(w, gradient, step_size)
    ws.append(next_w)
    errs.append(loss_fn(next_w))
    next_mae = calc_mae(next_w)
    MAE_errs.append(next_mae)
    
    if abs(next_mae - mae) < tolerance:
      return (ws, MAE_errs, errs, grads)
    else:
      w, mae = next_w, next_mae  
      if counter % 10000 == 0:
        # plot the fitted curve with the current weights
        xs = np.linspace(min_X, max_X, 100)
        ys = [ f_x(w, i) for i in xs ] 
        clear_output(wait=True)  # to update the current plot
        plt.scatter(X, Y)
        plt.plot(xs, ys, color='green')
        plt.show()
      counter += 1
      
      
def f_x(coefs, x):
  return sum([ coefs[d] * x**d for d in range(deg + 1) ])


def error(w):
  # y(x, w) = w0 + w1*x
  # error = 1/2 Sigma (y(x_n,w) - t_n) ** 2  (the 1/2 coefficient is for making the math simple and it's not important)
  return sum( (f_x(w, item[0]) - item[1]) ** 2 for item in data )


def error_grad(w):
  # sum of squared errors
  # error = 1/2 Sigma (y(x_n,w) - t_n) ** 2
  # grad_error = [Sigma(y(x_n,w) - t_n), Sigma x_n(y(x_n,w) - t_n)]
  return [ sum((f_x(w, item[0]) - item[1]) * item[0] ** d for item in data) for d in range(deg + 1) ]
  
  # return [ sum((f_x(w, item[0]) - item[1]) * item[0] ** d for item in data) for d in range(deg + 1) ]
        
        
def calc_mae(w):
  return sum([ abs(f_x(w, i[0]) - i[1] ) for i in data ]) / len(data)


deg = 2
random_weights = [random.uniform(-30, 30) for _ in range(deg + 1)]
#random_weights = [24, -9, 1]
all_weights, all_mae, all_errors, all_grads = minimize_batch(error, error_grad, random_weights)
print('# of Iterations: {}'.format(len(all_mae)))
print('Random Initial Weights: ', random_weights) 
print('Last MAE: {}'.format(all_mae[-1]))
print(list(zip([list(i) for i in all_weights], all_errors, all_mae, all_grads + [[]]))[-1])


###################################################################################################################
########## if the function is not differentiable or the derivative is not straightforward to calculate ############
###################################################################################################################

# we want to fit lines to some data. the loss for each point is the distance to the closest line.

def calc_loss(weights_set):  # slightly slower than the next function, but easier to understand
    y_preds = []
    for weights in weights_set:
        y_pred = np.dot(X, weights)
        y_preds.append(y_pred)
    error = np.minimum.reduce([ (y - i) ** 2 for i in y_preds])
    return np.sum(error)

def calc_loss(weights_set): 
    weights_set = np.array(weights_set)
    y_pred = np.dot(X, weights_set.reshape(weights_set.shape[:2]).T)
    error = np.min(np.subtract(y_pred, y) ** 2, axis=1)
    return np.sum(error)



X_ = np.array([[0], [1], [1.5], [2], [3], [4]])
y = np.array([[1.5], [2], [2], [0], [.5], [.5]])

# Add a column of ones to X for the bias term
X = np.hstack((np.ones((X_.shape[0], 1)), X_))

# Initialize the parameters
weights1 = np.array([[1.], [0.]])
weights2 = np.array([[2.], [0.]])

def fit(weights, learning_rate, purturbation_rate, epsilon, n=1000, plot=False):

    weights1, weights2 = weights

    for i in range(n + 1):

        if plot:
            if i % 500 == 0:
                print(weights1[1])
                print(weights2[1])
                clear_output(wait=True)  # to update the current plot
                plt.scatter(X_, y)
                plt.plot([0, 4], (weights1[0], weights1[0] + weights1[1] * 4) )
                plt.plot([0, 4], (weights2[0], weights2[0] + weights2[1] * 4) )
                plt.show()

        # Compute the gradient
        # For a given parameter, we perturb it slightly and evaluate the change in the 
        # loss function. The gradient with respect to that parameter is then approximated 
        # by the ratio of the change in the loss function to the perturbation.
        current_loss = calc_loss([weights1, weights2])

        new_weights1_0 = weights1 * [[purturbation_rate], [1]]
        new_weights1_1 = weights1 * [[1], [purturbation_rate]] + [[0], [np.random.choice([epsilon, -epsilon])]]

        grad1_0 = (calc_loss([new_weights1_0, weights2]) - current_loss) / current_loss
        grad1_1 = (calc_loss([new_weights1_1, weights2]) - current_loss) / current_loss

        new_weights2_0 = weights2 * [[purturbation_rate], [1]]
        new_weights2_1 = weights2 * [[1], [purturbation_rate]] + [[0], [np.random.choice([epsilon, -epsilon])]]

        grad2_0 = (calc_loss([weights1, new_weights2_0]) - current_loss) / current_loss
        grad2_1 = (calc_loss([weights1, new_weights2_1]) - current_loss) / current_loss

        gradients1 = np.array([[grad1_0], [grad1_1]])
        gradients2 = np.array([[grad2_0], [grad2_1]])

        # Update the parameters
        weights1 -= learning_rate * gradients1
        weights2 -= learning_rate * gradients2
    
    return current_loss

learning_rate = .01
purturbation_rate = 1.01
epsilon = -.0000001

fit([weights1, weights2], learning_rate, purturbation_rate, epsilon, 10000, plot=True)
