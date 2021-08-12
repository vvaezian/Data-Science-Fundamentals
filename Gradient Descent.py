# Gradient Descent on simplest linear model y(x, w) = w0 + w1*x1 + ... + wn*xn
# we start with an initial guess for w
# and update it by taking steps in opposite of the gradient of error function
# The error function we use here is sum of squared errors

from matplotlib import pyplot as plt
from IPython.display import clear_output
import random
import numpy as np

with open('...') as f:
  d = f.readlines()[1:]  # [1:] to exclude header
data = [ eval(line.rstrip('\n')) for line in d ]
X, Y = zip(*data)


def step(w, direction, step_size):
  """move step_size in the direction from p"""
  return [w_i + step_size * direction_i
          for w_i, direction_i in zip(w, direction)]


def minimize_batch(target_fn, gradient_fn, w_0, tolerance=.01):
  """
  use gradient descent to find theta that minimizes target function.
  tolerance: to stop the program if the pace of the improvement is less than this number
  """
  
  w = w_0  # initial guess for weights
  ws = [w_0]  # list containing all weights
  step_size = -0.000001  # to minimize, move towards opposite of gradient
  errs = [error(w_0)]  # list containing all squared errors
  mae = calc_mae(w_0)  
  MAE_errs = [mae]  # list containing all MAE errors
  grads = []  # list containing all gradients
  
  plt.scatter(X, Y)
  plt.show()

  min_X = min(i[0] for i in data)
  max_X = max(i[0] for i in data)
  
  while True:
    gradient = gradient_fn(w)
    grads.append(gradient)
    next_w = step(w, gradient, step_size)
    ws.append(next_w)
    errs.append(error(next_w))
    next_mae = calc_mae(next_w)
    MAE_errs.append(next_mae)
    
    if abs(next_mae - mae) < tolerance:
      return (ws, MAE_errs, errs, grads)
    else:
      w, mae = next_w, next_mae  
      # plot the fitted curve with the current weights
      xs = np.linspace(min_X, max_X, 100)
      ys = [ f_x(w, i) for i in xs ] 
      clear_output(wait=True)  # to update the current plot
      plt.scatter(X, Y)
      plt.plot(xs, ys, color='green')
      plt.show()
      
      
def f_x(coefs, x):
  return coefs[0] + coefs[1] * x 


def calc_mae(w):
  return sum([ abs(f_x(w, i[0]) - i[1] ) for i in data ])/len(data)


def error(w):
  # y(x, w) = w0 + w1*x
  # error = 1/2 Sigma (y(x_n,w) - t_n) ** 2  (the 1/2 coefficient is for making the math simple and it's not important)
  return sum((f_x(w, item[0]) - item[1]) ** 2
             for item in data)


def error_grad(w):
  # error = 1/2 Sigma (y(x_n,w) - t_n) ** 2
  # grad_error = [Sigma(y(x_n,w) - t_n), Sigma x_n(y(x_n,w) - t_n)]
  return [sum((f_x(w, item[0]) - item[1]) for item in data) / len(data),
          sum((f_x(w, item[0]) - item[1]) * item[0] for item in data) / len(data)]


all_weights, all_mae, all_errors, all_grads = minimize_batch(error, error_grad,
                                                             [random.uniform(0, 1000),
                                                              random.uniform(-10, 10)])
print('# of Iterations: {}'.format(len(all_mae)))
print('Last MAE: {}'.format(all_mae[-1]))
for i in zip([list(i) for i in all_weights], all_errors, all_mae, all_grads + [[]]):
  print(i)
