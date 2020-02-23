# Gradient Descent on simplest linear model y(x, w) = w0 + w1*x1 + ... + wn*xn
# we start with an initial guess for w
# and update it by taking steps in opposite of the gradient of error function
# The error function we use here is sum of squared errors

from matplotlib import pyplot as plt
from IPython.display import clear_output
import random
import time

with open('...') as f:
  d = f.readlines()[1:]  # [1:] to exclude header
data = [ eval(line.rstrip('\n')) for line in d ]
X, Y = zip(*data)

plt.scatter(X, Y)
plt.show()

min_X = min(i[0] for i in data)
max_X = max(i[0] for i in data)

def step(p, direction, step_size):
  """move step_size in the direction from p"""
  return [p_i + step_size * direction_i
          for p_i, direction_i in zip(p, direction)]

def minimize_batch(target_fn, gradient_fn, w_0, tolerance=30):
  """
  use gradient descent to find theta that minimizes target function.
  tolerance: to stop the program if the pace of the improvement is less than this number
  """

  w = w_0  # initial guess for weights
  step_size = -0.000001  # to minimize, move towards opposite of gradient
  value = target_fn(w)
  errs = []

  while True:
    gradient = gradient_fn(w)
    next_w = step(w, gradient, step_size)
    next_value = target_fn(next_w)
    if abs(next_value - value) < tolerance:
      errs.append(mae(w))
      return (w, errs)
    else:
      w, value = next_w, next_value
      # plot the fitted curve with the current weights
      x_s, y_s = min_X, f_x(w, min_X)  # start point for the line
      x_e, y_e = max_X, f_x(w, max_X)   # end point for the line
      clear_output(wait=True)  # to update the current plot
      plt.scatter(X, Y)
      plt.plot([x_s, x_e], [y_s, y_e], color='green')
      plt.show()
      errs.append(mae(w))
      #break


def f_x(coefs, x):
  return coefs[0] + coefs[1] * x 

def mae(w):
  return sum([ abs(f_x(w, i[0]) - i[1] ) for i in data ])/len(data)

def error(w):
  # y(x, w) = w0 + w1*x
  # error = 1/2 Sigma (y(x_n,w) - t_n) ** 2
  # the 1/2 coefficient is for making the math simple and not important
  return sum((f_x(w, item[0]) - item[1]) ** 2
             for item in data)

def error_grad(w):
  # error = 1/2 Sigma (y(x_n,w) - t_n) ** 2
  # grad_error = [Sigma(y(x_n,w) - t_n), Sigma x_n(y(x_n,w) - t_n)]
  return [sum((w[0] + w[1] * item[0] - item[1]) for item in data) / len(data),
          sum((w[0] + w[1] * item[0] - item[1]) * item[0] for item in data) / len(data)]

final_w = minimize_batch(error, error_grad,
                       [random.uniform(0, 1000),
                        random.uniform(-10, 10)])
print(final_w[1])
