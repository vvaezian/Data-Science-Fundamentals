# Gradient Descent on simplest linear model y(x, w) = w0 + w1*x1 + ... + wn*xn
# we start with an initial guess for w
# and update it by taking steps in opposite of the gradient of error function
# The error function we use here is sum of squared errors

import matplotlib.pyplot as plt
import random


def step(p, direction, step_size):
    """move step_size in the direction from p"""
    return [p_i + step_size * direction_i
            for p_i, direction_i in zip(p, direction)]


def minimize_batch(target_fn, gradient_fn, w_0, tolerance=0.01):
    """use gradient descent to find theta that minimizes target function"""
    w = w_0  # initial guess for weights
    value = target_fn(w)
    step_size = -0.000001  # to minimize, move towards opposite of gradient

    while True:
        print(w)
        gradient = gradient_fn(w)
        next_w = step(w, gradient, step_size)
        next_value = target_fn(next_w)

        if abs(next_value - value) < tolerance:
            return w
        else:
            w, value = next_w, next_value


# ---------- EXAMPLE ------------
def error(w):
    # y(x, w) = w0 + w1*x
    # error = 1/2 Sigma (y(x_n,w) - t_n) ** 2
    # the 1/2 coefficient is for making the math simple and not important
    return sum(((w[0] + w[1] * item[0]) - item[1]) ** 2
               for item in data)


def error_grad(w):
    # error = 1/2 Sigma (y(x_n,w) - t_n) ** 2
    # grad_error = [Sigma(y(x_n,w) - t_n), Sigma x_n(y(x_n,w) - t_n)]
    return [sum((w[0] + w[1] * item[0] - item[1]) for item in data) / len(data),
            sum((w[0] + w[1] * item[0] - item[1]) * item[0] for item in data) / len(data)]


# observed data:
f = open('data.txt')
data = [eval(line.rstrip('\n')) for line in f]
# "eval" converts strings into their intended meanings
# "rstrip('\n')" is for ignoring newline symbols from right (r-strip)
# which causes to print an empty line between lines

final_w = minimize_batch(error, error_grad,
                         [random.randrange(0, 10),
                          random.randrange(0, 10)])

x_list, y_list = zip(*data)
plt.scatter(x_list, y_list)

# drawing the fitting line
minx = min(i[0] for i in data)
maxx = max(i[0] for i in data)
x_1, y_1 = minx, final_w[0] + final_w[1] * minx
x_2, y_2 = maxx, final_w[0] + final_w[1] * maxx
plt.plot([x_1, x_2], [y_1, y_2], color='green')

plt.show()
