# Decision Boundary is \vec{w}.\vec{x} + b = 0
# Starting with initial values w^(0) as a vector of 0s and b^(0)=0,
# it updates by looping through all of the points and improving the
# weights and bias based on any misclassified points as follows:
# w^(k+1) = w^(k) + y_i * x_i
# b^(k+1) = b^(k) + y_i
#
# We know a point is misclassified if y_i * (w * x_i + b) <= 0  

import matplotlib.pyplot as plt
import numpy as np


def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    line1.set_data(x, y)


b = [0]
w = np.array([[0, 0]])
# observed data:
x = np.array([[1, 3], [2, 4], [3, 3], [1, 1], [2, 0], [3, 1]])
y = [1, 1, 1, -1, -1, -1]

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(0, 0, color='green')

b_index = 0
w_index = 0
x_index = 0
y_index = 0
counter = 0
iter_count = 0

while counter != len(x):  # while not all points are classified correctly

    x_index = x_index % len(x)  # looping through members
    y_index = y_index % len(x)

    # if the point is not classified correctly
    if y[y_index] * ((w[w_index]@x[x_index]) + b[b_index]) <= 0:
        counter = 0
        iter_count += 1

        # update weight
        tmp_w = np.add(w[w_index], y[y_index] * x[x_index])
        w = np.append(w, [tmp_w], axis=0)
        w_index += 1

        # update bias
        b.append(b[b_index] + y[y_index])
        b_index += 1

    else:
        counter += 1

    x_index += 1
    y_index += 1

    # The following x and y are different from x, y of the observed data
    # These x, y are for drawing the decision boundary
    eq_x = w[-1][0]  # w[-1] is the last updated weights
    eq_y = w[-1][1]
    eq_b = b[-1]
    # w1*x + w2*y + b = 0  ->  y = (-w1*x - b)/w2
    equation = '({}*x+{})/{}'.format(-eq_x, -eq_b, eq_y)

    graph(equation, range(0, 6))
    plt.title('iteration {}'.format(iter_count))
    plt.plot([1, 2, 3], [3, 4, 3], 'ro', color='blue')
    plt.plot([1, 2, 3], [1, 0, 1], 'ro', color='red')
    fig.canvas.draw()

plt.show(block=True)    # keep the plot window open at the end

print("w = " + str(w[-1]), "\nb = " + str(b[-1]))
