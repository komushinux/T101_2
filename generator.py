from random import random
from time import time

import matplotlib.pyplot as plt
import numpy
import numpy as np


def generate_linear(a, b, noise, filename, size=100):  # generate x,y for linear
    x = 2 * np.random.rand(size, 1) - 1
    y = a * x + b + noise * a * (np.random.rand(size, 1) - 0.5)
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')
    return x, y


def linear_regression_numpy(filename):  # linear regression with polyfit
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    print("shape of x and y are: ", np.shape(x), np.shape(y))

    time_start = time()
    modl = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 1)
    print('modl - ', modl)
    time_end = time()
    print(f"polyfit in {time_end - time_start} seconds")
    h = modl[0] * x + modl[1]

    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    return modl


def linear_regression_exact(filename):  # custom linear regression
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    time_start = time()
    tmp_x = np.hstack([np.ones((100, 1)), x])
    trans_x = np.transpose(tmp_x)
    res_theta = np.linalg.matrix_power(trans_x.dot(tmp_x), -1).dot(trans_x).dot(y)
    print("Res_theta: ", res_theta)
    print("Shape of x and y are: ", np.shape(x), np.shape(y))
    time_end = time()

    h = res_theta[1] * x + res_theta[0]
    print(f"Linear regression time:{time_end - time_start}")
    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    return res_theta


def check(modl, ground_truth):
    if len(modl) != len(ground_truth):
        print("Model is inconsistent")
        return False
    else:
        r = np.dot(modl - ground_truth, modl - ground_truth) / (np.dot(ground_truth, ground_truth))
        print("Result of check: ", r)
        if r < 0.0005:
            return True
        else:
            return False


def generate_poly(a, n, noise, filename, size=100):
    x = 2 * np.random.rand(size, 1) - 1
    y = np.zeros((size, 1))
    if len(a) != (n + 1):
        print(f'ERROR: Length of polynomial coefficients ({len(a)}) must be the same as polynomial degree {n}')
        return
    for i in range(0, n + 1):
        y = y + a[i] * np.power(x, i) + noise * (np.random.rand(size, 1) - 0.5)
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')


def polynomial_regression_numpy(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    print("Shape of x and y are: ", np.shape(x), np.shape(y))
    time_start = time()
    modl = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 2)
    time_end = time()
    print(f"Polynomial regression with polyfit in {time_end - time_start} seconds")
    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    x = np.sort(x, axis=0)
    h = modl[0] * x * x + modl[1] * x + modl[2]
    plt.plot(x, h, "r", label='modl')
    plt.legend()
    plt.show()
    return modl


# Ex.2 gradient descent for linear regression without regularization

# find minimum of function J(theta) using gradient descent
# alpha - speed of descend
# theta - vector of arguments, we're looking for the optimal ones (shape is 1 х N)
# J(theta) function which is being minimizing over theta (shape is 1 x 1 - scalar)
# dJ(theta) - gradient, i.e. partial derivatives of J over theta - dJ/dtheta_i (shape is 1 x N - the same as theta)
# x and y are both vectors

def gradient_descent_step(dJ, theta, alpha):
    new_tet = theta - alpha * dJ
    return new_tet


# get gradient over all xy dataset - gradient descent
def get_dJ(x, y, theta):
    x_trp = x.transpose()
    y_trp = y.transpose()
    h = theta.dot(x_trp)
    dJ = 1 / len(y) * (h - y_trp).dot(x)
    return dJ


def get_dJ_minibatch(x, y, theta):
    x_trp = x.transpose()
    h = theta.dot(x_trp)
    dJ = 1 / len(y) * (h - y).dot(x)
    return dJ


# get gradient over all minibatch of single sample from xy dataset - stochastic gradient descent
def get_dJ_sgd(x, y, theta, alpha):
    temp_x = x
    temp_y = y
    new_theta = np.ndarray
    new_theta.reshape(np.shape(theta))
    theta_trans = theta.transpose()
    h = np.dot(theta_trans, x)  # calculate hypothesis
    dj = [0] * len(theta)
    for i in range(len(theta)):  # calculate partial derivatives of J
        for k in range(len(x)):
            dj[i] += 1 / len(x) * (h[np.power(x, i)] - np.power(y, i)) * np.power(x[k], i)
    for i in range(len(x)):
        index = random.randint(0, len(x))
        new_theta.itemset(i, gradient_descent_step(dj[index], theta[index], alpha))
    return new_theta


def minimize(x_data, y_data, L, deg_stand):
    alpha = 0.15
    theta_count = np.ones((1, deg_stand + 1))  # you can try random initialization

    itr = [0] * L
    J_mass = [None] * L
    for i in range(L):
        dJ = get_dJ(x_data, y_data, theta_count)  # here you should try different gradient descents
        # theta_count = theta_count - alpha * dJ
        theta_count = gradient_descent_step(dJ, theta_count, alpha)
        alpha -= 0.0002
        # print('new theta = ', theta_count)
        h = np.dot(theta_count, x_data.transpose())
        J_mass[i] = 0.5 / len(y_data) * np.square(h - y_data.transpose()).sum(axis=1)
        itr[i] = i

    plt.title("Minimize task")
    plt.xlabel("iteration")
    plt.ylabel("Loss func")
    plt.plot(itr, J_mass, "g.")
    plt.show()
    return theta_count


def minimize_minibatch(x_data, y_data, L, M, deg_minibatch):
    alpha = 0.15
    shape_x = np.shape(x_data)[0]
    size_minibatch = shape_x / M
    x_data = np.vsplit(x_data, size_minibatch)
    y_data = y_data.transpose()
    y_data = np.hsplit(y_data, size_minibatch)
    theta_count = np.ones((1, deg_minibatch + 1))

    for i in range(L):
        buf = int(size_minibatch * random())
        dJ = get_dJ_minibatch(x_data[buf], y_data[buf], theta_count)  # here you should try different gradient descents
        # theta_count = theta_count - alpha * dJ
        theta_count = gradient_descent_step(dJ, theta_count, alpha)
        alpha -= 0.0002
        # print('new theta = ', theta_count)
        h = np.dot(theta_count, x_data[buf].transpose())
        J = 0.5 / 60 * np.square(h - y_data[buf]).sum(axis=1)
        plt.plot(i, J, "b.")

    plt.title("Minimize task")
    plt.xlabel("iteration")
    plt.ylabel("Loss func")
    plt.show()
    return theta_count


if __name__ == "__main__":
    # ex1. exact solution
    # generate_linear(1, -3, 1, 'linear.csv', 100)
    # model = np.squeeze(linear_regression_exact("linear.csv"))
    mod1 = np.squeeze(numpy.asarray(np.array(([-3], [1]))))
    # print(f"Is model correct? - {check(model, mod1)}")
    # print("*" * 40)
    #
    # # ex1. polynomial with numpy
    # generate_poly([1, 2, 3], 2, 0.5, 'polynomial.csv')
    # poly_model = polynomial_regression_numpy("polynomial.csv")
    # print("*" * 40)

    # ex2. find minimum with gradient descent
    # 0. generate date with function above
    generate_linear(1, -3, 1, 'linear.csv', 100)
    # 1. shuffle data into train - test - valid
    with open('linear.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    train_data = data[:60]
    test_data = data[60::1]
    valid_data = data[80::1]
    x, y = np.hsplit(train_data, 2)
    x = np.hstack([np.ones((60, 1)), x])

    # 2. call minimize(...) and plot J(i)
    # 3. call check(theta1, theta2) to check results for optimal theta
    deg = 1
    theta = np.squeeze(minimize(x, y, 60, deg))
    print('theta = ', theta)
    print(f"Is model correct? - {check(theta, mod1)}")
    print("*" * 40)

    theta = np.squeeze(minimize_minibatch(x, y, 60, 5, deg))
    print('theta = ', theta)
    print(f"Is model correct? - {check(theta, mod1)}")
    print("*" * 40)
    # ex3. polynomial regression
    # 0. generate date with function generate_poly for degree=3, use size = 10, 20, 30, ... 100
    # for each size:
    # 1. shuffle data into train - test - valid
    # Now we're going to try different degrees of model to aproximate our data, set degree=1 (linear regression)
    # 2. call minimize(...) and plot J(i)
    # 3. call check(theta1, theta2) to check results for optimal theta
    # 4. plot min(J_train), min(J_test) vs size: is it overfit or underfit?
    #
    # repeat 0-4 for degres = 2,3,4

    # ex3* the same with regularization
