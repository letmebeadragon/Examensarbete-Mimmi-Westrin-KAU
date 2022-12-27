# -*- coding: utf-8 -*-
"""
Example 2
"""

import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from time import process_time


def sigmoid(z):
    """
    Sigmoid activation function.

    :param z: The input value
    :return: The output of the activation function
    """
    return 1 / (1 + np.exp(-z))


def neural_network(parameters, x_in):
    """
    Sets up the neural network and calculates the output.

    :param parameters: The weights and biases of the network
    :param x_in: The input values
    :return: The output of the neural network
    """
    # Calculates the number of hidden layers.
    n_hidden = np.size(parameters) - 1
    # Gets the input values.
    num_values = np.size(x_in)
    x_in = x_in.reshape(-1, num_values)
    # Input layer consisting of just the input.
    x_input = x_in
    # References to the previous hidden layer.
    x_prev = x_input

    # Input goes through the hidden layers.
    for i in range(n_hidden):
        # Finds the correct weights and biases for the layer.
        w_hidden = parameters[i]
        # Adds a row of ones to include bias.
        x_prev = np.concatenate((np.ones((1, num_values)), x_prev), axis=0)
        # Calculates the output of the layer.
        z_hidden = np.matmul(w_hidden, x_prev)
        x_hidden = sigmoid(z_hidden)
        # Updates x_prev to point at current layer.
        x_prev = x_hidden

    # Gets the weights and bias of the output layer.
    w_output = parameters[-1]
    x_prev = np.concatenate((np.ones((1, num_values)), x_prev), axis=0)
    # Calculates the output.
    z_output = np.matmul(w_output, x_prev)
    x_output = z_output

    return x_output


def neural_network_solve_ode(x_in, num_neurons, num_iterations, learning_rate):
    """
    Sets up the parameters for the neural network and solves the given ODE.

    :param x_in: The input values
    :param num_neurons: The number of neurons in the network
    :param num_iterations: The number of iterations used for training
    :param learning_rate: The rate at which the network learns
    :return: The network parameters
    """
    # Calculates the number of hidden layers.
    n_hidden = np.size(num_neurons)
    # Sets up initial weights and biases.
    parameters = [None] * (n_hidden + 1)
    parameters[0] = npr.randn(num_neurons[0], 2)
    # Initialises the parameters to random values.
    for j in range(1, n_hidden):
        parameters[j] = npr.randn(num_neurons[j], num_neurons[j - 1] + 1)
    # Initialises the parameters for the output layer.
    parameters[-1] = npr.randn(1, num_neurons[-1] + 1)

    print('Initial cost: %g' % cost_function(parameters, x_in))

    cost_function_grad = grad(cost_function, 0)
    # Trains the network.
    for i in range(num_iter):
        # Evaluates the gradient.
        cost_grad = cost_function_grad(parameters, x_in)
        # Updates the parameters.
        for j in range(n_hidden + 1):
            parameters[j] = parameters[j] - learning_rate * cost_grad[j]

    print('Final cost: %g' % cost_function(parameters, x_in))

    return parameters


def f(x, trial):
    """
    The right side of the ODE.

    :param x: The input value
    :param trial: The value of the trial solution
    :return: The value of the function
    """
    return 1 / trial


def cost_function(parameters, x_in):
    """
    Returns the means square error of the approximation.

    :param parameters: The weights and biases of the network
    :param x_in: The input values
    :return: The mean squared error
    """
    # Evaluates the trial solution with the current parameters.
    g_t = g_trial_solution(x_in, parameters)
    # Finds the derivative of the trial function.
    d_g_t = elementwise_grad(g_trial_solution, 0)(x_in, parameters)

    right_side = f(x_in, g_t)

    err_sqr = (d_g_t - right_side) ** 2
    cost_sum = np.sum(err_sqr)

    return cost_sum / np.size(err_sqr)


def g_trial_solution(x, parameters):
    """
    Calculates the value of the trial solution at a point x.

    :param x: The input value
    :param parameters: The weights and biases of the network
    :return: The value of the trial solution at x
    """
    return 1 + x * neural_network(parameters,x)


def g_exact(x):
    """
    Calculates the values of the exact solution.

    :param x: The input values
    :return: The output values of the exact solution
    """
    return (2 * x + 1) ** (1/2)


def implicit_euler_residual(yp, ode, to, yo, tp):
    """
    Evaluates the residual of the implicit Euler.

    :param yp: Estimated solution value at the new time
    :param ode: The right hand side of the ODE
    :param to: The old time
    :param yo: The old solution value
    :param tp: The new time
    :return: The residual
    """
    value = yp - yo - (tp - to) * ode(tp, yp)
    return value


def implicit_euler(ode, tspan=np.array([0.0, 1.0]), y0=1, num_steps=10):
    """
    Numerical approximation of the solution to the ODE.

    :param ode: The ODE to be solved
    :param tspan: Start and end times
    :param y0: Initial condition
    :param num_steps: The number of steps
    :return: Returns the numerical approximation
    """
    if np.ndim(y0) == 0:
        m = 1
    else:
        m = len(y0)
    # Defines parameters.
    t = np.zeros(num_steps + 1)
    y = np.zeros([num_steps + 1, m])
    dt = (tspan[1] - tspan[0]) / float(num_steps)
    t[0] = tspan[0]
    y[0, :] = y0

    # Implicit Euler Method.
    for i in range(0, num_steps):
        to = t[i]
        yo = y[i, :]
        tp = t[i] + dt
        yp = yo + dt * ode(to, yo)
        # Solves equation.
        yp = fsolve(implicit_euler_residual, yp, args=(ode, to, yo, tp))
        # Updates values.
        t[i + 1] = tp
        y[i + 1, :] = yp[:]

    return y, t


if __name__ == '__main__':
    npr.seed(4155)
    nx = 10
    x = np.linspace(0, 1, nx)
    # Sets up the initial parameters.
    num_hidden_neurons = [200, 100]
    num_iter = 1000
    lmb = 1e-3

    t1_start = process_time()
    P = neural_network_solve_ode(x, num_hidden_neurons, num_iter, lmb)
    t1_stop = process_time()

    g_dnn_ag = g_trial_solution(x, P)
    g_analytical = g_exact(x)

    tnum_start = process_time()
    s, t = implicit_euler(lambda t, s: 1 / s, num_steps=nx)
    tnum_stop = process_time()

    # Finds the maximum absolute difference between the solutions.
    max_diff = np.max(np.abs(g_dnn_ag - g_analytical))
    print("The max absolute difference between the solutions is: %g" % max_diff)
    print("Elapsed time: ", t1_stop - t1_start)

    max_diff_num = np.max(np.abs(s - g_analytical))
    mse = np.square(s - g_analytical).mean()
    print("Mse: ", mse)
    print("The max abs diff is: %g" % max_diff_num)
    print("Elapsed time: ", tnum_stop - tnum_start)

    plt.figure(figsize=(10, 10))

    plt.plot(x, g_analytical, marker="o")
    plt.plot(x, g_dnn_ag[0, :])
    plt.plot(t, s)
    plt.legend(['Exact Solution', 'Neural Network Approximation', 'Implicit Euler Approximation'])
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('Approximated solutions to Example 2')
    plt.show()
