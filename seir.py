# -*- coding: utf-8 -*-
"""
SEIR System
"""
from torch import linspace
from neurodiffeq import diff
from neurodiffeq.solvers import Solver1D, Solver2D
from neurodiffeq.conditions import IVP, DirichletBVP2D
from neurodiffeq.networks import FCNN, SinActv
from neurodiffeq.monitors import Monitor1D
from matplotlib import pyplot as plt
from neurodiffeq.ode import solve_system
from time import process_time
from scipy.optimize import fsolve
import torch
import numpy as np

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


def implicit_euler(ode, y0, tspan, num_steps=10):
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

def seir_deriv(t, values):
    """
    Derivative of system equations.

    :param t: Input value
    :param rf: The initial conditions
    :return: The derivatives of the equations
    """
    s = values[0]
    e = values[1]
    i = values[2]
    r = values[3]

    a = 0.05
    b = 0.2
    c = 0.1

    dsdt = -a * s * i
    dedt = a * s * i - b * e
    didt = b * e - c * i
    drdt = c * i

    derivs = np.array([dsdt, dedt, didt, drdt])

    return derivs


def neural_network():
    """
    Solves the SEIR system of equations using a fully connected
    neural network.

    :return: The solutions to the system
    """
    beta, epsilon, gamma = 0.05, 0.2, 0.1
    seir = lambda s, e, i, r, t : [ diff(s, t) + (beta * s * i),
                                    diff(e, t) - (beta * s * i - epsilon * e),
                                    diff(i, t) - (epsilon * e - gamma * i),
                                    diff(r, t) - (gamma * i) ]
    # Initial conditions.
    init_vals_seir = [
        IVP(t_0=0.0, u_0=10),  
        IVP(t_0=0.0, u_0=1.0),  
        IVP(t_0=0.0, u_0=0.0),
        IVP(t_0=0.0, u_0=0.0)
    ]
    # Sets up the neural network.
    nets_seir = [
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(32, 32), actv=SinActv),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(32, 32), actv=SinActv),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(32, 32), actv=SinActv),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(32, 32), actv=SinActv)
    ]
    # Creates a monitor.
    monitor = Monitor1D(t_min=0.0, t_max=25.0, check_every=100)
    monitor_callback = monitor.to_callback()
    # Instantiates a solver instance.
    solver = Solver1D(
        ode_system=seir,
        conditions=init_vals_seir,
        t_min=0.1,
        t_max=25.0,
        nets=nets_seir,
    )
    # Trains the network.
    solver.fit(max_epochs=3000, callbacks=[monitor_callback])
    # Gets the solution.
    solution_seir = solver.get_solution()
    ts = np.linspace(0, 25, 100)
    s_net, e_net, i_net, r_net = solution_seir(ts, to_numpy=True)

    return ts, s_net, e_net, i_net, r_net


if __name__=='__main__':
    # Neural network approximation.
    t1_start = process_time()
    ts, s_net, e_net, i_net, r_net = neural_network()
    t1_stop = process_time()

    plt.clf()
    fig = plt.figure(figsize=(6, 5))
    plt.plot(ts, s_net, label='Susceptible')
    plt.plot(ts, e_net, label='Exposed')
    plt.plot(ts, i_net, label='Infected')
    plt.plot(ts, r_net, label='Removed')
    plt.ylabel('Population')
    plt.xlabel('Time')
    plt.title('Neural Network Approximation')
    plt.legend()
    plt.show()

    print('Elapsed time: ', t1_stop - t1_start)

    tspan = np.array([0.0, 25])
    y0 = np.array([10, 1, 0, 0])
    n = 100

    t2_start = process_time()
    s_num, t = implicit_euler(seir_deriv, y0, tspan, n)
    t2_stop = process_time()

    plt.clf()
    plt.plot(t, s_num)
    plt.legend(['Susceptible', 'Exposed', 'Infected', 'Removed'])
    plt.title('Numerical Approximation')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()

    print('Elapsed time: ', t2_stop - t2_start)

    plt.clf()
    plt.plot(ts, s_net, linewidth=1)
    plt.plot(ts, e_net, linewidth=1)
    plt.plot(ts, i_net, linewidth=1)
    plt.plot(ts, r_net, linewidth=1)
    plt.plot(t, s_num, '--', linewidth=2)
    plt.legend(['NN Susceptible', 'NN Exposed', 'NN Infected', 'NN Removed', 'NUM Susceptible', 'NUM Exposed', 'NUM Infected', 'NUM Removed'])
    plt.title('Approximated solutions to the SEIR System')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()
