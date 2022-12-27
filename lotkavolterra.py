# -*- coding: utf-8 -*-
"""
Lotka-Volterra
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


def implicit_euler(ode, y0, tspan=np.array([0.0, 1.0]), num_steps=10):
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


def predator_prey_deriv(t, rf):
    """
    Derivative of system equations.

    :param t: Input value
    :param rf: The initial conditions
    :return: The derivatives of the equations
    """
    r = rf[0]
    f = rf[1]
    drdt =    1.5 * r - 1 * r * f
    dfdt = - 1 * f + 1 * r * f
    drfdt = np.array ( [ drdt, dfdt ] )

    return drfdt


def neural_network():
    """
    Solves the Lotka-Volterra equations using a fully connected
    neural network.

    :return: The approximated solutions
    """
    alpha, beta, delta, gamma = 1.5, 1, 1, 1
    lotka_volterra = lambda u, v, t : [ diff(u, t) - (alpha*u  - beta*u*v),
                                    diff(v, t) - (delta*u*v - gamma*v), ]
    # Initial conditions.
    init_vals_lv = [
        IVP(t_0=0.0, u_0=1.5),  
        IVP(t_0=0.0, u_0=1.0)  
    ]
    # Sets up the network.
    nets_lv = [
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(32, 32), actv=SinActv),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(32, 32), actv=SinActv)
    ]
    # Creates a monitor.
    monitor = Monitor1D(t_min=0.0, t_max=30.0, check_every=100)
    monitor_callback = monitor.to_callback()
    # Instantiates a solver instance.
    solver = Solver1D(
        ode_system=lotka_volterra,
        conditions=init_vals_lv,
        t_min=0.1,
        t_max=30.0,
        nets=nets_lv,
    )
    # Trains the network.
    solver.fit(max_epochs=3000, callbacks=[monitor_callback])
    # Obtains the solution.
    solution_lv = solver.get_solution()
    ts = np.linspace(0, 30, 300)
    prey_net, pred_net = solution_lv(ts, to_numpy=True)

    return ts, prey_net, pred_net


if __name__=='__main__':

    # Neural network approximation.
    t1_start = process_time()
    ts, prey_net, pred_net = neural_network()
    t1_stop = process_time()

    plt.clf()
    fig = plt.figure(figsize=(6, 5))
    plt.plot(ts, prey_net, label='Prey')
    plt.plot(ts, pred_net, label='Predators')
    plt.ylabel('Population')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

    plt.clf()
    plt.figure(figsize=(6, 5))
    plt.plot(pred_net, prey_net)
    plt.ylabel('Predators')
    plt.xlabel('Prey')
    plt.show()

    print('Elapsed time: ', t1_stop - t1_start)

    tspan = np.array([0.0, 30])
    y0 = np.array([1.5, 1])
    n = 200

    t2_start = process_time()
    s, t = implicit_euler(predator_prey_deriv, y0, tspan, n)
    t2_stop = process_time()

    plt.clf()
    plt.plot(t, s[:, 0])
    plt.plot(t, s[:, 1])
    plt.legend(['Prey', 'Predators'])
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()

    plt.clf()
    plt.plot(s[:, 1], s[:, 0])
    plt.legend(['Numerical Approximation'])
    plt.ylabel('Predators')
    plt.xlabel('Prey')
    plt.show()

    print('Elapsed time: ', t2_stop - t2_start)

    plt.clf()
    plt.plot(ts, prey_net, linewidth=1)
    plt.plot(ts, pred_net, linewidth=1)
    plt.plot(t, s[:, 0], '--', linewidth=2)
    plt.plot(t, s[:, 1], '--', linewidth=2)
    plt.legend(['NN Prey', 'NN Predators', 'NUM Prey', 'NUM Predators'])
    plt.title('Approximated solutions to the Lotka-Volterra Equations')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()

    plt.clf()
    plt.plot(pred_net, prey_net, linewidth=1)
    plt.plot(s[:, 1], s[:, 0], '--', linewidth=2)
    plt.legend(['Neural Network', 'Implicit Euler'])
    plt.title('Phase space of the Lotka-Volterra Equations')
    plt.xlabel('Prey')
    plt.ylabel('Predators')
    plt.show()
