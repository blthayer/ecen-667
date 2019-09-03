"""Author: Brandon Thayer
Module for homework 1.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# Tolerance for Newton's method.
EPS = 0.0001


def newton(x_0: np.ndarray, f: Callable[[np.ndarray], np.ndarray],
           j: Callable[[np.ndarray], np.ndarray], eps=EPS):
    """Newton's method for root finding.

    :param x_0: Initial guess, one dimensional.
    :param f: Function of x.
    :param j: Jacobian of f.
    :param eps: epsilon: tolerance for stopping iteration.
    """
    # Initialize f(x).
    f_x = f(x_0)

    # Initialize x to our starting point.
    x = x_0

    # Track iterations.
    i = 0

    # Loop while we do not have all our entries within the given
    # tolerance. Use a hard-coded iteration cap for safety.
    while (not np.all(np.abs(f_x) < eps)) and (i < 100):
        # Increment iteration counter.
        i += 1

        # Perform the Newton update.
        x = x - np.linalg.solve(j(x), f_x)

        # Re-compute f_x for the next iteration.
        f_x = f(x)

    if i >= 100:
        raise UserWarning('Iteration count exceeded!')

    # All done. Return our final x.
    return x


def init_x(x_0, t_end, t_start, dt):
    """Helper to initialize a matrix to hold time series solutions for
    numerically solving differential equations.

    :param x_0: 1 dimensional numpy array with initial conditions.
    :param t_end: Float, ending time.
    :param t_start: Float, starting time.
    :param dt: Time step to use.

    :returns: x, numpy array with size (num time steps, len(x_0))
    """
    # Compute the number of time steps we'll use.
    n = int(np.ceil(t_end - t_start) / dt)

    # Initialize matrix.
    x = np.zeros(shape=(n, x_0.shape[0]))

    # Put initial conditions in.
    x[0, :] = x_0

    # Done.
    return x


def trap_int(x_0: np.ndarray, f: Callable[[np.ndarray], np.ndarray],
             j: Callable[[np.ndarray], np.ndarray], dt: float, t_end: float,
             t_start=0.0):
    """Implicit trapezoidal integration.

    :param x_0: Initial conditions, one dimensional.
    :param f: Function of x.
    :param j: Function to evaluate the Jacobian of f(x)
    :param dt: Time step to use.
    :param t_end: Ending time.
    :param t_start: Starting t.
    """
    # Initialize our x matrix.
    x = init_x(x_0=x_0, t_start=t_start, t_end=t_end, dt=dt)

    # Evaluate for each time step.
    for idx in range(1, x.shape[0]):
        # Grab this x for convenient.
        x_n = x[idx - 1, :]

        # Get a function for evaluating the trapezoidal equation.
        trap_eq = trap_eq_factory(x_n=x_n, dt=dt, f=f)

        # Get a function for evaluating the Jacobian of the trapezoidal
        # equation.
        trap_jac = trap_jac_factory(dt=dt, j=j)

        # Compute x at the next time step by solving the trapezoidal
        # equation via Newton's method.
        x[idx, :] = newton(x_0=x[idx, :], j=trap_jac, f=trap_eq)

    return x


def rk2(x_0: np.ndarray, f: Callable[[np.ndarray], np.ndarray], dt: float,
        t_end: float, t_start=0.0):
    """Second order Runge-Kutte method for numerical integration.

    :param x_0: Initial conditions. Should be one dimensional.
    :param f: Function of x.
    :param dt: Time step to use.
    :param t_end: Ending time. Float.
    :param t_start: Starting t, defaults to 0.
    """
    # Initialize our x matrix.
    x = init_x(x_0=x_0, t_start=t_start, t_end=t_end, dt=dt)

    # Evaluate for each time step.
    for idx in range(1, x.shape[0]):
        # Extract the previous x for convenience.
        x_n = x[idx - 1, :]

        k1 = dt * f(x_n)
        k2 = dt * f(x_n + k1)
        x[idx, :] = x_n + (k1 + k2) / 2

    return x


def main():
    # Problem 1:
    # TODO:

    # Problem 2:
    p2()


def p1_func(x):
    """Function f(x) for problem 1.

    Equations:
    x_1_dot = 2 / 3 * x_1 - 4/3 * x1 * x2
    x_2_dot = x_1 * x_2 - x_2

    :param x: numpy array with 2 elements.
    :returns: f(x), also a numpy array with 2 elements.
    """
    return np.array(
        [
            2/3 * x[0] - 4/3 * x[0] * x[1],
            x[0] * x[1] - x[1]
        ]
    )


def p1_jac(x):
    """Function to compute the Jacobian matrix for problem 1.
    """
    return np.array(
        [
            [
                2/3 - 4/3 * x[1],   # df1/dx1
                - 4/3 * x[0]        # df1/dx2
            ],
            [
                x[1],               # df2/dx1
                x[0] - 1            # df2/dx2
            ]
        ]
    )


def trap_eq_factory(x_n, dt, f):

    def trap_eq(x_n_1):
        return x_n_1 - x_n - dt / 2 * (f(x_n) + f(x_n_1))

    return trap_eq


def trap_jac_factory(j, dt):

    def trap_jac(x_n):
        """Function to compute the Jacobian of the implicit trapezoidal
        equation.
        """
        return np.identity(x_n.shape[0]) - dt / 2 * j(x_n)

    return trap_jac


def p1():
    """Problem 1."""
    x_0 = np.array([1, 1])

    # Test out our root finding method. Spoiler: it works.
    # root = newton(x_0, p1_func, p1_jac, eps=0.0001)
    dt = 0.01
    t_end = 5
    x_trap = trap_int(x_0=x_0, f=p1_func, j=p1_jac, dt=dt, t_end=t_end)
    plt.title('Trapezoidal')
    plt.plot(x_trap)
    plt.figure()
    x_rk2 = rk2(x_0=x_0, f=p1_func, dt=dt, t_end=t_end)
    plt.plot(x_rk2)
    plt.title('RK2')
    plt.show()

    pass


def p2_func(x):
    """Function f(x) for problem 2.

    Equations:
    x_1_dot = 10 * (x_2 - x_1)
    x_2_dot = x_1 * (28 - x_3) - x_2
    x_3_dot = x_1 * x_2 - 8/3 * x_3

    :param x: numpy array with three elements.
    :returns: f(x), also a numpy array with three elements.
    """
    return np.array(
        [
            10 * (x[1] - x[0]),
            x[0] * (28 - x[2]) - x[1],
            x[0] * x[1] - 8 / 3 * x[2]
        ]
    )


def p2():
    """Problem 2."""
    x_0 = np.array([5, 5, 5])

    # Notes on dt and t_end:
    #
    # dt = 0.1,     t_end = 1   --> function quickly blows up.
    # dt = 0.01,    t_end = 1   --> much better, but can't tell if we're
    #                               going to hit equilibrium.
    # dt = 0.001,   t_end = 1   --> Looks identical to case above.
    # dt = 0.01,    t_end = 5   --> clearly oscillating behavior with
    #                               growing amplitude.
    # dt = 0.01,    t_end = 10  --> Weird stuff at 6 seconds, but still
    #                               no clear equilibrium.
    # dt = 0.01,    t_end = 100 --> Very clear there's no equilibrium.
    #                               Some sort of larger oscillatory
    #                               behavior.
    dt = 0.01
    t_end = 5

    x = rk2(x_0=x_0, f=p2_func, dt=dt, t_end=t_end)
    plt.plot(x)
    plt.show()


if __name__ == '__main__':
    p1()
    # p2()
    # main()
