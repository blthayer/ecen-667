"""Author: Brandon Thayer
Module for homework 1.

There's a "main" method which solves problems 1 and 2.

The following web pages were useful:
https://en.wikipedia.org/wiki/Trapezoidal_rule_(differential_equations)
https://homepage.math.uiowa.edu/~ljay/publications.dir/EACM_Lobatto_Methods.pdf
http://www.math.pitt.edu/~sussmanm/2071/lab03/index.html
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
# Use scipy to confirm that my methods are working.
from scipy.integrate import solve_ivp

# Tolerance and maximum iterations for Newton's method.
EPS = 0.0001
MAX_ITERATIONS = 100


def newton(x_0: np.ndarray, f: Callable[[np.ndarray], np.ndarray],
           j: Callable[[np.ndarray], np.ndarray], eps=EPS):
    """Newton's method for root finding.

    :param x_0: Initial guess, one dimensional.
    :param f: Function of x.
    :param j: Jacobian of f.
    :param eps: epsilon: tolerance for stopping iteration.

    :returns: x, numpy array of the same shape as x_0, containing the
        root of f.

    :raises UserWarning: If MAX_ITERATIONS is exceeded when finding the
        root.
    """
    # Initialize f(x).
    f_x = f(x_0)

    # Initialize x to our starting point.
    x = x_0

    # Track iterations.
    i = 0

    # Loop while we do not have all our entries within the given
    # tolerance. Use a hard-coded iteration cap for safety.
    while (not np.all(np.abs(f_x) < eps)) and (i < MAX_ITERATIONS):
        # Increment iteration counter.
        i += 1

        # Perform the Newton update.
        x = x - np.linalg.solve(j(x), f_x)

        # Re-compute f_x for the next iteration.
        f_x = f(x)

    # Raise an exception if Newton's method didn't converge.
    if i >= MAX_ITERATIONS:
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


def implicit_trapezoidal_integration(
        x_0: np.ndarray, f: Callable[[np.ndarray], np.ndarray],
        j: Callable[[np.ndarray], np.ndarray], dt: float, t_end: float,
        t_start=0.0):
    """Implicit trapezoidal integration.

    :param x_0: Initial conditions, one dimensional.
    :param f: Function of x.
    :param j: Function to evaluate Jacobian(x), where the Jacobian is
        for the function f.
    :param dt: Time step to use.
    :param t_end: Ending time.
    :param t_start: Starting t.

    :returns: x, numpy array with as many rows as time steps, and as
        many columns as variables in x_0. Contains the numerical
        integration results for each time step.
    """
    # Initialize our x matrix.
    x = init_x(x_0=x_0, t_start=t_start, t_end=t_end, dt=dt)

    # Evaluate for each time step.
    for idx in range(1, x.shape[0]):
        # Grab this x for convenience.
        x_n = x[idx - 1, :]

        # Get a function for evaluating the trapezoidal equation.
        trap_eq = trap_eq_factory(x_n=x_n, dt=dt, f=f)

        # Get a function for evaluating the Jacobian of the trapezoidal
        # equation.
        trap_jac = trap_jac_factory(dt=dt, j=j)

        # Compute x at the next time step by solving the trapezoidal
        # equation via Newton's method.
        x[idx, :] = newton(x_0=x[idx, :], j=trap_jac, f=trap_eq)

    # All done, return the solution.
    return x


def rk2(x_0: np.ndarray, f: Callable[[np.ndarray], np.ndarray], dt: float,
        t_end: float, t_start=0.0):
    """Second order Runge-Kutte method for numerical integration.

    :param x_0: Initial conditions. Should be one dimensional.
    :param f: Function of x.
    :param dt: Time step to use.
    :param t_end: Ending time. Float.
    :param t_start: Starting t, defaults to 0.

    :returns: x, numpy array with as many rows as time steps, and as
        many columns as variables in x_0. Contains the numerical
        integration results for each time step.
    """
    # Initialize our x matrix.
    x = init_x(x_0=x_0, t_start=t_start, t_end=t_end, dt=dt)

    # Evaluate for each time step.
    for idx in range(1, x.shape[0]):
        # Extract the previous x for convenience.
        x_n = x[idx - 1, :]

        # Compute k1 and k2
        k1 = dt * f(x_n)
        k2 = dt * f(x_n + k1)

        # Evaluate x for this time step.
        x[idx, :] = x_n + (k1 + k2) / 2

    # All done, return the solution.
    return x


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


# noinspection PyUnusedLocal
def p1_func_wrapper(t, x):
    """Wrapper to call p1_func but accept a 't' argument. This is for
    verification with scipy.
    """
    return p1_func(x)


def p1_jac(x: np.ndarray):
    """Function to evaluate the Jacobian matrix for problem 1.

    :param x: Numpy array of shape (2,) corresponding to x1 and x2
        for problem 1.
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
    """Factory function to return a function for evaluating the
    trapezoidal integration function. This returns a function of
    x_(n+1) (x at the next time step) for the given current x (x_n),
    time step (dt), and function (f).

    :param x_n: one-dimensional numpy array giving the values of x at
        the current time step.
    :param dt: Time step.
    :param f: Function of x.

    :returns: trap_eq, callable which takes x_(n+1) and evaluates the
        trapezoidal formula.
    """

    def trap_eq(x_n_1):
        """Simply write up the trapezoidal formula."""
        return x_n_1 - x_n - dt / 2 * (f(x_n) + f(x_n_1))

    # Return our new function.
    return trap_eq


def trap_jac_factory(j, dt):
    """Factory function to return a function for evaluating the Jacobian
    of the trapezoidal formula. This returns a function of x_n (x at
    this time step).

    :param j: Jacobian of the function of x.
    :param dt: time step.

    :returns: trap_jac, callable which takes x_n and evaluates the
        Jacobian of the trapezoidal formula.
    """

    def trap_jac(x_n):
        """Function to compute the Jacobian of the implicit trapezoidal
        equation.
        """
        return np.identity(x_n.shape[0]) - dt / 2 * j(x_n)

    return trap_jac


def p1():
    """Problem 1.

    This function doesn't return anything, but saves hw1_p1.eps to file.
    """
    # Initial conditions.
    x_0 = np.array([1, 1])

    # Test out our root finding method. Spoiler: it works.
    # root = newton(x_0, p1_func, p1_jac, eps=0.0001)

    # Set our time step (dt) and ending time (t_end).
    dt = 0.01
    t_end = 20

    # Notes:
    # t_end = 5 is not long enough to see what's going on.
    # t_end = 10: same
    # t_end = 100: See lots of oscillations
    # t_end =

    # Perform the trapezoidal integration.
    x_trap = implicit_trapezoidal_integration(x_0=x_0, f=p1_func, j=p1_jac,
                                              dt=dt, t_end=t_end)
    plot(x=x_trap,
         title="Problem 1, Implicit Trapezoidal Integration via Newton's "
               "Method.", dt=dt, t_end=t_end, out_file='hw1_p1_trap.eps')

    # Ensure our trapezoidal method looks similar to the rk2 results.
    # Spoiler: they do.
    x_rk2 = rk2(x_0=x_0, f=p1_func, dt=dt, t_end=t_end)
    plot(x=x_rk2,
         title="Problem 2, Runge-Kutte.", dt=dt, t_end=t_end,
         out_file="hw1_p1_rk2.eps")

    # Confirm with scipy.
    result_scipy = solve_ivp(fun=p1_func_wrapper, t_span=(0, t_end),
                             y0=x_0, max_step=dt)
    plot(x=result_scipy['y'].T,
         title="Problem 1, Using Scipy.", dt=dt, t_end=t_end,
         out_file='hw1_p1_scipy.eps')


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


# noinspection PyUnusedLocal
def p2_func_wrapper(t, x):
    """Wrapper to call p2_func but accept a 't' argument. This is for
    verification with scipy.
    """
    return p2_func(x)


def p2_jac(x):
    """Function to evaluate the Jacobian for problem 2. While this
    is not needed for Runge-Kutte, it's useful so we can compare the
    RK results with trapezoidal integration.

    :param x: numpy array with three elements.
    :returns: J(x), a 3x3 numpy array.
    """
    return np.array(
        [
            [-10, 10, 0],
            [28 - x[2], -1, -x[0]],
            [x[1], x[0], -8/3]
        ]
    )


def p2():
    """Problem 2.

    This function doesn't return anything, but saves hw1_p1.eps to file.
    """
    # Initial conditions.
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

    x_rk2 = rk2(x_0=x_0, f=p2_func, dt=dt, t_end=t_end)
    plot(x=x_rk2, title='Problem 2, Runge-Kutte', dt=dt, t_end=t_end,
         out_file='hw_p2_rk2.eps')

    # Do trapezoidal integration to confirm.
    x_trap = implicit_trapezoidal_integration(x_0=x_0, f=p2_func,
                                              j=p2_jac, dt=dt, t_end=t_end)
    plot(x=x_trap, title="Problem 2, Implicit Trapezoidal via Newton's method",
         dt=dt, t_end=t_end, out_file='hw_p2_trap.eps')

    # Confirm with scipy.
    result_scipy = solve_ivp(fun=p2_func_wrapper, t_span=(0, t_end),
                             y0=x_0, max_step=dt)
    plot(x=result_scipy['y'].T,
         title="Problem 2, Using Scipy.", dt=dt, t_end=t_end,
         out_file='hw2_p1_scipy.eps')


def plot(x, title, dt, t_end, out_file):
    """Hacky helper for plotting."""
    plt.figure()
    plt.plot(x)
    plt.title('{}'.format(title) + '\n'
              + r"$\Delta t = {}$, End Time = {} seconds".format(dt, t_end))
    plt.xlabel('Time steps')
    plt.ylabel('$F(t, x)$')
    plt.legend(['x' + str(x) for x in range(1, x.shape[1] + 1)])
    plt.savefig(out_file)


# noinspection PyPep8Naming
def p3():
    """Problem 3."""
    # Given parameters/functions.

    # L prime (H/mi).
    l_p = 2.18 * 10**-3

    # C prime (F/mi).
    c_p = 0.0136 * 10**-6

    # Line length (mi).
    d = 225

    # Sending end perfect voltage source.
    def get_v_k(t):
        """Return sending end voltage in Volts, given t."""
        return 188000*np.cos(2 * np.pi * 60 * t)

    # Derived parameters.

    # velocity of propagation (mi/s)
    v_p = 1 / (l_p * c_p) ** 0.5

    # tao (time for wave to reach end of line, seconds)
    tao = d / v_p

    # Use time step of tao / 6
    ts = 6
    dt = tao / ts

    # Characteristic impedance (Ohms)
    z_c = (l_p / c_p) ** 0.5

    # noinspection PyPep8Naming
    def get_I_k(idx, i_m, v_m):
        try:
            # TODO: Is this idx - ts bit correct?
            return i_m[idx - ts] - (1 / z_c) * v_m[idx - ts]
        except IndexError:
            return 0

    # noinspection PyPep8Naming
    def get_I_m(idx, i_k, v_k):
        try:
            # TODO: Is this idx - ts bit correct?
            return i_k[idx - ts] + (1 / z_c) * v_k[idx - ts]
        except IndexError:
            return 0

    # noinspection PyPep8Naming
    def get_i_k(v_k, I_k):
        """
        This incorporates our 5 Ohm switch.
        i_k * 5 + i_(z_c) * z_c = v_k
        i_k - i_(z_c) = I_k

        :param v_k:
        :param I_k:
        :return:
        """
        # Define the matrix for our system of equations.
        a = np.array(
            [
                [5, z_c],
                [1, -1]
            ]
        )

        # Define the vector for our system of equations.
        b = np.array(
            [v_k, I_k]
        )

        result = np.linalg.solve(a, b)

        # In the way this is formulated, i_k is the first element of
        # the result.
        return result[0]

    # noinspection PyPep8Naming
    def get_v_m(I_m):
        """
        Combine Z_c and R_load in parallel to get Z_total. Then we
        can easily compute v_m.
        :param I_m:
        :return:
        """
        # This could be done outside the function...
        z_tot = z_c * 300 / (z_c + 300)
        return I_m * z_tot

    def get_i_m(v_m):
        """If we have v_m, i_m is computed via Ohm's law."""
        return v_m / 300

    # Get an array of our time steps for convenience.
    t_steps = np.arange(0, 0.04 + dt, dt)

    # Initialize arrays to hold our values.
    v_k_arr = np.zeros_like(t_steps)
    I_k_arr = np.zeros_like(t_steps)
    I_m_arr = np.zeros_like(t_steps)
    i_k_arr = np.zeros_like(t_steps)
    i_m_arr = np.zeros_like(t_steps)
    v_m_arr = np.zeros_like(t_steps)

    for i, t in enumerate(t_steps):
        # Start by computing v_k
        v_k_arr[i] = get_v_k(t)

        # Compute I_k and I_m
        I_k_arr[i] = get_I_k(i, i_m_arr, v_m_arr)
        I_m_arr[i] = get_I_m(i, i_k_arr, v_k_arr)

        # Compute sending current.
        i_k_arr[i] = get_i_k(v_k_arr[i], I_k_arr[i])

        # Compute receiving voltage.
        v_m_arr[i] = get_v_m(I_m_arr[i])

        # Compute receiving current.
        i_m_arr[i] = get_i_m(v_m_arr[i])

    plt.plot(t_steps, v_k_arr, t_steps, v_m_arr)
    plt.title('Problem 3, Voltages')
    plt.xlabel('Time')
    plt.ylabel('Voltage (Volts)')
    plt.legend(['Sending', 'Receiving'])

    plt.figure()
    plt.plot(t_steps, i_k_arr, t_steps, i_m_arr)
    plt.title('Problem 3, Currents')
    plt.xlabel('Time')
    plt.ylabel('Current (Amps)')
    plt.legend(['Sending', 'Receiving'])


def main():
    """Run methods to solve problems 1 and 2."""
    # # Problem 1:
    # p1()
    # # Problem 2:
    # p2()
    # Problem 3:
    p3()

    # Show plots.
    plt.show()


if __name__ == '__main__':
    main()
