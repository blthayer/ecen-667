"""Module for homework 6. This is some of the crappiest code I've
written in recent memory. Oh well.
"""
import cmath
import numpy as np
from hw1 import implicit_trapezoidal_integration, newton
import pandas as pd


def get_dq_matrix(delta):
    """Given delta in radians, get the dq matrix which transforms from
    the system reference frame to the machine dq reference frame.

    [V_d, V_q]^t = dq_matrix * [V_r, V_i]^t
    """
    return np.array(
        [
            [np.sin(delta), -np.cos(delta)],
            [np.cos(delta), np.sin(delta)]
        ]
    )


def get_dq(delta, n):
    dq_mat = get_dq_matrix(delta)

    ri = np.array([[n.real], [n.imag]])
    return np.matmul(dq_mat, ri).flatten()


# noinspection PyPep8Naming
def p1():
    # Line impedance on 100 MVA base.
    line_imp_100 = 1j*0.1

    # Generator real power.
    p = 300

    # Bus voltage
    inf_v = 1

    # GENSAL params (400 MVA base)
    H = 5
    D = 0
    R_s = 0
    X_d = 2.1
    X_q = 1.5
    X_p_d = 0.3
    # X''_d = X''_q
    X_p_p = 0.18
    X_l = 0.12
    T_p_do = 7
    T_pp_do = 0.035
    T_pp_qo = 0.05

    # Convert line impedance to 400 MVA base from 100 MVA base
    line_imp = line_imp_100 * 400 / 100
    # Combine impedances in parallel.
    line_imp_total = (line_imp**2 / (2 * line_imp))
    print(f'Total line impedance: {line_imp_total:.2f}')

    # Convert power to per unit
    p_pu = 300 / 400

    # Compute terminal conditions.
    # Current at infinite bus:
    I_initial = p_pu / inf_v + 1j*0
    print(f'Initial current: {I_initial}')

    # Compute terminal voltage.
    V_t = inf_v + I_initial * line_imp_total
    print(f'Initial terminal voltage: {cmath.polar(V_t)}')

    # Compute E in order to get delta
    E_initial = V_t + 1j * X_q * I_initial
    delta = cmath.phase(E_initial)
    print(f'Initial |E|: {abs(E_initial)}')
    print(f'Initial delta: {delta:.2f} radians, '
          f'{delta * 180 / cmath.pi:.2f} degrees.')

    # Convert to DQ
    V_dq = get_dq(delta=delta, n=V_t)
    I_dq = get_dq(delta=delta, n=I_initial)

    print(f'[V_d, V_q]: {V_dq}')
    print(f'[I_d, I_q]: {I_dq}')

    # Compute E''
    E_pp = V_t + (0 + 1j * X_p_p) * I_initial
    print(f"E'': {E_pp:.2f}")

    # Compute Psi''
    psi_pp = get_dq(delta=delta, n=E_pp)
    print(f"[-Psi''_q, Psi''_d]: {psi_pp}")

    # Flip the sign on the initial element of psi_pp
    psi_pp[0] *= 1

    # Time to solve the differential equations in the steady state
    # condition.
    I_d = I_dq[0]
    psi_pp_q = psi_pp[0]
    psi_p_p_d = psi_pp[1]
    a = (X_p_p - X_l) / (X_p_d - X_l)
    b = (X_p_d - X_p_p) / (X_p_d - X_l)

    mat = np.array([
        [1, -1, 0],
        [0, a, b],
        [0, 1, -1]
    ])

    vec = np.array([
        [I_d * (X_d - X_p_d)],
        [psi_p_p_d],
        [I_d * (X_p_d - X_l)]
    ])

    result = np.linalg.solve(mat, vec).flatten()

    E_fd = result[0]
    E_p_q = result[1]
    psi_p_d = result[2]

    # At this point, we have all our initial conditions complete.
    print('Initialization complete.')
    print('*' * 80)
    print('Fault time.')

    # Compute Thevenin voltage for faulted system.
    x_line = line_imp.imag
    x_line2 = x_line / 2
    v_thev = x_line2 / (x_line + x_line2) * (1 + 1j * 0)

    # Compute Thevenin impedance for faulted system.
    x_thev = x_line * x_line2 / (x_line + x_line2)

    print(f'Thev V: {abs(v_thev):.2f}')
    print(f'Thev X: {x_thev:.2f}')

    # Compute the line current.
    I_fault = (E_pp - v_thev) / (X_p_p + x_thev)

    print(f'Initial fault current: {I_fault:.2f}')

    # Compute terminal voltage.
    V_t = E_pp - I_fault * X_p_p

    print(f'Terminal voltage immediately following fault: {V_t:.2f}')

    # Convert to DQ
    V_dq = get_dq(delta=delta, n=V_t)
    V_d = V_dq[0]
    V_q = V_dq[1]
    I_dq = get_dq(delta=delta, n=I_fault)
    I_d = I_dq[0]
    I_q = I_dq[1]

    # Speed.
    w = 2 * cmath.pi * 60
    w_s = 2 * cmath.pi * 60

    # Create function for computing the derivatives.
    def der_f():

        # Compute some terms to make the long diff'eq shorter.
        term1 = (X_p_d - X_p_p) / (X_p_d - X_l) ** 2
        term2 = (-psi_p_d - I_d * (X_p_d - X_l) + E_p_q)
        term3 = (X_d - X_p_d)

        return np.array([
            (-psi_pp_q - I_d * (X_q - X_p_p)) / T_pp_qo,
            (-psi_p_d - I_d * (X_p_d - X_l) + E_p_q) / T_pp_do,
            -(((term1 * term2 + I_d) * term3 + E_p_q) + E_fd) / T_p_do,
            w - w_s,
            p_pu / (2*H) - (w_s / (2 * H)) * (psi_p_p_d * I_q - psi_pp_q * I_d)
        ])

    # Do Euler's with the differential equations for two time steps.
    for _ in range(2):
        x = np.array([psi_pp_q, psi_p_d, E_p_q, delta, w])
        f_x = der_f()
        x_t_p_1 = x + 0.01 * f_x

        psi_pp_q = x_t_p_1[0]
        psi_p_d = x_t_p_1[1]
        E_p_q = x_t_p_1[2]
        delta = x_t_p_1[3]
        w = x_t_p_1[4]

    x = np.array([psi_pp_q, psi_p_d, E_p_q, delta, w])
    print(f'States after two steps of fault: {x}')
    pass


# noinspection PyPep8Naming
def p2():
    # start with current. Which in this case is also equal to the
    # active power (and thus mechanical power)
    i = 1.2

    # Solve for machine voltage and angle
    e = 1 + i * (1j*0.04 + 1j*0.05)
    E = abs(e)
    delta = cmath.phase(e)

    print(f'|E|: {E:.3f}, delta: {delta:.3f}')

    omega = 2 * cmath.pi * 60

    H = 5

    x_0 = np.array([delta, omega])

    def f(x):
        return np.array(
            [
                x[1] - omega,
                (i / omega - E / (1j*0.04) * np.sin(x[0])) * (omega / (2 * H))
            ]
        )

    def j(x):
        return np.array([
            [0, 1],
            [-E/(1j*0.04) * np.cos(x[0]) * omega / (2 * H), 0]
        ])

    results = implicit_trapezoidal_integration(x_0=x_0, f=f, j=j,
                                               dt=0.02, t_start=0,
                                               t_end=0.05)

    print(results)


def p3():
    # Line impedance
    line_imp = 1j * 0.15

    # Generator inertias (given)
    h1 = 3
    h2 = 6

    # Synchronous speed. Why divide by 100? Not sure.
    ws = 2 * cmath.pi * 60 / 100

    # Given terminal voltages from power flow.
    v1 = 1.07 * np.exp(1j*8.6*cmath.pi/180)
    v2 = 1 + 1j*0

    # Get real and imaginary components.
    v_d1 = v1.real
    v_q1 = v1.imag
    v_d2 = v2.real
    v_q2 = v2.imag

    # Solve for current across line.
    i = (v1 - v2) / (1j * line_imp)

    # Solve for machine voltage 1.
    e1 = v1 + i * 1j * 0.08
    e_p_1 = abs(e1)
    d1 = cmath.phase(e1)
    e_d1 = e1.real
    e_q1 = e1.imag

    # Solve for machine voltage 2.
    e2 = v2 - i * 1j * 0.02
    e_p_2 = abs(e2)
    d2 = cmath.phase(e2)
    e_d2 = e2.real
    e_q2 = e2.imag

    b1 = (1 / (1j * 0.08)).imag
    b2 = (1 / (1j * 0.02)).imag

    # Y-bus
    line_y = 1 / line_imp
    y11 = (line_y + 1j*b1).imag
    y12 = (-line_y).imag
    y21 = (-line_y).imag
    y22 = (line_y + 1j*b2).imag

    print(f'B1: {b1:.3f}')
    print(f"E'_1: {e_p_1:.3f}")
    print(f'delta 1: {d1:.3f}')
    print(f'E_D1: {e_d1:.3f}')
    print(f'E_Q1: {e_q1:.3f}')
    print('*' * 80)
    print(f'B2: {b2:.3f}')
    print(f"E'_2: {e_p_2:.3f}")
    print(f'delta 2: {d2:.3f}')
    print(f'E_D2: {e_d2:.3f}')
    print(f'E_Q2: {e_q2:.3f}')

    rows = ['d1', 'w1', 'd2', 'w2', 'id1', 'iq1',
            'id2', 'iq2']
    cols = ['d1', 'w1', 'd2', 'w2', 'vd1', 'vq1',
            'vd2', 'vq2']
    jac = pd.DataFrame(np.zeros((8, 8)), index=rows,
                       columns=cols)

    # Slide 21 (lecture 16)
    jac.loc['id1', 'd1'] = e_p_1 * np.cos(d1) * b1
    jac.loc['id2', 'd2'] = e_p_2 * np.cos(d2) * b2

    jac.loc['iq1', 'd1'] = e_p_1 * np.sin(d1) * b1
    jac.loc['iq2', 'd2'] = e_p_2 * np.sin(d2) * b2

    # Slide 22
    jac.loc['d1', 'w1'] = ws
    jac.loc['d2', 'w2'] = ws

    jac.loc['w1', 'vd1'] = 1 / (2 * h1) * e_q1 * b1
    jac.loc['w2', 'vd2'] = 1 / (2 * h2) * e_q2 * b2

    jac.loc['w1', 'vq1'] = 1 / (2 * h1) * -1 * e_d1 * b1
    jac.loc['w2', 'vq2'] = 1 / (2 * h2) * -1 * e_d2 * b2

    # Fill in first 4 diagonals with -1.
    jac.loc['d1', 'd1'] = -1
    jac.loc['w1', 'w1'] = -1
    jac.loc['d2', 'd2'] = -1
    jac.loc['w2', 'w2'] = -1

    # I/V w.r.t V/I
    jac.loc['id1', 'vq1'] = -y11
    jac.loc['iq1', 'vd1'] = y11

    jac.loc['iq2', 'vd1'] = y12
    jac.loc['iq1', 'vd2'] = y12

    jac.loc['id2', 'vq1'] = -y12
    jac.loc['id1', 'vq2'] = -y12

    jac.loc['id2', 'vq2'] = y22
    jac.loc['iq2', 'vd2'] = y22

    pass


def p4():
    # Given:
    r_s = 0.01
    x_s = 0.06
    x_m = 4
    r_r = 0.02
    x_r = 0.03
    w_s = cmath.pi * 2 * 60

    # Power: 100 MW on 125 MVA base.
    p_e = 100 / 125
    # Voltage: given, 0.995 angle 0
    v_d = 0.995
    v_q = 0

    # Solve for intermediate params
    x_p = x_s + (x_r * x_m) / (x_r + x_m)
    x = x_s + x_m
    t_p_0 = (x_r + x_m) / (w_s * r_r)

    # Build our vector of states (have to make initial guesses):
    # s, I_D, I_Q, E'_D, E'_Q
    s_idx = 0
    i_d_idx = 1
    i_q_idx = 2
    e_p_d_idx = 3
    e_p_q_idx = 4
    # Use some values from lecture for our guess
    x_0 = np.array([0.01, 1, 1, 1, 1])

    # # Let's get a better guess for E'_D and E'_Q by using the
    # # equations.
    # A = np.array([
    #     [-1/t_p_0, w_s * x_0[s_idx]],
    #     [-w_s * x_0[s_idx], -1/t_p_0]
    # ]
    # )
    # b = np.array([
    #     [(x - x_p) * x_0[i_q_idx]],
    #     [-(x - x_p) * x_0[i_d_idx]]
    # ])
    #
    # e_arr = np.linalg.solve(A, b)
    #
    # x_0[e_p_d_idx] = e_arr[0][0]
    # x_0[e_p_q_idx] = e_arr[1][0]

    # Build function to evaluate f(x)
    def f(x_arr):
        # Order: P_E eq, V_D eq, V_Q eq, dE'_D eq, dE'_Q eq
        return np.array(
            [
                -p_e + v_d * x_arr[i_d_idx] + v_q * x_arr[i_q_idx],

                -v_d + x_arr[e_p_d_idx] + r_s * x_arr[i_d_idx]
                - x_p * x_arr[i_q_idx],

                -v_q + x_arr[e_p_q_idx] + r_s * x_arr[i_q_idx]
                + x_p * x_arr[i_d_idx],

                w_s * x_arr[s_idx] * x_arr[e_p_q_idx] - (1 / t_p_0)
                * (x_arr[e_p_d_idx] + (x - x_p) * x_arr[i_q_idx]),

                -w_s * x_arr[s_idx] * x_arr[e_p_d_idx] - (1 / t_p_0)
                * (x_arr[e_p_q_idx] - (x - x_p) * x_arr[i_d_idx])
            ]
        )

    # Build function to evaluate the Jacobian of f.
    def j(x_arr):
        # Row order: P_E eq, V_D eq, V_Q eq, dE'_D eq, dE'_Q eq
        # Column order: s, I_D, I_Q, E'_D, E'_Q
        return np.array([
            [0, v_d, v_q, 0, 0],
            [0, r_s, -x_p, 1, 0],
            [0, x_p, r_s, 0, 1],
            [w_s * x_arr[e_p_q_idx], 0, -(x - x_p) / t_p_0, -1 / t_p_0,
             w_s * x_arr[s_idx]],
            [-w_s * x_arr[e_p_d_idx], (x - x_p) / t_p_0, 0,
             -w_s * x_arr[s_idx], -1 / t_p_0]
        ]
        )

    result = newton(x_0=x_0, f=f, j=j)

    print(result)


def p5():
    # Given:
    r_s = 0.01
    x_s = 0.06
    x_m = 4
    r_r = 0.03  # From lecture, not problem 4
    x_r = 0.04  # From lecture, not problem 4
    w_s = cmath.pi * 2 * 60

    # Voltage: given, 0.995 angle 0
    v_d = 0.995
    v_q = 0

    # Solve for intermediate params
    x_p = x_s + (x_r * x_m) / (x_r + x_m)

    z_in = ((r_s + 1j*x_s)
            + (1j * x_m * (r_r + 1j*x_r)) / (r_r + 1j * (x_r + x_m)))

    i = (v_d + 1j * v_q) / z_in
    i_d = i.real
    i_q = i.imag

    term1 = (v_d - r_s * i_d + x_p * i_q) * i_d
    term2 = (v_q - r_s * i_q - x_p * i_d) * i_q

    torque = (term1 + term2) / w_s
    print(f'Starting torque: {torque:.5f}')


if __name__ == '__main__':
    p1()
    # p2()
    # p3()
    # p4()
    # p5()
