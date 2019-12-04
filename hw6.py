import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from hw1 import rk2


def main():
    # p1()
    p2()
    # p6()


def p1():
    # Do part 1.
    print('*' * 80)
    print('Problem 8.8, Part 1')
    a1 = np.array([
        [3, 8],
        [2, 3]
    ])

    _get_participation(a1)

    # Now part 2.
    print('*' * 80)
    print('Problem 8.8, Part 2')
    a2 = np.array([
        [1, 2, 1],
        [0, 3, 1],
        [0, 5, -1]
    ])

    _get_participation(a2)


def _get_participation(a1):
    # Get right eigenvectors.
    lambda1, v1 = np.linalg.eig(a1)

    # Get left eigenvectors.
    lambda_left, w1 = np.linalg.eig(a1.T)

    # Sort so that our eigenvectors line up.
    sort_1 = np.argsort(lambda1)
    sort_2 = np.argsort(lambda_left)

    # Check.
    np.testing.assert_allclose(lambda1[sort_1],
                               lambda_left[sort_2])

    print(f'Eigenvalues: {lambda1[sort_1]}')

    v1 = v1[:, sort_1]
    w1 = w1[:, sort_2]

    # Scale left eigenvectors so that w_i^t * v_i = 1.
    for idx in range(w1.shape[0]):
        w_i = w1[:, idx]
        v_i = v1[:, idx]
        p = np.matmul(w_i, v_i)
        if p == 0:
            continue
        c = 1 / np.matmul(w_i, v_i)
        w1[:, idx] = w1[:, idx] * c

    # Check.
    # Commenting this out since it doesn't work well with values very
    # near zero (e.g. 1e-17).
    # np.testing.assert_allclose(np.matmul(w1.T, v1), np.identity(a1.shape[0]))

    # The participation factors are simple elementwise multiplication.
    p_1 = v1 * w1
    print(f'Participation Factors:\n{p_1}')


def p2():
    # Given parameters
    m = 0.0133
    p_m = 0.91
    p_e = 3.24

    # Compute delta^s
    d_s = np.arcsin(p_m / p_e)

    # Compute V_cr
    v_cr = -p_m * (np.pi - 2 * d_s) + 2 * p_e * np.cos(d_s)

    # Initialize variables.
    t = 0
    dt = 0.005
    delta = d_s
    w = 0

    # Function for computing w(t)
    def w_t():
        # Consider w_0 to be 0, since we're in the "delta w" frame.
        return p_m * t / m

    # Function for computing delta(t)
    def d_t():
        # Again, consider w_0 to be 0.
        return 0.5 * p_m * t**2 / m + d_s

    # Energy function.
    def v():
        return 0.5 * m * w**2 - p_m * (delta - d_s) - \
                p_e * (np.cos(delta) - np.cos(d_s))

    # Compute initial v
    v_t = v()
    v_list = [v_t]
    i = 0
    while v_t <= v_cr and i < 1000:
        t += dt
        # Compute delta and omega.
        delta = d_t()
        w = w_t()

        # Compute energy.
        v_t = v()
        v_list.append(v_t)

        i += 1

    if i >= 100:
        raise UserWarning('Maxed iterations.')

    print(f't_cr: {t:.3f}')


def p6():
    # Phase angles of vstab and speed
    vstab = -30.925
    speed = -45.306
    phi_deg = vstab + 360 - speed
    print(f'phi_deg: {phi_deg:.3f}')
    # Convert to radians, subtract 180 degrees, divide by 2.
    phi = (phi_deg - 180) / 2 * np.pi / 180

    # Frequency of our mode
    f = 1.67

    # Compute alpha
    alpha = (1 - np.sin(phi)) / (1 + np.sin(phi))
    print(f'alpha: {alpha:.3f}')

    # Now compute t1 and t2.
    t1 = 1 / (2 * np.pi * f * np.sqrt(alpha))
    t2 = alpha * t1
    print(f't1: {t1:.3f}')
    print(f't2: {t2:.3f}')
    pass


if __name__ == '__main__':
    main()
