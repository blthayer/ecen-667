import numpy as np


def main():
    p1()


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


if __name__ == '__main__':
    main()
