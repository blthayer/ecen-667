"""Module for homework 2.
"""
import numpy as np
import cmath
from utils import b_matrix


def p1():
    """Problem 1"""
    ####################################################################
    # Phase conductors, hence the "_p" notation
    ####################################################################
    # Geometric mean radius (GMR) (ft)
    gmr_p = 0.0523

    # Resistance per distance (Ohms/mile)
    r_p = 0.0612

    ####################################################################
    # Neutral conductors, hence the "_n" notation
    ####################################################################
    # GMR (ft)
    gmr_n = 0.0217

    # Resistance per distance (Ohms/mile)
    r_n = 0.35

    ####################################################################
    # Define conductor positions
    ####################################################################
    # Use Kersting's trick of specifying each conductor in Cartesian
    # coordinates using complex number notation. The origin will be at
    # ground directly below phase a.
    # c for coordinates
    coord_a = 0 + 1j*40
    coord_b = 12 + 1j*40
    coord_c = 24 + 1j*40
    coord_g1 = 5 + 1j*55
    coord_g2 = 19 + 1j*55

    # Create a matrix with differences between conductors.
    coordinate_array = np.array([coord_a, coord_b, coord_c, coord_g1,
                                 coord_g2])

    z_abc = get_phase_impedance(gmr_phase=gmr_p, gmr_neutral=gmr_n,
                                resistance_phase=r_p, resistance_neutral=r_n,
                                n_phase_conductors=3,
                                coordinate_array=coordinate_array,
                                rho=120)

    z_012 = phase_to_sequence(z_abc)

    print('Z_abc for Problem 1:')
    print(b_matrix(z_abc))

    print('Z_012 for Problem 1:')
    print(b_matrix(z_012))


def example_4_1():
    """Example 4.1 from Distribution System Modeling and Analysis,
    Third Edition by William H. Kersting. Used to verify code is
    working properly.
    """
    z_abc = get_phase_impedance(
        gmr_phase=0.0244, gmr_neutral=0.00814, resistance_phase=0.306,
        resistance_neutral=0.5920, n_phase_conductors=3,
        coordinate_array=np.array([0+1j*29, 2.5+1j*29, 7+1j*29, 4+1j*25]),
        rho=100
    )

    print("Z_abc for Example 4.1 From Kersting's book:")
    print(b_matrix(z_abc))


def get_phase_impedance(gmr_phase, gmr_neutral, resistance_phase,
                        resistance_neutral, n_phase_conductors,
                        coordinate_array, freq=60, rho=100):
    """Compute the phase impedance matrix for an overhead line. To keep
    things simple, it is assumed all phase conductors and all neutral
    conductors are of the same type. The ordering of the
    coordinate_array matters, so check the documentation.

    :param gmr_phase: Geometric mean radius (GMR) of the phase
        conductors (ft.).
    :param gmr_neutral: GMR of neutral conductors (ft.).
    :param resistance_phase: Resistance of the phase conductors in
        Ohms/mile.
    :param resistance_neutral: Resistance of the neutral conductors in
        Ohms/mile
    :param n_phase_conductors: Number of phase conductors. E.g. 3 for a
        "single-circuit" three phase line.
    :param coordinate_array: Numpy ndarray defining the coordinates of
        each conductor in the complex plane. The origin should be at
        the ground level and directly below the left-most phase
        conductor.
    :param freq: System frequency, defaults to 60 Hz.
    :param rho: Earth resistivity (Ohm * m). Defaults to 100 Ohm*m.
    """

    ####################################################################
    # Create distance matrix.
    ####################################################################
    # Create a matrix with differences between conductors.
    n_cond = len(coordinate_array)
    distance_mat = np.zeros((n_cond, n_cond))

    # Just use a crappy double for-loop. No need to over-optimize.
    # No, this is not the most efficient data-type either.
    for row in range(n_cond):
        for col in range(n_cond):
            if row != col:
                # Take the absolute difference between the positions.
                distance_mat[row, col] = \
                    abs(coordinate_array[row] - coordinate_array[col])
            else:
                # Fill in diagonal with the appropriate GMR.
                if row < n_phase_conductors:
                    distance_mat[row, row] = gmr_phase
                else:
                    distance_mat[row, row] = gmr_neutral

    # Fill in the diagonals with GMRs.
    for idx in range(n_phase_conductors):
        distance_mat[idx, idx] = gmr_phase

    for idx in range(n_phase_conductors, n_cond):
        distance_mat[idx, idx] = gmr_neutral

    ####################################################################
    # Constants for modified Carson equations
    ####################################################################

    # Constants which I'm too lazy too look up meanings/dimensions:
    real_constant = 0.00158836 * freq
    imag_constant = 1j * 0.00202237 * freq
    rf_constant = 7.6786 + 0.5 * np.log(rho / freq)

    ####################################################################
    # Functions for modified Carson equations
    ####################################################################

    def carson_self(r, gmr):
        """Compute the self-impedance of a conductor in Ohms/mile

        :param r: Resistance of conductor in Ohms/mile
        :param gmr: Geometric mean radius of conductor in feet.

        :returns: Self-impedance in Ohms/mile
        """
        return (r + real_constant
                + imag_constant * (np.log(1 / gmr) + rf_constant))

    def carson_mutual(d_ij):
        """Compute mutual impedance between conductors in Ohms/mile.

        :param d_ij: Distance between the conductors (ft).

        :returns: Mutual impedance in Ohms/mile
        """
        return real_constant + imag_constant * (np.log(1/d_ij) + rf_constant)

    ####################################################################
    # Primitive impedance matrix
    ####################################################################
    # Initialize the primitive impedance matrix.
    z_primitive = 1j * np.zeros_like(distance_mat)

    # Sanity check
    assert z_primitive.shape[0] == n_cond
    assert z_primitive.shape[1] == n_cond

    # Use another double for loop to fill it in.
    for i in range(z_primitive.shape[0]):
        for j in range(z_primitive.shape[1]):
            # Off-diagonal terms.
            if i != j:
                # Compute the mutual impedance, which only depends on
                # the distance between conductors.
                z_primitive[i, j] = carson_mutual(distance_mat[i, j])
            else:
                # Self impedance. This depends on the resistance as
                # well as the GMR. Note that i = j in this case.
                if i < n_phase_conductors:
                    # Use phase resistance.
                    this_r = resistance_phase
                else:
                    # Use neutral resistance.
                    this_r = resistance_neutral

                # Compute the self impedance.
                z_primitive[i, j] = carson_self(this_r, distance_mat[i, j])

    ####################################################################
    # Kron reduction to get phase impedance matrix
    ####################################################################
    # Extract the phase portion of the matrix.
    z_ij = z_primitive[0:n_phase_conductors, 0:n_phase_conductors]

    # Extract phase to neutral portion.
    z_in = z_primitive[0:n_phase_conductors, n_phase_conductors:]

    # Extract the neutral to phase portion.
    z_nj = z_primitive[n_phase_conductors:, 0:n_phase_conductors]

    # Extract the neutral to neutral portion.
    z_nn = z_primitive[n_phase_conductors:, n_phase_conductors:]

    # Sanity checks
    assert z_ij.shape[0] + z_nj.shape[0] == z_primitive.shape[0]
    assert z_ij.shape[1] + z_in.shape[1] == z_primitive.shape[1]
    assert z_nj.shape[1] + z_nn.shape[1] == z_primitive.shape[1]
    assert z_in.shape[0] + z_nn.shape[0] == z_primitive.shape[0]

    # Perform Kron reduction to get the phase impedance matrix.
    return z_ij - np.matmul(np.matmul(z_in, np.linalg.inv(z_nn)), z_nj)


def phase_to_sequence(z):
    """Convert the 3x3 phase impedance matrix to the 3x3 sequence
    impedance matrix.

    :param z: 3x3 numpy array representing the phase impedance matrix.

    :returns z_012: 3x3 numpy array representing the sequence impedance
        matrix.
    """
    # In a "real" implementation we'd want to move all this junk outside
    # the function.
    a_s = cmath.rect(1, 120 * np.pi / 180)
    o = 1 + 1j*0
    a = np.array([
        [o, o, o],
        [o, a_s**2, a_s],
        [o, a_s, a_s**2]
    ])

    return np.matmul(np.matmul(np.linalg.inv(a), z), a)


if __name__ == '__main__':
    # Run problem 1.
    p1()
    # Run example.
    # example_4_1()
