"""Module for homework 2.
"""
import numpy as np

def main():
    # Define given variables.

    ####################################################################
    # Phase conductors, hence the "_p" notation
    ####################################################################
    # Geometric mean radius (GMR) (ft)
    gmr_p = 0.0523

    # Resistance per distance (Ohms/mile)
    r_p = 0.0612

    ####################################################################
    # Ground conductors, hence the "_g" notation
    ####################################################################
    # GMR (ft)
    gmr_g = 0.0217

    # Resistance per distance (Ohms/mile)
    r_g = 0.35

    ####################################################################
    # Indices for phases
    ####################################################################
    i_a = 0
    i_b = 1
    i_c = 2
    i_g1 = 3
    i_g2 = 4

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
    coord_arr = np.array([coord_a, coord_b, coord_c, coord_g1, coord_g2])
    coord_mat = np.zeros((len(coord_arr), len(coord_arr)))

    # Just use a crappy double for-loop. No need to over-optimize.
    # No, this is not the most efficient data-type either.
    for row in range(len(coord_arr)):
        for col in range(len(coord_arr)):
            coord_mat[row, col] = abs(coord_arr[row] - coord_arr[col])

    # Fill in the diagonals with GMRs.
    for idx in [i_a, i_b, i_c]:
        coord_mat[idx, idx] = gmr_p

    for idx in [i_g1, i_g2]:
        coord_mat[idx, idx] = gmr_g

    ####################################################################
    # Compute geometric mean distances
    ####################################################################
    # Phase GMD (ft.)
    gmd_p = (coord_mat[i_a, i_b] * coord_mat[i_b, i_c]
             * coord_mat[i_c, i_a]) ** (1/3)

    # Ground GMD (ft.) (note it's the same for g1 or g2 due to symmetry)
    gmd_g = (coord_mat[i_a, i_g1] * coord_mat[i_b, i_g1]
             * coord_mat[i_c, i_g1]) ** (1/3)

    ####################################################################
    # Constants for modified Carson equations
    ####################################################################
    # Frequency (Hz)
    freq = 60

    # Earth resistivity (Ohm-m)
    rho = 120

    # Constants which I'm too lazy too look up it's meaning/dimensions:
    real_constant = 0.00158836 * freq
    imag_constant = 1j * 0.00202237 * freq
    rf_constant = 7.6786 + 0.5 * np.log(rho / freq)

    ####################################################################
    # Function for modified Carson equations
    ####################################################################

    def carson_self(r, gmr):
        """Compute the self-impedance of a conductor in Ohms/mile

        :param r: Resistance of conductor in Ohms/mile
        :param gmr: Geometric mean radius of conductor in feet.

        :returns: Self-impedance in Ohms/mile
        """
        return (r + real_constant
                + imag_constant * (np.log(1 / gmr) + rf_constant))

    def carson_mutual(gmd):
        """Compute mutual impedance between conductors in Ohms/mile.

        :param gmd: Geometric mean distance between the conductors.

        :returns: Mutual impedance in Ohms/mile
        """
        return real_constant + imag_constant * (np.log(1/gmd) + rf_constant)

    ####################################################################
    # Primitive impedance matrix
    ####################################################################
    # Initialize the primitive impedance matrix.
    z_primitive = np.zeros_like(coord_mat) + 1j * np.zeros_like(coord_mat)

    # Sanity check
    assert z_primitive.shape[0] == 5
    assert z_primitive.shape[1] == 5

    # Use another double for loop to fill it in.
    for i in range(z_primitive.shape[0]):
        for j in range(z_primitive.shape[1]):
            # Off-diagonal terms.
            if i != j:
                # WARNING: HARD-CODING.
                # Assume matrix order is [a, b, c, g1, g2].
                if (i < i_g1) and (j < i_g1):
                    # Use the GMD between phases.
                    this_gmd = gmd_p
                else:
                    # Use the GMD from phase to ground/neutral.
                    this_gmd = gmd_g

                # Fill it in.
                z_primitive[i, j] = carson_mutual(this_gmd)
            else:
                # Diagonal terms.
                # WARNING: HARD-CODING.
                # Assume matrix order is [a, b, c, g1, g2].
                if (i < i_g1) and (j < i_g1):
                    this_gmr = gmr_p
                    this_r = r_p
                else:
                    this_gmr = gmr_g
                    this_r = r_g

                z_primitive[i, j] = carson_self(this_r, this_gmr)

    pass


if __name__ == '__main__':
    main()
