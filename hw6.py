import cmath
import numpy as np


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
    X_p_p_d = 0.18
    X_p_p_q = 0.18
    X_1 = 0.12
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
    dq_mat = np.array(
        [
            [np.sin(delta), -np.cos(delta)],
            [np.cos(delta), np.sin(delta)]
        ]
    )

    V_ri = np.array([[V_t.real], [V_t.imag]])
    V_dq = np.matmul(dq_mat, V_ri).flatten()

    I_ri = np.array([[I_initial.real], [I_initial.imag]])
    I_dq = np.matmul(dq_mat, I_ri).flatten()

    print(f'[V_d, V_q]: {V_dq}')
    print(f'[I_d, I_q]: {I_dq}')

    # Compute E''
    E_pp = V_t + (0 + 1j * X_p_p_d) * I_initial

    print(f"E'': {E_pp:.2f}")

    E_ri = np.array([[E_pp.real], [E_pp.imag]])
    psi_pp = np.matmul(dq_mat, E_ri).flatten()

    print(f"[-Psi''_q, Psi''_d]: {psi_pp}")

    # Flip the sign on the initial element of psi_pp
    psi_pp[0] *= 1


    pass

if __name__ == '__main__':
    p1()