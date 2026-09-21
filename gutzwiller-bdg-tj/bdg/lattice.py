import numpy as np

# ---------------------------------------------------------------- lattice --

def square_lattice(Lx, Ly):
    N = Lx * Ly
    idx = np.arange(N).reshape(Ly, Lx)

    nn_bonds = []
    nnn_bonds = []
    for y in range(Ly):
        for x in range(Lx):
            i = idx[y, x]
            j = idx[y, (x + 1) % Lx]
            nn_bonds.append((i, j, 'x'))
            j = idx[(y + 1) % Ly, x]
            nn_bonds.append((i, j, 'y'))
            j = idx[(y + 1) % Ly, (x + 1) % Lx]
            nnn_bonds.append((i, j))
            j = idx[(y - 1) % Ly, (x + 1) % Lx]
            nnn_bonds.append((i, j))
    return N, nn_bonds, nnn_bonds


# ---------------------------------------------------- Gutzwiller factors --
# Scalar versions -- used by solve_homogeneous.py.

def g_t(delta, m, sigma):
    delta = np.clip(delta, 1e-6, 1 - 1e-6)
    num = 2 * delta * (1 - delta)
    den = 1 - delta**2 + 4 * m**2
    den = np.where(den < 1e-8, 1e-8, den)
    a = (1 + delta + sigma * 2 * m) / np.clip(1 + delta - sigma * 2 * m, 1e-8, None)
    return np.sqrt(np.clip(num / den, 0, None) * np.clip(a, 0, None))


def g_sxy(delta, m):
    delta = np.clip(delta, 1e-6, 1 - 1e-6)
    den = 1 - delta**2 + 4 * m**2
    den = np.where(den < 1e-8, 1e-8, den)
    return 2 * (1 - delta) / den


def g_sz_bond(delta_i, delta_j, m_i, m_j, Delta_ij, chi_ij, gsxy_i, gsxy_j):
    gsxy_ij = gsxy_i * gsxy_j
    d2c2 = Delta_ij**2 + chi_ij**2
    denom_i = (1 - delta_i**2 + 4 * m_i**2)
    denom_j = (1 - delta_j**2 + 4 * m_j**2)
    denom_i = denom_i if denom_i > 1e-8 else 1e-8
    denom_j = denom_j if denom_j > 1e-8 else 1e-8
    X = 1 + 12 * (1 - delta_i) * (1 - delta_j) * d2c2 / np.sqrt(denom_i * denom_j)
    num = 2 * d2c2 - 4 * m_i * m_j * X**2
    den = 2 * d2c2 - 4 * m_i * m_j
    if abs(den) < 1e-10:
        den = 1e-10 * np.sign(den) if den != 0 else 1e-10
    return gsxy_ij * num / den


def g_sz_bond_and_derivs(delta_i, delta_j, m_i, m_j, Delta_ij, chi_ij, gsxy_i, gsxy_j):
    gsxy_ij = gsxy_i * gsxy_j
    D = Delta_ij**2 + chi_ij**2
    A = 4 * m_i * m_j
    denom_i = max(1 - delta_i**2 + 4 * m_i**2, 1e-8)
    denom_j = max(1 - delta_j**2 + 4 * m_j**2, 1e-8)
    c = 12 * (1 - delta_i) * (1 - delta_j) / np.sqrt(denom_i * denom_j)
    X = 1 + c * D
    den = 2 * D - A
    if abs(den) < 1e-10:
        den = 1e-10 * np.sign(den) if den != 0 else 1e-10
    f = (2 * D - A * X**2) / den
    gsz = gsxy_ij * f
    dnum_dD = 2 - A * 2 * X * c
    df_dD = (dnum_dD * den - (2 * D - A * X**2) * 2) / den**2
    dgsz_dDelta = gsxy_ij * df_dD * 2 * Delta_ij
    dgsz_dchi = gsxy_ij * df_dD * 2 * chi_ij
    return gsz, dgsz_dDelta, dgsz_dchi


def g_t_dn(delta_i, m_i, sigma_prime, sigma_field, eps=1e-6):
    ddelta, dm = -eps, sigma_field / 2.0 * eps
    gp = g_t(delta_i + ddelta, m_i + dm, sigma_prime)
    gm = g_t(delta_i - ddelta, m_i - dm, sigma_prime)
    return (gp - gm) / (2 * eps)


def g_sxy_dn(delta_i, m_i, sigma_field, eps=1e-6):
    ddelta, dm = -eps, sigma_field / 2.0 * eps
    gp = g_sxy(delta_i + ddelta, m_i + dm)
    gm = g_sxy(delta_i - ddelta, m_i - dm)
    return (gp - gm) / (2 * eps)


def g_sz_bond_dn_i(delta_i, delta_j, m_i, m_j, Delta_ij, chi_ij, sigma_field, eps=1e-6):
    ddelta, dm = -eps, sigma_field / 2.0 * eps
    gsxy_j = g_sxy(delta_j, m_j)
    gsxy_i_p = g_sxy(delta_i + ddelta, m_i + dm)
    gp, _, _ = g_sz_bond_and_derivs(delta_i + ddelta, delta_j, m_i + dm, m_j,
                                     Delta_ij, chi_ij, gsxy_i_p, gsxy_j)
    gsxy_i_m = g_sxy(delta_i - ddelta, m_i - dm)
    gm, _, _ = g_sz_bond_and_derivs(delta_i - ddelta, delta_j, m_i - dm, m_j,
                                     Delta_ij, chi_ij, gsxy_i_m, gsxy_j)
    return (gp - gm) / (2 * eps)


# ------------------------------------------------------- 
# Used by solve_disordered.py

def g_sz_bond_vec(delta_i, delta_j, m_i, m_j, Delta_ij, chi_ij, gsxy_i, gsxy_j):
    gsxy_ij = gsxy_i * gsxy_j
    D = Delta_ij**2 + chi_ij**2
    A = 4 * m_i * m_j
    denom_i = np.maximum(1 - delta_i**2 + 4*m_i**2, 1e-8)
    denom_j = np.maximum(1 - delta_j**2 + 4*m_j**2, 1e-8)
    X = 1 + 12*(1-delta_i)*(1-delta_j)*D/np.sqrt(denom_i*denom_j)
    num = 2*D - A*X**2
    den = 2*D - A
    den = np.where(np.abs(den) < 1e-10, np.where(den == 0, 1e-10, 1e-10*np.sign(den)), den)
    return gsxy_ij * num / den


def g_sz_bond_and_derivs_vec(delta_i, delta_j, m_i, m_j, Delta_ij, chi_ij, gsxy_i, gsxy_j):
    gsxy_ij = gsxy_i * gsxy_j
    D = Delta_ij**2 + chi_ij**2
    A = 4 * m_i * m_j
    denom_i = np.maximum(1 - delta_i**2 + 4*m_i**2, 1e-8)
    denom_j = np.maximum(1 - delta_j**2 + 4*m_j**2, 1e-8)
    c = 12*(1-delta_i)*(1-delta_j)/np.sqrt(denom_i*denom_j)
    X = 1 + c*D
    den = 2*D - A
    den = np.where(np.abs(den) < 1e-10, np.where(den == 0, 1e-10, 1e-10*np.sign(den)), den)
    f = (2*D - A*X**2) / den
    gsz = gsxy_ij * f
    dnum_dD = 2 - A*2*X*c
    df_dD = (dnum_dD*den - (2*D - A*X**2)*2) / den**2
    dgsz_dDelta = gsxy_ij * df_dD * 2 * Delta_ij
    dgsz_dchi = gsxy_ij * df_dD * 2 * chi_ij
    return gsz, dgsz_dDelta, dgsz_dchi


def g_sz_bond_dn_i_vec(delta_i, delta_j, m_i, m_j, Delta_ij, chi_ij, sigma_field, eps=1e-6):
    ddelta, dm = -eps, sigma_field/2.0*eps
    gsxy_j = g_sxy(delta_j, m_j)
    gsxy_i_p = g_sxy(delta_i+ddelta, m_i+dm)
    gp, _, _ = g_sz_bond_and_derivs_vec(delta_i+ddelta, delta_j, m_i+dm, m_j,
                                         Delta_ij, chi_ij, gsxy_i_p, gsxy_j)
    gsxy_i_m = g_sxy(delta_i-ddelta, m_i-dm)
    gm, _, _ = g_sz_bond_and_derivs_vec(delta_i-ddelta, delta_j, m_i-dm, m_j,
                                         Delta_ij, chi_ij, gsxy_i_m, gsxy_j)
    return (gp - gm) / (2*eps)


def g_sxy_dn_vec(delta_i, m_i, sigma_field, eps=1e-6):
    ddelta, dm = -eps, sigma_field/2.0*eps
    gp = g_sxy(delta_i+ddelta, m_i+dm)
    gm = g_sxy(delta_i-ddelta, m_i-dm)
    return (gp - gm) / (2*eps)


def g_t_dn_vec(delta_i, m_i, sigma_prime, sigma_field, eps=1e-6):
    ddelta, dm = -eps, sigma_field/2.0*eps
    gp = g_t(delta_i+ddelta, m_i+dm, sigma_prime)
    gm = g_t(delta_i-ddelta, m_i-dm, sigma_prime)
    return (gp - gm) / (2*eps)


# --------------------------------------------------------- BdG

def fermi(E, T):
    if T < 1e-9:
        return (E < 0).astype(float)
    return 1.0 / (1.0 + np.exp(np.clip(E / T, -500, 500)))


def diagonalize_bdg(H0up, H0dn, Deltamat):
    N = H0up.shape[0]
    HBdG = np.zeros((2 * N, 2 * N), dtype=complex)
    HBdG[:N, :N] = H0up
    HBdG[N:, N:] = -H0dn.conj()
    HBdG[:N, N:] = Deltamat
    HBdG[N:, :N] = Deltamat.conj().T
    E, psi = np.linalg.eigh(HBdG)
    return E, psi


def diagonalize_bdg_real(H0up, H0dn, Deltamat, N=None):
    if N is None:
        N = H0up.shape[0]
    HBdG = np.zeros((2 * N, 2 * N))
    HBdG[:N, :N] = H0up
    HBdG[N:, N:] = -H0dn
    HBdG[:N, N:] = Deltamat
    HBdG[N:, :N] = Deltamat.T
    return np.linalg.eigh(HBdG)
