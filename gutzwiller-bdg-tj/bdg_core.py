"""
Real-space Gutzwiller-BdG solver for the t-J model, following
Christensen, Hirschfeld, Andersen, PRB 84, 184511 (2011).

Stage 1 target: homogeneous (clean) case -> reproduce Fig. 1
(m, Delta, chi vs doping).

Conventions:
  - Square lattice, Lx x Ly sites, periodic boundary conditions.
  - NN bonds (hopping t) carry the d-wave pairing; NNN bonds (hopping t').
  - Nambu basis per site: (c_i-up, c_i-down^dagger).
  - Simplified EGFs exactly as Eqs. (8)-(13) of the paper.
  - Magnetic + pairing channels of the J-term mean-field decoupled with
    the standard "3/8 J" (pairing) / "3/4 J" (Ising/magnetic) RMFT
    prefactors used throughout the t-J RMFT literature (Zhang-Gros-Rice-Shiba
    school). This is the piece NOT literally spelled out as a closed formula
    in the paper (it's implicit in Eqs. 16-18); everything upstream of it
    (the EGFs themselves) is taken verbatim from the paper.
"""
import numpy as np

# ---------------------------------------------------------------- lattice --

def square_lattice(Lx, Ly):
    """Return site index grid, NN bond list (with x/y flag for d-wave sign),
    and NNN bond list, all with periodic boundary conditions."""
    N = Lx * Ly
    idx = np.arange(N).reshape(Ly, Lx)  # idx[y, x]

    nn_bonds = []   # (i, j, 'x' or 'y')
    nnn_bonds = []  # (i, j)
    for y in range(Ly):
        for x in range(Lx):
            i = idx[y, x]
            # +x neighbor
            j = idx[y, (x + 1) % Lx]
            nn_bonds.append((i, j, 'x'))
            # +y neighbor
            j = idx[(y + 1) % Ly, x]
            nn_bonds.append((i, j, 'y'))
            # NNN: (+1,+1) and (+1,-1)
            j = idx[(y + 1) % Ly, (x + 1) % Lx]
            nnn_bonds.append((i, j))
            j = idx[(y - 1) % Ly, (x + 1) % Lx]
            nnn_bonds.append((i, j))
    return N, nn_bonds, nnn_bonds


# ---------------------------------------------------- Gutzwiller factors --

def g_t(delta, m, sigma):
    """Eq. (9): simplified EGF for the hopping term, site-dependent,
    sigma = +1 (up) or -1 (down)."""
    delta = np.clip(delta, 1e-6, 1 - 1e-6)
    num = 2 * delta * (1 - delta)
    den = 1 - delta**2 + 4 * m**2
    den = np.where(den < 1e-8, 1e-8, den)
    a = (1 + delta + sigma * 2 * m) / np.clip(1 + delta - sigma * 2 * m, 1e-8, None)
    return np.sqrt(np.clip(num / den, 0, None) * np.clip(a, 0, None))


def g_sxy(delta, m):
    """Eq. (11)."""
    delta = np.clip(delta, 1e-6, 1 - 1e-6)
    den = 1 - delta**2 + 4 * m**2
    den = np.where(den < 1e-8, 1e-8, den)
    return 2 * (1 - delta) / den


def g_sz_bond(delta_i, delta_j, m_i, m_j, Delta_ij, chi_ij, gsxy_i, gsxy_j):
    """Eqs. (12)-(13). Delta_ij, chi_ij here are the *bar* (spin-averaged)
    bond order parameters (real, single number per bond)."""
    gsxy_ij = gsxy_i * gsxy_j
    d2c2 = Delta_ij**2 + chi_ij**2
    denom_i = (1 - delta_i**2 + 4 * m_i**2)
    denom_j = (1 - delta_j**2 + 4 * m_j**2)
    denom_i = denom_i if denom_i > 1e-8 else 1e-8
    denom_j = denom_j if denom_j > 1e-8 else 1e-8
    # Eq. (13): denominator is sqrt(denom_i * denom_j), NOT the plain product.
    X = 1 + 12 * (1 - delta_i) * (1 - delta_j) * d2c2 / np.sqrt(denom_i * denom_j)
    num = 2 * d2c2 - 4 * m_i * m_j * X**2
    den = 2 * d2c2 - 4 * m_i * m_j
    if abs(den) < 1e-10:
        den = 1e-10 * np.sign(den) if den != 0 else 1e-10
    return gsxy_ij * num / den


def g_sz_bond_and_derivs(delta_i, delta_j, m_i, m_j, Delta_ij, chi_ij, gsxy_i, gsxy_j):
    """g_sz plus its exact partial derivatives w.r.t. Delta_ij and chi_ij
    (Eqs. 12-13 differentiated w.r.t. the bond order parameters D=Delta^2+chi^2
    enters through X too)."""
    gsxy_ij = gsxy_i * gsxy_j
    D = Delta_ij**2 + chi_ij**2
    A = 4 * m_i * m_j
    denom_i = max(1 - delta_i**2 + 4 * m_i**2, 1e-8)
    denom_j = max(1 - delta_j**2 + 4 * m_j**2, 1e-8)
    # Eq. (13): denominator is sqrt(denom_i * denom_j), NOT the plain product
    # (this was previously wrong -- confirmed against the clean arXiv:1109.1708
    # source, not the OCR'd PDF).
    c = 12 * (1 - delta_i) * (1 - delta_j) / np.sqrt(denom_i * denom_j)
    X = 1 + c * D
    den = 2 * D - A
    if abs(den) < 1e-10:
        den = 1e-10 * np.sign(den) if den != 0 else 1e-10
    f = (2 * D - A * X**2) / den
    gsz = gsxy_ij * f
    # df/dD
    dnum_dD = 2 - A * 2 * X * c
    df_dD = (dnum_dD * den - (2 * D - A * X**2) * 2) / den**2
    dgsz_dDelta = gsxy_ij * df_dD * 2 * Delta_ij
    dgsz_dchi = gsxy_ij * df_dD * 2 * chi_ij
    return gsz, dgsz_dDelta, dgsz_dchi


# ------------------------------------------- derivatives w.r.t. n_i,sigma --
# Eq. (18) needs dg^t/dn_i,sigma, dg^{s,xy}/dn_i,sigma, dg^{s,z}_ij/dn_i,sigma:
# these were previously OMITTED (only the leading AFM term of Eq. 18 was
# implemented). Site density and Neel moment are related to occupations by
#   delta_i = 1 - n_i,up - n_i,down ,   m_i = (n_i,up - n_i,down)/2
# so holding n_i,-sigma fixed while varying n_i,sigma gives the chain rule
#   d(delta_i)/d(n_i,sigma) = -1 ,   d(m_i)/d(n_i,sigma) = sigma/2 .
# Computed here by central finite differences (robust vs. hand-differentiating
# the already-complicated Eqs. 9/11/12-13 through this chain rule).

def g_t_dn(delta_i, m_i, sigma_prime, sigma_field, eps=1e-6):
    """d g^t_{i,sigma_prime} / d n_{i,sigma_field}."""
    ddelta, dm = -eps, sigma_field / 2.0 * eps
    gp = g_t(delta_i + ddelta, m_i + dm, sigma_prime)
    gm = g_t(delta_i - ddelta, m_i - dm, sigma_prime)
    return (gp - gm) / (2 * eps)


def g_sxy_dn(delta_i, m_i, sigma_field, eps=1e-6):
    """d g^{s,xy}_i / d n_{i,sigma_field}."""
    ddelta, dm = -eps, sigma_field / 2.0 * eps
    gp = g_sxy(delta_i + ddelta, m_i + dm)
    gm = g_sxy(delta_i - ddelta, m_i - dm)
    return (gp - gm) / (2 * eps)


def g_sz_bond_dn_i(delta_i, delta_j, m_i, m_j, Delta_ij, chi_ij, sigma_field, eps=1e-6):
    """d g^{s,z}_ij / d n_{i,sigma_field}, differentiating only the site-i
    dependence (site j, and the bond order parameters Delta_ij/chi_ij, held
    fixed). g^{s,xy}_i re-evaluated at the shifted point since it also
    depends on delta_i, m_i."""
    ddelta, dm = -eps, sigma_field / 2.0 * eps
    gsxy_j = g_sxy(delta_j, m_j)
    gsxy_i_p = g_sxy(delta_i + ddelta, m_i + dm)
    gp, _, _ = g_sz_bond_and_derivs(delta_i + ddelta, delta_j, m_i + dm, m_j,
                                     Delta_ij, chi_ij, gsxy_i_p, gsxy_j)
    gsxy_i_m = g_sxy(delta_i - ddelta, m_i - dm)
    gm, _, _ = g_sz_bond_and_derivs(delta_i - ddelta, delta_j, m_i - dm, m_j,
                                     Delta_ij, chi_ij, gsxy_i_m, gsxy_j)
    return (gp - gm) / (2 * eps)


# --------------------------------------------------------- BdG machinery --

def fermi(E, T):
    if T < 1e-9:
        return (E < 0).astype(float)
    return 1.0 / (1.0 + np.exp(np.clip(E / T, -500, 500)))


def diagonalize_bdg(H0up, H0dn, Deltamat):
    """Build and diagonalize the 2N x 2N BdG matrix.
    Basis: (c_1up,...,c_Nup, c_1dn^+,...,c_Ndn^+)."""
    N = H0up.shape[0]
    HBdG = np.zeros((2 * N, 2 * N), dtype=complex)
    HBdG[:N, :N] = H0up
    HBdG[N:, N:] = -H0dn.conj()
    HBdG[:N, N:] = Deltamat
    HBdG[N:, :N] = Deltamat.conj().T
    E, psi = np.linalg.eigh(HBdG)
    return E, psi
