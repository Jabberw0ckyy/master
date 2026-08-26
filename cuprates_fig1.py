import numpy as np
from scipy.optimize import brentq
from bdg_core import (square_lattice, g_t, g_sxy, g_sz_bond_and_derivs, diagonalize_bdg,
                       g_t_dn, g_sxy_dn, g_sz_bond_dn_i)

# ---------------------------------------------------------------- params --
Lx, Ly = 8, 8          # even x even so the Neel pattern is commensurate w/ PBC
t, tp, J = 1.0, -0.25, 0.3
N, nn_bonds, nnn_bonds = square_lattice(Lx, Ly)
xs = np.array([i % Lx for i in range(N)])
ys = np.array([i // Lx for i in range(N)])
neel_sign = (-1.0) ** (xs + ys)

# neighbor lists per site (for the magnetic mean field and averaging g_sz)
nn_of = [[] for _ in range(N)]
for (i, j, d) in nn_bonds:
    nn_of[i].append((j, d))
    nn_of[j].append((i, d))

nnn_of = [[] for _ in range(N)]
for (i, j) in nnn_bonds:
    nnn_of[i].append(j)
    nnn_of[j].append(i)


def bdg_expectation(H0up, H0dn, Deltamat):
    E, psi = diagonalize_bdg(H0up, H0dn, Deltamat)
    u = psi[:N, :]
    v = psi[N:, :]
    neg = E < -1e-9
    pos = E > 1e-9
    n_up = np.sum(np.abs(u[:, neg]) ** 2, axis=1)
    n_dn = np.sum(np.abs(v[:, pos]) ** 2, axis=1)
    # F_ij = <c_i,up c_j,dn>  = sum_{E_n>0} u_n(i) v_n*(j)
    Upos = u[:, pos]
    Vpos = v[:, pos]
    F = Upos @ Vpos.conj().T
    # G_ij = <c_i,dn c_j,up>  = sum_{E_n<0} v_n*(i) u_n(j)
    Uneg = u[:, neg]
    Vneg = v[:, neg]
    G = Vneg.conj() @ Uneg.T
    # singlet pairing amplitude on bond ij: Delta_ij ~ (F_ij - G_ij)/2
    Delta_bond = 0.5 * (F - G)
    # bond kinetic <c_i,up^+ c_j,up> = sum_{E_n<0} u_n(i)* u_n(j)
    Chi = (Uneg.conj() @ Uneg.T)
    return n_up, n_dn, Delta_bond, Chi


def solve_at_doping(delta_target, seed=None, verbose=False, max_iter=200, mix=0.3):
    # state: m_i (Neel amplitude), delta_i (per site, start uniform),
    # chi0 (uniform NN bond kinetic, real), Delta0 (uniform NN d-wave amplitude),
    # chi_nnn0 (uniform NNN bond kinetic, real -- needed for the dg^t/dn_i,sigma
    # feedback term on t' bonds, Eq. 18)
    if seed is None:
        m0, chi0, Delta0, mu, chi_nnn0 = 0.15, 0.15, 0.08, -0.3, 0.1
    else:
        if len(seed) == 5:
            m0, chi0, Delta0, mu, chi_nnn0 = seed
        else:
            m0, chi0, Delta0, mu = seed
            chi_nnn0 = 0.1

    delta_i = np.full(N, delta_target)
    m_i = m0 * neel_sign

    for it in range(max_iter):
        gsxy_i = g_sxy(delta_i, m_i)
        gt_up = g_t(delta_i, m_i, +1)
        gt_dn = g_t(delta_i, m_i, -1)
        i0 = 0
        j0 = [j for (j, d) in nn_of[i0]][0]
        jn0 = nnn_of[i0][0]

        gsz, dgsz_dDelta, dgsz_dchi = g_sz_bond_and_derivs(
            delta_i[i0], delta_i[j0], m_i[i0], m_i[j0],
            Delta0, chi0, gsxy_i[i0], gsxy_i[j0])

        # 2D-A factor from Eqs (16)-(17): (|Delta|^2*2 + |chi|^2*2 - 4 m_i m_j)
        twoD_minus_A = 2 * (Delta0**2 + chi0**2) - 4 * m_i[i0] * m_i[j0]
        twoD = 2 * (Delta0**2 + chi0**2)

        # --- Eq. (18), full: leading AFM term + the three dg/dn_i,sigma
        # feedback terms that were previously omitted entirely. Since the
        # lattice only has two sublattices (Neel checkerboard), every site
        # of a given sublattice sees an identical local environment, so it
        # suffices to evaluate everything once at the representative site
        # i0 (sublattice with m_i0 = +m0) for both spin channels, then use
        # the sublattice-swap = spin-flip symmetry of the problem to get
        # the other sublattice's fields (total_B,up = total_A,down, etc.)
        # instead of separately re-deriving/re-evaluating them.
        def total_field(sigma_field):
            # leading term: (sigma/2) * J * sum_{j in NN(i0)} gsz * m_j
            s = sum(gsz * m_i[j] for (j, d) in nn_of[i0])
            leading = 0.5 * sigma_field * J * s

            # term 2: -(J/4) sum_{j in NN} (2D-A) * dgsz_ij/dn_i,sigma
            dgsz_dn = g_sz_bond_dn_i(delta_i[i0], delta_i[j0], m_i[i0], m_i[j0],
                                      Delta0, chi0, sigma_field)
            term2 = -(J / 4.0) * len(nn_of[i0]) * twoD_minus_A * dgsz_dn

            # term 3: -(J/2) sum_{j in NN} 2D * dgsxy_ij/dn_i,sigma
            #   gsxy_ij = gsxy_i * gsxy_j (Eq. 10, product), only the i-factor
            #   depends on n_i,sigma -> product rule
            dgsxy_i_dn = g_sxy_dn(delta_i[i0], m_i[i0], sigma_field)
            dgsxy_ij_dn = dgsxy_i_dn * gsxy_i[j0]
            term3 = -(J / 2.0) * len(nn_of[i0]) * twoD * dgsxy_ij_dn

            # term 4 (paper: the *dominant* contribution): -sum_j,sigma' t_ij *
            #   dg^t_ij,sigma'/dn_i,sigma * (chi_ij,sigma' + chi*_ij,sigma')
            #   = -sum_j,sigma' t_ij * [dg^t_i,sigma'/dn_i,sigma * g^t_j,sigma'] * 2*chi_ij
            #   summed separately over NN bonds (t, chi0) and NNN bonds (t', chi_nnn0)
            term4 = 0.0
            for sigma_p in (+1.0, -1.0):
                dgt_i_dn = g_t_dn(delta_i[i0], m_i[i0], sigma_p, sigma_field)
                gt_j0_sp = g_t(delta_i[j0], m_i[j0], sigma_p)
                term4 += -len(nn_of[i0]) * t * (dgt_i_dn * gt_j0_sp) * 2 * chi0
                gt_jn0_sp = g_t(delta_i[jn0], m_i[jn0], sigma_p)
                term4 += -len(nnn_of[i0]) * tp * (dgt_i_dn * gt_jn0_sp) * 2 * chi_nnn0

            return leading + term2 + term3 + term4

        total_up_A = total_field(+1.0)
        total_dn_A = total_field(-1.0)
        # sublattice B (m = -m0) by symmetry: total_B,sigma = total_A,-sigma
        total_up_B = total_dn_A
        total_dn_B = total_up_A

        h_up = np.where(neel_sign > 0, total_up_A, total_up_B)
        h_dn = np.where(neel_sign > 0, total_dn_A, total_dn_B)

        # bond g^{s,xy}_ij = g^{s,xy}_i * g^{s,xy}_j, Eq. (10) -- this is a BOND
        # quantity (product of the two site factors), not a single site value.
        gsxy_ij = gsxy_i[i0] * gsxy_i[j0]

        # exchange-generated hopping on NN bonds, Eq. (16):
        #   coeff = J*(gsz/4 + gsxy_ij/2)*chi0  [leading]
        #         + (J/4)*(2D-A)*dgsz/dchi   [feedback term, ADDITIVE not multiplicative]
        exch_hop_field = J * (0.25 * gsz + 0.5 * gsxy_ij) * chi0 \
            + 0.25 * J * twoD_minus_A * dgsz_dchi
        # pairing field on NN bonds, Eq. (17), same structure:
        exch_pair_field = J * (0.25 * gsz + 0.5 * gsxy_ij) * Delta0 \
            + 0.25 * J * twoD_minus_A * dgsz_dDelta

        H0up = np.zeros((N, N))
        H0dn = np.zeros((N, N))
        for (i, j, d) in nn_bonds:
            # Eq. (8): g^t_ijσ = g^t_iσ * g^t_jσ  (product, not arithmetic mean)
            hop = -(gt_up[i] * gt_up[j]) * t - exch_hop_field
            H0up[i, j] += hop; H0up[j, i] += hop
            hop_dn = -(gt_dn[i] * gt_dn[j]) * t - exch_hop_field
            H0dn[i, j] += hop_dn; H0dn[j, i] += hop_dn
        for (i, j) in nnn_bonds:
            hop = -(gt_up[i] * gt_up[j]) * tp
            H0up[i, j] += hop; H0up[j, i] += hop
            hop_dn = -(gt_dn[i] * gt_dn[j]) * tp
            H0dn[i, j] += hop_dn; H0dn[j, i] += hop_dn
        # Eq. (18), full version (leading AFM term + the three feedback terms
        # computed above via total_field/h_up/h_dn).
        for i in range(N):
            H0up[i, i] += h_up[i]
            H0dn[i, i] += h_dn[i]

        Deltamat = np.zeros((N, N), dtype=complex)
        for (i, j, d) in nn_bonds:
            sign = +1.0 if d == 'x' else -1.0
            val = sign * exch_pair_field
            Deltamat[i, j] += val
            Deltamat[j, i] += val

        # chemical potential root-find to hit target doping
        def total_delta(mu_):
            n_up, n_dn, Db, Chi = bdg_expectation(H0up - mu_ * np.eye(N),
                                                   H0dn - mu_ * np.eye(N), Deltamat)
            return np.mean(1 - (n_up + n_dn)) - delta_target

        try:
            mu = brentq(total_delta, -8, 8, xtol=1e-7)
        except ValueError:
            lo_hi = np.linspace(-8, 8, 33)
            vals = [total_delta(x) for x in lo_hi]
            sgn = np.sign(vals)
            idx = np.where(np.diff(sgn) != 0)[0]
            if len(idx) == 0:
                raise RuntimeError("no mu bracket found")
            mu = brentq(total_delta, lo_hi[idx[0]], lo_hi[idx[0] + 1], xtol=1e-7)

        n_up, n_dn, Db, Chi = bdg_expectation(H0up - mu * np.eye(N),
                                               H0dn - mu * np.eye(N), Deltamat)
        new_delta_i = 1 - (n_up + n_dn)
        new_m_i = 0.5 * (n_up - n_dn)

        # bond-average new chi, Delta magnitudes (should be uniform in homogeneous phase)
        chi_vals, Delta_vals, chi_nnn_vals = [], [], []
        for (i, j, d) in nn_bonds:
            c = np.real(0.5 * (Chi[i, j] + Chi[j, i]))
            chi_vals.append(c)
            f = np.real(0.5 * (Db[i, j] + Db[j, i]))
            sign = +1.0 if d == 'x' else -1.0
            Delta_vals.append(sign * f)
        for (i, j) in nnn_bonds:
            chi_nnn_vals.append(np.real(0.5 * (Chi[i, j] + Chi[j, i])))
        new_chi0 = float(np.mean(chi_vals))
        new_Delta0 = float(np.mean(Delta_vals))
        new_m0 = float(np.mean(np.abs(new_m_i)))
        new_chi_nnn0 = float(np.mean(chi_nnn_vals))

        d_m = abs(new_m0 - m0)
        d_chi = abs(new_chi0 - chi0)
        d_Delta = abs(new_Delta0 - Delta0)
        d_chi_nnn = abs(new_chi_nnn0 - chi_nnn0)

        m0 = (1 - mix) * m0 + mix * new_m0
        chi0 = (1 - mix) * chi0 + mix * new_chi0
        Delta0 = (1 - mix) * Delta0 + mix * new_Delta0
        chi_nnn0 = (1 - mix) * chi_nnn0 + mix * new_chi_nnn0
        m_i = m0 * neel_sign
        delta_i = np.full(N, delta_target)

        if verbose and it % 20 == 0:
            print(f"  it={it:3d} m0={m0:.4f} chi0={chi0:.4f} Delta0={Delta0:.4f} mu={mu:.4f}")

        if max(d_m, d_chi, d_Delta, d_chi_nnn) < 1e-5 and it > 5:
            break

    return dict(m=m0, chi=chi0, Delta=abs(Delta0), mu=mu, chi_nnn=chi_nnn0, iters=it)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # headless-safe; remove this line (and the
                            # matplotlib.use call) for an interactive window
    import matplotlib.pyplot as plt

    # ---------------------------------------------------------- 1. scan --
    deltas = np.arange(0.02, 0.26, 0.01)
    m_ours, Delta_ours, chi_ours = [], [], []
    seed = None
    print("Running self-consistency scan (this takes a few minutes)...")
    for d in deltas:
        res = solve_at_doping(d, seed=seed)
        seed = (res['m'] if res['m'] > 1e-3 else 0.05, res['chi'],
                 res['Delta'] if res['Delta'] > 1e-3 else 0.05, res['mu'], res['chi_nnn'])
        m_ours.append(res['m'])
        Delta_ours.append(res['Delta'])
        chi_ours.append(res['chi'])
        print(f"delta={d:.3f}  m={res['m']:.4f}  Delta={res['Delta']:.4f}  "
              f"chi={res['chi']:.4f}  iters={res['iters']}")

    np.save("fig1_results.npy",
            [dict(delta=d, m=m, Delta=D, chi=c)
             for d, m, D, c in zip(deltas, m_ours, Delta_ours, chi_ours)])

    # ----------------------------------------- 2. digitized paper reference --
    # Hand-read directly off the Fig. 1 image (page 3 of the PDF), not OCR text.
    paper_delta = [0.026, 0.05, 0.08, 0.10, 0.115, 0.15, 0.20, 0.25]
    paper_chi   = [0.175, 0.185, 0.195, 0.20, 0.205, 0.215, 0.23, 0.24]
    paper_Delta = [0.155, 0.145, 0.135, 0.128, 0.122, 0.115, 0.095, 0.08]
    paper_m     = [0.16, 0.145, 0.11, 0.08, 0.0, 0.0, 0.0, 0.0]

    # -------------------------------------------------------------- 3. plot --
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(deltas, chi_ours, color="green", label=r"$\chi$ (ours)")
    ax.plot(deltas, m_ours, color="red", label=r"$m$ (ours)")
    ax.plot(deltas, Delta_ours, color="blue", label=r"$\Delta$ (ours)")
    ax.plot(paper_delta, paper_chi, "o--", color="darkgreen", alpha=0.6,
             label=r"$\chi$ (paper, digitized)")
    ax.plot(paper_delta, paper_m, "o--", color="darkred", alpha=0.6,
             label=r"$m$ (paper, digitized)")
    ax.plot(paper_delta, paper_Delta, "o--", color="navy", alpha=0.6,
             label=r"$\Delta$ (paper, digitized)")
    ax.set_xlabel(r"doping $\delta$")
    ax.set_ylabel("order parameter")
    ax.set_title("Homogeneous Gutzwiller-BdG solution vs. Fig. 1\n"
                 "(solid = this code, dashed = paper)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.26)
    ax.set_ylim(0, 0.30)
    plt.tight_layout()
    plt.savefig("fig1_comparison.png", dpi=150)
    print("\nSaved fig1_comparison.png")
