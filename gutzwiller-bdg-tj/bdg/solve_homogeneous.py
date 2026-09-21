import numpy as np
from scipy.optimize import brentq
from lattice import (square_lattice, g_t, g_sxy, g_sz_bond_and_derivs, diagonalize_bdg,
                      g_t_dn, g_sxy_dn, g_sz_bond_dn_i)

Lx, Ly = 24, 24
t, tp, J = 1.0, -0.25, 0.3
N, nn_bonds, nnn_bonds = square_lattice(Lx, Ly)
xs = np.array([i % Lx for i in range(N)])
ys = np.array([i // Lx for i in range(N)])
neel_sign = (-1.0) ** (xs + ys)

nn_of = [[] for _ in range(N)]
for (i, j, d) in nn_bonds:
    nn_of[i].append((j, d)); nn_of[j].append((i, d))
nnn_of = [[] for _ in range(N)]
for (i, j) in nnn_bonds:
    nnn_of[i].append(j); nnn_of[j].append(i)


def bdg_expectation(H0up, H0dn, Deltamat):
    E, psi = diagonalize_bdg(H0up, H0dn, Deltamat)
    u = psi[:N, :]; v = psi[N:, :]
    neg = E < -1e-9; pos = E > 1e-9
    n_up = np.sum(np.abs(u[:, neg]) ** 2, axis=1)
    n_dn = np.sum(np.abs(v[:, pos]) ** 2, axis=1)
    Upos = u[:, pos]; Vpos = v[:, pos]
    F = Upos @ Vpos.conj().T
    Uneg = u[:, neg]; Vneg = v[:, neg]
    G = Vneg.conj() @ Uneg.T
    Delta_bond = 0.5 * (F - G)
    Chi = (Uneg.conj() @ Uneg.T)
    return n_up, n_dn, Delta_bond, Chi


def solve_at_doping(delta_target, seed=None, verbose=False, max_iter=200, mix=0.3):
    if seed is None:
        m0, chi0, Delta0, mu, chi_nnn0 = 0.15, 0.15, 0.08, -0.3, 0.1
    else:
        if len(seed) == 5:
            m0, chi0, Delta0, mu, chi_nnn0 = seed
        else:
            m0, chi0, Delta0, mu = seed; chi_nnn0 = 0.1

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
            delta_i[i0], delta_i[j0], m_i[i0], m_i[j0], Delta0, chi0, gsxy_i[i0], gsxy_i[j0])

        twoD_minus_A = 2 * (Delta0**2 + chi0**2) - 4 * m_i[i0] * m_i[j0]
        twoD = 2 * (Delta0**2 + chi0**2)

        def total_field(sigma_field):
            s = sum(gsz * m_i[j] for (j, d) in nn_of[i0])
            leading = 0.5 * sigma_field * J * s
            dgsz_dn = g_sz_bond_dn_i(delta_i[i0], delta_i[j0], m_i[i0], m_i[j0], Delta0, chi0, sigma_field)
            term2 = -(J / 4.0) * len(nn_of[i0]) * twoD_minus_A * dgsz_dn
            dgsxy_i_dn = g_sxy_dn(delta_i[i0], m_i[i0], sigma_field)
            dgsxy_ij_dn = dgsxy_i_dn * gsxy_i[j0]
            term3 = -(J / 2.0) * len(nn_of[i0]) * twoD * dgsxy_ij_dn
            term4 = 0.0
            for sigma_p in (+1.0, -1.0):
                dgt_i_dn = g_t_dn(delta_i[i0], m_i[i0], sigma_p, sigma_field)
                gt_j0_sp = g_t(delta_i[j0], m_i[j0], sigma_p)
                term4 += -len(nn_of[i0]) * t * (dgt_i_dn * gt_j0_sp) * 2 * chi0
                gt_jn0_sp = g_t(delta_i[jn0], m_i[jn0], sigma_p)
                term4 += -len(nnn_of[i0]) * tp * (dgt_i_dn * gt_jn0_sp) * 2 * chi_nnn0
            return leading + term2 + term3 + term4

        total_up_A = total_field(+1.0); total_dn_A = total_field(-1.0)
        total_up_B = total_dn_A; total_dn_B = total_up_A
        h_up = np.where(neel_sign > 0, total_up_A, total_up_B)
        h_dn = np.where(neel_sign > 0, total_dn_A, total_dn_B)

        gsxy_ij = gsxy_i[i0] * gsxy_i[j0]
        exch_hop_field = J * (0.25 * gsz + 0.5 * gsxy_ij) * chi0 + 0.25 * J * twoD_minus_A * dgsz_dchi
        exch_pair_field = J * (0.25 * gsz + 0.5 * gsxy_ij) * Delta0 + 0.25 * J * twoD_minus_A * dgsz_dDelta

        H0up = np.zeros((N, N)); H0dn = np.zeros((N, N))
        for (i, j, d) in nn_bonds:
            hop = -(gt_up[i] * gt_up[j]) * t - exch_hop_field
            H0up[i, j] += hop; H0up[j, i] += hop
            hop_dn = -(gt_dn[i] * gt_dn[j]) * t - exch_hop_field
            H0dn[i, j] += hop_dn; H0dn[j, i] += hop_dn
        for (i, j) in nnn_bonds:
            hop = -(gt_up[i] * gt_up[j]) * tp
            H0up[i, j] += hop; H0up[j, i] += hop
            hop_dn = -(gt_dn[i] * gt_dn[j]) * tp
            H0dn[i, j] += hop_dn; H0dn[j, i] += hop_dn
        for i in range(N):
            H0up[i, i] += h_up[i]; H0dn[i, i] += h_dn[i]

        Deltamat = np.zeros((N, N), dtype=complex)
        for (i, j, d) in nn_bonds:
            sign = +1.0 if d == 'x' else -1.0
            val = sign * exch_pair_field
            Deltamat[i, j] += val; Deltamat[j, i] += val

        def total_delta(mu_):
            n_up, n_dn, Db, Chi = bdg_expectation(H0up - mu_ * np.eye(N), H0dn - mu_ * np.eye(N), Deltamat)
            return np.mean(1 - (n_up + n_dn)) - delta_target
        try:
            mu = brentq(total_delta, -8, 8, xtol=1e-7)
        except ValueError:
            lo_hi = np.linspace(-8, 8, 33)
            vals = [total_delta(x) for x in lo_hi]
            sgn = np.sign(vals); idx = np.where(np.diff(sgn) != 0)[0]
            if len(idx) == 0: raise RuntimeError("no mu bracket found")
            mu = brentq(total_delta, lo_hi[idx[0]], lo_hi[idx[0] + 1], xtol=1e-7)

        n_up, n_dn, Db, Chi = bdg_expectation(H0up - mu * np.eye(N), H0dn - mu * np.eye(N), Deltamat)
        new_delta_i = 1 - (n_up + n_dn)
        new_m_i = 0.5 * (n_up - n_dn)

        chi_vals, Delta_vals, chi_nnn_vals = [], [], []
        for (i, j, d) in nn_bonds:
            c = np.real(0.5 * (Chi[i, j] + Chi[j, i])); chi_vals.append(c)
            f = np.real(0.5 * (Db[i, j] + Db[j, i]))
            sign = +1.0 if d == 'x' else -1.0
            Delta_vals.append(sign * f)
        for (i, j) in nnn_bonds:
            chi_nnn_vals.append(np.real(0.5 * (Chi[i, j] + Chi[j, i])))
        new_chi0 = float(np.mean(chi_vals)); new_Delta0 = float(np.mean(Delta_vals))
        new_m0 = float(np.mean(np.abs(new_m_i))); new_chi_nnn0 = float(np.mean(chi_nnn_vals))

        d_m = abs(new_m0 - m0); d_chi = abs(new_chi0 - chi0)
        d_Delta = abs(new_Delta0 - Delta0); d_chi_nnn = abs(new_chi_nnn0 - chi_nnn0)

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
    import os
    import pickle
    import matplotlib.pyplot as plt

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    deltas = np.arange(0.02, 0.26, 0.01)
    seed = None
    results = []
    for d in deltas:
        res = solve_at_doping(d, seed=seed, max_iter=200, mix=0.3, verbose=True)
        seed = (res['m'] if res['m'] > 1e-3 else 0.05, res['chi'],
                 res['Delta'] if res['Delta'] > 1e-3 else 0.05, res['mu'], res['chi_nnn'])
        results.append((d, res['m'], res['chi'], res['Delta']))
        print(f"delta={d:.3f}  m={res['m']:.4f}  chi={res['chi']:.4f}  "
              f"Delta={res['Delta']:.4f}  iters={res['iters']}")

    with open(os.path.join(data_dir, "homogeneous_scan_24x24.pkl"), "wb") as f:
        pickle.dump(results, f)

    deltas_arr = np.array([r[0] for r in results])
    m_ours = np.array([r[1] for r in results])
    chi_ours = np.array([r[2] for r in results])
    Delta_ours = np.array([r[3] for r in results])

    # digitized directly off the paper's Fig. 1 image
    paper_delta = [0.026, 0.05, 0.08, 0.10, 0.115, 0.15, 0.20, 0.25]
    paper_chi   = [0.175, 0.185, 0.195, 0.20, 0.205, 0.215, 0.23, 0.24]
    paper_Delta = [0.155, 0.145, 0.135, 0.128, 0.122, 0.115, 0.095, 0.08]
    paper_m     = [0.16, 0.145, 0.11, 0.08, 0.0, 0.0, 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(deltas_arr, chi_ours, color="green", label=r"$\chi$ (ours)")
    ax.plot(deltas_arr, m_ours, color="red", label=r"$m$ (ours)")
    ax.plot(deltas_arr, Delta_ours, color="blue", label=r"$\Delta$ (ours)")
    ax.plot(paper_delta, paper_chi, "o--", color="darkgreen", alpha=0.6, label=r"$\chi$ (paper)")
    ax.plot(paper_delta, paper_m, "o--", color="darkred", alpha=0.6, label=r"$m$ (paper)")
    ax.plot(paper_delta, paper_Delta, "o--", color="navy", alpha=0.6, label=r"$\Delta$ (paper)")
    ax.set_xlabel(r"doping $\delta$")
    ax.set_ylabel("order parameter")
    ax.set_title(f"Homogeneous GBdG, {Lx}x{Ly} lattice")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.26)
    ax.set_ylim(0, 0.30)
    plt.tight_layout()
    out_path = os.path.join(data_dir, "fig1_check_24x24.png")
    plt.savefig(out_path, dpi=140)
    plt.show()
    print(f"\nsaved {out_path}")
