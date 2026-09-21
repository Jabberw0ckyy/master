
#eigenvalue-spectrum check for a near-zero BdG mode concentrated on site 240 
import numpy as np
from scipy.optimize import brentq
from lattice import square_lattice, g_t, g_sxy
from gutzwiller_vectorized import (g_sz_bond_vec, g_sz_bond_and_derivs_vec,
                             g_sz_bond_dn_i_vec, g_sxy_dn_vec, g_t_dn_vec)
from solve_disordered import (Lx, Ly, N, nn_bonds, nnn_bonds, xs, ys, neel_sign,
                            nn_i, nn_j, nn_sign, nnn_i, nnn_j,
                            t, tp, J, afm_field, diagonalize_bdg_real,
                            bdg_expectation)

SITE_TO_CHECK = 240
STOP_AT_ITER = 100


def run_and_inspect(delta_target, V_i, max_iter=STOP_AT_ITER + 1, mix=0.15, seed=None):
    if seed is None:
        delta_i = np.full(N, delta_target)
        m_i = 0.12 * neel_sign
        chi_nn = np.full(len(nn_bonds), 0.18)
        Delta_nn = 0.10 * nn_sign
        chi_nnn = np.full(len(nnn_bonds), 0.05)
        mu = -0.5
    else:
        delta_i, m_i, chi_nn, Delta_nn, chi_nnn, mu = seed

    for it in range(max_iter):
        gsxy_i = g_sxy(delta_i, m_i)
        gt_up = g_t(delta_i, m_i, +1)
        gt_dn = g_t(delta_i, m_i, -1)

        gsz_nn = g_sz_bond_vec(delta_i[nn_i], delta_i[nn_j], m_i[nn_i], m_i[nn_j],
                                Delta_nn, chi_nn, gsxy_i[nn_i], gsxy_i[nn_j])
        gsxy_nn = gsxy_i[nn_i] * gsxy_i[nn_j]

        exch_hop = J * (0.25 * gsz_nn + 0.5 * gsxy_nn) * chi_nn
        exch_pair = J * (0.25 * gsz_nn + 0.5 * gsxy_nn) * Delta_nn

        hop_up = -(gt_up[nn_i] * gt_up[nn_j]) * t - exch_hop
        hop_dn = -(gt_dn[nn_i] * gt_dn[nn_j]) * t - exch_hop
        hop_up_nnn = -(gt_up[nnn_i] * gt_up[nnn_j]) * tp
        hop_dn_nnn = -(gt_dn[nnn_i] * gt_dn[nnn_j]) * tp

        H0up = np.zeros((N, N))
        H0dn = np.zeros((N, N))
        H0up[nn_i, nn_j] += hop_up; H0up[nn_j, nn_i] += hop_up
        H0dn[nn_i, nn_j] += hop_dn; H0dn[nn_j, nn_i] += hop_dn
        H0up[nnn_i, nnn_j] += hop_up_nnn; H0up[nnn_j, nnn_i] += hop_up_nnn
        H0dn[nnn_i, nnn_j] += hop_dn_nnn; H0dn[nnn_j, nnn_i] += hop_dn_nnn

        h_up = afm_field(delta_i, m_i, gsxy_i, chi_nn, Delta_nn, chi_nnn, +1.0)
        h_dn = afm_field(delta_i, m_i, gsxy_i, chi_nn, Delta_nn, chi_nnn, -1.0)

        np.fill_diagonal(H0up, H0up.diagonal() + h_up + V_i)
        np.fill_diagonal(H0dn, H0dn.diagonal() + h_dn + V_i)

        Deltamat = np.zeros((N, N))
        Deltamat[nn_i, nn_j] += nn_sign * exch_pair
        Deltamat[nn_j, nn_i] += nn_sign * exch_pair

        def avg_delta(mu_):
            n_up, n_dn, _, _ = bdg_expectation(H0up - mu_ * np.eye(N),
                                                H0dn - mu_ * np.eye(N), Deltamat)
            return np.mean(1 - (n_up + n_dn)) - delta_target

        try:
            mu = brentq(avg_delta, mu - 1.0, mu + 1.0, xtol=1e-6)
        except ValueError:
            try:
                mu = brentq(avg_delta, -8, 8, xtol=1e-6)
            except ValueError:
                xs_ = np.linspace(-8, 8, 41)
                vals = [avg_delta(x) for x in xs_]
                idx = np.where(np.diff(np.sign(vals)) != 0)[0]
                if len(idx) == 0:
                    raise RuntimeError("no mu bracket")
                mu = brentq(avg_delta, xs_[idx[0]], xs_[idx[0] + 1], xtol=1e-6)

        if it == STOP_AT_ITER:
            E, psi = diagonalize_bdg_real(H0up - mu * np.eye(N),
                                           H0dn - mu * np.eye(N), Deltamat)
            print(f"\n=== inspecting it={it}, mu={mu:.4f} ===")
            print("10 smallest |E|:", np.sort(np.abs(E))[:10])

            near_zero = np.where(np.abs(E) < 1e-6)[0]
            print(f"modes with |E|<1e-6: {len(near_zero)}  ->  E = {E[near_zero]}")

            near_loose = np.where(np.abs(E) < 1e-3)[0]
            print(f"modes with |E|<1e-3: {len(near_loose)}")
            for m_idx in near_loose:
                u_w = psi[SITE_TO_CHECK, m_idx] ** 2
                v_w = psi[N + SITE_TO_CHECK, m_idx] ** 2
                print(f"    mode {m_idx}: E={E[m_idx]:.3e}  "
                      f"|u({SITE_TO_CHECK})|^2={u_w:.4f}  |v({SITE_TO_CHECK})|^2={v_w:.4f}")
            all_u2 = np.sum(psi[SITE_TO_CHECK, :] ** 2)
            all_v2 = np.sum(psi[N + SITE_TO_CHECK, :] ** 2)
            print(f"sum over ALL modes of |u({SITE_TO_CHECK})|^2 = {all_u2:.4f}, "
                  f"|v({SITE_TO_CHECK})|^2 = {all_v2:.4f}")

            n_up_site = np.sum(psi[SITE_TO_CHECK, E < -1e-9] ** 2)
            n_dn_site = np.sum(psi[N + SITE_TO_CHECK, E > 1e-9] ** 2)
            print(f"current n_up[{SITE_TO_CHECK}]={n_up_site:.4f}  "
                  f"n_dn[{SITE_TO_CHECK}]={n_dn_site:.4f}  "
                  f"-> delta_i={1-(n_up_site+n_dn_site):.4f}")

        n_up, n_dn, Db, Chi = bdg_expectation(H0up - mu * np.eye(N),
                                               H0dn - mu * np.eye(N), Deltamat)
        new_delta_i = 1 - (n_up + n_dn)
        new_m_i = 0.5 * (n_up - n_dn)
        new_chi_nn = 0.5 * (Chi[nn_i, nn_j] + Chi[nn_j, nn_i])
        new_Delta_nn = nn_sign * 0.5 * (Db[nn_i, nn_j] + Db[nn_j, nn_i])
        new_chi_nnn = 0.5 * (Chi[nnn_i, nnn_j] + Chi[nnn_j, nnn_i])

        if it == STOP_AT_ITER:
            print(f"new_delta_i[{SITE_TO_CHECK}] = {new_delta_i[SITE_TO_CHECK]:.4f}")
            break

        delta_i = (1 - mix) * delta_i + mix * new_delta_i
        m_i = (1 - mix) * m_i + mix * new_m_i
        chi_nn = (1 - mix) * chi_nn + mix * new_chi_nn
        Delta_nn = (1 - mix) * Delta_nn + mix * new_Delta_nn
        chi_nnn = (1 - mix) * chi_nnn + mix * new_chi_nnn


if __name__ == "__main__":
    import numpy as np
    from solve_disordered import N, t

    rng = np.random.default_rng(1)
    n_imp = max(1, int(round(0.16 * N)))
    imp_sites = rng.choice(N, size=n_imp, replace=False)
    V_i = np.zeros(N)
    V_i[imp_sites] = t

    run_and_inspect(0.16, V_i, mix=0.15)
