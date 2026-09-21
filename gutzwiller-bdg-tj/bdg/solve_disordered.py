import numpy as np
from scipy.optimize import brentq
from lattice import (square_lattice, g_t, g_sxy, g_sz_bond_vec,
                      g_sz_bond_and_derivs_vec, g_sz_bond_dn_i_vec,
                      g_sxy_dn_vec, g_t_dn_vec, diagonalize_bdg_real)

Lx, Ly = 24, 24
t, tp, J = 1.0, -0.25, 0.3
N, nn_bonds, nnn_bonds = square_lattice(Lx, Ly)
xs = np.array([i % Lx for i in range(N)])
ys = np.array([i // Lx for i in range(N)])
neel_sign = (-1.0) ** (xs + ys)

nn_i = np.array([b[0] for b in nn_bonds])
nn_j = np.array([b[1] for b in nn_bonds])
nn_sign = np.array([1.0 if b[2] == 'x' else -1.0 for b in nn_bonds])
nnn_i = np.array([b[0] for b in nnn_bonds])
nnn_j = np.array([b[1] for b in nnn_bonds])

DEFAULT_T = 0.1


def fermi_smeared(E, T):
    if T < 1e-9:
        return (E < 0).astype(float)
    return 1.0 / (1.0 + np.exp(np.clip(E / T, -500, 500)))


def bdg_expectation(H0up, H0dn, Deltamat, T=DEFAULT_T):
    E, psi = diagonalize_bdg_real(H0up, H0dn, Deltamat)
    u, v = psi[:N, :], psi[N:, :]
    f = fermi_smeared(E, T)
    n_up = np.sum(f[None, :] * u ** 2, axis=1)
    n_dn = np.sum((1.0 - f)[None, :] * v ** 2, axis=1)
    F = ((1.0 - f)[None, :] * u) @ v.T
    G = (f[None, :] * v) @ u.T
    Delta_bond = 0.5 * (F - G)
    Chi = (f[None, :] * u) @ u.T
    return n_up, n_dn, Delta_bond, Chi


def place_impurities(n_imp_frac, V, rng):
    V_i = np.zeros(N)
    n_imp = max(1, int(round(n_imp_frac * N)))
    sites = rng.choice(N, size=n_imp, replace=False)
    V_i[sites] = V
    return V_i, sites


def place_impurities_isolated(n_imp_frac, V, rng, min_dist=6, max_tries=5000):
    n_imp = max(1, int(round(n_imp_frac * N)))
    chosen = []
    tries = 0
    while len(chosen) < n_imp and tries < max_tries:
        tries += 1
        cand = int(rng.integers(0, N))
        if cand in chosen:
            continue
        cx, cy = cand % Lx, cand // Lx
        ok = True
        for s in chosen:
            sx, sy = s % Lx, s // Lx
            dx = min(abs(cx - sx), Lx - abs(cx - sx))
            dy = min(abs(cy - sy), Ly - abs(cy - sy))
            if (dx**2 + dy**2) ** 0.5 < min_dist:
                ok = False
                break
        if ok:
            chosen.append(cand)
    if len(chosen) < n_imp:
        raise RuntimeError(
            f"only placed {len(chosen)}/{n_imp} impurities with min_dist={min_dist} "
            f"after {max_tries} tries -- lower min_dist or n_imp_frac"
        )
    sites = np.array(chosen)
    V_i = np.zeros(N)
    V_i[sites] = V
    return V_i, sites


def afm_field(delta_i, m_i, gsxy_i, chi_nn, Delta_nn, chi_nnn, sigma_field):
    D_nn = Delta_nn**2 + chi_nn**2
    A_nn = 4 * m_i[nn_i] * m_i[nn_j]
    twoD_minus_A = 2 * D_nn - A_nn
    twoD = 2 * D_nn

    gsz_nn, _, _ = g_sz_bond_and_derivs_vec(delta_i[nn_i], delta_i[nn_j],
                                             m_i[nn_i], m_i[nn_j],
                                             Delta_nn, chi_nn,
                                             gsxy_i[nn_i], gsxy_i[nn_j])

    field = np.zeros(N)

    np.add.at(field, nn_i, 0.5 * sigma_field * J * gsz_nn * m_i[nn_j])
    np.add.at(field, nn_j, 0.5 * sigma_field * J * gsz_nn * m_i[nn_i])

    dgsz_at_i = g_sz_bond_dn_i_vec(delta_i[nn_i], delta_i[nn_j], m_i[nn_i], m_i[nn_j],
                                    Delta_nn, chi_nn, sigma_field)
    dgsz_at_j = g_sz_bond_dn_i_vec(delta_i[nn_j], delta_i[nn_i], m_i[nn_j], m_i[nn_i],
                                    Delta_nn, chi_nn, sigma_field)
    np.add.at(field, nn_i, -(J / 4.0) * twoD_minus_A * dgsz_at_i)
    np.add.at(field, nn_j, -(J / 4.0) * twoD_minus_A * dgsz_at_j)

    dgsxy_at_i = g_sxy_dn_vec(delta_i[nn_i], m_i[nn_i], sigma_field) * gsxy_i[nn_j]
    dgsxy_at_j = g_sxy_dn_vec(delta_i[nn_j], m_i[nn_j], sigma_field) * gsxy_i[nn_i]
    np.add.at(field, nn_i, -(J / 2.0) * twoD * dgsxy_at_i)
    np.add.at(field, nn_j, -(J / 2.0) * twoD * dgsxy_at_j)

    for sigma_p in (+1.0, -1.0):
        dgt_at_i_nn = g_t_dn_vec(delta_i[nn_i], m_i[nn_i], sigma_p, sigma_field)
        dgt_at_j_nn = g_t_dn_vec(delta_i[nn_j], m_i[nn_j], sigma_p, sigma_field)
        gt_j_sp_nn = g_t(delta_i[nn_j], m_i[nn_j], sigma_p)
        gt_i_sp_nn = g_t(delta_i[nn_i], m_i[nn_i], sigma_p)
        np.add.at(field, nn_i, -t * dgt_at_i_nn * gt_j_sp_nn * 2 * chi_nn)
        np.add.at(field, nn_j, -t * dgt_at_j_nn * gt_i_sp_nn * 2 * chi_nn)

        dgt_at_i_nnn = g_t_dn_vec(delta_i[nnn_i], m_i[nnn_i], sigma_p, sigma_field)
        dgt_at_j_nnn = g_t_dn_vec(delta_i[nnn_j], m_i[nnn_j], sigma_p, sigma_field)
        gt_j_sp_nnn = g_t(delta_i[nnn_j], m_i[nnn_j], sigma_p)
        gt_i_sp_nnn = g_t(delta_i[nnn_i], m_i[nnn_i], sigma_p)
        np.add.at(field, nnn_i, -tp * dgt_at_i_nnn * gt_j_sp_nnn * 2 * chi_nnn)
        np.add.at(field, nnn_j, -tp * dgt_at_j_nnn * gt_i_sp_nnn * 2 * chi_nnn)

    return field


def solve_disordered(delta_target, V_i, seed=None, max_iter=150, mix=0.15, tol=2e-4,
                      verbose=False, T=DEFAULT_T):
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
                                                H0dn - mu_ * np.eye(N), Deltamat, T=T)
            return np.mean(1 - (n_up + n_dn)) - delta_target

        try:
            mu_bisected = brentq(avg_delta, mu - 1.0, mu + 1.0, xtol=1e-6)
        except ValueError:
            try:
                mu_bisected = brentq(avg_delta, -8, 8, xtol=1e-6)
            except ValueError:
                xs_ = np.linspace(-8, 8, 41)
                vals = [avg_delta(x) for x in xs_]
                idx = np.where(np.diff(np.sign(vals)) != 0)[0]
                if len(idx) == 0:
                    raise RuntimeError("no mu bracket")
                mu_bisected = brentq(avg_delta, xs_[idx[0]], xs_[idx[0] + 1], xtol=1e-6)
        mu = (1 - mix) * mu + mix * mu_bisected

        n_up, n_dn, Db, Chi = bdg_expectation(H0up - mu * np.eye(N),
                                               H0dn - mu * np.eye(N), Deltamat, T=T)
        new_delta_i = 1 - (n_up + n_dn)
        new_m_i = 0.5 * (n_up - n_dn)
        new_chi_nn = 0.5 * (Chi[nn_i, nn_j] + Chi[nn_j, nn_i])
        new_Delta_nn = nn_sign * 0.5 * (Db[nn_i, nn_j] + Db[nn_j, nn_i])
        new_chi_nnn = 0.5 * (Chi[nnn_i, nnn_j] + Chi[nnn_j, nnn_i])

        diffs = {
            'delta_i': np.max(np.abs(new_delta_i - delta_i)),
            'm_i': np.max(np.abs(new_m_i - m_i)),
            'chi_nn': np.max(np.abs(new_chi_nn - chi_nn)),
            'Delta_nn': np.max(np.abs(new_Delta_nn - Delta_nn)),
        }
        diff = max(diffs.values())
        worst = max(diffs, key=diffs.get)

        delta_i = (1 - mix) * delta_i + mix * new_delta_i
        m_i = (1 - mix) * m_i + mix * new_m_i
        chi_nn = (1 - mix) * chi_nn + mix * new_chi_nn
        Delta_nn = (1 - mix) * Delta_nn + mix * new_Delta_nn
        chi_nnn = (1 - mix) * chi_nnn + mix * new_chi_nnn

        if verbose and it % 10 == 0:
            msg = (f"    it={it:3d}  diff={diff:.4f} (worst={worst})  "
                   f"mean|m|={np.mean(np.abs(m_i)):.4f}  mu={mu:.3f}")
            if worst == 'delta_i':
                bad = np.argmax(np.abs(new_delta_i - delta_i))
                msg += (f"  | site={bad} (x={bad % Lx},y={bad // Lx})"
                        f" V_i={V_i[bad]:.2f} delta_i={delta_i[bad]:.4f}"
                        f" new={new_delta_i[bad]:.4f}")
            print(msg)
        if diff < tol and it > 8:
            break

    return dict(delta_i=delta_i, m_i=m_i, chi_nn=chi_nn, Delta_nn=Delta_nn,
                chi_nnn=chi_nnn, mu=mu, iters=it, diff=diff, T=T)


def run_fig2(deltas=(0.06, 0.10, 0.135, 0.145, 0.155, 0.16), max_iter=200, mix=0.15,
             seed_rng=1, T=DEFAULT_T):
    rng = np.random.default_rng(seed_rng)
    n_imp_max = max(1, int(round(max(deltas) * N)))
    all_sites = rng.choice(N, size=n_imp_max, replace=False)

    results = []
    for d in deltas:
        n_imp = max(1, int(round(d * N)))
        imp_sites = all_sites[:n_imp]
        V_i = np.zeros(N)
        V_i[imp_sites] = t
        res = solve_disordered(d, V_i, seed=None, max_iter=max_iter, mix=mix,
                                tol=1e-4, verbose=True, T=T)
        res['imp_sites'] = imp_sites
        res['delta_target'] = d
        res['V_i'] = V_i.copy()
        results.append(res)
        print(f"delta={d:.3f} done: iters={res['iters']} diff={res['diff']:.3e} "
              f"mean|m|={np.mean(np.abs(res['m_i'])):.3f}")
    return results


def run_fig3(panels=((0.01, 0.14), (0.02, 0.14), (0.01, 0.15), (0.02, 0.15)),
             V_strong_factor=100.0, max_iter=200, mix=0.10, seed_rng=3, T=DEFAULT_T):
    results = []
    for k, (n_imp_frac, d) in enumerate(panels):
        rng = np.random.default_rng(seed_rng + k)
        V_i, imp_sites = place_impurities(n_imp_frac, V_strong_factor * t, rng)
        res = solve_disordered(d, V_i, seed=None, max_iter=max_iter, mix=mix,
                                tol=1e-4, verbose=True, T=T)
        res['imp_sites'] = imp_sites
        res['delta_target'] = d
        res['n_imp_frac'] = n_imp_frac
        res['V_i'] = V_i.copy()
        results.append(res)
        print(f"n_imp={n_imp_frac*100:.0f}%  delta={d:.3f}  iters={res['iters']} "
              f"diff={res['diff']:.3e}  mean|m|={np.mean(np.abs(res['m_i'])):.3f}  "
              f"max|m|={np.max(np.abs(res['m_i'])):.3f}")
    return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pickle
    deltas = (0.115, 0.125, 0.13, 0.135, 0.145, 0.155)
    results = run_fig2(deltas=deltas, max_iter=200, mix=0.15, T=DEFAULT_T)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.3), constrained_layout=True)
    for ax, res in zip(axes, results):
        m_map = res["m_i"].reshape(Ly, Lx)
        im = ax.imshow(m_map, cmap="jet", vmin=-0.12, vmax=0.12, origin="lower")
        imp_y = res["imp_sites"] // Lx
        imp_x = res["imp_sites"] % Lx
        ax.scatter(imp_x, imp_y, c="k", s=35)
        converged = "OK" if res["diff"] < 1e-4 else "not converged"
        ax.set_title(rf"$\delta={res['delta_target']:.3f}$"
                     rf"   $\langle|m|\rangle={np.mean(np.abs(res['m_i'])):.3f}$"
                     f"\n{converged} (diff={res['diff']:.1e}, {res['iters']} it)",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label=r"$m_i$")
    fig.suptitle(f"Type I disorder, nested impurity sets, T={DEFAULT_T}")
    plt.savefig("fig2_type1_maps.png", dpi=140)
    plt.show()

    import os
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    for res in results:
        fname = os.path.join(data_dir, f"type1_delta{res['delta_target']:.3f}_T{DEFAULT_T}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(res, f)
        print(f"saved {fname}")
