import os
import time
import numpy as np
from scipy.optimize import brentq
from lattice import (square_lattice, g_t, g_sxy, g_sz_bond_vec,
                     g_sz_bond_and_derivs_vec, g_sz_bond_dn_i_vec,
                     g_sxy_dn_vec, g_t_dn_vec, diagonalize_bdg_real)

Lx, Ly = 24, 24
t, tp, J = 1.0, -0.25, 0.3
KAPPA = 1.0 / 16.0
N, nn_bonds, nnn_bonds = square_lattice(Lx, Ly)
xs = np.array([i % Lx for i in range(N)])
ys = np.array([i // Lx for i in range(N)])
neel_sign = (-1.0) ** (xs + ys)

nn_i = np.array([b[0] for b in nn_bonds])
nn_j = np.array([b[1] for b in nn_bonds])
nn_sign = np.array([1.0 if b[2] == 'x' else -1.0 for b in nn_bonds])
nnn_i = np.array([b[0] for b in nnn_bonds])
nnn_j = np.array([b[1] for b in nnn_bonds])

DEFAULT_T = 0.01
MIX_FLUCT = 0.03


def fermi_smeared(E, T):
    if T < 1e-9:
        return (E < 0).astype(float)
    return 1.0 / (1.0 + np.exp(np.clip(E / T, -500, 500)))


# ------------------------------------------------------------------ state ---

def initial_state(delta_target, pairing=True):
    nb, nnb = len(nn_bonds), len(nnn_bonds)
    return dict(delta_i=np.full(N, delta_target), m_i=0.12 * neel_sign,
                chi_up=np.full(nb, 0.18), chi_dn=np.full(nb, 0.18),
                Delta_nn=(0.10 if pairing else 0.0) * nn_sign,
                chin_up=np.full(nnb, 0.05), chin_dn=np.full(nnb, 0.05), mu=0.5)


def _gfac(s):
    d, m = s['delta_i'], s['m_i']
    return g_sxy(d, m), g_t(d, m, +1), g_t(d, m, -1)


def bracket(s):
    m = s['m_i']
    return (2 * s['Delta_nn'] ** 2 + s['chi_up'] ** 2 + s['chi_dn'] ** 2
            - 4 * m[nn_i] * m[nn_j])


# --------------------------------------------------- mean-field Hamiltonian --

def onsite_field(s, sigma_f, gsxy, B):
    d, m = s['delta_i'], s['m_i']
    chib = np.maximum(np.sqrt(0.5 * (s['chi_up'] ** 2 + s['chi_dn'] ** 2)), 1e-12)
    Dl = s['Delta_nn']
    gsz, _, _ = g_sz_bond_and_derivs_vec(d[nn_i], d[nn_j], m[nn_i], m[nn_j], Dl, chib,
                                         gsxy[nn_i], gsxy[nn_j])
    field = np.zeros(N)
    np.add.at(field, nn_i, 0.5 * sigma_f * J * gsz * m[nn_j])
    np.add.at(field, nn_j, 0.5 * sigma_f * J * gsz * m[nn_i])

    dz_i = g_sz_bond_dn_i_vec(d[nn_i], d[nn_j], m[nn_i], m[nn_j], Dl, chib, sigma_f)
    dz_j = g_sz_bond_dn_i_vec(d[nn_j], d[nn_i], m[nn_j], m[nn_i], Dl, chib, sigma_f)
    np.add.at(field, nn_i, -(J / 4.0) * B * dz_i)
    np.add.at(field, nn_j, -(J / 4.0) * B * dz_j)

    cross = 2.0 * (s['chi_up'] * s['chi_dn'] + Dl ** 2)
    dx_i = g_sxy_dn_vec(d[nn_i], m[nn_i], sigma_f) * gsxy[nn_j]
    dx_j = g_sxy_dn_vec(d[nn_j], m[nn_j], sigma_f) * gsxy[nn_i]
    np.add.at(field, nn_i, -(J / 2.0) * cross * dx_i)
    np.add.at(field, nn_j, -(J / 2.0) * cross * dx_j)

    for sp, cnn, cnnn in ((+1.0, s['chi_up'], s['chin_up']), (-1.0, s['chi_dn'], s['chin_dn'])):
        for (bi, bj, hop, chi) in ((nn_i, nn_j, t, cnn), (nnn_i, nnn_j, tp, cnnn)):
            dgi = g_t_dn_vec(d[bi], m[bi], sp, sigma_f)
            dgj = g_t_dn_vec(d[bj], m[bj], sp, sigma_f)
            gi, gj = g_t(d[bi], m[bi], sp), g_t(d[bj], m[bj], sp)
            np.add.at(field, bi, -hop * dgi * gj * 2 * chi)
            np.add.at(field, bj, -hop * dgj * gi * 2 * chi)
    return field


def build_hamiltonian(s, V_i, kappa=KAPPA):
    d, m = s['delta_i'], s['m_i']
    gsxy, gtu, gtd = _gfac(s)
    chib = np.maximum(np.sqrt(0.5 * (s['chi_up'] ** 2 + s['chi_dn'] ** 2)), 1e-12)
    Dl = s['Delta_nn']
    gsz, dgD, dgc = g_sz_bond_and_derivs_vec(d[nn_i], d[nn_j], m[nn_i], m[nn_j], Dl, chib,
                                             gsxy[nn_i], gsxy[nn_j])
    gxy = gsxy[nn_i] * gsxy[nn_j]
    B = bracket(s)

    ex_hop_up = J * (0.25 * gsz * s['chi_up'] + 0.5 * gxy * s['chi_dn']) + kappa * J * B * dgc * s['chi_up'] / chib
    ex_hop_dn = J * (0.25 * gsz * s['chi_dn'] + 0.5 * gxy * s['chi_up']) + kappa * J * B * dgc * s['chi_dn'] / chib
    ex_pair = J * (0.25 * gsz + 0.5 * gxy) * Dl + kappa * J * B * dgD

    H0up = np.zeros((N, N)); H0dn = np.zeros((N, N))
    for H, g, ex, cnn in ((H0up, gtu, ex_hop_up, None), (H0dn, gtd, ex_hop_dn, None)):
        h1 = -(g[nn_i] * g[nn_j]) * t - ex
        H[nn_i, nn_j] += h1; H[nn_j, nn_i] += h1
        h2 = -(g[nnn_i] * g[nnn_j]) * tp
        H[nnn_i, nnn_j] += h2; H[nnn_j, nnn_i] += h2
    h_up = onsite_field(s, +1.0, gsxy, B)
    h_dn = onsite_field(s, -1.0, gsxy, B)
    np.fill_diagonal(H0up, H0up.diagonal() + h_up + V_i)
    np.fill_diagonal(H0dn, H0dn.diagonal() + h_dn + V_i)

    Dm = np.zeros((N, N))
    Dm[nn_i, nn_j] += nn_sign * ex_pair
    Dm[nn_j, nn_i] += nn_sign * ex_pair
    return H0up, H0dn, Dm


def free_energy_terms(s):
    d, m = s['delta_i'], s['m_i']
    gsxy, gtu, gtd = _gfac(s)
    chib = np.maximum(np.sqrt(0.5 * (s['chi_up'] ** 2 + s['chi_dn'] ** 2)), 1e-12)
    gsz = g_sz_bond_vec(d[nn_i], d[nn_j], m[nn_i], m[nn_j], s['Delta_nn'], chib,
                        gsxy[nn_i], gsxy[nn_j])
    B = bracket(s)
    gxy = gsxy[nn_i] * gsxy[nn_j]
    EJ = -(J / 4) * gsz * B - (J / 2) * gxy * 2 * (s['chi_up'] * s['chi_dn'] + s['Delta_nn'] ** 2)
    Ek = -t * 2 * (gtu[nn_i] * gtu[nn_j] * s['chi_up'] + gtd[nn_i] * gtd[nn_j] * s['chi_dn'])
    En = -tp * 2 * (gtu[nnn_i] * gtu[nnn_j] * s['chin_up'] + gtd[nnn_i] * gtd[nnn_j] * s['chin_dn'])
    return (EJ.sum() + Ek.sum() + En.sum()) / N

def bdg_eigen(H0up, H0dn, Dm, mu, T):
    HB = np.zeros((2 * N, 2 * N))
    HB[:N, :N] = H0up - mu * np.eye(N)
    HB[N:, N:] = -(H0dn - mu * np.eye(N))
    HB[:N, N:] = Dm; HB[N:, :N] = Dm.T
    E, psi = np.linalg.eigh(HB)
    return E, psi


def correlators(E, psi, T, shift=None):
    u, v = psi[:N], psi[N:]
    f = fermi_smeared(E if shift is None else E + shift, T)
    n_up = (u ** 2) @ f
    n_dn = (v ** 2) @ (1.0 - f)
    F = ((1.0 - f)[None, :] * u) @ v.T
    G = (f[None, :] * v) @ u.T
    Db = 0.5 * (F - G)
    Cu = (f[None, :] * u) @ u.T
    Cd = ((1.0 - f)[None, :] * v) @ v.T
    return n_up, n_dn, Db, Cu, Cd


def solve_mu(H0up, H0dn, Dm, mu0, delta_target, T):
    def avg(mu_):
        E, psi = bdg_eigen(H0up, H0dn, Dm, mu_, T)
        n_up, n_dn, *_ = correlators(E, psi, T)
        return np.mean(1 - (n_up + n_dn)) - delta_target
    for w in (0.03, 0.12, 0.5, 1.5):
        lo, hi = mu0 - w, mu0 + w
        flo, fhi = avg(lo), avg(hi)
        if flo * fhi < 0:
            return brentq(avg, lo, hi, xtol=1e-8)
    try:
        return brentq(avg, mu0 - 1.0, mu0 + 1.0, xtol=1e-7)
    except ValueError:
        grid = np.linspace(-8, 8, 41)
        vals = [avg(x) for x in grid]
        k = np.where(np.diff(np.sign(vals)) != 0)[0]
        if len(k) == 0:
            raise RuntimeError("no mu bracket")
        return brentq(avg, grid[k[0]], grid[k[0] + 1], xtol=1e-7)


def solve_mu_fast(H0up, H0dn, Dm, mu0, delta_target, T, tol=2e-8, max_eval=12):
    def exact(mu_):
        E, psi = bdg_eigen(H0up, H0dn, Dm, mu_, T)
        n_up, n_dn, *_ = correlators(E, psi, T)
        return np.mean(1 - (n_up + n_dn)) - delta_target, E, psi

    f0, E, psi = exact(mu0)
    if abs(f0) < tol:
        return mu0, E, psi
    tau = (psi[:N] ** 2).sum(0) - (psi[N:] ** 2).sum(0)

    def pred(dmu):
        n_up, n_dn, *_ = correlators(E, psi, T, shift=-dmu * tau)
        return np.mean(1 - (n_up + n_dn)) - delta_target
    try:
        dmu = brentq(pred, -0.5, 0.5, xtol=1e-10)
    except ValueError:
        dmu = 0.05 * np.sign(-f0)
    pts = [(mu0, f0)]
    mu1 = mu0 + dmu
    for _ in range(max_eval):
        f1, E1, psi1 = exact(mu1)
        if abs(f1) < tol:
            return mu1, E1, psi1
        pts.append((mu1, f1))
        (ma, fa), (mb, fb) = pts[-2], pts[-1]
        if fb == fa:
            mu_next = mb + 0.01
        else:
            mu_next = mb - fb * (mb - ma) / (fb - fa)
        mu1 = mu_next
    return mu1, E1, psi1


def _as_state(seed):
    if isinstance(seed, dict):
        return dict(seed)
    delta_i, m_i, chi_nn, Delta_nn, chi_nnn, mu = seed[:6]
    return dict(delta_i=np.array(delta_i, float), m_i=np.array(m_i, float),
                chi_up=np.array(chi_nn, float), chi_dn=np.array(chi_nn, float),
                Delta_nn=np.array(Delta_nn, float), chin_up=np.array(chi_nnn, float),
                chin_dn=np.array(chi_nnn, float), mu=float(mu))


# ----------------------------------------------------------------- solver ---

def solve_disordered(delta_target, V_i, seed=None, max_iter=300, mix=0.15, tol=1e-4,
                     verbose=False, T=DEFAULT_T, pairing=True, mix_fluct=MIX_FLUCT,
                     kappa=KAPPA, mu_mode="secant", warmup=6, mix_mu=1.0, sym_spin=False):
    s = _as_state(seed) if seed is not None else initial_state(delta_target, pairing)
    mu = s['mu']
    diff = np.inf
    mix_mu = mix if mix_mu is None else mix_mu
    for it in range(max_iter):
        H0up, H0dn, Dm = build_hamiltonian(s, V_i, kappa)
        if not pairing:
            Dm = np.zeros_like(Dm)

        if mu_mode == "secant":
            if it < warmup:
                mu_star = solve_mu(H0up, H0dn, Dm, mu, delta_target, T)
                mu_new = mu_star
            else:
                mu_star, E, psi = solve_mu_fast(H0up, H0dn, Dm, mu, delta_target, T)
                mu_new = (1 - mix_mu) * mu + mix_mu * mu_star
            if it < warmup or mix_mu != 1.0:
                E, psi = bdg_eigen(H0up, H0dn, Dm, mu_new, T)
            shift = None
        elif mu_mode == "bisect" or it < warmup:
            mu_star = solve_mu(H0up, H0dn, Dm, mu, delta_target, T)
            mu_new = mu_star if it < warmup else (1 - mix_mu) * mu + mix_mu * mu_star
            E, psi = bdg_eigen(H0up, H0dn, Dm, mu_new, T)
            shift = None
        else:
            E, psi = bdg_eigen(H0up, H0dn, Dm, mu, T)
            tau = (psi[:N] ** 2).sum(0) - (psi[N:] ** 2).sum(0)

            def avg(dmu):
                n_up, n_dn, *_ = correlators(E, psi, T, shift=-dmu * tau)
                return np.mean(1 - (n_up + n_dn)) - delta_target
            try:
                dmu_star = brentq(avg, -1.0, 1.0, xtol=1e-9)
            except ValueError:
                dmu_star = 0.0
            dmu = mix_mu * dmu_star
            mu_new = mu + dmu
            shift = -dmu * tau
        n_up, n_dn, Db, Cu, Cd = correlators(E, psi, T, shift=shift)

        new = dict(
            delta_i=1 - (n_up + n_dn), m_i=0.5 * (n_up - n_dn),
            chi_up=0.5 * (Cu[nn_i, nn_j] + Cu[nn_j, nn_i]),
            chi_dn=0.5 * (Cd[nn_i, nn_j] + Cd[nn_j, nn_i]),
            Delta_nn=(nn_sign * 0.5 * (Db[nn_i, nn_j] + Db[nn_j, nn_i])) if pairing
            else np.zeros(len(nn_i)),
            chin_up=0.5 * (Cu[nnn_i, nnn_j] + Cu[nnn_j, nnn_i]),
            chin_dn=0.5 * (Cd[nnn_i, nnn_j] + Cd[nnn_j, nnn_i]))
        if sym_spin:       # force chi_{ij,up} = chi_{ij,down} (spin-averaged bond correlators)
            new['chi_up'] = new['chi_dn'] = 0.5 * (new['chi_up'] + new['chi_dn'])
            new['chin_up'] = new['chin_dn'] = 0.5 * (new['chin_up'] + new['chin_dn'])
        diffs = {k: np.max(np.abs(new[k] - s[k])) for k in new}
        diff = max(diffs.values())

        for k in new:
            if k == 'delta_i' and mix_fluct is not None:
                a0, a1 = s[k].mean(), new[k].mean()
                s[k] = (1 - mix) * a0 + mix * a1 + \
                       (1 - mix_fluct) * (s[k] - a0) + mix_fluct * (new[k] - a1)
            else:
                s[k] = (1 - mix) * s[k] + mix * new[k]
        mu = mu_new
        s['mu'] = mu
        if verbose and it % 10 == 0:
            print(f"    it={it:4d} diff={diff:.2e} (worst={max(diffs, key=diffs.get)}) "
                  f"<|m|>={np.mean(np.abs(s['m_i'])):.4f} <|D|>={np.mean(np.abs(s['Delta_nn'])):.4f} "
                  f"delta_i in [{s['delta_i'].min():+.3f},{s['delta_i'].max():.3f}] mu={mu:+.4f}")
        if diff < tol and it > 8:
            break

    out = dict(s)
    out.update(chi_nn=0.5 * (s['chi_up'] + s['chi_dn']), chi_nnn=0.5 * (s['chin_up'] + s['chin_dn']),
               iters=it, diff=diff, T=T, pairing=pairing, delta_target=delta_target, V_i=V_i.copy())
    return out


# -------------------------------------------------------- impurity helpers ---

def place_impurities(n_imp_frac, V, rng):
    V_i = np.zeros(N)
    n_imp = max(1, int(round(n_imp_frac * N)))
    sites = rng.choice(N, size=n_imp, replace=False)
    V_i[sites] = V
    return V_i, sites


def place_impurities_isolated(n_imp_frac, V, rng, min_dist=6, max_tries=5000):
    n_imp = max(1, int(round(n_imp_frac * N)))
    chosen, tries = [], 0
    while len(chosen) < n_imp and tries < max_tries:
        tries += 1
        cand = int(rng.integers(0, N))
        if cand in chosen:
            continue
        cx, cy = cand % Lx, cand // Lx
        ok = True
        for s_ in chosen:
            sx, sy = s_ % Lx, s_ // Lx
            dx = min(abs(cx - sx), Lx - abs(cx - sx)); dy = min(abs(cy - sy), Ly - abs(cy - sy))
            if (dx ** 2 + dy ** 2) ** 0.5 < min_dist:
                ok = False; break
        if ok:
            chosen.append(cand)
    if len(chosen) < n_imp:
        raise RuntimeError("could not place isolated impurities")
    V_i = np.zeros(N); V_i[np.array(chosen)] = V
    return V_i, np.array(chosen)


# ===================================================================== legacy API ===

def afm_field(delta_i, m_i, gsxy_i, chi_nn, Delta_nn, chi_nnn, sigma_field):
    cn_up, cn_dn = chi_nnn if isinstance(chi_nnn, tuple) else (chi_nnn, chi_nnn)
    s = dict(delta_i=delta_i, m_i=m_i, chi_up=chi_nn, chi_dn=chi_nn, Delta_nn=Delta_nn,
             chin_up=np.broadcast_to(cn_up, (len(nnn_i),)), chin_dn=np.broadcast_to(cn_dn, (len(nnn_i),)),
             mu=0.0)
    return onsite_field(s, sigma_field, gsxy_i, bracket(s))


def bdg_expectation(H0up, H0dn, Deltamat, T=DEFAULT_T):
    E, psi = bdg_eigen(H0up, H0dn, Deltamat, 0.0, T)
    n_up, n_dn, Db, Cu, _ = correlators(E, psi, T)
    return n_up, n_dn, Db, Cu

# ================================================================ Type I / Type II drivers ===

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DELTA_SHIFT = 0.045
PAPER_DELTAS_FIG2 = (0.115, 0.125, 0.135, 0.145, 0.155, 0.165)


def stamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _seed_state(delta, m0):
    st = initial_state(delta, True)
    if m0 is not None:
        st['m_i'] = m0 * neel_sign
    return st


def solve_best(d, V_i, m0, max_iter, mix, tol, T, compare=True):
    starts = (m0, 0.0) if compare else (m0,)
    best = None
    for m_start in starts:
        r = solve_disordered(d, V_i, seed=_seed_state(d, m_start), max_iter=max_iter, mix=mix,
                             tol=tol, verbose=True, T=T)
        r['energy'] = free_energy_terms(r)
        r['m_start'] = m_start
        print(f"    start m0={m_start:g}: E/N={r['energy']:.6f} <|m|>={np.mean(np.abs(r['m_i'])):.4f} diff={r['diff']:.2e}")
        if best is None or r['energy'] < best['energy']:
            best = r
    return best


def run_fig2(deltas, max_iter=3000, mix=0.15, seed_rng=1, T=DEFAULT_T, tol=1e-5, V=None, m0=0.12, compare=True):
    rng = np.random.default_rng(seed_rng)
    n_imp_max = max(1, int(round(max(deltas) * N)))
    all_sites = rng.choice(N, size=n_imp_max, replace=False)
    results = []
    for d in deltas:
        n_imp = max(1, int(round(d * N)))
        imp_sites = all_sites[:n_imp]
        V_i = np.zeros(N)
        V_i[imp_sites] = t if V is None else V
        print(f"\n=== delta={d:.3f}  n_imp={n_imp}/{N}  V={V_i.max():g} ===")
        res = solve_best(d, V_i, m0, max_iter, mix, tol, T, compare)
        res.update(imp_sites=imp_sites, delta_target=d, n_imp_frac=n_imp / N, V_i=V_i.copy())
        results.append(res)
        print(f"delta={d:.3f} done: iters={res['iters']} diff={res['diff']:.3e} "
              f"mean|m|={np.mean(np.abs(res['m_i'])):.3f}")
    return results


def run_fig3(panels, V_strong_factor=100.0, max_iter=7000, mix=0.15, seed_rng=3, T=DEFAULT_T,
             tol=1e-5, m0=0.02):
    results = []
    for k, (n_imp_frac, d) in enumerate(panels):
        rng = np.random.default_rng(seed_rng + k)
        V_i, imp_sites = place_impurities(n_imp_frac, V_strong_factor * t, rng)
        print(f"\n=== n_imp={n_imp_frac*100:.0f}%  delta={d:.3f}  ({len(imp_sites)} impurities) ===")
        res = solve_disordered(d, V_i, seed=_seed_state(d, m0), max_iter=max_iter, mix=mix,
                               tol=tol, verbose=True, T=T)
        res.update(imp_sites=imp_sites, delta_target=d, n_imp_frac=n_imp_frac, V_i=V_i.copy())
        results.append(res)
        print(f"n_imp={n_imp_frac*100:.0f}%  delta={d:.3f}  iters={res['iters']} "
              f"diff={res['diff']:.3e}  mean|m|={np.mean(np.abs(res['m_i'])):.3f}  "
              f"max|m|={np.max(np.abs(res['m_i'])):.3f}")
    return results


def plot_panels(results, title):
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.3), constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, res in zip(axes, results):
        im = ax.imshow(res["m_i"].reshape(Ly, Lx), cmap="jet", vmin=-0.12, vmax=0.12, origin="lower")
        ax.scatter(res["imp_sites"] % Lx, res["imp_sites"] // Lx, c="k", s=35)
        converged = "OK" if res["diff"] < 1e-4 else "not converged"
        ax.set_title(rf"$\delta={res['delta_target']:.3f}$"
                     rf"   $\langle|m|\rangle={np.mean(np.abs(res['m_i'])):.3f}$"
                     f"\n{converged} (diff={res['diff']:.1e}, {res['iters'] + 1} it)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label=r"$m_i$")
    fig.suptitle(title)
    return fig


if __name__ == "__main__":
    import argparse
    import pickle
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description="Type I magnetization maps (solver v2)")
    ap.add_argument("--deltas", type=float, nargs="+", default=None,
                    help=f"dopings (default: the paper's {PAPER_DELTAS_FIG2} shifted by +{DELTA_SHIFT})")
    ap.add_argument("--paper-dopings", action="store_true", dest="paper_dopings",
                    help="use the paper's dopings without the shift")
    ap.add_argument("--T", type=float, default=DEFAULT_T)
    ap.add_argument("--iters", type=int, default=3000, help="max iterations per panel")
    ap.add_argument("--mix", type=float, default=0.15)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--m0", type=float, default=0.12, help="initial staggered m")
    ap.add_argument("--no-compare", action="store_true", dest="no_compare")
    ap.add_argument("--save", action="store_true", help="also write one result pickle per panel to ../data")
    a = ap.parse_args()

    if a.deltas is not None:
        deltas = tuple(a.deltas)
    elif a.paper_dopings:
        deltas = PAPER_DELTAS_FIG2
    else:
        deltas = tuple(round(d + DELTA_SHIFT, 4) for d in PAPER_DELTAS_FIG2)

    results = run_fig2(deltas, max_iter=a.iters, mix=a.mix, seed_rng=a.seed, T=a.T, tol=a.tol, m0=a.m0, compare=not a.no_compare)

    shift = "" if (a.deltas is not None or a.paper_dopings) else f", dopings shifted by +{DELTA_SHIFT}"
    fig = plot_panels(results, f"Type I disorder, nested impurity sets, T={a.T}{shift}")
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, f"fig2_type1_maps_d{deltas[0]:g}-{deltas[-1]:g}_n{len(deltas)}"
                             f"_T{a.T:g}_it{a.iters}_s{a.seed}_{stamp()}.png")
    fig.savefig(out, dpi=140)
    print("saved", out)
    if a.save:
        for res in results:
            fname = os.path.join(DATA, f"type1_delta{res['delta_target']:.3f}_T{a.T}.pkl")
            with open(fname, "wb") as f:
                pickle.dump(res, f)
            print("saved", fname)
    plt.show()
