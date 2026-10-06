import os
import sys
import pickle
import argparse
import numpy as np
from scipy.optimize import brentq

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
for _p in (os.path.join(_PROJECT_ROOT, "bdg"), _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from solve_disordered import (N, Lx, Ly, xs, ys, nn_i, nn_j, nnn_i, nnn_j, neel_sign,
                              fermi_smeared, build_hamiltonian)
from continuum import rebuild_bdg_from_result, _v2_state
from wannier import wannier_envelope

GAMMA = 1.5


# ----------------------------------------------------------- 1. uniform host --

def solve_uniform_af(delta, T=0.02, mix=0.3, tol=1e-8, max_iter=600,
                     m0=0.15, chi0=0.18, chin0=0.05, mu0=1.6, verbose=False):
    nb, nnb = len(nn_i), len(nnn_i)
    m, chi_u, chi_d = m0, chi0, chi0
    cn_u, cn_d = np.full(nnb, chin0), np.full(nnb, chin0)
    mu = mu0
    zero_bond = np.zeros(nb)
    for it in range(max_iter):
        state = dict(delta_i=np.full(N, delta), m_i=m * neel_sign, chi_up=np.full(nb, chi_u),
                     chi_dn=np.full(nb, chi_d), Delta_nn=zero_bond, chin_up=cn_u, chin_dn=cn_d, mu=mu)
        H0up, H0dn, _ = build_hamiltonian(state, np.zeros(N))
        eu, pu = np.linalg.eigh(H0up)
        ed, pd = np.linalg.eigh(H0dn)

        def dop(mu_):
            occ = fermi_smeared(eu - mu_, T).sum() + fermi_smeared(ed - mu_, T).sum()
            return 1.0 - occ / N - delta
        try:
            mu = brentq(dop, mu - 0.5, mu + 0.5, xtol=1e-12)
        except ValueError:
            mu = brentq(dop, -10, 10, xtol=1e-12)

        fu, fd = fermi_smeared(eu - mu, T), fermi_smeared(ed - mu, T)
        n_up, n_dn = (pu ** 2) @ fu, (pd ** 2) @ fd
        Cu, Cd = (pu * fu) @ pu.T, (pd * fd) @ pd.T
        m_new = np.mean(neel_sign * 0.5 * (n_up - n_dn))
        cu_new = np.mean(0.5 * (Cu[nn_i, nn_j] + Cu[nn_j, nn_i]))
        cd_new = np.mean(0.5 * (Cd[nn_i, nn_j] + Cd[nn_j, nn_i]))
        cnu_new = 0.5 * (Cu[nnn_i, nnn_j] + Cu[nnn_j, nnn_i])
        cnd_new = 0.5 * (Cd[nnn_i, nnn_j] + Cd[nnn_j, nnn_i])
        diff = max(abs(m_new - m), abs(cu_new - chi_u), abs(cd_new - chi_d),
                   np.max(np.abs(cnu_new - cn_u)), np.max(np.abs(cnd_new - cn_d)))
        m += mix * (m_new - m)
        chi_u += mix * (cu_new - chi_u); chi_d += mix * (cd_new - chi_d)
        cn_u = cn_u + mix * (cnu_new - cn_u); cn_d = cn_d + mix * (cnd_new - cn_d)
        if verbose and it % 10 == 0:
            print(f"    it={it:3d} diff={diff:.2e} m={m:.5f} chi={0.5*(chi_u+chi_d):.5f} mu={mu:.5f}")
        if diff < tol and it > 5:
            break
    chi = 0.5 * (chi_u + chi_d)
    return dict(delta_i=np.full(N, delta), m_i=m * neel_sign, chi_up=np.full(nb, chi_u),
                chi_dn=np.full(nb, chi_d), Delta_nn=zero_bond.copy(), chin_up=cn_u, chin_dn=cn_d,
                chi_nn=np.full(nb, chi), chi_nnn=0.5 * (cn_u + cn_d), mu=mu, iters=it, diff=diff,
                T=T, pairing=False, delta_target=delta, m=m, chi=chi)


# ----------------------------------------------------------- 2. impurity -----

def site_index(x, y):
    return (int(y) % Ly) * Lx + (int(x) % Lx)


def impurity_potential(site, V):
    V_i = np.zeros(N)
    V_i[site] = V
    return V_i


def bdg_eigenpairs(res, V_i):
    _, _, _, E, psi = rebuild_bdg_from_result(res, V_i)
    return E, psi


# ----------------------------------------------------------- 3. Green fn -----

def nambu_green_full(E, psi, z):
    return (psi / (z - E)[None, :]) @ psi.T


def projected_green(E, psi, W, omega, eta):
    Ut, Vt = W @ psi[:N], W @ psi[N:]
    gp = 1.0 / (omega + 1j * eta - E)
    gh = 1.0 / (-omega + 1j * eta - E)
    return (Ut * gp) @ Ut.T, (Vt * gh) @ Vt.T


def ldos_spectra(E, psi, W, omegas, eta):
    omegas = np.asarray(omegas, float)
    Ut2 = (W @ psi[:N]) ** 2
    Vt2 = (W @ psi[N:]) ** 2
    gp = 1.0 / (omegas[None, :] + 1j * eta - E[:, None])
    gh = 1.0 / (-omegas[None, :] + 1j * eta - E[:, None])
    rho_up = -(Ut2 @ gp).imag / np.pi
    rho_dn = -(Vt2 @ gh).imag / np.pi
    return rho_up, rho_dn, rho_up + rho_dn


# ----------------------------------------------------------- 4. Wannier ------

def wannier_matrix(rx, ry, gamma=GAMMA, a=1.0):
    rx = np.atleast_1d(np.asarray(rx, float))
    ry = np.atleast_1d(np.asarray(ry, float))
    dx = rx[:, None] - xs[None, :]
    dy = ry[:, None] - ys[None, :]
    dx -= Lx * np.round(dx / Lx)
    dy -= Ly * np.round(dy / Ly)
    return wannier_envelope(dx, dy, gamma=gamma, a=a)


def site_matrix(sites):
    sites = np.atleast_1d(sites)
    W = np.zeros((len(sites), N))
    W[np.arange(len(sites)), sites] = 1.0
    return W

def host_tb_params(res):
    H0up, H0dn, _ = build_hamiltonian(_v2_state(res), np.zeros(N))
    iA, iB = site_index(0, 0), site_index(1, 0)
    prm = dict(mu=res["mu"])
    for name, H in (("up", H0up), ("dn", H0dn)):
        p = dict(hA=H[iA, iA], hB=H[iB, iB], t1=H[iA, iB],
                 tA=H[iA, site_index(1, 1)], tB=H[iB, site_index(2, 1)])
        # uniformity checks (would fail for a non-uniform result)
        assert abs(H[site_index(3, 3), site_index(3, 4)] - p["t1"]) < 1e-10
        assert abs(H[site_index(2, 2), site_index(2, 2)] - p["hA"]) < 1e-10
        assert abs(H[site_index(2, 1), site_index(2, 1)] - p["hB"]) < 1e-10
        prm[name] = p
    return prm


def _k_grid(Nk):
    k = 2 * np.pi * np.arange(Nk) / Nk
    return np.meshgrid(k, k, indexing="ij")


def host_green_real(prm, spin, z, Nk):
    p = prm[spin]
    KX, KY = _k_grid(Nk)
    cc = np.cos(KX) * np.cos(KY)
    dA = p["hA"] - prm["mu"] + 4 * p["tA"] * cc
    dB = p["hB"] - prm["mu"] + 4 * p["tB"] * cc
    t = 2 * p["t1"] * (np.cos(KX) + np.cos(KY))
    det = (z - dA) * (z - dB) - t * t
    return (np.fft.ifft2((z - dB) / det), np.fft.ifft2((z - dA) / det),
            np.fft.ifft2(t / det))


class _TipGeometry:

    def __init__(self, tips, Nk, alpha0, cells=5, gamma=GAMMA):
        tips = np.asarray(tips, float)
        P = len(tips)
        d1 = np.arange(-cells, cells + 1)
        self.Nk, self.alpha0 = Nk, alpha0
        self.P, self.M = P, len(d1) ** 2
        base = np.floor(tips).astype(int)
        DX, DY = np.meshgrid(d1, d1, indexing="ij")
        self.dx = base[:, 0, None] + DX.ravel()[None, :]      # (P, M) site offsets
        self.dy = base[:, 1, None] + DY.ravel()[None, :]
        self.K = wannier_envelope(tips[:, 0, None] - self.dx, tips[:, 1, None] - self.dy,
                                  gamma=gamma)
        self.alpha = (alpha0 + self.dx + self.dy) % 2          # sublattice of each site
        ddx = self.dx[:, :, None] - self.dx[:, None, :]
        ddy = self.dy[:, :, None] - self.dy[:, None, :]
        self.ix, self.iy = ddx % Nk, ddy % Nk
        self.nix, self.niy = (-ddx) % Nk, (-ddy) % Nk
        a, b = self.alpha[:, :, None], self.alpha[:, None, :]
        self.code = 2 * a + b                                   # 0 AA, 1 AB, 2 BA, 3 BB
        self.i0x, self.i0y = self.dx % Nk, self.dy % Nk         # offset of site i from impurity
        self.n0x, self.n0y = (-self.dx) % Nk, (-self.dy) % Nk


def _gather_G0(geo, gAA, gBB, gAB):
    c = geo.code
    return np.where(c == 0, gAA[geo.ix, geo.iy],
           np.where(c == 3, gBB[geo.ix, geo.iy],
           np.where(c == 1, gAB[geo.ix, geo.iy], gAB[geo.nix, geo.niy])))


def _gather_Gi0(geo, gAA, gBB, gAB):
    if geo.alpha0 == 0:      # impurity on A:  AA for even sites, BA(=AB reversed) for odd
        return np.where(geo.alpha == 0, gAA[geo.i0x, geo.i0y], gAB[geo.n0x, geo.n0y])
    return np.where(geo.alpha == 1, gBB[geo.i0x, geo.i0y], gAB[geo.i0x, geo.i0y])


def infinite_ldos(res, V, tips, omegas, eta, Nk=512, alpha0=0, cells=5, gamma=GAMMA):
    prm = host_tb_params(res)
    geo = _TipGeometry(tips, Nk, alpha0, cells, gamma)
    omegas = np.asarray(omegas, float)
    rho_imp = np.zeros((geo.P, len(omegas)))
    rho_cl = np.zeros_like(rho_imp)
    for n, om in enumerate(omegas):
        for spin in ("up", "dn"):
            gAA, gBB, gAB = host_green_real(prm, spin, om + 1j * eta, Nk)
            G0 = _gather_G0(geo, gAA, gBB, gAB)
            host = np.einsum("pi,pij,pj->p", geo.K, G0, geo.K)
            g00 = (gAA if alpha0 == 0 else gBB)[0, 0]
            T = V / (1.0 - V * g00)
            A = np.einsum("pi,pi->p", geo.K, _gather_Gi0(geo, gAA, gBB, gAB))
            rho_cl[:, n] += -host.imag / np.pi
            rho_imp[:, n] += -(host + T * A * A).imag / np.pi
    return rho_imp, rho_cl


def infinite_map(res, V, omega, eta, half=6, sub=6, Nk=512, alpha0=0, cells=5, gamma=GAMMA):
    prm = host_tb_params(res)
    s1 = (np.arange(sub) + 0.5) / sub - 0.5
    SX, SY = np.meshgrid(s1, s1, indexing="ij")
    tips = np.stack([SX.ravel(), SY.ravel()], 1)
    geos = [_TipGeometry(tips, Nk, (alpha0 + q) % 2, cells, gamma) for q in (0, 1)]
    geo = geos[0]
    R1 = np.arange(-half, half + 1)
    RX, RY = np.meshgrid(R1, R1, indexing="ij")
    RX, RY = RX.ravel(), RY.ravel()
    qR = (RX + RY) % 2
    D = np.arange(Nk)
    par = (D[:, None] + D[None, :]) % 2
    neg = (-D) % Nk
    rho_imp = np.zeros((len(RX), len(tips)))
    rho_cl = np.zeros_like(rho_imp)
    for spin in ("up", "dn"):
        gAA, gBB, gAB = host_green_real(prm, spin, omega + 1j * eta, Nk)
        host_q = np.array([np.einsum("pi,pij,pj->p", geo.K, _gather_G0(g, gAA, gBB, gAB), geo.K)
                           for g in geos])                                  # (2, P)
        if alpha0 == 0:
            GA = np.where(par == 0, gAA, gAB[neg[:, None], neg[None, :]])
            g00 = gAA[0, 0]
        else:
            GA = np.where(par == 0, gBB, gAB)
            g00 = gBB[0, 0]
        T = V / (1.0 - V * g00)
        S = GA[(RX[:, None, None] + geo.dx[None]) % Nk, (RY[:, None, None] + geo.dy[None]) % Nk]
        A = np.einsum("rpm,pm->rp", S, geo.K)
        host = host_q[qR]                                                   # (nR, P)
        rho_cl += -host.imag / np.pi
        rho_imp += -(host + T * A * A).imag / np.pi
    nR = len(R1)

    def to_img(a):
        return a.reshape(nR, nR, sub, sub).transpose(1, 3, 0, 2).reshape(nR * sub, nR * sub)
    ext = [-half - 0.5, half + 0.5, -half - 0.5, half + 0.5]
    return to_img(rho_imp), to_img(rho_cl), ext


# ----------------------------------------------------------- 5. STM ---------

def tip_positions(far_d=20):
    d = far_d
    near = {
        "tip on impurity (0,0)  [zero by symmetry]": (0.0, 0.0),
        "nn (1,0)": (1.0, 0.0),
        "bond centre (0.5,0)": (0.5, 0.0),
        "(2,0)": (2.0, 0.0),
        "(3,0)": (3.0, 0.0),
        "off-axis (2,1)": (2.0, 1.0),
    }
    far = {
        f"far ({d},0), same sublattice": (float(d), 0.0),
        f"far ({d - 1},0), other sublattice": (float(d - 1), 0.0),
        f"far bond centre ({d - 0.5:g},0)": (d - 0.5, 0.0),
        f"far off-axis ({d - 2},{d // 3})": (float(d - 2), float(d // 3)),
    }
    return near, far


def spectra_infinite(positions, res, V, omegas, eta, Nk=512):
    pts = np.array(list(positions.values()), float)
    r_imp, r_cl = infinite_ldos(res, V, pts, omegas, eta, Nk=Nk)
    keys = list(positions.keys())
    return ({k: r_imp[i] for i, k in enumerate(keys)},
            {k: r_cl[i] for i, k in enumerate(keys)})


def spectra_finite(site, positions, E, psi, omegas, eta):
    pts = np.array(list(positions.values()), float) + np.array([xs[site], ys[site]])
    W = wannier_matrix(pts[:, 0], pts[:, 1])
    return ldos_spectra(E, psi, W, omegas, eta)[2]


def find_resonances(s_imp, s_cl, omegas, keys, n_peaks=3):
    sig = sum(np.abs(s_imp[k] - s_cl[k]) for k in keys)
    loc = np.where((sig[1:-1] > sig[:-2]) & (sig[1:-1] >= sig[2:]))[0] + 1
    loc = loc[np.argsort(sig[loc])[::-1]][:n_peaks]
    return omegas[loc]


def ldos_maps(E, psi, window, omegas, eta, pts_per_a=6, gamma=GAMMA):
    x0, x1, y0, y1 = window
    gx = np.linspace(x0, x1, int(round((x1 - x0) * pts_per_a)) + 1)
    gy = np.linspace(y0, y1, int(round((y1 - y0) * pts_per_a)) + 1)
    GX, GY = np.meshgrid(gx, gy)
    W = wannier_matrix(GX.ravel(), GY.ravel(), gamma)
    _, _, tot = ldos_spectra(E, psi, W, np.asarray(omegas, float), eta)
    return GX, GY, tot.T.reshape(len(omegas), *GX.shape)


# ----------------------------------------------------------- plots ----------

def _panel_grid(n):
    import matplotlib.pyplot as plt
    ncol = 3 if n > 4 else 2
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow), sharex=True)
    return fig, np.atleast_1d(axs).ravel()


def plot_spectra(title, omegas, s_imp, s_cl, show_rel=False, save=None):
    fig, axs = _panel_grid(len(s_imp))
    for ax, k in zip(axs, s_imp):
        ax.plot(omegas, s_imp[k], "k", label="with impurity")
        ax.plot(omegas, s_cl[k], "r:", label="clean AF")
        t = k
        if show_rel:
            rel = np.max(np.abs(s_imp[k] - s_cl[k])) / np.max(s_cl[k])
            t += f"\nmax|drho|/max rho_clean = {rel:.1e}"
        ax.set_title(t, fontsize=9)
        ax.set_ylim(bottom=0)
    for ax in axs[-3:]:
        ax.set_xlabel(r"bias $\omega/t$")
    axs[0].legend(fontsize=8)
    axs[0].set_ylabel(r"$\rho(\mathbf{r},\omega)$ (arb.)")
    fig.suptitle(title)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140)
    return fig


def plot_maps(R, R0, extent, omegas_map, save=None):
    import matplotlib.pyplot as plt
    n = len(omegas_map)
    fig, axs = plt.subplots(2, n, figsize=(4.4 * n, 8.6), squeeze=False)
    for c, om in enumerate(omegas_map):
        im = axs[0, c].imshow(R[c], extent=extent, origin="lower", cmap="inferno")
        axs[0, c].set_title(rf"$\rho(\mathbf{{r}},\omega={om:+.3f}t)$")
        fig.colorbar(im, ax=axs[0, c], shrink=0.75)
        dr = R[c] - R0[c]
        v = np.abs(dr).max()
        im = axs[1, c].imshow(dr, extent=extent, origin="lower", cmap="bwr", vmin=-v, vmax=v)
        axs[1, c].set_title(r"$\rho - \rho_{clean}$")
        fig.colorbar(im, ax=axs[1, c], shrink=0.75)
        for ax in axs[:, c]:
            ax.plot(0, 0, "c+", ms=10)
            ax.set_xlabel("x/a  (relative to impurity)"); ax.set_ylabel("y/a")
    fig.suptitle("Wannier-filtered LDOS maps (STM conductance maps), impurity marked +")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140)
    return fig


def plot_decay(res, V, omega, eta, Nk, site=None, E=None, psi=None, E0=None, psi0=None,
               dmax=60, save=None):
    import matplotlib.pyplot as plt
    d = np.arange(1, dmax + 1)
    tips = np.stack([d, np.zeros_like(d)], 1).astype(float)
    r_imp, r_cl = infinite_ldos(res, V, tips, [omega], eta, Nk=Nk)
    rel = np.abs(r_imp[:, 0] - r_cl[:, 0]) / r_cl[:, 0]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ev = d % 2 == 0
    ax.semilogy(d[ev], rel[ev], "o", color="tab:blue", label="infinite lattice, same sublattice")
    ax.semilogy(d[~ev], rel[~ev], "s", color="tab:orange", label="infinite lattice, other sublattice")
    if E is not None:
        df = d[d <= Lx // 2]
        tf = np.stack([df, np.zeros_like(df)], 1).astype(float)
        a = spectra_finite(site, {i: tuple(p) for i, p in enumerate(tf)}, E, psi, np.array([omega]), eta)
        b = spectra_finite(site, {i: tuple(p) for i, p in enumerate(tf)}, E0, psi0, np.array([omega]), eta)
        ax.semilogy(df, np.abs(a[:, 0] - b[:, 0]) / b[:, 0], "k+", ms=9,
                    label=f"{Lx}x{Ly} periodic lattice")
    ax.set_xlabel("distance of tip from impurity along x [a]")
    ax.set_ylabel(r"$|\delta\rho|/\rho_{clean}$")
    ax.set_title(rf"STM signal vs distance at $\omega={omega:+.3f}t$, eta={eta}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140)
    return fig, d, rel


# ----------------------------------------------------------- driver ---------

def _host_path(delta, T):
    return os.path.join(_DATA_DIR, f"af_host_delta{delta:.3f}_T{T}.pkl")


def cmd_host(args):
    os.makedirs(_DATA_DIR, exist_ok=True)
    res = solve_uniform_af(args.delta, T=args.T, verbose=True)
    print(f"host: delta={args.delta}  m={res['m']:.5f}  chi_nn={res['chi']:.5f}  "
          f"mu={res['mu']:.5f}  iters={res['iters']}  diff={res['diff']:.1e}")
    with open(_host_path(args.delta, args.T), "wb") as f:
        pickle.dump(res, f)
    return res


def cmd_stm(args, res=None):
    import matplotlib.pyplot as plt
    os.makedirs(_DATA_DIR, exist_ok=True)
    if res is None:
        with open(_host_path(args.delta, args.T), "rb") as f:
            res = pickle.load(f)
    eta, V, Nk = args.eta, args.V, args.Nk

    site = site_index(Lx // 2, Ly // 2)
    E0, psi0 = bdg_eigenpairs(res, np.zeros(N))
    if args.pkl:
        with open(args.pkl, "rb") as f:
            res_sc = pickle.load(f)
        site = int(res_sc["imp_sites"][0])
        E, psi = bdg_eigenpairs(res_sc, res_sc["V_i"])
        print("finite-lattice maps use the self-consistent run", os.path.basename(args.pkl))
    else:
        E, psi = bdg_eigenpairs(res, impurity_potential(site, V))
    print(f"host: delta={res['delta_target']}  m={res['m']:.4f}  mu={res['mu']:.4f}")
    print(f"impurity V={V}t on site ({xs[site]},{ys[site]}); eta={eta}; infinite-lattice Nk={Nk}")

    near, far = tip_positions(args.far)
    omegas = np.linspace(-args.wmax, args.wmax, args.n_omega)
    s_imp, s_cl = spectra_infinite({**near, **far}, res, V, omegas, eta, Nk=Nk)
    print("max|rho_imp - rho_clean| / max rho_clean per tip (infinite lattice):")
    for k in s_imp:
        print(f"  {k:44s} {np.abs(s_imp[k] - s_cl[k]).max() / s_cl[k].max():.2e}")

    # cross-check: finite periodic lattice, near tips
    f_imp = spectra_finite(site, near, E, psi, omegas, eta)
    f_cl = spectra_finite(site, near, E0, psi0, omegas, eta)
    for i, k in enumerate(near):
        if "zero" in k:
            continue
        dr_inf = s_imp[k] - s_cl[k]
        print(f"  finite({Lx}x{Ly}) vs infinite d-rho at {k:20s}: "
              f"{np.abs((f_imp[i] - f_cl[i]) - dr_inf).max() / np.abs(dr_inf).max():.2e}")

    keys_res = [k for k in near if k.startswith(("nn", "bond", "(2"))]
    pk = find_resonances(s_imp, s_cl, omegas, keys_res)
    print("strongest impurity-induced peaks (omega/t):", np.round(pk, 3))
    om0 = args.omega[0] if args.omega else float(pk[0])

    import time
    tag = (f"V{V:g}_d{args.delta:.3f}_T{args.T:g}_eta{eta:g}_far{args.far}_"
           f"{time.strftime('%Y%m%d-%H%M%S')}")
    P = lambda name: os.path.join(_DATA_DIR, f"af_stm_{name}_{tag}.png")
    plot_spectra("STM spectra NEAR the impurity (infinite lattice, T-matrix)", omegas,
                 {k: s_imp[k] for k in near}, {k: s_cl[k] for k in near}, save=P("near"))
    plot_spectra(f"STM spectra FAR from the impurity (distance ~{args.far}a)", omegas,
                 {k: s_imp[k] for k in far}, {k: s_cl[k] for k in far}, show_rel=True,
                 save=P("far"))
    om_map = list(args.omega) if args.omega else [float(x) for x in pk[:2]] + [0.5 * args.wmax]
    if args.pkl:       # self-consistent impurity: finite lattice maps (no T-matrix available)
        x0, y0 = xs[site], ys[site]
        h = float(args.half)
        win = (x0 - h, x0 + h, y0 - h, y0 + h)
        R = ldos_maps(E, psi, win, om_map, eta)[2]
        R0 = ldos_maps(E0, psi0, win, om_map, eta)[2]
        plot_maps(list(R), list(R0), [-h, h, -h, h], om_map, save=P("maps"))
    else:
        out = [infinite_map(res, V, om, eta, half=args.half, Nk=Nk) for om in om_map]
        plot_maps([o[0] for o in out], [o[1] for o in out], out[0][2], om_map, save=P("maps"))
    _, d, rel = plot_decay(res, V, om0, eta, Nk, site=site, E=E, psi=psi, E0=E0, psi0=psi0,
                           save=P("decay"))
    for dd in (5, 10, 20, 40):
        if dd <= d.max():
            print(f"  |drho|/rho_clean at ({dd},0), omega={om0:+.3f}: {rel[dd - 1]:.2e}")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pure-AF + single impurity: GF / STM")
    ap.add_argument("mode", choices=["host", "stm", "all"])
    ap.add_argument("--delta", type=float, default=0.12)
    ap.add_argument("--V", type=float, default=1.0, help="impurity potential in units of t")
    ap.add_argument("--T", type=float, default=0.02, help="smearing in the host self-consistency")
    ap.add_argument("--eta", type=float, default=0.02, help="Lorentzian broadening")
    ap.add_argument("--wmax", type=float, default=1.0)
    ap.add_argument("--n_omega", type=int, default=301)
    ap.add_argument("--Nk", type=int, default=512, help="k-mesh of the infinite-lattice GF")
    ap.add_argument("--half", type=int, default=6, help="half-size of the map window [a]")
    ap.add_argument("--far", type=int, default=20, help="distance of the 'far' tips [a]")
    ap.add_argument("--omega", type=float, nargs="*", help="bias values for the maps/decay")
    ap.add_argument("--pkl", type=str, default=None,
                    help="self-consistent solve_disordered pkl (with V_i, imp_sites): used for the "
                         "finite-lattice maps instead of the frozen-host impurity")
    a = ap.parse_args()
    if a.mode == "host":
        cmd_host(a)
    elif a.mode == "stm":
        cmd_stm(a)
    else:
        cmd_stm(a, res=cmd_host(a))
