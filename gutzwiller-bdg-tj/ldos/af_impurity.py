import os
import sys
import pickle
import argparse
import numpy as np
from scipy.optimize import brentq
from scipy.ndimage import label

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


def solve_uniform_af(delta, T=0.02, mix=0.3, tol=1e-8, max_iter=600,
                     m0=0.15, chi0=0.18, chin0=0.05, mu0=1.6, verbose=False, m_fixed=None):
    delta_t = delta
    if m_fixed is not None:
        delta = max(delta, 0.002)
        m0 = m_fixed
    nb, nnb = len(nn_i), len(nnn_i)
    m, chi_u, chi_d = m0, chi0, chi0
    cn_u, cn_d = np.full(nnb, chin0), np.full(nnb, chin0)
    mu = mu0
    zero_bond = np.zeros(nb)
    gap_eig = np.nan
    m_new = m
    for it in range(max_iter):
        state = dict(delta_i=np.full(N, delta), m_i=m * neel_sign, chi_up=np.full(nb, chi_u),
                     chi_dn=np.full(nb, chi_d), Delta_nn=zero_bond, chin_up=cn_u, chin_dn=cn_d, mu=mu)
        H0up, H0dn, _ = build_hamiltonian(state, np.zeros(N))
        eu, pu = np.linalg.eigh(H0up)
        ed, pd = np.linalg.eigh(H0dn)

        def dop(mu_):
            occ = fermi_smeared(eu - mu_, T).sum() + fermi_smeared(ed - mu_, T).sum()
            return 1.0 - occ / N - delta_t
        try:
            mu = brentq(dop, mu - 0.5, mu + 0.5, xtol=1e-12)
        except ValueError:
            lo_, hi_ = min(eu.min(), ed.min()) - 1.0, max(eu.max(), ed.max()) + 1.0
            mu = brentq(dop, lo_, hi_, xtol=1e-12)
        if m_fixed is not None:
            Es = np.sort(np.concatenate([eu, ed]))
            n_occ = int(round((1.0 - delta_t) * N))
            if abs((1.0 - delta_t) * N - n_occ) < 1e-9 and 0 < n_occ < 2 * N:
                gap_eig = Es[n_occ] - Es[n_occ - 1]
                if gap_eig > 4 * T:
                    mu = 0.5 * (Es[n_occ] + Es[n_occ - 1])

        fu, fd = fermi_smeared(eu - mu, T), fermi_smeared(ed - mu, T)
        n_up, n_dn = (pu ** 2) @ fu, (pd ** 2) @ fd
        Cu, Cd = (pu * fu) @ pu.T, (pd * fd) @ pd.T
        m_new = np.mean(neel_sign * 0.5 * (n_up - n_dn))
        cu_new = np.mean(0.5 * (Cu[nn_i, nn_j] + Cu[nn_j, nn_i]))
        cd_new = np.mean(0.5 * (Cd[nn_i, nn_j] + Cd[nn_j, nn_i]))
        cnu_new = 0.5 * (Cu[nnn_i, nnn_j] + Cu[nnn_j, nnn_i])
        cnd_new = 0.5 * (Cd[nnn_i, nnn_j] + Cd[nnn_j, nnn_i])
        dm = 0.0 if m_fixed is not None else abs(m_new - m)
        diff = max(dm, abs(cu_new - chi_u), abs(cd_new - chi_d),
                   np.max(np.abs(cnu_new - cn_u)), np.max(np.abs(cnd_new - cn_d)))
        if m_fixed is None:
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
                T=T, pairing=False, delta_target=delta_t, m=m, chi=chi, m_out=m_new, gap_eig=gap_eig)


def site_index(x, y):
    return (int(y) % Ly) * Lx + (int(x) % Lx)


def impurity_potential(site, V):
    V_i = np.zeros(N)
    V_i[site] = V
    return V_i


def bdg_eigenpairs(res, V_i):
    _, _, _, E, psi = rebuild_bdg_from_result(res, V_i)
    return E, psi


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
        self.dx = base[:, 0, None] + DX.ravel()[None, :]
        self.dy = base[:, 1, None] + DY.ravel()[None, :]
        self.K = wannier_envelope(tips[:, 0, None] - self.dx, tips[:, 1, None] - self.dy,
                                  gamma=gamma)
        self.alpha = (alpha0 + self.dx + self.dy) % 2
        ddx = self.dx[:, :, None] - self.dx[:, None, :]
        ddy = self.dy[:, :, None] - self.dy[:, None, :]
        self.ix, self.iy = ddx % Nk, ddy % Nk
        self.nix, self.niy = (-ddx) % Nk, (-ddy) % Nk
        a, b = self.alpha[:, :, None], self.alpha[:, None, :]
        self.code = 2 * a + b
        self.i0x, self.i0y = self.dx % Nk, self.dy % Nk
        self.n0x, self.n0y = (-self.dx) % Nk, (-self.dy) % Nk


def _gather_G0(geo, gAA, gBB, gAB):
    c = geo.code
    return np.where(c == 0, gAA[geo.ix, geo.iy],
           np.where(c == 3, gBB[geo.ix, geo.iy],
           np.where(c == 1, gAB[geo.ix, geo.iy], gAB[geo.nix, geo.niy])))


def _gather_Gi0(geo, gAA, gBB, gAB):
    if geo.alpha0 == 0:
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


def infinite_ldos_multiV(res, Vs, tips, omegas, eta, Nk=512, alpha0=0, cells=5, gamma=GAMMA):
    prm = host_tb_params(res)
    geo = _TipGeometry(tips, Nk, alpha0, cells, gamma)
    Vs = np.asarray(Vs, float)
    omegas = np.asarray(omegas, float)
    rho_imp = np.zeros((len(Vs), geo.P, len(omegas)))
    rho_cl = np.zeros((geo.P, len(omegas)))
    for n, om in enumerate(omegas):
        for spin in ("up", "dn"):
            gAA, gBB, gAB = host_green_real(prm, spin, om + 1j * eta, Nk)
            G0 = _gather_G0(geo, gAA, gBB, gAB)
            host = np.einsum("pi,pij,pj->p", geo.K, G0, geo.K)
            g00 = (gAA if alpha0 == 0 else gBB)[0, 0]
            A = np.einsum("pi,pi->p", geo.K, _gather_Gi0(geo, gAA, gBB, gAB))
            Tm = Vs / (1.0 - Vs * g00)
            rho_cl[:, n] += -host.imag / np.pi
            rho_imp[:, :, n] += -(host[None, :] + Tm[:, None] * A[None, :] ** 2).imag / np.pi
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
                           for g in geos])
        if alpha0 == 0:
            GA = np.where(par == 0, gAA, gAB[neg[:, None], neg[None, :]])
            g00 = gAA[0, 0]
        else:
            GA = np.where(par == 0, gBB, gAB)
            g00 = gBB[0, 0]
        T = V / (1.0 - V * g00)
        S = GA[(RX[:, None, None] + geo.dx[None]) % Nk, (RY[:, None, None] + geo.dy[None]) % Nk]
        A = np.einsum("rpm,pm->rp", S, geo.K)
        host = host_q[qR]
        rho_cl += -host.imag / np.pi
        rho_imp += -(host + T * A * A).imag / np.pi
    nR = len(R1)

    def to_img(a):
        return a.reshape(nR, nR, sub, sub).transpose(1, 3, 0, 2).reshape(nR * sub, nR * sub)
    ext = [-half - 0.5, half + 0.5, -half - 0.5, half + 0.5]
    return to_img(rho_imp), to_img(rho_cl), ext


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


def g00_mean(prm, spin, z, Nk, alpha0=0):
    p = prm[spin]
    KX, KY = _k_grid(Nk)
    cc = np.cos(KX) * np.cos(KY)
    dA = p["hA"] - prm["mu"] + 4 * p["tA"] * cc
    dB = p["hB"] - prm["mu"] + 4 * p["tB"] * cc
    t1 = 2 * p["t1"] * (np.cos(KX) + np.cos(KY))
    det = (z - dA) * (z - dB) - t1 * t1
    return np.mean(((z - dB) if alpha0 == 0 else (z - dA)) / det)


def pole_scan(res, Vs, wlim, Nk=256, eta=5e-4, n=1600, n_fine=3200, wfine=0.8, tol=0.3, alpha0=0):
    prm = host_tb_params(res)
    om = np.unique(np.concatenate([np.linspace(-wlim, wlim, n), np.linspace(-wfine, wfine, n_fine)]))
    G = {sp: np.array([g00_mean(prm, sp, w + 1j * eta, Nk, alpha0) for w in om]) for sp in ("up", "dn")}
    free_both = (np.abs(G["up"].imag) < tol) & (np.abs(G["dn"].imag) < tol)
    i0 = np.argmin(np.abs(om))
    glo = ghi = 0.0
    if free_both[i0]:
        l = r = i0
        while l > 0 and free_both[l - 1]:
            l -= 1
        while r < len(om) - 1 and free_both[r + 1]:
            r += 1
        glo, ghi = om[l], om[r]
    out = {float(V): [] for V in Vs}
    for spin in ("up", "dn"):
        gr = G[spin].real
        free = np.abs(G[spin].imag) < tol
        lab, _ = label(free)
        keep = {lab[0], lab[-1], lab[i0]} - {0}
        free = np.isin(lab, list(keep))
        for V in Vs:
            f = 1.0 - V * gr
            idx = np.where((f[:-1] * f[1:] < 0) & free[:-1] & free[1:])[0]
            for i in idx:
                w0 = om[i] - f[i] * (om[i + 1] - om[i]) / (f[i + 1] - f[i])
                out[float(V)].append((float(w0), spin))
    for V in out:
        out[V].sort()
    return out, (glo, ghi)


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
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140)
    return fig


def plot_maps(R, R0, extent, omegas_map, save=None, info=""):
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
    fig.suptitle("Wannier-filtered LDOS maps (STM conductance maps), impurity marked +\n" + info, fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140)
    return fig


def plot_decay(res, V, omega, eta, Nk, site=None, E=None, psi=None, E0=None, psi0=None, info="",
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
    ax.set_title(rf"STM signal vs distance at $\omega={omega:+.3f}t$" + "\n" + info, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140)
    return fig, d, rel


def _host_path(delta, T, m=None):
    mt = "" if m is None else f"_m{m:g}"
    return os.path.join(_DATA_DIR, f"af_host_delta{delta:.3f}_T{T}{mt}.pkl")


def cmd_host(args):
    os.makedirs(_DATA_DIR, exist_ok=True)
    res = solve_uniform_af(args.delta, T=args.T, verbose=True, m_fixed=args.m)
    print(f"host: delta={args.delta}  m={res['m']:.5f}  chi_nn={res['chi']:.5f}  "
          f"mu={res['mu']:.5f}  iters={res['iters']}  diff={res['diff']:.1e}")
    with open(_host_path(args.delta, args.T, args.m), "wb") as f:
        pickle.dump(res, f)
    return res


def cmd_stm(args, res=None):
    import matplotlib.pyplot as plt
    os.makedirs(_DATA_DIR, exist_ok=True)
    if res is None:
        with open(_host_path(args.delta, args.T, args.m), "rb") as f:
            res = pickle.load(f)
    eta, V, Nk = args.eta, args.V, args.Nk

    site = site_index(Lx // 2, Ly // 2)
    if not args.only_ldos:
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

    if not args.only_ldos:
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

    pls, (glo, ghi) = pole_scan(res, [V], wlim=V + 3.0, Nk=min(Nk, 256))
    pl = pls[float(V)]
    ing = [w for w, _ in pl if glo < w < ghi] if ghi > glo else []
    outp = [w for w, _ in pl if not (ghi > glo and glo < w < ghi)]
    fm = lambda L: " ".join(f"{w:+.3f}" for w in L) if L else "-"
    print(f"gap at mu from Green function: [{glo:.3f}, {ghi:.3f}]" if ghi > glo else "no gap at mu (metal)")
    print(f"T-matrix poles V={V:g}: in gap: {fm(ing)} | outside bands: {fm(outp)}")

    import time
    tag = (f"V{V:g}_d{args.delta:.3f}_T{args.T:g}_eta{eta:g}_far{args.far}_"
           f"{time.strftime('%Y%m%d-%H%M%S')}")
    tag = f"m{res['m']:.3f}_" + tag
    mt = (rf"$m={res['m']:.3f}$, $\delta={res['delta_target']:g}$, $V={V:g}t$, "
          rf"$T={args.T:g}$, $\eta={eta:g}$, $N_k={Nk}$")
    P = lambda name: os.path.join(_DATA_DIR, f"af_stm_{name}_{tag}.png")
    plot_spectra(f"STM spectra NEAR the impurity (T-matrix)\n{mt}", omegas,
                 {k: s_imp[k] for k in near}, {k: s_cl[k] for k in near}, save=P("near"))
    plot_spectra(f"STM spectra FAR from the impurity (~{args.far}a)\n{mt}", omegas,
                 {k: s_imp[k] for k in far}, {k: s_cl[k] for k in far}, show_rel=True,
                 save=P("far"))
    if args.only_ldos:
        plt.close("all")
        return
    om_map = list(args.omega) if args.omega else [float(x) for x in pk[:2]] + [0.5 * args.wmax]
    if args.pkl:
        x0, y0 = xs[site], ys[site]
        h = float(args.half)
        win = (x0 - h, x0 + h, y0 - h, y0 + h)
        R = ldos_maps(E, psi, win, om_map, eta)[2]
        R0 = ldos_maps(E0, psi0, win, om_map, eta)[2]
        plot_maps(list(R), list(R0), [-h, h, -h, h], om_map, save=P("maps"), info=mt)
    else:
        out = [infinite_map(res, V, om, eta, half=args.half, Nk=Nk) for om in om_map]
        plot_maps([o[0] for o in out], [o[1] for o in out], out[0][2], om_map, save=P("maps"), info=mt)
    _, d, rel = plot_decay(res, V, om0, eta, Nk, site=site, E=E, psi=psi, E0=E0, psi0=psi0, info=mt,
                           save=P("decay"))
    for dd in (5, 10, 20, 40):
        if dd <= d.max():
            print(f"  |drho|/rho_clean at ({dd},0), omega={om0:+.3f}: {rel[dd - 1]:.2e}")
    plt.show()


def cmd_mscan(args):
    import matplotlib.pyplot as plt
    os.makedirs(_DATA_DIR, exist_ok=True)
    ms = args.ms if args.ms else [round(x, 3) for x in np.arange(0.05, 0.451, 0.05)]
    omegas = np.linspace(-args.wmax, args.wmax, args.n_omega)
    rows = []
    print(f"m scan at delta={args.delta}  (m_out = m from one self-consistency step at fixed input m)")
    print(f"{'m':>7} {'m_out':>8} {'m_out-m':>9} {'gap_eig':>8} {'rho(0)/max':>11} {'mu':>8}")
    for m in ms:
        res = solve_uniform_af(args.delta, T=args.T, m_fixed=m)
        prm = host_tb_params(res)
        rho = np.zeros(len(omegas))
        for n, om in enumerate(omegas):
            for spin in ("up", "dn"):
                gAA, gBB, _ = host_green_real(prm, spin, om + 1j * args.eta, args.Nk)
                rho[n] += -(gAA[0, 0] + gBB[0, 0]).imag / np.pi
        r0 = rho[np.argmin(np.abs(omegas))] / rho.max()
        rows.append((m, res["m_out"], res["gap_eig"], r0, res["mu"]))
        print(f"{m:7.3f} {res['m_out']:8.4f} {res['m_out'] - m:+9.4f} {res['gap_eig']:8.3f} {r0:11.3e} {res['mu']:8.4f}")
    arr = np.array(rows)
    f = arr[:, 1] - arr[:, 0]
    roots = []
    for i in range(len(ms) - 1):
        if f[i] * f[i + 1] < 0:
            g = lambda x: solve_uniform_af(args.delta, T=args.T, m_fixed=x)["m_out"] - x
            roots.append(brentq(g, ms[i], ms[i + 1], xtol=1e-4))
    for mstar in roots:
        res = solve_uniform_af(args.delta, T=args.T, m_fixed=mstar)
        print(f"\nself-consistent m* = {mstar:.4f}   gap = {res['gap_eig']:.3f}t   mu = {res['mu']:.4f}")
    if not roots:
        print("\nno sign change of m_out - m in the scanned range")
    import time
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(arr[:, 0], arr[:, 1], "o-", label=r"$m_{out}(m_{in})$")
    axs[0].plot(arr[:, 0], arr[:, 0], "k:", label="$m_{out}=m_{in}$")
    for mstar in roots:
        axs[0].axvline(mstar, color="r", lw=0.8)
    axs[0].set_xlabel(r"$m_{in}$"); axs[0].legend()
    if np.all(np.isnan(arr[:, 2])):
        axs[1].plot(arr[:, 0], arr[:, 3], "o-")
        axs[1].set_ylabel(r"$\rho(0)/\rho_{max}$ (no gap: metal)")
    else:
        axs[1].plot(arr[:, 0], arr[:, 2], "o-")
        axs[1].set_ylabel("gap at mu / t")
    axs[1].set_xlabel(r"$m_{in}$")
    for mstar in roots:
        axs[1].axvline(mstar, color="r", lw=0.8)
    ttl = rf"$\delta={args.delta:g}$, $T={args.T:g}$, $\eta={args.eta:g}$, $N_k={args.Nk}$"
    if roots:
        ttl += rf", $m^*={roots[0]:.3f}$"
    fig.suptitle(ttl, fontsize=10)
    fig.tight_layout()
    fn = os.path.join(_DATA_DIR, f"af_mscan_d{args.delta:.3f}_{time.strftime('%Y%m%d-%H%M%S')}.png")
    fig.savefig(fn, dpi=140)
    print("saved", fn)


def cmd_stack(args):
    import matplotlib.pyplot as plt
    import time
    os.makedirs(_DATA_DIR, exist_ok=True)
    path = _host_path(args.delta, args.T, args.m)
    if os.path.exists(path):
        with open(path, "rb") as f:
            res = pickle.load(f)
    else:
        res = solve_uniform_af(args.delta, T=args.T, m_fixed=args.m)
        with open(path, "wb") as f:
            pickle.dump(res, f)
    Vs = list(args.Vs) if args.Vs else [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    tip = np.array([args.tip], float)
    omegas = np.linspace(-args.wmax, args.wmax, args.n_omega)
    r_imp, r_cl = infinite_ldos_multiV(res, Vs, tip, omegas, args.eta, Nk=args.Nk)
    pls, (glo, ghi) = pole_scan(res, Vs, wlim=max(Vs) + 3.0, Nk=min(args.Nk, 256))
    step = 0.85 * r_imp.max()
    fig, ax = plt.subplots(figsize=(6.4, 1.4 + 1.1 * len(Vs)))
    if ghi > glo:
        ax.axvspan(glo, ghi, color="0.92", zorder=0)
    ax.plot(omegas, r_cl[0], "r:", lw=1.3)
    ax.text(omegas[-1], 0, "  clean", fontsize=8, va="bottom", color="r")
    for i, V in enumerate(Vs):
        off = (i + 1) * step
        y = r_imp[i, 0] + off
        ax.plot(omegas, y, "k", lw=1.1)
        ax.plot(omegas, np.full_like(omegas, off), color="0.8", lw=0.5, zorder=0)
        ax.text(omegas[-1], off, f"  V={V:g}t", fontsize=8, va="bottom")
    ax.set_xlim(-args.wmax, args.wmax)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.set_xlabel(r"bias $\omega/t$")
    ax.set_title(rf"STM spectrum at tip ({args.tip[0]:g},{args.tip[1]:g})" + "\n"
                 + rf"$m={res['m']:.3f}$, $\delta={res['delta_target']:g}$, $T={args.T:g}$, "
                 rf"$\eta={args.eta:g}$, $N_k={args.Nk}$", fontsize=9)
    fig.tight_layout()
    fn = os.path.join(_DATA_DIR, f"af_stack_m{res['m']:.3f}_d{args.delta:.3f}_T{args.T:g}_eta{args.eta:g}_"
                                 f"Nk{args.Nk}_{time.strftime('%Y%m%d-%H%M%S')}.png")
    fig.savefig(fn, dpi=140)
    print("saved", fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pure-AF + single impurity: GF / STM")
    ap.add_argument("mode", choices=["host", "stm", "all", "mscan", "stack"])
    ap.add_argument("--delta", type=float, default=0.12)
    ap.add_argument("--V", type=float, default=1.0, help="impurity potential in units of t")
    ap.add_argument("--m", type=float, default=None,
                    help="fixed staggered moment: insulating AF host (default: self-consistent m)")
    ap.add_argument("--ms", type=float, nargs="+", default=None,
                    help="mscan: input m values (default 0.05 ... 0.45)")
    ap.add_argument("--tip", type=float, nargs=2, default=[1.0, 0.0],
                    help="stack: tip position (x y) relative to the impurity")
    ap.add_argument("--only_ldos", action="store_true",
                    help="stm: only the LDOS spectra near and far from the impurity")
    ap.add_argument("--Vs", type=float, nargs="+", default=None,
                    help="several impurity potentials; host is solved once")
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
    elif a.mode == "mscan":
        cmd_mscan(a)
    elif a.mode == "stack":
        cmd_stack(a)
    else:
        host = cmd_host(a) if a.mode == "all" else None
        for V in (a.Vs if a.Vs else [a.V]):
            a.V = V
            print(f"\n######## V = {V:g} ########")
            cmd_stm(a, res=host)
