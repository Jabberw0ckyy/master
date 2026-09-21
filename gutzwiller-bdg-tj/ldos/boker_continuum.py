import os
import numpy as np
import matplotlib.pyplot as plt
from wannier import wannier_envelope, plot_wannier

T1, T2, T3, MU = -590.8, 96.2, -130.6, 156.6
DELTA0 = 31.0
GAMMA = 1.5
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(os.path.dirname(_THIS_DIR), "data")


def xi_k(kx, ky):
    return (MU + 0.5 * T1 * (np.cos(kx) + np.cos(ky)) + T2 * np.cos(kx) * np.cos(ky)
            + 0.5 * T3 * (np.cos(2 * kx) + np.cos(2 * ky)))


def gap_k(kx, ky, kind="d"):
    g = 0.5 * DELTA0 * (np.cos(kx) - np.cos(ky))
    return g if kind == "d" else np.abs(g)


def k_mesh(Nk):
    k = 2 * np.pi * np.arange(Nk) / Nk
    return np.meshgrid(k, k, indexing="ij")


def nambu_G(z, KX, KY, kind="d"):
    x, d = xi_k(KX, KY), gap_k(KX, KY, kind)
    D = z * z - x * x - d * d
    return (z + x) / D, d / D, (z - x) / D


def w_k(r, KX, KY, cells=6, gamma=GAMMA):
    rx, ry = r
    Rx = np.arange(int(np.floor(rx)) - cells, int(np.floor(rx)) + cells + 2)
    Ry = np.arange(int(np.floor(ry)) - cells, int(np.floor(ry)) + cells + 2)
    W = wannier_envelope(rx - Rx[:, None], ry - Ry[None, :], gamma=gamma)
    kx, ky = KX[:, 0], KY[0, :]
    return np.exp(1j * np.outer(kx, Rx)) @ W @ np.exp(1j * np.outer(Ry, ky))


def tmatrix(g0_11, g0_12, g0_22, V0):
    M = np.array([[1.0 / V0 - g0_11, -g0_12], [-g0_12, -1.0 / V0 - g0_22]])
    return np.linalg.inv(M)


def spectrum(r, omegas, V0=None, Nk=1024, delta=1.0, kind="d", lattice=False):
    KX, KY = k_mesh(Nk)
    wk = None if lattice else w_k(r, KX, KY)
    out = np.zeros(len(omegas))
    for n, om in enumerate(omegas):
        G11, G12, G22 = nambu_G(om + 1j * delta, KX, KY, kind)
        if lattice:
            out[n] = -np.mean(G11).imag / np.pi
            continue
        val = np.mean(np.abs(wk) ** 2 * G11)
        if V0 is not None:
            T = tmatrix(G11.mean(), G12.mean(), G22.mean(), V0)
            A = np.array([[np.mean(wk * G11), np.mean(wk * G12)],
                          [np.mean(wk * G12), np.mean(wk * G22)]])
            val += (A @ T @ A)[0, 0]
        out[n] = -val.imag / np.pi
    return out * 1000.0


def lattice_green(omega, Nk=1024, delta=1.0, kind="d"):
    KX, KY = k_mesh(Nk)
    G11, G12, G22 = nambu_G(omega + 1j * delta, KX, KY, kind)
    return tuple(np.fft.ifft2(G) for G in (G11, G12, G22))


def conductance_map(omega, V0, half_size=20, sub=8, Nk=1024, delta=1.0,
                    kind="d", cells=5):
    g11, g12, g22 = lattice_green(omega, Nk, delta, kind)
    d = np.arange(-cells, cells + 1)
    DX, DY = np.meshgrid(d, d, indexing="ij")
    dxf, dyf = DX.ravel(), DY.ravel()
    R0 = np.arange(-half_size, half_size + 1)

    def shifted(g):
        ix = (R0[None, :, None] + dxf[:, None, None]) % Nk
        iy = (R0[None, None, :] + dyf[:, None, None]) % Nk
        return g[ix, iy]
    S11, S12, S22 = shifted(g11), shifted(g12), shifted(g22)

    Gmat = g11[(dxf[:, None] - dxf[None, :]) % Nk, (dyf[:, None] - dyf[None, :]) % Nk]
    T = tmatrix(g11[0, 0], g12[0, 0], g22[0, 0], V0)

    s_vals = (np.arange(sub) + 0.5) / sub - 0.5
    n = len(R0) * sub
    rho_tot = np.zeros((n, n))
    rho_host = np.zeros((n, n))
    for a, sx in enumerate(s_vals):
        for b, sy in enumerate(s_vals):
            K = wannier_envelope(sx - dxf, sy - dyf)
            host = K @ Gmat @ K
            A11 = np.tensordot(K, S11, axes=(0, 0))
            A12 = np.tensordot(K, S12, axes=(0, 0))
            A22 = np.tensordot(K, S22, axes=(0, 0))
            imp = (A11 * A11 * T[0, 0] + A11 * A12 * (T[0, 1] + T[1, 0])
                   + A12 * A12 * T[1, 1])
            tot = -(host + imp).imag / np.pi * 1000.0
            hst = -host.imag / np.pi * 1000.0
            rho_tot[b::sub, a::sub] = tot.T
            rho_host[b::sub, a::sub] = hst
    xs = (R0[:, None] + s_vals[None, :]).ravel()
    X, Y = np.meshgrid(xs, xs)
    return X, Y, rho_tot, rho_host


def plot_spectra_clean(Nk=1024, delta=1.0, n_om=201, save=None):
    om = np.linspace(-1.5, 1.5, n_om) * DELTA0
    panels = [
        ("lattice-site LDOS N_i(w)", dict(r=(0, 0), lattice=True), "black"),
        ("continuum LDOS, r = Cu site / orbital lobe (0, 0)", dict(r=(0.0, 0.0)), "tab:red"),
        ("continuum LDOS, r = bond midpoint (+0.5a, 0)", dict(r=(0.5, 0.0)), "tab:blue"),
        ("continuum LDOS, r = plaquette center (+0.5a, +0.5a)", dict(r=(0.5, 0.5)), "tab:green"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    fig.suptitle("Continuum LDOS of the d-wave host at different STM tip positions ")
    for ax, (title, kw, col) in zip(axs.ravel(), panels):
        y = spectrum(omegas=om, Nk=Nk, delta=delta, **kw)
        ax.plot(om / DELTA0, y, color=col)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(r"$\rho$ [1/eV]")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(bottom=0)
    for ax in axs[1]:
        ax.set_xlabel(r"$\omega/\Delta_0$")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140)
    return fig


def plot_impurity_spectra(V0s=(-20.0, -150.0, -1000.0), Nk=1024, delta=1.0,
                          n_om=201, save=None):
    om = np.linspace(-1.2, 1.2, n_om) * DELTA0
    fig, axs = plt.subplots(1, len(V0s), figsize=(4.2 * len(V0s), 3.6))
    clean = spectrum((0.0, 0.0), om, Nk=Nk, delta=delta)
    for ax, V0 in zip(axs, V0s):
        imp = spectrum((0.0, 0.0), om, V0=V0, Nk=Nk, delta=delta)
        ax.plot(om / DELTA0, imp, "k", label="impurity site")
        ax.plot(om / DELTA0, clean, "k:", label="no impurity")
        ax.set_title(rf"$V_0={V0:g}$ meV", fontsize=10)
        ax.set_xlabel(r"$\omega/\Delta_0$")
        ax.set_ylim(bottom=0)
    axs[0].set_ylabel(r"$\rho(\omega)/\alpha_\omega^2$ [1/eV]")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140)
    return fig


def plot_conductance_map(omega, V0=-1000.0, half_size=20, show=12, save=None, **kw):
    X, Y, rho, host = conductance_map(omega, V0, half_size=half_size, **kw)
    sub = X.shape[0] // (2 * half_size + 1)
    drho = rho - host
    F = np.fft.fftshift(np.fft.fft2(drho))
    q = np.fft.fftshift(np.fft.fftfreq(drho.shape[0], d=1.0 / sub)) * 2
    c0, w = drho.shape[0] // 2, show * sub
    crop = np.s_[c0 - w:c0 + w + 1, c0 - w:c0 + w + 1]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    im = axs[0].imshow(rho[crop], extent=[-show - 0.5, show + 0.5] * 2, origin="lower",
                       cmap="Blues", vmax=np.percentile(rho[crop], 99.5))
    axs[0].set_title(rf"$\rho(\mathbf{{r}},\omega)$, $\omega/\Delta_0={omega / DELTA0:.2f}$, "
                     rf"$V_0={V0:g}$ meV")
    axs[0].set_xlabel("x/a"); axs[0].set_ylabel("y/a")
    fig.colorbar(im, ax=axs[0], shrink=0.8, label="[1/eV]")
    im2 = axs[1].imshow(np.abs(F), extent=[q[0], q[-1], q[0], q[-1]], origin="lower",
                        cmap="Blues", vmax=np.percentile(np.abs(F), 99.7))
    axs[1].set_xlim(-2, 2); axs[1].set_ylim(-2, 2)
    axs[1].set_title(r"$|\delta\rho(\mathbf{q},\omega)|$")
    axs[1].set_xlabel(r"$q_x/\pi$"); axs[1].set_ylabel(r"$q_y/\pi$")
    fig.colorbar(im2, ax=axs[1], shrink=0.8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140)
    return fig


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    what = args[0] if args else "all"
    fast = "--fast" in sys.argv
    kw = dict(Nk=512, delta=2.0, n_om=121) if fast else {}
    os.makedirs(_OUT_DIR, exist_ok=True)
    p = lambda name: os.path.join(_OUT_DIR, name)
    if what in ("fig4", "all"):
        plot_wannier(gamma=GAMMA, n=400, save_path=p("boker_fig4_wannier.png"))
    if what in ("spectra", "all"):
        plot_spectra_clean(save=p("boker_spectra_clean.png"), **kw)
    if what in ("fig8", "all"):
        plot_impurity_spectra(save=p("boker_fig8_insets.png"), **kw)
    if what in ("fig9", "all"):
        mk = dict(Nk=512, delta=2.0) if fast else {}
        plot_conductance_map(-0.3 * DELTA0, save=p("boker_fig9_map.png"), **mk)
    plt.show()
