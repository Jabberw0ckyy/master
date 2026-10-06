import sys
import os
import glob
import pickle
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_BDG_DIR = os.path.join(_PROJECT_ROOT, "bdg")
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
if _BDG_DIR not in sys.path:
    sys.path.insert(0, _BDG_DIR)

from lattice import g_t, g_sxy, diagonalize_bdg_real, g_sz_bond_vec
from solve_disordered import (
    N, Lx, Ly, xs, ys, t, tp, J,
    nn_i, nn_j, nn_sign, nnn_i, nnn_j,
    build_hamiltonian,
)
from wannier import wannier_envelope, plot_wannier


def _unique(fname):
    """../data/<name>_<time stamp>.<ext>: figures never overwrite each other."""
    import time
    base, ext = os.path.splitext(fname)
    return os.path.join(_DATA_DIR, f"{base}_{time.strftime('%Y%m%d-%H%M%S')}{ext}")


def lorentzian(x, eta):
    return (1.0 / np.pi) * eta / (x ** 2 + eta ** 2)


def continuum_ldos(rx, ry, xs_, ys_, E, psi, N_, omegas, eta=0.02, a=1.0, gamma=1.5,
                   Lx_=None, Ly_=None):
    dx = rx - xs_
    dy = ry - ys_
    if Lx_ is not None:
        dx = dx - Lx_ * np.round(dx / Lx_)
    if Ly_ is not None:
        dy = dy - Ly_ * np.round(dy / Ly_)
    w = wannier_envelope(dx, dy, gamma=gamma, a=a)
    u = psi[:N_, :]
    v = psi[N_:, :]
    Ut = w @ u
    Vt = w @ v
    om = omegas[:, None]
    Ep = E[None, :]
    rho = (Ut[None, :] ** 2 * lorentzian(om - Ep, eta)
           + Vt[None, :] ** 2 * lorentzian(om + Ep, eta)).sum(axis=1)
    return rho


def lattice_site_ldos(site_idx, E, psi, N_, omegas, eta=0.02):
    u = psi[site_idx, :]
    v = psi[N_ + site_idx, :]
    om = omegas[:, None]
    Ep = E[None, :]
    rho = (u[None, :] ** 2 * lorentzian(om - Ep, eta)
           + v[None, :] ** 2 * lorentzian(om + Ep, eta)).sum(axis=1)
    return rho


def continuum_ldos_map(rx_grid, ry_grid, xs_, ys_, E, psi, N_, omega, eta=0.02,
                        a=1.0, gamma=1.5, site_cutoff=6.0, Lx_=None, Ly_=None):
    shape = rx_grid.shape
    rxf, ryf = rx_grid.ravel(), ry_grid.ravel()
    out = np.zeros(len(rxf))
    u = psi[:N_, :]
    v = psi[N_:, :]
    om = np.array([omega])
    for k in range(len(rxf)):
        dx = rxf[k] - xs_
        dy = ryf[k] - ys_
        if Lx_ is not None:
            dx = dx - Lx_ * np.round(dx / Lx_)
        if Ly_ is not None:
            dy = dy - Ly_ * np.round(dy / Ly_)
        dist = np.sqrt(dx ** 2 + dy ** 2)
        near = dist < site_cutoff
        w = np.zeros(N_)
        w[near] = wannier_envelope(dx[near], dy[near], gamma=gamma, a=a)
        Ut = w @ u
        Vt = w @ v
        val = (Ut ** 2 * lorentzian(om - E, eta) + Vt ** 2 * lorentzian(om + E, eta)).sum()
        out[k] = val
    return out.reshape(shape)


def _v2_state(res):
    if 'chi_up' in res:
        return dict(delta_i=res['delta_i'], m_i=res['m_i'], chi_up=res['chi_up'], chi_dn=res['chi_dn'],
                    Delta_nn=res['Delta_nn'], chin_up=res['chin_up'], chin_dn=res['chin_dn'],
                    mu=res['mu'])
    chi_nnn = np.broadcast_to(res['chi_nnn'], (len(nnn_i),)).copy()
    return dict(delta_i=res['delta_i'], m_i=res['m_i'], chi_up=res['chi_nn'], chi_dn=res['chi_nn'],
                Delta_nn=res['Delta_nn'], chin_up=chi_nnn, chin_dn=chi_nnn, mu=res['mu'])


def rebuild_bdg_from_result(res, V_i):
    s = _v2_state(res)
    H0up, H0dn, Deltamat = build_hamiltonian(s, V_i)
    mu = res['mu']
    E, psi = diagonalize_bdg_real(H0up - mu * np.eye(N), H0dn - mu * np.eye(N), Deltamat)
    return H0up, H0dn, Deltamat, E, psi


def zero_bias_map(res, V_i, eta=0.02, grid_n=None, site_cutoff=6.0, gamma=1.5):
    if grid_n is None:
        grid_n = Lx
    _, _, _, E, psi = rebuild_bdg_from_result(res, V_i)
    rx = np.linspace(0, Lx, grid_n, endpoint=False)
    ry = np.linspace(0, Ly, grid_n, endpoint=False)
    RX, RY = np.meshgrid(rx, ry)
    rho_map = continuum_ldos_map(RX, RY, xs, ys, E, psi, N, 0.0,
                                  eta=eta, gamma=gamma,
                                  site_cutoff=site_cutoff, Lx_=Lx, Ly_=Ly)
    return RX, RY, rho_map, E, psi


def lattice_zero_bias(res, V_i, eta=0.02):
    _, _, _, E, psi = rebuild_bdg_from_result(res, V_i)
    out = np.zeros(N)
    for i in range(N):
        out[i] = lattice_site_ldos(i, E, psi, N, np.array([0.0]), eta=eta)[0]
    return out.reshape(Ly, Lx), E, psi


def load_type1_results(data_dir=_DATA_DIR, pattern="type1_delta*.pkl"):
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {data_dir} -- run "
            f"bdg/solve_disordered.py first to generate them."
        )
    results = []
    for p in paths:
        with open(p, "rb") as f:
            res = pickle.load(f)
        results.append((p, res))
    return results


def plot_ldos_map(E, psi, xs_, ys_, N_, omega, window, Lx_=None, Ly_=None,
                  cu_x=None, cu_y=None, pts_per_a=20, eta=0.02, gamma=1.5,
                  site_cutoff=6.0, save_path=None):
    import matplotlib.pyplot as plt

    x0, x1, y0, y1 = window
    nx = int(round((x1 - x0) * pts_per_a))
    ny = int(round((y1 - y0) * pts_per_a))
    GX, GY = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    ldos_map = continuum_ldos_map(GX, GY, xs_, ys_, E, psi, N_, omega,
                                  eta=eta, gamma=gamma, site_cutoff=site_cutoff,
                                  Lx_=Lx_, Ly_=Ly_)

    fig = plt.figure(figsize=(7, 7))
    plt.imshow(ldos_map, extent=[x0, x1, y0, y1], origin="lower", cmap="inferno")
    plt.colorbar(label="LDOS (arb.)")
    if cu_x is not None:
        plt.scatter(cu_x, cu_y, color="cyan", marker="+", s=30, label="Cu sites")
        plt.legend(loc="upper right")
    plt.title(f"Continuum LDOS map at $\\omega = {omega}t$")
    plt.xlabel("x/a")
    plt.ylabel("y/a")
    if save_path:
        plt.savefig(save_path, dpi=140)
    return fig


def plot_ldos_spectra(E, psi, xs_, ys_, N_, site_idx, omegas, eta=0.02,
                      gamma=1.5, Lx_=None, Ly_=None, save_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    cx, cy = xs_[site_idx], ys_[site_idx]
    r_lobe = (cx + 1.0, cy)
    r_bond = (cx + 0.5, cy)
    r_plaq = (cx + 0.5, cy + 0.5)

    ldos_lattice = lattice_site_ldos(site_idx, E, psi, N_, omegas, eta=eta)
    ldos_lobe = continuum_ldos(*r_lobe, xs_, ys_, E, psi, N_, omegas, eta=eta, gamma=gamma,
                              Lx_=Lx_, Ly_=Ly_)
    ldos_bond = continuum_ldos(*r_bond, xs_, ys_, E, psi, N_, omegas, eta=eta, gamma=gamma,
                              Lx_=Lx_, Ly_=Ly_)
    ldos_plaq = continuum_ldos(*r_plaq, xs_, ys_, E, psi, N_, omegas, eta=eta, gamma=gamma,
                              Lx_=Lx_, Ly_=Ly_)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Continuum LDOS shape at different STM tip positions "
                 "(independent y-scales)")

    plots_data = [
        (axs[0, 0], ldos_lattice, "black",
         "lattice-site LDOS N_i(w) [old method, point-like]"),
        (axs[0, 1], ldos_lobe, "tab:red",
         "continuum LDOS, r = orbital lobe peak (+1a, 0)"),
        (axs[1, 0], ldos_bond, "tab:blue",
         "continuum LDOS, r = bond midpoint (+0.5a, 0)"),
        (axs[1, 1], ldos_plaq, "tab:green",
         "continuum LDOS, r = plaquette center (+0.5a,+0.5a)"),
    ]
    for ax, data, color, title in plots_data:
        ax.plot(omegas, data, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("$\\omega/t$")
        ax.set_ylabel("LDOS")
        ax.set_xlim(omegas[0], omegas[-1])
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140)
    return fig


def run_demo(omega_map=-0.28):
    import matplotlib.pyplot as plt
    os.makedirs(_DATA_DIR, exist_ok=True)

    N_side = 6
    xs_d = np.array([float(ix) for ix in range(1, N_side + 1)
                     for iy in range(1, N_side + 1)])
    ys_d = np.array([float(iy) for ix in range(1, N_side + 1)
                     for iy in range(1, N_side + 1)])
    N_d = len(xs_d)

    rng = np.random.RandomState(42)
    E_d = np.sort(np.concatenate([np.linspace(-0.6, 0.6, N_d),
                                  np.linspace(-0.6, 0.6, N_d)]))
    psi_d = rng.randn(2 * N_d, 2 * N_d) * 0.1

    cu_x, cu_y = np.meshgrid(np.arange(1, 8), np.arange(1, 8))
    plot_wannier(save_path=_unique("fig_ldos_wannier.png"))
    plot_ldos_map(E_d, psi_d, xs_d, ys_d, N_d, omega_map,
                  window=(1.5, 6.5, 1.5, 6.5), Lx_=6.0, Ly_=6.0,
                  cu_x=cu_x.ravel(), cu_y=cu_y.ravel(),
                  save_path=_unique("fig_ldos_map.png"))
    plot_ldos_spectra(E_d, psi_d, xs_d, ys_d, N_d, N_d // 2,
                      np.linspace(-0.6, 0.6, 300), Lx_=6.0, Ly_=6.0,
                      save_path=_unique("fig_ldos_spectra.png"))
    plt.show()


def run_real(file_index=-1, omega_map=-0.28, omega_max=0.6, eta=0.02,
             half_window=4.0):
    import matplotlib.pyplot as plt
    os.makedirs(_DATA_DIR, exist_ok=True)

    path, res = load_type1_results()[file_index]
    print(f"using {os.path.basename(path)}  delta={res['delta_target']:.3f}  "
          f"diff={res['diff']:.2e}")
    _, _, _, E, psi = rebuild_bdg_from_result(res, res["V_i"])

    if "imp_sites" in res and len(res["imp_sites"]) > 0:
        site_idx = int(res["imp_sites"][0])
    else:
        site_idx = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
    cx, cy = xs[site_idx], ys[site_idx]
    print(f"tip reference site: index {site_idx} at ({cx:g}, {cy:g})")

    win = (cx - half_window, cx + half_window, cy - half_window, cy + half_window)
    near = (np.abs(xs - cx) <= half_window + 1) & (np.abs(ys - cy) <= half_window + 1)

    plot_wannier(save_path=_unique("fig_ldos_wannier.png"))
    plot_ldos_map(E, psi, xs, ys, N, omega_map, window=win, Lx_=Lx, Ly_=Ly,
                  cu_x=xs[near], cu_y=ys[near], eta=eta,
                  save_path=_unique("fig_ldos_map_real.png"))
    plot_ldos_spectra(E, psi, xs, ys, N, site_idx,
                      np.linspace(-omega_max, omega_max, 300), eta=eta, Lx_=Lx, Ly_=Ly,
                      save_path=_unique("fig_ldos_spectra_real.png"))
    plt.show()


def run_type1_maps():
    import matplotlib.pyplot as plt

    results = load_type1_results()
    print(f"found {len(results)} result files in {_DATA_DIR}:")
    for p, res in results:
        print(f"  {os.path.basename(p)}  delta={res['delta_target']:.3f}  "
              f"diff={res['diff']:.2e}")

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.3), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (path, res) in zip(axes, results):
        RX, RY, rho0, E, psi = zero_bias_map(res, res['V_i'], grid_n=Lx, site_cutoff=6.0)
        im = ax.pcolormesh(RX, RY, rho0, cmap="inferno", shading="auto")
        imp_y = res["imp_sites"] // Lx
        imp_x = res["imp_sites"] % Lx
        ax.scatter(imp_x, imp_y, c="c", s=25, marker='x')
        converged = "OK" if res["diff"] < 1e-4 else "NOT CONVERGED"
        ax.set_title(rf"$\delta={res['delta_target']:.3f}$"
                     f"\n{converged} (diff={res['diff']:.1e})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.8, label=r"$\rho(r,\omega=0)$")
    fig.suptitle("Type I disorder: zero-bias continuum LDOS maps")
    out_path = _unique("fig_ldos_type1.png")
    plt.savefig(out_path, dpi=140)
    plt.show()
    print(f"saved {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    ap.add_argument("mode", nargs="?", default="type1",
                    choices=["type1", "demo", "real"],
                    help="type1: zero-bias maps of all pkl results (default); "
                         "demo: synthetic-data figures; "
                         "real: the three figures from a converged pkl")
    ap.add_argument("--file", type=int, default=-1,
                    help="index into the sorted pkl list (real mode)")
    ap.add_argument("--omega", type=float, default=-0.28,
                    help="bias of the LDOS map, in units of t")
    args = ap.parse_args()

    if args.mode == "demo":
        run_demo(omega_map=args.omega)
    elif args.mode == "real":
        run_real(file_index=args.file, omega_map=args.omega)
    else:
        run_type1_maps()
