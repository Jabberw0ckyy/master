import os
import sys
import time
import pickle
import argparse
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
for _p in (os.path.join(_PROJECT_ROOT, "bdg"), _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import solve_disordered as sd
from solve_disordered import N, Lx, Ly, neel_sign
from af_jacobian import NAMES, OFFS, DIM, pack, unpack, load_base
from af_selfcons import imp_site, distance_from


class Residual:
    def __init__(self, base, delta, T, V):
        self.base, self.delta, self.T = base, delta, T
        self.V_i = np.zeros(N)
        self.V_i[imp_site()] = V
        self.calls = 0
        self.mu = base["mu"]

    def F(self, x):
        self.calls += 1
        b = dict(self.base)
        b["mu"] = self.mu
        r = sd.solve_disordered(self.delta, self.V_i, seed=unpack(x, b), max_iter=1, mix=1.0, mix_fluct=1.0,
                                mix_mu=1.0, warmup=0, T=self.T, pairing=False)
        self.mu = r["mu"]
        return pack(r), r

    def R(self, x):
        fx, r = self.F(x)
        return fx - x, r


def newton_step(res, x, Rx, gm_tol, gm_iters, eps):
    fx = Rx + x

    def matvec(v):
        nv = np.linalg.norm(v)
        if nv == 0:
            return np.zeros_like(v)
        h = eps / nv
        f2, _ = res.F(x + h * v)
        return (f2 - fx) / h - v

    A = LinearOperator((DIM, DIM), matvec=matvec, dtype=float)
    dx, info = gmres(A, -Rx, rtol=gm_tol, restart=gm_iters, maxiter=1)
    return dx, info


def solve_at_V(base, delta, T, V, x_start, args, t_start):
    res = Residual(base, delta, T, V)
    x = x_start.copy()
    Rx, r = res.R(x)
    hist = [float(np.abs(Rx).max())]
    print(f"  V={V:g}: start residual max|R| = {hist[0]:.2e}", flush=True)
    for step in range(1, args.newton_iters + 1):
        if hist[-1] < args.tol:
            break
        calls0 = res.calls
        dx, info = newton_step(res, x, Rx, args.gm_tol, args.gm_iters, args.eps)
        lam = 1.0
        for _ in range(5):
            xn = x + lam * dx
            Rn, rn = res.R(xn)
            if np.abs(Rn).max() < np.abs(Rx).max():
                break
            lam *= 0.5
        x, Rx, r = xn, Rn, rn
        hist.append(float(np.abs(Rx).max()))
        print(f"  V={V:g}: Newton step {step}: max|R| = {hist[-1]:.2e}  (damping {lam:g}, {res.calls - calls0} F-calls, "
              f"gmres info {info})  [{time.time() - t_start:.0f}s]", flush=True)
        if args.budget and time.time() - t_start > args.budget:
            return x, hist, r, False
    return x, hist, r, hist[-1] < args.tol


def sol_path(delta, T, V):
    return os.path.join(_DATA_DIR, f"af_newton_sol_d{delta:.3f}_T{T:g}_V{V:g}.pkl")


def run(args):
    base, clean = load_base(args.delta, args.T)
    x_prev = pack(base)
    t0 = time.time()
    for V in args.Vs:
        path = sol_path(args.delta, args.T, V)
        if os.path.exists(path):
            with open(path, "rb") as f:
                sol = pickle.load(f)
            if sol["converged"]:
                x_prev = sol["x"]
                print(f"V={V:g}: already solved (residual {sol['hist'][-1]:.1e})")
                continue
            x_prev = sol["x"]
        x, hist, r, ok = solve_at_V(base, args.delta, args.T, V, x_prev, args, t0)
        with open(path, "wb") as f:
            pickle.dump(dict(x=x, hist=hist, converged=ok, V=V, delta=args.delta, T=args.T,
                             delta_i=r["delta_i"], m_i=r["m_i"], mu=r["mu"]), f)
        if not ok:
            print("not finished (budget or no convergence): run the same command again to continue")
            return
        x_prev = x
    print("all V done")


def load_solution(delta, T, V, base):
    with open(sol_path(delta, T, V), "rb") as f:
        sol = pickle.load(f)
    st = unpack(sol["x"], base)
    st["delta_i"] = np.asarray(st["delta_i"], float)
    return sol, st


def report(args):
    import matplotlib.pyplot as plt
    base, clean = load_base(args.delta, args.T)
    site = imp_site()
    dx, dy, r = distance_from(site)
    nn = np.where(r == 1.0)[0]
    far = r > 8.0
    m_c = neel_sign * base["m_i"]
    n_c = 1 - base["delta_i"]
    e_c = sd.free_energy_terms(dict(base))
    rows = []
    sols = {}
    for V in args.Vs:
        if not os.path.exists(sol_path(args.delta, args.T, V)):
            continue
        sol, st = load_solution(args.delta, args.T, V, base)
        m = neel_sign * st["m_i"]
        n = 1 - st["delta_i"]
        e = sd.free_energy_terms(st)
        sols[V] = (m, n, sol["hist"][-1])
        rows.append((V, sol["hist"][-1], m[site], st["delta_i"][site], m[nn].mean(), (n - n_c)[nn].mean(),
                     np.abs(m - m_c)[far].max(), np.abs(n - n_c)[far].max(), (e - e_c) * N))
    print(f"clean: m = {m_c.mean():.4f}, delta = {base['delta_i'].mean():.4f}, E/N = {e_c:.6f}")
    print(f"{'V':>6} {'resid':>8} {'m_imp':>8} {'delta_imp':>10} {'m_nn':>8} {'dn_nn':>8} {'far|dm|':>9} {'far|dn|':>9} {'E-E_clean':>10}")
    for row in rows:
        print(f"{row[0]:6.2f} {row[1]:8.1e} {row[2]:+8.4f} {row[3]:+10.4f} {row[4]:+8.4f} {row[5]:+8.4f} {row[6]:9.1e} {row[7]:9.1e} {row[8]:+10.4f}")
    Vp = args.V_plot
    if Vp not in sols:
        return
    m, n, res_p = sols[Vp]
    h = args.half
    win = (np.abs(dx) <= h) & (np.abs(dy) <= h)

    def crop(a):
        sel = np.where(win)[0]
        img = np.full((2 * h + 1, 2 * h + 1), np.nan)
        img[dy[sel] + h, dx[sel] + h] = a[sel]
        return img

    ext = [-h - 0.5, h + 0.5, -h - 0.5, h + 0.5]
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 9))
    im = axs[0, 0].imshow(crop(m), extent=ext, origin="lower", cmap="viridis")
    axs[0, 0].set_title(r"staggered moment $(-1)^{x+y}m_i$"); fig.colorbar(im, ax=axs[0, 0], shrink=0.8)
    dm = m - m_c
    v = np.nanmax(np.abs(crop(dm)))
    im = axs[0, 1].imshow(crop(dm), extent=ext, origin="lower", cmap="bwr", vmin=-v, vmax=v)
    axs[0, 1].set_title(r"$\Delta m_i$ = with impurity $-$ clean"); fig.colorbar(im, ax=axs[0, 1], shrink=0.8)
    dn = n - n_c
    v = np.nanmax(np.abs(crop(dn)))
    im = axs[0, 2].imshow(crop(dn), extent=ext, origin="lower", cmap="PiYG", vmin=-v, vmax=v)
    axs[0, 2].set_title(r"charge change $\Delta n_i$"); fig.colorbar(im, ax=axs[0, 2], shrink=0.8)
    for ax in axs[0]:
        ax.plot(0, 0, "k+", ms=10); ax.set_xlabel("x/a"); ax.set_ylabel("y/a")
    row = np.where(ys_arr() == ys_arr()[site])[0]
    row = row[np.argsort(dx[row])]
    for V, (mm, nn_, _) in sols.items():
        axs[1, 0].plot(dx[row], mm[row], "o-", ms=3, label=f"V={V:g}t")
    axs[1, 0].axhline(m_c.mean(), color="k", ls=":", label="clean")
    axs[1, 0].set_xlabel("x/a along the row through the impurity"); axs[1, 0].set_ylabel(r"$(-1)^{x+y}m_i$")
    axs[1, 0].legend(fontsize=7)
    for lab, d_, mk in ((r"$|\Delta m|$", np.abs(dm), "o-"), (r"$|\Delta n|$", np.abs(dn), "s-")):
        u = np.unique(np.round(r, 6))
        axs[1, 1].semilogy(u[1:], np.array([d_[np.round(r, 6) == x].mean() for x in u])[1:] + 1e-12, mk, ms=4, label=lab)
    axs[1, 1].set_xlabel("distance from impurity r/a"); axs[1, 1].set_ylabel("average over sites at distance r")
    axs[1, 1].set_title(f"V={Vp:g}t", fontsize=9); axs[1, 1].legend(fontsize=8)
    Vv = [x[0] for x in rows]
    axs[1, 2].plot(Vv, [x[2] for x in rows], "o-", label="m on impurity site")
    axs[1, 2].plot(Vv, [x[4] for x in rows], "s-", label="m on nn sites")
    axs[1, 2].axhline(m_c.mean(), color="k", ls=":", lw=0.8)
    axs[1, 2].set_xlabel("V/t"); axs[1, 2].set_ylabel(r"staggered moment")
    ax2 = axs[1, 2].twinx()
    ax2.plot(Vv, [x[3] for x in rows], "^--", color="tab:green", label=r"$\delta$ on impurity site")
    ax2.set_ylabel(r"$\delta_{imp}$")
    h1, l1 = axs[1, 2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axs[1, 2].legend(h1 + h2, l1 + l2, fontsize=7)
    fig.suptitle(rf"self-consistent (Newton-Krylov) {Lx}x{Ly} lattice, $\delta={args.delta:g}$, $T={args.T:g}$, "
                 rf"residual max|F(x)-x| = {res_p:.1e} at V={Vp:g}t", fontsize=11)
    fig.tight_layout()
    import time as _t
    fn = os.path.join(_DATA_DIR, f"af_newton_report_d{args.delta:.3f}_V{Vp:g}_{_t.strftime('%Y%m%d-%H%M%S')}.png")
    fig.savefig(fn, dpi=140)
    print("saved", fn)


def ys_arr():
    return sd.ys


def compare(args):
    import matplotlib.pyplot as plt
    import af_impurity as af
    base, clean = load_base(args.delta, args.T)
    host = af.solve_uniform_af(args.delta, T=args.T)
    print(f"infinite-lattice host: m = {host['m']:.4f}, mu = {host['mu']:.4f}")
    site = imp_site()
    omegas = np.linspace(args.wmin, args.wmax, args.n_omega)
    tips = {"nn (1,0)": (1.0, 0.0), "(2,0)": (2.0, 0.0), "bond (0.5,0)": (0.5, 0.0)}
    Vs = [V for V in args.Vs if os.path.exists(sol_path(args.delta, args.T, V))]
    E0, psi0 = af.bdg_eigenpairs(dict(base), np.zeros(N))
    clean_fin = af.spectra_finite(site, tips, E0, psi0, omegas, args.eta)
    fig, axs = plt.subplots(1, len(Vs), figsize=(5.2 * len(Vs), 4.4), squeeze=False)
    print(f"{'V':>6} {'tip':>14} {'self-consistent':>16} {'frozen (finite)':>16} {'frozen (T-matrix)':>18}   peak of rho_imp - rho_clean in [{args.pmin:g},{args.pmax:g}]")
    for ax, V in zip(axs[0], Vs):
        sol, st = load_solution(args.delta, args.T, V, base)
        V_i = np.zeros(N)
        V_i[site] = V
        Esc, psisc = af.bdg_eigenpairs(st, V_i)
        Efr, psifr = af.bdg_eigenpairs(dict(base), V_i)
        sc = af.spectra_finite(site, tips, Esc, psisc, omegas, args.eta)
        fr = af.spectra_finite(site, tips, Efr, psifr, omegas, args.eta)
        s_imp, s_cl = af.spectra_infinite(tips, host, V, omegas, args.eta, Nk=args.Nk)
        win = (omegas >= args.pmin) & (omegas <= args.pmax)
        for i, k in enumerate(tips):
            pk = []
            for y, y0 in ((sc[i], clean_fin[i]), (fr[i], clean_fin[i]), (s_imp[k], s_cl[k])):
                d = np.where(win, y - y0, -np.inf)
                pk.append(omegas[int(np.argmax(d))])
            print(f"{V:6.2f} {k:>14} {pk[0]:+16.3f} {pk[1]:+16.3f} {pk[2]:+18.3f}")
        ax.plot(omegas, clean_fin[0], color="0.6", ls=":", label="clean (finite)")
        ax.plot(omegas, fr[0], "b--", lw=1.1, label="frozen host, finite lattice")
        ax.plot(omegas, s_imp["nn (1,0)"], "r-", lw=1.0, label="frozen host, T-matrix (infinite)")
        ax.plot(omegas, sc[0], "k-", lw=1.6, label="self-consistent, finite lattice")
        ax.set_title(f"nn (1,0) tip, V={V:g}t", fontsize=10)
        ax.set_xlabel(r"bias $\omega/t$")
        ax.set_ylim(bottom=0)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(rf"frozen vs self-consistent host: $\delta={args.delta:g}$, $T={args.T:g}$, $\eta={args.eta:g}$", fontsize=11)
    fig.tight_layout()
    import time as _t
    fn = os.path.join(_DATA_DIR, f"af_newton_compare_d{args.delta:.3f}_eta{args.eta:g}_{_t.strftime('%Y%m%d-%H%M%S')}.png")
    fig.savefig(fn, dpi=140)
    print("saved", fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Newton-Krylov self-consistent impurity with continuation in V")
    ap.add_argument("--delta", type=float, default=0.12)
    ap.add_argument("--T", type=float, default=0.02)
    ap.add_argument("--Vs", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--tol", type=float, default=1e-4, help="Newton tolerance on max|F(x)-x|")
    ap.add_argument("--newton_iters", type=int, default=8)
    ap.add_argument("--gm_tol", type=float, default=0.05)
    ap.add_argument("--gm_iters", type=int, default=50)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--report", action="store_true", help="table and figure from the solved states")
    ap.add_argument("--compare", action="store_true", help="LDOS: self-consistent vs frozen host")
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--wmin", type=float, default=-0.6)
    ap.add_argument("--wmax", type=float, default=0.9)
    ap.add_argument("--n_omega", type=int, default=301)
    ap.add_argument("--pmin", type=float, default=0.0)
    ap.add_argument("--pmax", type=float, default=0.6)
    ap.add_argument("--Nk", type=int, default=256)
    ap.add_argument("--V_plot", type=float, default=0.5)
    ap.add_argument("--half", type=int, default=7)
    a = ap.parse_args()
    if a.compare:
        compare(a)
    elif a.report:
        report(a)
    else:
        run(a)
