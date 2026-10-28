import os
import sys
import time
import pickle
import argparse
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
for _p in (os.path.join(_PROJECT_ROOT, "bdg"), _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from solve_disordered import (N, Lx, Ly, xs, ys, neel_sign, solve_disordered, initial_state)

KEYS = ("delta_i", "m_i", "chi_up", "chi_dn", "Delta_nn", "chin_up", "chin_dn", "mu")


def imp_site():
    return (Ly // 2) * Lx + Lx // 2


def distance_from(site):
    dx = (xs - xs[site] + Lx // 2) % Lx - Lx // 2
    dy = (ys - ys[site] + Ly // 2) % Ly - Ly // 2
    return dx, dy, np.sqrt(dx ** 2 + dy ** 2)


def run_chunked(delta, V_i, m0, T, mix, tol, max_iter, chunk=10, tag="", seed=None, ckpt=None, budget=None,
                mix_fluct=0.03, mix_mu=1.0):
    if ckpt and os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            saved = pickle.load(f)
        seed, hist, done = saved["seed"], saved["hist"], saved["done"]
        print(f"  [{tag}] resuming from {os.path.basename(ckpt)} at iteration {done}", flush=True)
    else:
        if seed is None:
            seed = initial_state(delta, pairing=False)
            seed["m_i"] = m0 * neel_sign
        hist, done = [], 0
    site = imp_site()
    _, _, r = distance_from(site)
    far = r > 8.0
    t_start = time.time()
    res = None
    finished = False
    while done < max_iter:
        res = solve_disordered(delta, V_i, seed=seed, max_iter=chunk, mix=mix, tol=tol, T=T,
                               pairing=False, mix_fluct=mix_fluct, mix_mu=mix_mu,
                               warmup=(6 if done == 0 else 0))
        done += res["iters"] + 1
        seed = {k: res[k] for k in KEYS}
        mst = neel_sign * res["m_i"]
        hist.append(dict(it=done, diff=res["diff"], m_far=mst[far].mean(), m_imp=mst[site],
                         d_imp=res["delta_i"][site], mu=res["mu"]))
        print(f"  [{tag}] it={done:4d} diff={res['diff']:.2e} <m>_far={mst[far].mean():.4f} "
              f"m_imp={mst[site]:+.4f} delta_imp={res['delta_i'][site]:+.4f} mu={res['mu']:+.4f}", flush=True)
        if ckpt:
            with open(ckpt, "wb") as f:
                pickle.dump(dict(seed=seed, hist=hist, done=done), f)
        if res["unphysical"]:
            print(f"  [{tag}] aborted: some site has n_i > 1 (unphysical)")
            finished = True
            break
        if res["diff"] < tol:
            finished = True
            break
        if budget is not None and time.time() - t_start > budget:
            print(f"  [{tag}] time budget reached: run the same command again to continue", flush=True)
            break
    else:
        finished = True
    res["history"] = hist
    res["finished"] = finished
    res["converged"] = bool(res["diff"] < tol and not res["unphysical"])
    return res


def radial_profile(vals, r, nbins=None):
    rr = np.round(r, 6)
    u = np.unique(rr)
    return u, np.array([vals[rr == x].mean() for x in u])


def make_figure(clean, imp, args, save):
    import matplotlib.pyplot as plt
    site = imp_site()
    dx, dy, r = distance_from(site)
    m_c, m_i = neel_sign * clean["m_i"], neel_sign * imp["m_i"]
    n_c, n_i = 1 - clean["delta_i"], 1 - imp["delta_i"]
    shape = (Ly, Lx)
    h = args.half
    win = (np.abs(dx) <= h) & (np.abs(dy) <= h)
    order = np.lexsort((xs, ys))

    def crop(a):
        sel = np.where(win)[0]
        xx = dx[sel]; yy = dy[sel]
        img = np.full((2 * h + 1, 2 * h + 1), np.nan)
        img[yy + h, xx + h] = a[sel]
        return img

    ext = [-h - 0.5, h + 0.5, -h - 0.5, h + 0.5]
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 9))
    im = axs[0, 0].imshow(crop(m_i), extent=ext, origin="lower", cmap="viridis")
    axs[0, 0].set_title(r"staggered moment $(-1)^{x+y}m_i$"); fig.colorbar(im, ax=axs[0, 0], shrink=0.8)
    dm = m_i - m_c
    v = np.nanmax(np.abs(crop(dm)))
    im = axs[0, 1].imshow(crop(dm), extent=ext, origin="lower", cmap="bwr", vmin=-v, vmax=v)
    axs[0, 1].set_title(r"$\Delta m_i$ = impurity run $-$ clean run"); fig.colorbar(im, ax=axs[0, 1], shrink=0.8)
    dn = n_i - n_c
    v = np.nanmax(np.abs(crop(dn)))
    im = axs[0, 2].imshow(crop(dn), extent=ext, origin="lower", cmap="PiYG", vmin=-v, vmax=v)
    axs[0, 2].set_title(r"charge change $\Delta n_i = \Delta(1-\delta_i)$"); fig.colorbar(im, ax=axs[0, 2], shrink=0.8)
    for ax in axs[0]:
        ax.plot(0, 0, "k+", ms=10)
        ax.set_xlabel("x/a"); ax.set_ylabel("y/a")

    row = np.where(ys == ys[site])[0]
    row = row[np.argsort(dx[row])]
    axs[1, 0].plot(dx[row], m_i[row], "o-", label="with impurity (each site re-solved)")
    axs[1, 0].axhline(m_c.mean(), color="r", ls=":", label=f"clean run, m = {m_c.mean():.3f}")
    axs[1, 0].set_xlabel("x/a along the row through the impurity"); axs[1, 0].set_ylabel(r"$(-1)^{x+y}m_i$")
    axs[1, 0].legend(fontsize=8)

    ru, pm = radial_profile(np.abs(dm), r)
    ru2, pn = radial_profile(np.abs(dn), r)
    axs[1, 1].semilogy(ru[1:], pm[1:] + 1e-12, "o-", label=r"$|\Delta m|$")
    axs[1, 1].semilogy(ru2[1:], pn[1:] + 1e-12, "s-", label=r"$|\Delta n|$")
    axs[1, 1].set_xlabel("distance from impurity r/a"); axs[1, 1].set_ylabel("average over sites at distance r")
    axs[1, 1].legend(fontsize=8)

    for res, lab in ((clean, "clean (V=0)"), (imp, f"impurity V={args.V:g}t")):
        hh = res["history"]
        axs[1, 2].semilogy([x["it"] for x in hh], [x["diff"] for x in hh], "o-", ms=3, label=lab)
    axs[1, 2].axhline(args.tol, color="k", ls=":", lw=0.8)
    axs[1, 2].set_xlabel("iteration"); axs[1, 2].set_ylabel("max change per iteration (residual)")
    axs[1, 2].legend(fontsize=8)
    axs[1, 2].set_title(f"convergence (tol={args.tol:g}); converged: clean={clean['converged']}, imp={imp['converged']}", fontsize=9)

    fig.suptitle(rf"self-consistent {Lx}x{Ly} lattice, $\delta={args.delta:g}$, $V={args.V:g}t$, $T={args.T:g}$, "
                 rf"mix={args.mix:g}, start $m_0={args.m0:g}$", fontsize=11)
    fig.tight_layout()
    fig.savefig(save, dpi=140)
    return fig


def _paths(a):
    base = f"d{a.delta:.3f}_T{a.T:g}"
    mu = "_mud" if a.mu_damped else ""
    return (os.path.join(_DATA_DIR, f"af_selfcons_ckpt_clean_{base}.pkl"),
            os.path.join(_DATA_DIR, f"af_selfcons_ckpt_imp_{base}_V{a.V:g}{mu}.pkl"),
            os.path.join(_DATA_DIR, f"af_selfcons_final_clean_{base}.pkl"),
            os.path.join(_DATA_DIR, f"af_selfcons_final_imp_{base}_V{a.V:g}{mu}.pkl"))


def report(clean, imp, a):
    site = imp_site()
    dx, dy, r = distance_from(site)
    mc, mi = neel_sign * clean["m_i"], neel_sign * imp["m_i"]
    nc, ni = 1 - clean["delta_i"], 1 - imp["delta_i"]
    dm, dn = mi - mc, ni - nc
    nn = np.where(r == 1.0)[0]
    far = r > 8.0
    print(f"\nclean run : <m>={mc.mean():.5f} spread={mc.max() - mc.min():.1e} <delta>={clean['delta_i'].mean():+.5f} "
          f"mu={clean['mu']:+.4f} converged={clean['converged']} residual={clean['diff']:.1e}")
    print(f"impurity  : converged={imp['converged']} residual={imp['diff']:.1e} iterations={imp['history'][-1]['it']} "
          f"<delta>={imp['delta_i'].mean():+.5f}")
    print(f"impurity site : m = {mi[site]:+.4f} (clean {mc[site]:+.4f}),  n = {ni[site]:.4f} (clean {nc[site]:.4f})")
    print(f"nn sites      : m = {mi[nn].mean():+.4f},  dn = {dn[nn].mean():+.4f}")
    print(f"far (r>8)     : max|dm| = {np.abs(dm[far]).max():.1e},  max|dn| = {np.abs(dn[far]).max():.1e}")
    print(f"frozen-host check: max|dm| over all sites = {np.abs(dm).max():.3f} (would be 0 for a frozen host)")
    print(f"total change  : sum dm = {dm.sum():+.4f}, sum dn = {dn.sum():+.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="self-consistent single impurity in the AF state (finite lattice)")
    ap.add_argument("mode", choices=["clean", "imp", "plot", "all"])
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--V", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=0.02)
    ap.add_argument("--mix", type=float, default=0.15)
    ap.add_argument("--mix_fluct", type=float, default=0.03, help="mixing of the site-to-site charge fluctuations")
    ap.add_argument("--mu_damped", action="store_true", help="damp mu with the same mix (mix_mu=None)")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--m0", type=float, default=0.163)
    ap.add_argument("--half", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--budget", type=float, default=None, help="stop after this many seconds (rerun to continue)")
    a = ap.parse_args()
    os.makedirs(_DATA_DIR, exist_ok=True)
    site = imp_site()
    ck_c, ck_i, fin_c, fin_i = _paths(a)
    V_clean = np.zeros(N)
    V_imp = np.zeros(N)
    V_imp[site] = a.V
    print(f"impurity site ({xs[site]},{ys[site]}), V={a.V}, delta={a.delta}, T={a.T}, mix={a.mix}, tol={a.tol}")
    if a.mode in ("clean", "all"):
        clean = run_chunked(a.delta, V_clean, a.m0, a.T, a.mix, a.tol, a.iters, a.chunk, "clean",
                            ckpt=ck_c, budget=a.budget, mix_fluct=a.mix_fluct,
                            mix_mu=(None if a.mu_damped else 1.0))
        with open(fin_c, "wb") as f:
            pickle.dump(clean, f)
        if not clean["finished"]:
            sys.exit(0)
    if a.mode in ("imp", "all"):
        if a.mode == "imp":
            with open(fin_c, "rb") as f:
                clean = pickle.load(f)
        seed = {k: clean[k] for k in KEYS}
        imp = run_chunked(a.delta, V_imp, a.m0, a.T, a.mix, a.tol, a.iters, a.chunk, "imp  ",
                          seed=seed, ckpt=ck_i, budget=a.budget, mix_fluct=a.mix_fluct,
                          mix_mu=(None if a.mu_damped else 1.0))
        with open(fin_i, "wb") as f:
            pickle.dump(imp, f)
        if not imp["finished"]:
            sys.exit(0)
    if a.mode in ("plot", "all"):
        with open(fin_c, "rb") as f:
            clean = pickle.load(f)
        with open(fin_i, "rb") as f:
            imp = pickle.load(f)
        report(clean, imp, a)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        mc = neel_sign * clean["m_i"]
        fn = os.path.join(_DATA_DIR, f"af_selfcons_d{a.delta:.3f}_V{a.V:g}_T{a.T:g}_m{mc.mean():.3f}_{stamp}.png")
        make_figure(clean, imp, a, fn)
        print("saved", fn)
