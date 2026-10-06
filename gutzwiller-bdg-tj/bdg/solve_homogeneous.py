import os
import sys
import pickle
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solve_disordered as sd
import uniform_k as uk
from solve_disordered import Lx, Ly, t, tp, J, N, nn_bonds, nnn_bonds      # v1 module-level names

T_DEFAULT = 0.02
NK_DEFAULT = 48


def solve_at_doping(delta_target, seed=None, verbose=False, max_iter=200, mix=0.3,
                    T=T_DEFAULT, Nk=NK_DEFAULT, kappa=sd.KAPPA):
    if seed is None:
        init = (0.15, 0.185, 0.12, 0.03, 0.03)
    else:
        m0, chi0, D0 = seed[:3]
        cn = seed[4] if len(seed) > 4 else 0.03
        init = (max(m0, 0.05), chi0, max(D0, 0.05), cn, cn)
    r = uk.solve_k(delta_target, T, kappa, Nk=Nk, init=init, tol=1e-8, mix=max(mix, 0.3),
                   it_max=max(max_iter, 400), verbose=verbose)
    return dict(m=r['m'], chi=r['chi'], Delta=abs(r['Delta']), mu=r['mu'],
                chi_nnn=0.5 * (r['cAu'] + r['cAd']), iters=r['iters'], diff=r['diff'])


def solve_at_doping_real(delta_target, max_iter=600, T=T_DEFAULT, verbose=False):
    r = sd.solve_disordered(delta_target, np.zeros(N), max_iter=max_iter, mix=0.15, tol=1e-7, T=T,
                            verbose=verbose)
    return dict(m=float(np.mean(np.abs(r['m_i']))), chi=float(np.mean(r['chi_nn'])),
                Delta=float(np.mean(np.abs(r['Delta_nn']))), mu=r['mu'],
                chi_nnn=float(np.mean(r['chi_nnn'])), iters=r['iters'], diff=r['diff'])

PAPER_DELTA = [0.026, 0.05, 0.08, 0.10, 0.115, 0.15, 0.20, 0.25]
PAPER_CHI = [0.175, 0.185, 0.195, 0.20, 0.205, 0.215, 0.23, 0.24]
PAPER_DELTA_SC = [0.150, 0.145, 0.135, 0.125, 0.120, 0.112, 0.095, 0.085]
PAPER_M = [0.16, 0.14, 0.105, 0.085, 0.035, 0.0, 0.0, 0.0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--real", type=float, nargs="*", default=None,
                    help="real-space check at these dopings (no scan)")
    ap.add_argument("--T", type=float, default=T_DEFAULT)
    ap.add_argument("--Nk", type=int, default=NK_DEFAULT)
    ap.add_argument("--dmin", type=float, default=0.02)
    ap.add_argument("--dmax", type=float, default=0.25)
    ap.add_argument("--step", type=float, default=0.01)
    a = ap.parse_args()

    if a.real is not None:
        for d in a.real:
            k = solve_at_doping(d, T=a.T, Nk=a.Nk)
            r = solve_at_doping_real(d, T=a.T)
            print(f"delta={d:.3f}  k-space: m={k['m']:.4f} chi={k['chi']:.4f} Delta={k['Delta']:.4f} | "
                  f"real-space {Lx}x{Ly}: m={r['m']:.4f} chi={r['chi']:.4f} Delta={r['Delta']:.4f} "
                  f"(it={r['iters']+1}, diff={r['diff']:.1e})")
        sys.exit(0)

    data_dir = os.path.join(os.path.dirname(HERE), "data")
    os.makedirs(data_dir, exist_ok=True)
    import matplotlib.pyplot as plt

    deltas = np.arange(a.dmin, a.dmax + 1e-9, a.step)
    seed, results = None, []
    for d in deltas:
        res = solve_at_doping(d, seed=seed, T=a.T, Nk=a.Nk)
        seed = (res['m'], res['chi'], res['Delta'], res['mu'], res['chi_nnn'])
        results.append((d, res['m'], res['chi'], res['Delta']))
        print(f"delta={d:.3f}  m={res['m']:.4f}  chi={res['chi']:.4f}  Delta={res['Delta']:.4f}  "
              f"iters={res['iters']}", flush=True)
    with open(os.path.join(data_dir, "homogeneous_scan_24x24.pkl"), "wb") as f:
        pickle.dump(results, f)

    d_arr = np.array([r[0] for r in results])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(d_arr, [r[2] for r in results], color="green", label=r"$\chi$ (ours)")
    ax.plot(d_arr, [r[1] for r in results], color="red", label=r"$m$ (ours)")
    ax.plot(d_arr, [r[3] for r in results], color="blue", label=r"$\Delta$ (ours)")
    ax.plot(PAPER_DELTA, PAPER_CHI, "o--", color="darkgreen", alpha=0.6, label=r"$\chi$ (paper)")
    ax.plot(PAPER_DELTA, PAPER_M, "o--", color="darkred", alpha=0.6, label=r"$m$ (paper)")
    ax.plot(PAPER_DELTA, PAPER_DELTA_SC, "o--", color="navy", alpha=0.6, label=r"$\Delta$ (paper)")
    ax.set_xlabel(r"doping $\delta$"); ax.set_ylabel("order parameter")
    ax.set_title(f"Homogeneous GBdG (v2 model), thermodynamic limit, T={a.T}")
    ax.legend(fontsize=8); ax.set_xlim(0, 0.26); ax.set_ylim(0, 0.30)
    plt.tight_layout()
    out = os.path.join(data_dir, f"fig1_homogeneous_T{a.T:g}_Nk{a.Nk}_d{a.dmin:g}-{a.dmax:g}_{sd.stamp()}.png")
    plt.savefig(out, dpi=140)
    print("saved", out)
    plt.show()
