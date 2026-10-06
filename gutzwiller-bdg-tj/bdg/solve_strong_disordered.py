import os
import sys
import pickle
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solve_disordered as sd
from solve_disordered import t, N, Lx, Ly, DEFAULT_T, DELTA_SHIFT

PANELS = [
    ('a', 0.01, 0.14),
    ('b', 0.02, 0.14),
    ('c', 0.01, 0.15),
    ('d', 0.02, 0.15),
]
V_STRONG_FACTOR = 100.0
SEED_RNG = 3

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Type II magnetization maps (solver v2)")
    ap.add_argument("--panels", nargs="+", default=[p[0] for p in PANELS])
    ap.add_argument("--paper-dopings", action="store_true", dest="paper_dopings")
    ap.add_argument("--T", type=float, default=DEFAULT_T)
    ap.add_argument("--iters", type=int, default=1000, help="max iterations per panel")
    ap.add_argument("--mix", type=float, default=0.15)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--m0", type=float, default=0.02, help="initial staggered m (small = nucleation test)")
    ap.add_argument("--save", action="store_true", help="also write one result pickle per panel to ../data")
    a = ap.parse_args()

    import matplotlib.pyplot as plt
    chosen = [p for p in PANELS if p[0] in a.panels]
    shift = 0.0 if a.paper_dopings else DELTA_SHIFT
    panels = [(n_frac, round(d + shift, 4)) for _, n_frac, d in chosen]

    results = []
    for (label, n_frac, d_paper), (_, d) in zip(chosen, panels):
        k = [p[0] for p in PANELS].index(label)
        res = sd.run_fig3(panels=((n_frac, d),), V_strong_factor=V_STRONG_FACTOR, max_iter=a.iters,
                          mix=a.mix, seed_rng=SEED_RNG + k, T=a.T, tol=a.tol, m0=a.m0)[0]
        results.append((label, res))
        print(f"    panel ({label}): mean|m|={np.mean(np.abs(res['m_i'])):.5f}  max|m|={np.max(np.abs(res['m_i'])):.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    ax_map = {'a': axes[0, 0], 'b': axes[0, 1], 'c': axes[1, 0], 'd': axes[1, 1]}
    im = None
    for label, res in results:
        ax = ax_map[label]
        im = ax.imshow(res['m_i'].reshape(Ly, Lx), cmap='jet', vmin=-0.1, vmax=0.1, origin='lower')
        ax.scatter(res['imp_sites'] % Lx, res['imp_sites'] // Lx, c='white', s=40, edgecolors='k')
        conv = "OK" if res['diff'] < 1e-4 else "NOT CONV."
        ax.set_title(f"({label}) n_imp={res['n_imp_frac']*100:.0f}%, δ={res['delta_target']:.3f}\n"
                     f"⟨|m|⟩={np.mean(np.abs(res['m_i'])):.4f}  ({conv}, diff={res['diff']:.1e}, "
                     f"{res['iters'] + 1} it)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.7, label=r"$m_i$")
    fig.suptitle(f"Type II disorder, V=100t, T={a.T}" + ("" if a.paper_dopings else f"  (dopings shifted by +{DELTA_SHIFT})"))

    os.makedirs(sd.DATA, exist_ok=True)
    tag = "".join(l for l, _ in results) + f"_T{a.T:g}_it{a.iters}" + ("_paperdop" if a.paper_dopings else f"_shift{DELTA_SHIFT:g}")
    out = os.path.join(sd.DATA, f"fig3_type2_maps_{tag}_{sd.stamp()}.png")
    plt.savefig(out, dpi=140)
    print("\nsaved", out)
    if a.save:
        for label, res in results:
            fname = os.path.join(sd.DATA, f"type2_panel{label}_n{res['n_imp_frac']*100:.0f}pct_delta{res['delta_target']:.3f}.pkl")
            with open(fname, "wb") as f:
                pickle.dump(res, f)
            print("saved", fname)
    plt.show()
