"""
V_i = 100t, panels:
(a) n_imp=1%, delta=0.14, (b) n_imp=2%, delta=0.14
(c) n_imp=1%, delta=0.15, d n_imp=2%, delta=0.15

"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from solve_disordered import solve_disordered, place_impurities, t, N, Lx, Ly, DEFAULT_T

PANELS = [
    ('a', 0.01, 0.14),
    ('b', 0.02, 0.14),
    ('c', 0.01, 0.15),
    ('d', 0.02, 0.15),
]
V_STRONG_FACTOR = 100.0
MAX_ITER = 700
MIX = 0.10
SEED_RNG = 3

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    results = {}
    for k, (label, n_imp_frac, d) in enumerate(PANELS):
        rng = np.random.default_rng(SEED_RNG + k)
        V_i, imp_sites = place_impurities(n_imp_frac, V_STRONG_FACTOR * t, rng)

        print(f"\n=== panel ({label}): n_imp={n_imp_frac*100:.0f}%  delta={d:.3f}  "
              f"({len(imp_sites)} impurities) ===")
        res = solve_disordered(d, V_i, seed=None, max_iter=MAX_ITER, mix=MIX,
                                tol=1e-4, verbose=True, T=DEFAULT_T)
        res['imp_sites'] = imp_sites
        res['delta_target'] = d
        res['n_imp_frac'] = n_imp_frac
        res['V_i'] = V_i.copy()
        results[label] = res

        mean_m = np.mean(np.abs(res['m_i']))
        max_m = np.max(np.abs(res['m_i']))
        conv = "OK" if res['diff'] < 1e-4 else "not converged"
        print(f"    panel ({label}) done: iters={res['iters']} diff={res['diff']:.2e}  "
              f"mean|m|={mean_m:.5f}  max|m|={max_m:.4f}  {conv}")

        fname = os.path.join(data_dir, f"type2_panel{label}_n{n_imp_frac*100:.0f}pct_delta{d:.3f}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(res, f)
        print(f"    saved {fname}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    ax_map = {'a': axes[0, 0], 'b': axes[0, 1], 'c': axes[1, 0], 'd': axes[1, 1]}

    for label, n_imp_frac, d in PANELS:
        res = results[label]
        ax = ax_map[label]
        m_map = res['m_i'].reshape(Ly, Lx)
        im = ax.imshow(m_map, cmap='jet', vmin=-0.1, vmax=0.1, origin='lower')
        imp_y = res['imp_sites'] // Lx
        imp_x = res['imp_sites'] % Lx
        ax.scatter(imp_x, imp_y, c='white', s=40, edgecolors='k')
        conv = "OK" if res['diff'] < 1e-4 else "NOT CONV."
        ax.set_title(f"({label}) n_imp={n_imp_frac*100:.0f}%, δ={d:.2f}\n"
                      f"⟨|m|⟩={np.mean(np.abs(res['m_i'])):.4f}  ({conv}, diff={res['diff']:.1e})",
                      fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.7, label=r"$m_i$")
    fig.suptitle(f"Type II disorder, V=100t, T={DEFAULT_T}")
    out_path = os.path.join(data_dir, "fig3_type2_maps.png")
    plt.savefig(out_path, dpi=140)
    plt.show()
    print(f"\nsaved {out_path}")
