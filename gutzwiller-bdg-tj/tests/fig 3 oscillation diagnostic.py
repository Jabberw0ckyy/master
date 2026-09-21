import numpy as np
from solve_disordered import solve_disordered, place_impurities, t

#panel (d)
rng = np.random.default_rng(3 + 3)
V_i, imp_sites = place_impurities(0.02, 100.0 * t, rng)

res = solve_disordered(0.15, V_i, seed=None, max_iter=60, mix=0.10,
                        tol=1e-4, verbose=True, T=0.1)

print(f"\nFINAL: iters={res['iters']} diff={res['diff']:.2e} "
      f"mean|m|={np.mean(np.abs(res['m_i'])):.4f}")