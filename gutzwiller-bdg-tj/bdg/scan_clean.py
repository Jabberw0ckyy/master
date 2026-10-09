"""Clean (V=0) m(delta) scan on the production lattice -> find the AF boundary of THIS lattice size.
Continuation: scan from low to high delta, seeding each point from the previous converged state.
Stops each delta as soon as diff < tol.   usage: python scan_clean.py 0.10 0.12 0.14 0.15 0.16 0.17
"""
import sys, pickle, numpy as np
import solve_disordered as S

deltas = [float(x) for x in sys.argv[1:]] or [0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
V = np.zeros(S.N)
seed, out = None, []
for d in sorted(deltas):
    st = S._seed_state(d, 0.12) if seed is None else seed
    r = S.solve_disordered(d, V, seed=st, max_iter=1500, mix=0.15, tol=1e-5, T=S.DEFAULT_T)
    ok = r['diff'] < 1e-5
    m = np.abs(r['m_i']).mean()
    print(f"L={S.Lx} delta={d:.3f}  <|m|>={m:.4f}  <|Delta|>={np.abs(r['Delta_nn']).mean():.4f}  "
          f"chi={r['chi_up'].mean():.4f}  diff={r['diff']:.1e}  it={r['iters']+1}  {'OK' if ok else 'NOT CONVERGED'}",
          flush=True)
    out.append((d, m, ok))
    seed = r if ok else None          # never continue from a non-converged state
    pickle.dump(out, open(f"scan_clean_L{S.Lx}.pkl", "wb"))
