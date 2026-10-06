# v2 changes (Oct 2026)

## Replaced
- `bdg/solve_disordered.py`  -> solver v2 (equations of PRB 84, 184511 Eqs. 8-13, 16-18 literally):
  spin-resolved chi_{ij,sigma} (nn and nnn), bracket B_ij, d g^{sz}/d chi and d g^{sz}/d Delta
  terms (coefficient J/16, `KAPPA`), separate mixing of delta_i fluctuations (`MIX_FLUCT=0.03`),
  exact mu every iteration (`mu_mode="secant"`).  Defaults: T=0.02 (was 0.1), pairing on.
  `__main__` is again the Type I figure: 6 panels in one row (nested impurity sets, n_imp = delta*N,
  V = t), only m_i maps, plt.show(), figure saved as ../data/fig2_type1_maps_<params>_<time stamp>.png.
  Dopings = paper's (0.115 ... 0.155) + DELTA_SHIFT (0.055), `--paper-dopings` / `--deltas ...` to change.
  `run_fig2`, `run_fig3` (solver loops) are in the file; `afm_field`, `bdg_expectation`, 6-tuple `seed`
  are kept as v1 shims.  Public names used by other files are unchanged.
- `ldos/continuum.py`  -> imports `build_hamiltonian` instead of `afm_field`; `rebuild_bdg_from_result`
  uses the v2 Hamiltonian (old-format pkl files still load, spin-symmetric -- regenerate them).

## New
- `bdg/uniform_k.py`     uniform (pi,pi)-AF + d-wave SC, thermodynamic limit, same Hamiltonian.
- `ldos/af_impurity.py`  pure-AF host + one impurity: Green function, T-matrix, Wannier-filtered
                          STM near / far from the impurity (v2 equations).
- `tests/check_gradients.py`  Hamiltonian == derivative of the energy functional (expect ~1e-8).

## Also changed (were inconsistent with v2)
- `bdg/solve_homogeneous.py`  rewritten as a thin layer over the v2 Hamiltonian (k-space, thermodynamic limit,
  T=0.02); old API `solve_at_doping(...)` kept; `--real` cross-checks on the Lx x Ly lattice.
- `bdg/solve_strong_disordered.py`  v2; same 2x2 figure as before, dopings shifted by +0.055 by default
  (`--paper-dopings` for the paper's values, which lie inside the clean AF phase of this model).

## Saving
Nothing but the figure is written by default (no pickles, no checkpoints).  `--save` writes the result
pickles (type1_delta<d>_T<T>.pkl / type2_panel<x>_...pkl) that ldos/continuum.py reads.

## Not changed
- `bdg/lattice.py`, `ldos/wannier.py`, `ldos/boker_continuum.py`.
- `tests/check_continuum_vs_paper.py`, `diagnose_eigenvalues.py`, `diagnose_oscillation.py`,
  `fig 3 oscillation diagnostic.py`, `README.md`, `archive-kagome/`  (not seen by me; the shims above are the only
  compatibility measure -- upload them if any fails).
- `data/*.pkl` (T=0.1, old equations) are obsolete: regenerate with `solve_disordered.py` /
  `solve_strong_disordered.py` / `solve_homogeneous.py`.

## Figures
All scripts that save images now put parameters and a time stamp in the file name
(solve_disordered, solve_strong_disordered, solve_homogeneous, continuum, af_impurity): no overwriting.

## Known open points
- Clean AF boundary of the model is delta ~ 0.17 (paper Fig. 1: ~0.12); m(delta) larger than in Fig. 1;
  chi(delta) flat (paper: rises 0.175 -> 0.24).  Unexplained.
- Near the boundary there is a soft mode at q = (pi,pi) +- (0, 2pi/8); `diff` ~ 1e-4 does NOT mean converged.
