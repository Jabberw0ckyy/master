# Repository for the code for Master Thesis
There will be all code used in my thesis, starting with...
## Gutzwiller-BdG Solver for the t-J Model
Real-space renormalized MF solver for the t-J
model, following Christensen, Hirschfeld, Andersen,
[*Phys. Rev. B* **84**, 184511 (2011)](https://doi.org/10.1103/PhysRevB.84.184511).

Computes the self-consistent coexistence of AFM order, d-wave superconductivity, and kinetic bond order for the
homogeneous t-J model on a square lattice. Yet with no disorder.

This sweeps doping $\delta \in [0.02, 0.25]$ and prints/saves $m(\delta)$,
$\Delta(\delta)$, $\chi(\delta)$.

## Model parameters

```python
t, tp, J = 1.0, -0.25, 0.3   # NN hop, NNN hop, superexchange (units of t)
Lx, Ly = 24, 24                 # lattice size (periodic boundary conditions)
```

`J/t = 0.3` and `t'/t = -0.25` are values for YBCO/LSCO-like
cuprates.

## Repository structure

```text
    ├── README.md
    ├── archive-kagome/
    │   └── kagome_delta_sweep.py
    └── gutzwiller-bdg-tj/
        ├── bdg/
        │   ├── lattice.py
        │   ├── solve_disordered.py
        │   ├── solve_homogeneous.py
        │   └── solve_strong_disordered.py
        ├── data/
        │   ├── type1_delta0.115_T0.1.pkl
        │   ├── type1_delta0.125_T0.1.pkl
        │   ├── type1_delta0.130_T0.1.pkl
        │   ├── type1_delta0.135_T0.1.pkl
        │   ├── type1_delta0.145_T0.1.pkl
        │   ├── type1_delta0.155_T0.1.pkl
        │   ├── type2_panela_n1pct_delta0.140.pkl
        │   ├── type2_panelb_n2pct_delta0.140.pkl
        │   ├── type2_panelc_n1pct_delta0.150.pkl
        │   └── type2_paneld_n2pct_delta0.150.pkl
        ├── ldos/
        │   ├── boker_continuum.py
        │   ├── continuum.py
        │   └── wannier.py
        └── tests/
            ├── check_continuum_vs_paper.py
            ├── diagnose_eigenvalues.py
            ├── diagnose_oscillation.py
            └── fig 3 oscillation diagnostic.py
```

lattice.py
- lay out the lattice and its bonds
- compute the Gutzwiller renormalization factors from local density and magnetism
- turn a Hamiltonian matrix into energy levels

solve_disordered.py
- runs the self-consistency loop.
- builds the mean-field Hamiltonian from the current guess
- places impurities on the lattice.
- plots Type I and Type II magnetization map

continuum.py
-takes an already converged answer from solve_disordered.py and caculates how this look under an STM tip


solve_homogeneous.py

-solves uniform state with no impurities at each doping
-scans doping and plots magnetization, kinetic term and pairing against the paper's Fig. 1 curves

solve_strong_disordered.py

-runs Type II magnetization maps (1% and 2% strong impurities, two dopings)
-shifts the dopings to match where this model's antiferromagnetism ends, unless told otherwise

uniform_k.py (to find where the antiferromagnetism disappears)

-solves the clean uniform AF plus SC in k space in the infinite-lattice limit
-same Hamiltonian as the real-space solver

ldos/

wannier.py

-defines the shape of the orbital that the STM tip "sees" around each lattice site
-plots shape

boker_continuum.py

-reproduces the Böker et al. approach for a d-wave SC with one impurity
-computes spectra and conductance maps at any tip position, using the impurity scattering T-matrix and the Wannier orbital

continuum.py

-takes a converged result from the solver and rebuilds its energy levels
-turns them into STM-like local density of states at any tip position: spectra at chosen points and maps at chosen bias

af_impurity.py

-treats the pure AF with a single impurity
-computes the response via Green's functions and the impurity scattering, on an infinite lattice
-produces STM spectra and maps near and far from the impurity, plus how the signal decays with distance
