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

lattice.py
- lay out the lattice and its bonds
- compute the Gutzwiller renormalization factors from local density and magnetism
- turn a Hamiltonian matrix into energy levels

solve_disordered.py
- runs the actual self-consistency loop: guess an answer, compute what the physics predicts from that guess, nudge the guess toward the prediction, repeat until agreed. 
- places impurities on the lattice.

continuum.py
-takes an already converged answer from solve_disordered.py and caculates how this look under an STM tip

## What might be problematic
The Type I disorder scan hasn't really converged, (in and out of self-consistensy loop)
//new_input = (1 - mix) * old_input + mix * output; then  DIIS/ Anderson mixing  (doesn't help)
Introducing small finite temperature helped.
But the paper says as well that their equations do misbehave close to half-filling/very low doping, for reasons built into their simplified formulas. they avoided testing that region.
