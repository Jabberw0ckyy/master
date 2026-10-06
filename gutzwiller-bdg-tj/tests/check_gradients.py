"""Checks that the v2 mean-field Hamiltonian is the exact derivative of the energy functional W
(Eqs. 16-18 of PRB 84, 184511) on a random, non-uniform, spin-resolved state.
Expected: (a), (b), (b') ~1e-8; (c) a constant ratio (+-4) on every bond.   Run: python check_gradients.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bdg'))
import numpy as np
import solve_disordered as sd
from solve_disordered import *
rng = np.random.default_rng(1)
nb, nnb = len(nn_i), len(nnn_i)
s = dict(delta_i=0.15 + 0.05*rng.random(N), m_i=0.12*neel_sign*(0.5+rng.random(N)) + 0.02*rng.standard_normal(N),
         chi_up=0.18 + 0.03*rng.standard_normal(nb), chi_dn=0.18 + 0.03*rng.standard_normal(nb),
         Delta_nn=nn_sign*(0.09 + 0.02*rng.standard_normal(nb)),
         chin_up=0.05 + 0.02*rng.standard_normal(nnb), chin_dn=0.05 + 0.02*rng.standard_normal(nnb), mu=0.0)
W = lambda st: free_energy_terms(st)*N
H0up, H0dn, Dm = build_hamiltonian(s, np.zeros(N))

# (a) on-site fields dW/dn_{i,sigma}
gsxy, _, _ = sd._gfac(s); B = bracket(s)
e = 1e-6; worst = 0
for sig, h in ((+1.0, onsite_field(s, +1.0, gsxy, B)), (-1.0, onsite_field(s, -1.0, gsxy, B))):
    for i in rng.choice(N, 8, replace=False):
        sp = {k: np.array(v, dtype=float).copy() for k, v in s.items()}; sm = {k: np.array(v, dtype=float).copy() for k, v in s.items()}
        sp['delta_i'][i] -= e; sp['m_i'][i] += sig*e/2
        sm['delta_i'][i] += e; sm['m_i'][i] -= sig*e/2
        num = (W(sp) - W(sm))/(2*e)
        worst = max(worst, abs(num - h[i]))
print("(a) max |dW/dn_i,sigma  -  onsite field| over 16 random (site,spin):", f"{worst:.2e}")

# (b) nn hopping: matrix element vs (1/2) dW/dchi_sigma
worst = 0
for name, key, H in (("up", 'chi_up', H0up), ("dn", 'chi_dn', H0dn)):
    for b in rng.choice(nb, 10, replace=False):
        sp = {k: np.array(v, dtype=float).copy() for k, v in s.items()}; sm = {k: np.array(v, dtype=float).copy() for k, v in s.items()}
        sp[key][b] += e; sm[key][b] -= e
        num = 0.5*(W(sp) - W(sm))/(2*e)
        worst = max(worst, abs(num - H[nn_i[b], nn_j[b]]))
print("(b) max |1/2 dW/dchi_nn  -  hopping element| over 20 random bonds:", f"{worst:.2e}")
# (b') nnn
worst = 0
for key, H in (('chin_up', H0up), ('chin_dn', H0dn)):
    for b in rng.choice(nnb, 10, replace=False):
        sp = {k: np.array(v, dtype=float).copy() for k, v in s.items()}; sm = {k: np.array(v, dtype=float).copy() for k, v in s.items()}
        sp[key][b] += e; sm[key][b] -= e
        worst = max(worst, abs(0.5*(W(sp) - W(sm))/(2*e) - H[nnn_i[b], nnn_j[b]]))
print("(b') nnn:", f"{worst:.2e}")
# (c) pairing: matrix element vs dW/dDelta (unsigned), up to the BdG sign/normalisation
ratios = []
for b in rng.choice(nb, 10, replace=False):
    sp = {k: np.array(v, dtype=float).copy() for k, v in s.items()}; sm = {k: np.array(v, dtype=float).copy() for k, v in s.items()}
    sp['Delta_nn'][b] += e*nn_sign[b]; sm['Delta_nn'][b] -= e*nn_sign[b]
    num = (W(sp) - W(sm))/(2*e)
    ratios.append(num/(nn_sign[b]*Dm[nn_i[b], nn_j[b]]))
print("(c) dW/dDelta / (pairing element): ", np.round(ratios, 6))
