import numpy as np
import boker_continuum as bc
import continuum as cn

L, eta = 12, 4.0
N = L * L
xs = np.repeat(np.arange(L), L).astype(float)
ys = np.tile(np.arange(L), L).astype(float)
idx = lambda x, y: (x % L) * L + (y % L)

H = np.zeros((N, N)); D = np.zeros((N, N))
for x in range(L):
    for y in range(L):
        i = idx(x, y)
        H[i, i] += bc.MU
        for dx, dy, tt in [(1, 0, bc.T1 / 4), (0, 1, bc.T1 / 4), (1, 1, bc.T2 / 4),
                           (1, -1, bc.T2 / 4), (2, 0, bc.T3 / 4), (0, 2, bc.T3 / 4)]:
            j = idx(x + dx, y + dy); H[i, j] += tt; H[j, i] += tt
        for dx, dy, s in [(1, 0, 1), (0, 1, -1)]:
            j = idx(x + dx, y + dy); D[i, j] += s * bc.DELTA0 / 4; D[j, i] += s * bc.DELTA0 / 4
E, psi = np.linalg.eigh(np.block([[H, D], [D, -H]]))

k = 2 * np.pi * np.arange(L) / L
KX, KY = np.meshgrid(k, k, indexing="ij")
om = np.linspace(-1.5, 1.5, 13) * bc.DELTA0
pos = E > 0

def kspace(r):
    wk = bc.w_k(r, KX, KY)
    return np.array([-np.mean(abs(wk) ** 2 * bc.nambu_G(o + 1j * eta, KX, KY)[0]).imag / np.pi
                     for o in om])

def realspace(r, modes):
    return cn.continuum_ldos(r[0], r[1], xs, ys, E[modes], psi[:, modes], N, om,
                             eta=eta, Lx_=L, Ly_=L)

ok = True
for r in [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.3, 0.8)]:
    ref = kspace(r)
    r_all = realspace(r, np.ones(2 * N, bool)) / ref
    r_pos = realspace(r, pos) / ref
    print(f"r={r}:  all 2N modes / paper = {r_all.mean():.4f} (spread {np.ptp(r_all):.1e});"
          f"  E>0 modes only / paper = {r_pos.mean():.4f} (spread {np.ptp(r_pos):.1e})")
    ok &= np.allclose(r_all, 2.0) and np.allclose(r_pos, 1.0)
print("PASS" if ok else "FAIL")
