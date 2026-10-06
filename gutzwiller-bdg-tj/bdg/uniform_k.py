import numpy as np
from scipy.optimize import brentq
import solve_disordered as sd
from solve_disordered import N, Lx, nn_i, nn_j, nnn_i, nnn_j, nn_sign, neel_sign, fermi_smeared


def params(delta, m, chi_up, chi_dn, Dl, chn_Au, chn_Ad, kappa):
    nb, nnb = len(nn_i), len(nnn_i)
    isA = (neel_sign[nnn_i] > 0)
    s = dict(delta_i=np.full(N, delta), m_i=m * neel_sign,
             chi_up=np.full(nb, chi_up), chi_dn=np.full(nb, chi_dn), Delta_nn=Dl * nn_sign,
             chin_up=np.where(isA, chn_Au, chn_Ad), chin_dn=np.where(isA, chn_Ad, chn_Au), mu=0.0)
    H0up, H0dn, Dm = sd.build_hamiltonian(s, np.zeros(N), kappa)
    idx = lambda x, y: (y % sd.Ly) * Lx + (x % Lx)
    p = dict(
        hAu=H0up[idx(0, 0), idx(0, 0)], hBu=H0up[idx(1, 0), idx(1, 0)],
        hAd=H0dn[idx(0, 0), idx(0, 0)], hBd=H0dn[idx(1, 0), idx(1, 0)],
        t1u=H0up[idx(0, 0), idx(1, 0)], t1d=H0dn[idx(0, 0), idx(1, 0)],
        tAu=H0up[idx(0, 0), idx(1, 1)], tBu=H0up[idx(1, 0), idx(2, 1)],
        tAd=H0dn[idx(0, 0), idx(1, 1)], tBd=H0dn[idx(1, 0), idx(2, 1)],
        P=Dm[idx(0, 0), idx(1, 0)])
    return p


def solve_k(delta, T, kappa, Nk=96, mix=0.3, tol=1e-9, it_max=3000, init=(0.15, 0.185, 0.11, 0.03, 0.03), verbose=False):
    k = 2 * np.pi * (np.arange(Nk) + 0.5) / Nk
    KX, KY = np.meshgrid(k, k, indexing="ij")
    gam = 2 * (np.cos(KX) + np.cos(KY)); dk = 2 * (np.cos(KX) - np.cos(KY)); cc = np.cos(KX) * np.cos(KY)
    cn = 0.5 * (np.cos(KX) + np.cos(KY))        # bond-average phase factor
    cd = 0.5 * (np.cos(KX) - np.cos(KY))
    m, chi, Dl, cAu, cAd = init
    mu = 1.6
    for it in range(it_max):
        p = params(delta, m, chi, chi, Dl, cAu, cAd, kappa)
        H = np.zeros(KX.shape + (4, 4))
        H[..., 0, 0] = p['hAu'] + 4 * p['tAu'] * cc; H[..., 1, 1] = p['hBu'] + 4 * p['tBu'] * cc
        H[..., 0, 1] = H[..., 1, 0] = p['t1u'] * gam
        H[..., 2, 2] = -(p['hAd'] + 4 * p['tAd'] * cc); H[..., 3, 3] = -(p['hBd'] + 4 * p['tBd'] * cc)
        H[..., 2, 3] = H[..., 3, 2] = -p['t1d'] * gam
        H[..., 0, 3] = H[..., 3, 0] = p['P'] * dk; H[..., 1, 2] = H[..., 2, 1] = p['P'] * dk
        tau = np.diag([1.0, 1.0, -1.0, -1.0])
        def solve_state(mu_):
            E, V = np.linalg.eigh(H - mu_ * tau)
            return E, V
        def dop(mu_):
            E, V = solve_state(mu_)
            f = fermi_smeared(E, T)
            nu = (np.abs(V[..., 0, :]) ** 2 + np.abs(V[..., 1, :]) ** 2) * f
            nd = (np.abs(V[..., 2, :]) ** 2 + np.abs(V[..., 3, :]) ** 2) * (1 - f)
            return 1 - (nu.sum(-1).mean() + nd.sum(-1).mean()) / 2 - delta
        mu = brentq(dop, mu - 0.5, mu + 0.5, xtol=1e-12) if dop(mu - 0.5) * dop(mu + 0.5) < 0 else brentq(dop, -6, 6, xtol=1e-12)
        E, V = solve_state(mu)
        f = fermi_smeared(E, T)
        uA, uB, vA, vB = V[..., 0, :], V[..., 1, :], V[..., 2, :], V[..., 3, :]
        nAu = (np.abs(uA) ** 2 * f).sum(-1).mean(); nAd = (np.abs(vA) ** 2 * (1 - f)).sum(-1).mean()
        nBu = (np.abs(uB) ** 2 * f).sum(-1).mean(); nBd = (np.abs(vB) ** 2 * (1 - f)).sum(-1).mean()
        m_new = 0.25 * ((nAu - nAd) - (nBu - nBd))
        chi_u = (np.real(uA.conj() * uB) * f).sum(-1) * cn
        chi_d = (np.real(vA.conj() * vB) * (1 - f)).sum(-1) * cn
        chi_new = 0.5 * (chi_u.mean() + chi_d.mean())
        cAu_n = 0.5 * (((np.abs(uA) ** 2 * f).sum(-1) * cc).mean() + ((np.abs(vB) ** 2 * (1 - f)).sum(-1) * cc).mean())   # A-A up  <-> B-B down
        cAd_n = 0.5 * (((np.abs(vA) ** 2 * (1 - f)).sum(-1) * cc).mean() + ((np.abs(uB) ** 2 * f).sum(-1) * cc).mean())   # A-A down <-> B-B up
        F_AB = ((1 - f) * uA * vB.conj()).sum(-1); G_AB = (f * vA * uB.conj()).sum(-1)
        F_BA = ((1 - f) * uB * vA.conj()).sum(-1); G_BA = (f * vB * uA.conj()).sum(-1)
        Db = 0.25 * (np.real(F_AB - G_AB) + np.real(F_BA - G_BA))
        Dl_new = (Db * cd).mean() * 1.0
        diff = max(abs(m_new - m), abs(chi_new - chi), abs(Dl_new - Dl), abs(cAu_n - cAu), abs(cAd_n - cAd))
        m += mix * (m_new - m); chi += mix * (chi_new - chi); Dl += mix * (Dl_new - Dl); cAu += mix * (cAu_n - cAu); cAd += mix * (cAd_n - cAd)
        if verbose and it % 50 == 0:
            print(it, f"diff={diff:.1e} m={m:.4f} chi={chi:.4f} D={Dl:.4f}")
        if diff < tol and it > 10:
            break
    return dict(m=m, chi=chi, Delta=Dl, cAu=cAu, cAd=cAd, mu=mu, iters=it, diff=diff)


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser(description="uniform (pi,pi) AF + d-wave SC, thermodynamic limit")
    ap.add_argument("--T", type=float, default=0.02)
    ap.add_argument("--kappa", type=float, default=sd.KAPPA)
    ap.add_argument("--Nk", type=int, default=64)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.10, 0.13, 0.15, 0.17, 0.20])
    a = ap.parse_args()
    init = (0.15, 0.185, 0.11, 0.03, 0.03)
    print(f"{'delta':>6} {'m':>8} {'chi':>8} {'Delta':>8} {'mu':>8}   (T={a.T}, kappa={a.kappa}, Nk={a.Nk})")
    for d in a.deltas:
        r = solve_k(d, a.T, a.kappa, Nk=a.Nk, init=init, tol=1e-8, mix=0.4, it_max=500)
        init = (max(r['m'], 0.05), r['chi'], r['Delta'], r['cAu'], r['cAd'])
        print(f"{d:6.3f} {r['m']:8.4f} {r['chi']:8.4f} {r['Delta']:8.4f} {r['mu']:8.4f}")
