"""af_sc.py: single non-magnetic impurity in the doped antiferromagnet, STM spectra with a responsive host.
Run with -h for all options."""
import os
import sys
import time
import pickle
import argparse
import numpy as np
from scipy.special import polygamma
from scipy.optimize import newton_krylov

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
for _p in (os.path.join(_PROJECT_ROOT, "bdg"), _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import solve_disordered as sd
import af_impurity as af
from wannier import wannier_envelope

N, Lx, Ly = sd.N, sd.Lx, sd.Ly
IMP_X, IMP_Y = 12, 12
SITE = IMP_Y * Lx + IMP_X


def cluster(c):
    d1 = np.arange(-c, c + 1)
    DX, DY = np.meshgrid(d1, d1, indexing="ij")
    dx, dy = DX.ravel(), DY.ravel()
    return np.stack([dx, dy], 1), (dx + dy) % 2


def neighborhood(tip, cells=5, gamma=af.GAMMA):
    base = np.floor(np.asarray(tip, float)).astype(int)
    d1 = np.arange(-cells, cells + 1)
    DX, DY = np.meshgrid(d1, d1, indexing="ij")
    dx, dy = base[0] + DX.ravel(), base[1] + DY.ravel()
    K = wannier_envelope(tip[0] - dx, tip[1] - dy, gamma=gamma)
    return np.stack([dx, dy], 1), K, (dx + dy) % 2


def g0_pairs(pa, aa, pb, ab, gAA, gBB, gAB, Nk):
    ddx = pa[:, None, 0] - pb[None, :, 0]
    ddy = pa[:, None, 1] - pb[None, :, 1]
    ix, iy = ddx % Nk, ddy % Nk
    nix, niy = (-ddx) % Nk, (-ddy) % Nk
    code = 2 * aa[:, None] + ab[None, :]
    return np.where(code == 0, gAA[ix, iy], np.where(code == 3, gBB[ix, iy],
                    np.where(code == 1, gAB[ix, iy], gAB[nix, niy])))


def sub_offsets(sub):
    s1 = (np.arange(sub) + 0.5) / sub - 0.5
    SX, SY = np.meshgrid(s1, s1, indexing="ij")
    return SX.ravel(), SY.ravel()


def embed_spectra(host, perts, tip, omegas, eta, Nk, c):
    prm = af.host_tb_params(host)
    N_pos, K, N_a = neighborhood(tip)
    C_pos, C_a = cluster(c)
    eye = np.eye(len(C_pos))
    rho = {k: np.zeros(len(omegas)) for k in perts}
    rho_cl = np.zeros(len(omegas))
    for n, om in enumerate(omegas):
        for spin in ("up", "dn"):
            gAA, gBB, gAB = af.host_green_real(prm, spin, om + 1j * eta, Nk)
            G0NN = g0_pairs(N_pos, N_a, N_pos, N_a, gAA, gBB, gAB, Nk)
            G0NC = g0_pairs(N_pos, N_a, C_pos, C_a, gAA, gBB, gAB, Nk)
            G0CC = g0_pairs(C_pos, C_a, C_pos, C_a, gAA, gBB, gAB, Nk)
            host_term = K @ G0NN @ K
            w = G0NC.T @ K
            rho_cl[n] += -host_term.imag / np.pi
            for key, dh in perts.items():
                d = dh[spin]
                T = np.linalg.solve(eye - d @ G0CC, d)
                rho[key][n] += -(host_term + w @ T @ w).imag / np.pi
    return rho, rho_cl


def embed_map(host, perts, omega, eta, Nk, c, half, sub, cells=5):
    prm = af.host_tb_params(host)
    C_pos, C_a = cluster(c)
    eye = np.eye(len(C_pos))
    sx, sy = sub_offsets(sub)
    fx, fy = np.floor(sx).astype(int), np.floor(sy).astype(int)
    R1 = np.arange(-half, half + 1)
    nR = len(R1)
    RX, RY = np.meshgrid(R1, R1, indexing="ij")
    RX, RY = RX.ravel(), RY.ravel()
    d1 = np.arange(-cells, cells + 1)
    DX, DY = np.meshgrid(d1, d1, indexing="ij")
    DX, DY = DX.ravel(), DY.ravel()
    out = {k: np.zeros((len(RX), len(sx))) for k in perts}
    for spin in ("up", "dn"):
        gAA, gBB, gAB = af.host_green_real(prm, spin, omega + 1j * eta, Nk)
        G0CC = g0_pairs(C_pos, C_a, C_pos, C_a, gAA, gBB, gAB, Nk)
        Ts = {k: np.linalg.solve(eye - dh[spin] @ G0CC, dh[spin]) for k, dh in perts.items()}
        for gx in (-1, 0):
            for gy in (-1, 0):
                sel = np.where((fx == gx) & (fy == gy))[0]
                if len(sel) == 0:
                    continue
                K = wannier_envelope(sx[sel][None, :] - (gx + DX)[:, None],
                                     sy[sel][None, :] - (gy + DY)[:, None], gamma=af.GAMMA)
                for ir in range(len(RX)):
                    Npos = np.stack([RX[ir] + gx + DX, RY[ir] + gy + DY], 1)
                    Na = (Npos[:, 0] + Npos[:, 1]) % 2
                    W = g0_pairs(Npos, Na, C_pos, C_a, gAA, gBB, gAB, Nk).T @ K
                    for k in perts:
                        out[k][ir, sel] += -np.einsum("ks,kl,ls->s", W, Ts[k], W).imag / np.pi

    def to_img(a):
        return a.reshape(nR, nR, sub, sub).transpose(1, 3, 0, 2).reshape(nR * sub, nR * sub)
    ext = [-half - 0.5, half + 0.5, -half - 0.5, half + 0.5]
    return {k: to_img(v) for k, v in out.items()}, ext


def frozen_delta_h(V, pos):
    n = len(pos)
    m = np.zeros((n, n))
    c0 = int(np.where((pos[:, 0] == 0) & (pos[:, 1] == 0))[0][0])
    m[c0, c0] = V
    return {"up": m.copy(), "dn": m.copy()}


def lattice_index(pos, site=SITE):
    x0, y0 = int(sd.xs[site]), int(sd.ys[site])
    return ((y0 + pos[:, 1]) % Ly) * Lx + (x0 + pos[:, 0]) % Lx


def base_state(host):
    nb, nnb = len(sd.nn_i), len(sd.nnn_i)
    return dict(delta_i=np.full(N, host["delta_target"]), m_i=host["m_i"].copy(),
                chi_up=host["chi_up"].copy(), chi_dn=host["chi_dn"].copy(),
                Delta_nn=np.zeros(nb), chin_up=host["chin_up"].copy(),
                chin_dn=host["chin_dn"].copy(), mu=host["mu"])


class SelfConsistentEmbedding:

    def __init__(self, host, c=5, Nk=256, n_mats=300, T=None, build_g0=True):
        self.host, self.c, self.Nk = host, c, Nk
        self.T = host["T"] if T is None else T
        self.base = base_state(host)
        self.pos, self.sub = cluster(c)
        self.nC = len(self.pos)
        self.idx = lattice_index(self.pos)
        self.cl_of = -np.ones(N, int)
        self.cl_of[self.idx] = np.arange(self.nC)
        self.nn_in = np.where((self.cl_of[sd.nn_i] >= 0) & (self.cl_of[sd.nn_j] >= 0))[0]
        self.nnn_in = np.where((self.cl_of[sd.nnn_i] >= 0) & (self.cl_of[sd.nnn_j] >= 0))[0]
        self.ai, self.aj = self.cl_of[sd.nn_i[self.nn_in]], self.cl_of[sd.nn_j[self.nn_in]]
        self.bi, self.bj = self.cl_of[sd.nnn_i[self.nnn_in]], self.cl_of[sd.nnn_j[self.nnn_in]]
        self.n_mats = n_mats
        wn = (2 * np.arange(n_mats) + 1) * np.pi * self.T
        prm = af.host_tb_params(host)
        self.G0 = {}
        for spin in (("up", "dn") if build_g0 else ()):
            G = np.empty((n_mats, self.nC, self.nC), complex)
            for n, w in enumerate(wn):
                gAA, gBB, gAB = af.host_green_real(prm, spin, 1j * w, Nk)
                G[n] = g0_pairs(self.pos, self.sub, self.pos, self.sub, gAA, gBB, gAB, Nk)
            self.G0[spin] = G
        self.tail = polygamma(1, n_mats + 0.5) / (2 * np.pi ** 2 * self.T)
        self.dim = 2 * self.nC + 2 * len(self.nn_in) + 2 * len(self.nnn_in)
        self.H0 = sd.build_hamiltonian(dict(self.base), np.zeros(N))

    def unpack(self, x):
        nC, nb, nn = self.nC, len(self.nn_in), len(self.nnn_in)
        o = np.cumsum([0, nC, nC, nb, nb, nn, nn])
        return [x[o[i]:o[i + 1]] for i in range(6)]

    def pack(self, parts):
        return np.concatenate(parts)

    def state(self, x):
        dd, dm, dcu, dcd, dnu, dnd = self.unpack(x)
        s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.base.items()}
        s["delta_i"][self.idx] += dd
        s["m_i"][self.idx] += dm
        s["chi_up"][self.nn_in] += dcu
        s["chi_dn"][self.nn_in] += dcd
        s["chin_up"][self.nnn_in] += dnu
        s["chin_dn"][self.nnn_in] += dnd
        return s

    def delta_h(self, st, V):
        V_i = np.zeros(N)
        V_i[SITE] = V
        Hu, Hd, _ = sd.build_hamiltonian(st, V_i)
        ix = np.ix_(self.idx, self.idx)
        return {"up": (Hu - self.H0[0])[ix], "dn": (Hd - self.H0[1])[ix]}

    def delta_corr(self, dh):
        out = {}
        eye = np.eye(self.nC)
        for sp in ("up", "dn"):
            G0 = self.G0[sp]
            d = dh[sp]
            Tm = np.linalg.solve(eye[None] - d[None] @ G0, np.broadcast_to(d, G0.shape))
            dG = G0 @ Tm @ G0
            out[sp] = 2 * self.T * dG.real.sum(0) - d * self.tail
        return out

    def fields_from_corr(self, C):
        dn_u, dn_d = np.diag(C["up"]), np.diag(C["dn"])
        return [-(dn_u + dn_d), 0.5 * (dn_u - dn_d),
                C["up"][self.ai, self.aj], C["dn"][self.ai, self.aj],
                C["up"][self.bi, self.bj], C["dn"][self.bi, self.bj]]

    def F(self, x, V):
        st = self.state(x)
        dh = self.delta_h(st, V)
        return self.pack(self.fields_from_corr(self.delta_corr(dh)))

    def residual(self, x, V):
        return self.F(x, V) - x

    def solve(self, V, x0=None, tol=1e-6, maxiter=30, verbose=True):
        x0 = np.zeros(self.dim) if x0 is None else x0
        hist = []

        def R(x):
            r = self.residual(x, V)
            hist.append(float(np.abs(r).max()))
            return r
        t0 = time.time()
        try:
            x = newton_krylov(R, x0, f_tol=tol, maxiter=maxiter, method="lgmres",
                              inner_maxiter=20, verbose=False)
            ok = True
        except Exception as e:
            x = x0
            ok = False
            if verbose:
                print("   solver stopped:", type(e).__name__)
        r = float(np.abs(self.residual(x, V)).max())
        if verbose:
            print(f"   V={V:g}: |F(x)-x|max = {r:.2e}  ({len(hist)} F-calls, {time.time() - t0:.0f}s)", flush=True)
        return x, hist, r




def _pkl_name(args):
    return os.path.join(_DATA_DIR, f"embsc_d{args.delta:.3f}_T{args.T:g}_c{args.solve_cluster}_Nk{args.Nk_solve}_nm{args.n_mats}.pkl")


def _save(fn, sols, meta, args):
    with open(fn, "wb") as f:
        pickle.dump(dict(sols=sols, delta=args.delta, T=args.T, cluster=args.solve_cluster, Nk=args.Nk_solve,
                         n_mats=args.n_mats, tol=args.tol, meta=meta), f)


def _predict(sols, good, V):
    below = [v for v in good if v < V]
    if not below:
        return None
    if len(below) >= 2:
        va, vb = below[-2], below[-1]
        xa, xb = sols[va]["x"], sols[vb]["x"]
        return xb + (xb - xa) * (V - vb) / (vb - va)
    return sols[below[-1]]["x"]


def continue_to(emb, sols, meta, V_target, args, fn):
    """Walk the solution branch from the nearest converged V below V_target up to V_target.
    On failure the step is halved (down to --dV_min) with a secant predictor; if that also fails
    the end of the branch is recorded in meta['branch_end'] = (last converged V, first failed V)."""
    ok = lambda v: sols[v]["resid"] < args.tol * 2
    good = sorted(v for v in sols if ok(v))
    if V_target in sols and ok(V_target):
        print(f"   V={V_target:g}: already solved (residual {sols[V_target]['resid']:.1e})")
        return True
    be = meta.get("branch_end")
    if be is not None and V_target >= be[1]:
        print(f"   V={V_target:g}: beyond the end of the solution branch ({be[0]:g} < V* <= {be[1]:g}), skipped")
        return False
    below = [v for v in good if v < V_target]
    V0 = below[-1] if below else 0.0
    step = V_target - V0
    fails = 0
    while True:
        V = V0 + step if step > 1e-12 else V_target
        V = min(V, V_target)
        if V in sols and ok(V):
            good = sorted(v for v in sols if ok(v))
            V0 = V
        else:
            x0 = _predict(sols, good, V)
            x, hist, r = emb.solve(V, x0=x0, tol=args.tol, maxiter=args.maxiter)
            if r < args.tol * 2:
                sols[V] = dict(x=x, hist=hist, resid=r)
                good = sorted(v for v in sols if ok(v))
                V0 = V
                _save(fn, sols, meta, args)
            else:
                fails += 1
                step = (V - V0) / 2
                if not args.adaptive or step < args.dV_min or fails >= args.max_fails:
                    meta["branch_end"] = (V0, V)
                    _save(fn, sols, meta, args)
                    print(f"   no self-consistent solution found above V={V0:g}: branch end between {V0:g} and {V:g}; larger V are skipped")
                    return False
                print(f"   -> retrying with a smaller step (V={V0 + step:g})")
                continue
        if abs(V0 - V_target) < 1e-12:
            return True
        step = V_target - V0



# ====================================================================== finite-lattice solver
FIN_KEYS = ["delta_i", "m_i", "chi_up", "chi_dn", "chin_up", "chin_dn"]


class FiniteLattice:
    """Full 24x24 Gutzwiller mean-field model with one impurity (the map used by af_newton.py)."""

    def __init__(self, host, delta, T):
        self.delta, self.T = delta, T
        self.base = base_state(host)
        self.offs = np.cumsum([0] + [len(self.base[k]) for k in FIN_KEYS])

    def pack(self, st):
        return np.concatenate([np.asarray(st[k], float) for k in FIN_KEYS])

    def unpack(self, x, mu):
        s = dict(self.base)
        for k, a, b in zip(FIN_KEYS, self.offs[:-1], self.offs[1:]):
            s[k] = np.asarray(x[a:b], float).copy()
        s["mu"] = mu
        return s

    def map(self, st, V):
        V_i = np.zeros(N)
        V_i[SITE] = V
        return sd.solve_disordered(self.delta, V_i, seed=dict(st), max_iter=1, mix=1.0, mix_fluct=1.0,
                                   mix_mu=1.0, warmup=0, T=self.T, pairing=False)

    def residual(self, st, V):
        r = self.map(st, V)
        return max(np.abs(np.asarray(r[k]) - np.asarray(st[k])).max() for k in FIN_KEYS)

    def solve(self, V, st_seed, tol, maxiter, verbose=True):
        mu = {"v": st_seed["mu"]}
        best = {"x": self.pack(st_seed), "r": np.inf, "calls": 0}

        def R(x):
            r = self.map(self.unpack(x, mu["v"]), V)
            mu["v"] = r["mu"]
            best["calls"] += 1
            return self.pack(r) - x

        def cb(x, f):
            if np.abs(f).max() < best["r"]:
                best["x"], best["r"] = x.copy(), float(np.abs(f).max())
            if verbose:
                print(f"      Newton step: calls {best['calls']:5d}  max|F(x)-x| = {np.abs(f).max():.2e}", flush=True)
        t0 = time.time()
        try:
            x = newton_krylov(R, best["x"], f_tol=tol, maxiter=maxiter, method="lgmres", inner_maxiter=10,
                              callback=cb, verbose=False)
        except Exception:
            x = best["x"]
        st = self.unpack(x, mu["v"])
        res = self.residual(st, V)
        if verbose:
            print(f"   V={V:g}: finite lattice, max|F(x)-x| = {res:.2e}  ({best['calls']} F-calls, {time.time() - t0:.0f}s)", flush=True)
        return st, res


def finite_path(delta, V):
    return os.path.join(_DATA_DIR, f"finite_nk_d{delta:.3f}_V{V:g}.pkl")


# ====================================================================== states for every requested V
def embedding_states(args, host, Vs, solve=True):
    emb = SelfConsistentEmbedding(host, c=args.solve_cluster, Nk=args.Nk_solve, n_mats=args.n_mats, build_g0=solve)
    fn = _pkl_name(args)
    sols, meta = {}, {}
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            old = pickle.load(f)
        sols, meta = old["sols"], old.get("meta", {})
    if solve:
        for V in sorted(Vs):
            continue_to(emb, sols, meta, V, args, fn)
        _save(fn, sols, meta, args)
    out = {}
    for V in Vs:
        if V in sols and sols[V]["resid"] < args.tol * 2:
            out[V] = dict(st=emb.state(sols[V]["x"]), resid=sols[V]["resid"], hist=sols[V].get("hist"))
    return out


def finite_states(args, host, Vs, solve=True):
    fl = FiniteLattice(host, args.delta, args.T)
    out, accepted = {}, {}
    for V in sorted(Vs):
        fn = finite_path(args.delta, V)
        st = resid = None
        if os.path.exists(fn):
            d = pickle.load(open(fn, "rb"))
            st = fl.unpack(d["x"], d["mu"])
            resid = d.get("resid")
            if resid is None:
                resid = fl.residual(st, V)
        elif solve:
            lower = [v for v in accepted if v < V]
            if lower:
                seed = accepted[max(lower)]
            else:
                es = embedding_states(args, host, [V], solve=True)
                seed = es[V]["st"] if V in es else fl.base
                print(f"   finite lattice, first V={V:g}: seeded with the " + ("embedding solution" if V in es else "clean state"))
            st, resid = fl.solve(V, seed, args.tol_finite, args.maxiter_finite)
            pickle.dump(dict(x=fl.pack(st), keys=FIN_KEYS, offs=fl.offs, mu=st["mu"], resid=resid), open(fn, "wb"))
        if st is not None and resid <= args.max_resid:
            accepted[V] = st
            out[V] = dict(st=st, resid=resid, hist=None)
        elif st is not None:
            print(f"   V={V:g}: finite-lattice state not converged (max|F(x)-x| = {resid:.1e} > {args.max_resid:g}); larger V are skipped")
            break
    return out


# ====================================================================== observables
def observables(host, st):
    base = base_state(host)
    dx = (sd.xs - sd.xs[SITE] + Lx // 2) % Lx - Lx // 2
    dy = (sd.ys - sd.ys[SITE] + Ly // 2) % Ly - Ly // 2
    dm = (st["m_i"] - base["m_i"]) * sd.neel_sign
    dd = st["delta_i"] - base["delta_i"]
    return dict(dx=dx.astype(int), dy=dy.astype(int), r=np.hypot(dx, dy), sub=((dx + dy) % 2).astype(int), dm=dm, dd=dd,
                m_imp=float(st["m_i"][SITE] * sd.neel_sign[SITE]), delta_imp=float(st["delta_i"][SITE]),
                dN=float(-dd.sum()), dSz=float(st["m_i"].sum() - base["m_i"].sum()))


def site_image(obs, key, h):
    img = np.full((2 * h + 1, 2 * h + 1), np.nan)
    k = (np.abs(obs["dx"]) <= h) & (np.abs(obs["dy"]) <= h)
    img[obs["dy"][k] + h, obs["dx"][k] + h] = obs[key][k]
    return img


def print_summary(args, host, states, dhs, emb_c):
    i0 = int(np.where((emb_c.pos[:, 0] == 0) & (emb_c.pos[:, 1] == 0))[0][0])
    print(f"\nSUMMARY  delta={args.delta:g}  method={args.method}  clean m={host['m']:.4f}")
    print("     V    resid    m_imp  m/m0  delta_imp  dH_imp(up/dn)    dN_total   dSz_total")
    rows = []
    for V in sorted(states):
        o = observables(host, states[V]["st"])
        dh = dhs[V]
        print(f"{V:6.3f}  {states[V]['resid']:.1e}  {o['m_imp']:+.4f}  {o['m_imp'] / host['m']:+.2f}   {o['delta_imp']:.4f}   "
              f"{dh['up'][i0, i0]:.3f}/{dh['dn'][i0, i0]:.3f}    {o['dN']:+.3f}    {o['dSz']:+.4f}")
        rows.append((V, states[V]["resid"], o["m_imp"], o["m_imp"] / host["m"], o["delta_imp"], dh["up"][i0, i0], dh["dn"][i0, i0], o["dN"], o["dSz"]))
    fn = os.path.join(_DATA_DIR, f"afsc_summary_{args.method}_d{args.delta:.3f}.csv")
    with open(fn, "w") as f:
        f.write("V,resid,m_imp,m_over_m0,delta_imp,dH_imp_up,dH_imp_dn,dN_total,dSz_total\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
    print("saved", fn)


# ====================================================================== bound-state test
def run_poles(args, host, states, emb_c):
    prm = af.host_tb_params(host)
    om = np.linspace(-0.3, 0.7, 401)
    rho = np.zeros_like(om)
    for sp in ("up", "dn"):
        for n, w in enumerate(om):
            gAA, gBB, gAB = af.host_green_real(prm, sp, w + 3e-3j, 512)
            rho[n] += -(gAA[0, 0].imag + gBB[0, 0].imag) / (2 * np.pi)
    k = np.where((om > 0.12) & (om < 0.45))[0]
    low = k[rho[k] < 0.01 * rho.max()]
    if len(low) == 0:
        print("\nBOUND-STATE TEST: the clean host has no empty gap region at this doping (host LDOS never below 1% of its peak); skipped")
        return
    lo, hi = om[low.min()], om[low.max()]
    print(f"\nBOUND-STATE TEST  clean gap region (host LDOS < 1% of its peak): [{lo:.3f}, {hi:.3f}]t")
    oms = np.arange(lo + 0.004, hi - 0.004, args.d_om)
    G = {}
    for sp in ("up", "dn"):
        G[sp] = np.empty((len(oms), emb_c.nC, emb_c.nC))
        for n, w in enumerate(oms):
            gAA, gBB, gAB = af.host_green_real(prm, sp, w + 1e-4j, args.Nk)
            G[sp][n] = g0_pairs(emb_c.pos, emb_c.sub, emb_c.pos, emb_c.sub, gAA, gBB, gAB, args.Nk).real

    def poles_of(dh, lam=1.0):
        out = []
        for sp in ("up", "dn"):
            M = np.eye(emb_c.nC) - lam * dh[sp][None] @ G[sp]
            sg, _ = np.linalg.slogdet(M)
            out += [(sp, float(0.5 * (oms[i] + oms[i + 1]))) for i in np.where(sg[1:] * sg[:-1] < 0)[0]]
        return out

    def margin(dh):
        for lam in np.arange(0.1, args.lam_max + 1e-9, 0.05):
            if poles_of(dh, lam):
                return float(lam)
        return None
    f = lambda m: (f"{m:.2f}" if m else f">{args.lam_max:g}")
    print("margin = smallest factor by which the whole dH must be scaled to create a bound state (> 1: none exists)")
    print(f"{'V':>6} | {'self-consistent: poles':>24} {'margin':>8} | {'frozen host: poles':>22} {'margin':>8}")
    for V in sorted(args.Vs):
        fr = frozen_delta_h(V, emb_c.pos)
        fp, mf = poles_of(fr), margin(fr)
        if V in states:
            dh = emb_c.delta_h(states[V]["st"], V)
            ps, ms = poles_of(dh), margin(dh)
            s_txt = ', '.join(f'{a}:{b:.3f}' for a, b in ps) or 'none'
            s_mar = f(ms)
        else:
            s_txt, s_mar = "(no state)", ""
        print(f"{V:6.3f} | {s_txt:>24} {s_mar:>8} | {(', '.join(f'{a}:{b:.3f}' for a, b in fp) or 'none'):>22} {f(mf):>8}")


# ====================================================================== plots
def _ax_off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plot_stack(args, host, states, dhs, emb_c):
    import matplotlib.pyplot as plt
    Vs = sorted(args.Vs)
    wmax = 1.5 if args.wmax is None else args.wmax
    wmin = -wmax if args.wmin is None else args.wmin
    omegas = np.linspace(wmin, wmax, args.n_omega)
    tip = tuple(args.tip)
    C = emb_c.c
    perts = {("fr", V): frozen_delta_h(V, emb_c.pos) for V in Vs}
    perts.update({("sc", V): dhs[V] for V in states})
    rho, rho_cl = embed_spectra(host, perts, tip, omegas, args.eta, args.Nk, C)
    have = sorted(states)
    side = set(args.side)
    maps, oms, ext = {}, {}, None
    if "map" in side:
        win = (omegas >= args.pmin) & (omegas <= args.pmax)
        for V in Vs:
            if args.omega is not None:
                oms[V] = float(args.omega)
            else:
                y = rho[("sc", V)] if V in states else rho[("fr", V)]
                oms[V] = float(omegas[int(np.argmax(np.where(win, np.abs(y - rho_cl), -1)))])
            mp = {"fr": perts[("fr", V)]}
            if V in states:
                mp["sc"] = dhs[V]
            maps[V], ext = embed_map(host, mp, oms[V], args.eta, args.Nk, C, args.half, args.sub)
            print(f"   map V={V:g}: omega = {oms[V]:+.3f}t", flush=True)
    obs = {V: observables(host, states[V]["st"]) for V in states}
    cols = []
    if "map" in side:
        cols += ["fmap", "smap"]
    cols += ["spec"]
    cols += [c for c in ("dm", "dd") if c in side]
    wr = {"fmap": 1.3, "smap": 1.3, "spec": 3.4, "dm": 1.3, "dd": 1.3}
    nrow = len(Vs) + 1
    fig = plt.figure(figsize=(sum(wr[c] for c in cols) * 1.9 + 0.8, 1.35 * nrow + 1.2))
    gs = fig.add_gridspec(nrow, len(cols), width_ratios=[wr[c] for c in cols], hspace=0.0, wspace=0.12,
                          left=0.03, right=0.98, top=0.92, bottom=0.08)
    ymax = 1.08 * max(max(rho[("fr", V)].max() for V in Vs), max([rho[("sc", V)].max() for V in have] + [0]), rho_cl.max())
    ispec = cols.index("spec")
    titles = {"fmap": r"frozen host: $\Delta\rho(r)$", "smap": r"self-consistent: $\Delta\rho(r)$",
              "dm": r"$\Delta m_{stag}$ per site", "dd": r"$\Delta\delta$ per site"}
    for row in range(nrow):
        V = Vs[nrow - 2 - row] if row < nrow - 1 else None
        for ic, c in enumerate(cols):
            ax = fig.add_subplot(gs[row, ic])
            if row == 0 and c in titles:
                p = ax.get_position()
                fig.text(p.x0 + p.width / 2, p.y1 + 0.004, titles[c], ha="center", va="bottom", fontsize=8)
            if c == "spec":
                if V is None:
                    ax.plot(omegas, rho_cl, "r:", lw=1.4)
                    ax.text(0.99, 0.85, "V=0 (clean)", transform=ax.transAxes, ha="right", fontsize=9, color="r")
                else:
                    ax.plot(omegas, rho_cl, "r:", lw=0.8)
                    ax.plot(omegas, rho[("fr", V)], color="tab:blue", lw=1.0, label="frozen host")
                    if V in states:
                        ax.plot(omegas, rho[("sc", V)], "k", lw=1.3, label="self-consistent")
                    lab = f"V={V:g}t" + ("" if V in states else "  (no self-consistent state)")
                    ax.text(0.99, 0.85, lab, transform=ax.transAxes, ha="right", fontsize=9)
                    if "map" in side:
                        ax.axvline(oms[V], color="0.6", lw=0.6, ls="--")
                    if row == 0:
                        ax.legend(fontsize=7, loc="upper left")
                ax.set_xlim(wmin, wmax)
                ax.set_ylim(0, ymax)
                ax.set_yticks([])
                if row < nrow - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(r"bias $\omega/t$")
                continue
            if V is None or (c in ("smap", "dm", "dd") and V not in states):
                ax.axis("off")
                continue
            if c in ("fmap", "smap"):
                img = maps[V]["fr" if c == "fmap" else "sc"]
                v = max(np.abs(maps[V][k]).max() for k in maps[V])
                ax.imshow(img, extent=ext, origin="lower", cmap="bwr", vmin=-v, vmax=v)
                ax.plot(0, 0, "k+", ms=8)
                ax.plot(args.tip[0], args.tip[1], "gx", ms=5)
                _ax_off(ax)
                if c == "fmap":
                    ax.text(0.03, 0.95, rf"$\omega={oms[V]:+.3f}t$", transform=ax.transAxes, va="top", fontsize=7)
            else:
                img = site_image(obs[V], c, args.site_half)
                v = np.nanmax(np.abs(img))
                ax.imshow(img, extent=[-args.site_half - .5, args.site_half + .5] * 2, origin="lower",
                          cmap="RdBu_r" if c == "dm" else "PuOr", vmin=-v, vmax=v)
                _ax_off(ax)
                ax.text(0.03, 0.95, f"max {v:.3f}", transform=ax.transAxes, va="top", fontsize=7)
    fig.suptitle(rf"nn tip ({tip[0]:g},{tip[1]:g}), $\delta={args.delta:g}$, $T={args.T:g}$, $\eta={args.eta:g}$, $N_k={args.Nk}$, "
                 f"method: {args.method}, cluster half-width {C}", fontsize=9)
    fn = os.path.join(_DATA_DIR, f"afsc_stack_{args.method}_d{args.delta:.3f}_{time.strftime('%Y%m%d-%H%M%S')}.png")
    fig.savefig(fn, dpi=140)
    print("saved", fn)


def plot_zoom(args, host, states, dhs, emb_c):
    import matplotlib.pyplot as plt
    Vs = sorted(args.Vs)
    om = np.linspace(args.zoom[0], args.zoom[1], args.n_omega_zoom)
    perts = {("fr", V): frozen_delta_h(V, emb_c.pos) for V in Vs}
    perts.update({("sc", V): dhs[V] for V in states})
    rho, rcl = embed_spectra(host, perts, tuple(args.tip), om, args.eta, args.Nk, emb_c.c)
    nc = 3
    nr = int(np.ceil(len(Vs) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.0 * nr), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for a, V in zip(ax.ravel(), Vs):
        a.axis("on")
        a.plot(om, rcl, "r:", label="clean")
        a.plot(om, rho[("fr", V)], "b", label="frozen host")
        if V in states:
            a.plot(om, rho[("sc", V)], "k", label="self-consistent")
        a.set_title(f"V={V:g}t", fontsize=9)
        a.set_xlim(*args.zoom)
        a.set_ylim(bottom=0)
        a.set_xlabel(r"$\omega/t$")
    ax.ravel()[0].legend(fontsize=7)
    fig.suptitle(rf"zoom, $\delta={args.delta:g}$, method {args.method}; every panel has its own vertical scale", fontsize=10)
    fig.tight_layout()
    fn = os.path.join(_DATA_DIR, f"afsc_zoom_{args.method}_d{args.delta:.3f}_{time.strftime('%Y%m%d-%H%M%S')}.png")
    fig.savefig(fn, dpi=130)
    print("saved", fn)


def plot_profiles(args, host, states):
    import matplotlib.pyplot as plt
    Vs = sorted(states)
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(Vs), 2)))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    for col, V in zip(cols, Vs):
        o = observables(host, states[V]["st"])
        for s, mk in ((0, "o"), (1, "s")):
            k = (o["sub"] == s) & (o["r"] > 0)
            axs[0].semilogy(o["r"][k], np.abs(o["dm"][k]) + 1e-8, mk, ms=3, color=col, alpha=0.7, label=f"V={V:g}" if s == 0 else None)
            axs[1].semilogy(o["r"][k], np.abs(o["dd"][k]) + 1e-8, mk, ms=3, color=col, alpha=0.7)
        axs[0].semilogy([0], [abs(o["dm"][SITE])], "*", color=col, ms=9)
        axs[1].semilogy([0], [abs(o["dd"][SITE]) + 1e-8], "*", color=col, ms=9)
        if states[V].get("hist"):
            axs[2].semilogy(np.arange(len(states[V]["hist"])), states[V]["hist"], color=col, label=f"V={V:g}")
    axs[0].set_title(r"$|\Delta m_{stag}|$ vs distance (o same sublattice as impurity, s other, star impurity)", fontsize=8)
    axs[1].set_title(r"$|\Delta\delta|$ vs distance", fontsize=9)
    axs[0].legend(fontsize=7)
    for a in axs[:2]:
        a.set_xlabel("r / a")
    if axs[2].has_data():
        axs[2].set_title("convergence of the embedding solver (F-calls)", fontsize=9)
        axs[2].set_xlabel("F evaluations")
        axs[2].axhline(args.tol, color="k", ls=":", lw=0.8)
    else:
        axs[2].axis("off")
        txt = "  V     m_imp   delta_imp    dN_tot    resid\n" + "\n".join(
            f"{V:4.2f}  {observables(host, states[V]['st'])['m_imp']:+.4f}   {observables(host, states[V]['st'])['delta_imp']:.4f}   "
            f"{observables(host, states[V]['st'])['dN']:+.3f}   {states[V]['resid']:.1e}" for V in Vs)
        axs[2].text(0.0, 0.95, txt, family="monospace", fontsize=9, va="top")
    fig.tight_layout()
    fn = os.path.join(_DATA_DIR, f"afsc_profiles_{args.method}_d{args.delta:.3f}_{time.strftime('%Y%m%d-%H%M%S')}.png")
    fig.savefig(fn, dpi=130)
    print("saved", fn)


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(
        description="Single non-magnetic impurity in the doped AF (Gutzwiller t-J), STM spectra with a responsive host.\n"
                    "One command does everything: solve the self-consistent states (cached, resumable), then draw what you ask for.\n"
                    "Example: python af_sc.py --delta 0.10 --Vs 0.1 0.25 0.5 --side map dm dd --show stack summary poles",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("physics")
    g.add_argument("--delta", type=float, default=0.10, help="hole doping")
    g.add_argument("--T", type=float, default=0.02)
    g.add_argument("--Vs", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0, 2.0], help="impurity potentials (units of t)")
    g.add_argument("--method", choices=["embed", "finite"], default="embed",
                   help="embed: self-consistent cluster inside the infinite clean host (fast, cluster-size limited); "
                        "finite: full 24x24 lattice, Newton-Krylov (slow, stronger response)")
    g = ap.add_argument_group("what to draw")
    g.add_argument("--show", nargs="+", default=["stack", "summary"], choices=["stack", "summary", "poles", "zoom", "profiles"],
                   help="stack: spectra stacked in V; summary: table; poles: bound-state test; zoom: gap-region panels; profiles: |dm|,|ddelta| vs distance")
    g.add_argument("--side", nargs="*", default=[], choices=["map", "dm", "dd"],
                   help="extra panels beside every row of the stack: map = real-space STM difference maps (frozen and self-consistent) on the left; "
                        "dm / dd = site maps of the moment and hole-density change on the right")
    g.add_argument("--no_solve", action="store_true", help="do not solve anything, use what is already saved in ../data")
    g.add_argument("--no_frozen", action="store_true", help="(reserved) frozen curves are always drawn")
    g = ap.add_argument_group("spectra and maps")
    g.add_argument("--eta", type=float, default=0.01)
    g.add_argument("--Nk", type=int, default=512, help="k-mesh for spectra and maps")
    g.add_argument("--tip", type=float, nargs=2, default=[1.0, 0.0], help="tip position (x y) relative to the impurity")
    g.add_argument("--wmin", type=float, default=None)
    g.add_argument("--wmax", type=float, default=None, help="stack: bias range -wmax..wmax (default 1.5)")
    g.add_argument("--n_omega", type=int, default=301)
    g.add_argument("--zoom", type=float, nargs=2, default=[0.0, 0.5], help="energy window of the zoom panels")
    g.add_argument("--n_omega_zoom", type=int, default=201)
    g.add_argument("--omega", type=float, default=None, help="map energy for every row (default: energy of the strongest change, per V)")
    g.add_argument("--pmin", type=float, default=0.05, help="energy window for the automatic map energy")
    g.add_argument("--pmax", type=float, default=0.45)
    g.add_argument("--half", type=int, default=6, help="half-size of the real-space map window (lattice constants)")
    g.add_argument("--sub", type=int, default=4, help="map pixels per lattice constant")
    g.add_argument("--site_half", type=int, default=6, help="half-size of the dm / dd site maps")
    g = ap.add_argument_group("solvers")
    g.add_argument("--cluster", type=int, default=None, help="cluster half-width for the spectra (default: 4 for embed, 7 for finite)")
    g.add_argument("--solve_cluster", type=int, default=4, help="embed: cluster half-width of the self-consistency")
    g.add_argument("--Nk_solve", type=int, default=256)
    g.add_argument("--n_mats", type=int, default=100, help="embed: Matsubara frequencies")
    g.add_argument("--tol", type=float, default=1e-4, help="embed: solver tolerance")
    g.add_argument("--maxiter", type=int, default=30, help="embed: Newton steps per V")
    g.add_argument("--no_adaptive", dest="adaptive", action="store_false", help="embed: do not halve the V step on failure")
    g.add_argument("--dV_min", type=float, default=0.005)
    g.add_argument("--max_fails", type=int, default=5)
    g.add_argument("--tol_finite", type=float, default=1e-3, help="finite: solver tolerance")
    g.add_argument("--maxiter_finite", type=int, default=14, help="finite: Newton steps per V")
    g.add_argument("--max_resid", type=float, default=None, help="largest residual accepted for plotting (default 1.5e-3 finite, 2*tol embed)")
    g.add_argument("--d_om", type=float, default=0.002, help="poles: energy step")
    g.add_argument("--lam_max", type=float, default=4.0, help="poles: largest scale factor for the margin")
    args = ap.parse_args()
    args.Vs = sorted(args.Vs)
    if args.max_resid is None:
        args.max_resid = 1.5e-3
    host = af.solve_uniform_af(args.delta, T=args.T)
    solve = not args.no_solve
    if args.method == "embed":
        states = embedding_states(args, host, args.Vs, solve=solve)
        cl = args.solve_cluster if args.cluster is None else args.cluster
    else:
        states = finite_states(args, host, args.Vs, solve=solve)
        cl = 7 if args.cluster is None else args.cluster
    missing = [V for V in args.Vs if V not in states]
    if missing:
        print("no self-consistent state for V =", ", ".join(f"{V:g}" for V in missing), "(frozen host only for those)")
    if not states:
        print("no self-consistent state available; only frozen curves will be drawn")
    emb_c = SelfConsistentEmbedding(host, c=cl, Nk=256, n_mats=100, build_g0=False)
    dhs = {V: emb_c.delta_h(states[V]["st"], V) for V in states}
    if "summary" in args.show and states:
        print_summary(args, host, states, dhs, emb_c)
    if "poles" in args.show:
        run_poles(args, host, states, emb_c)
    if "stack" in args.show:
        plot_stack(args, host, states, dhs, emb_c)
    if "zoom" in args.show:
        plot_zoom(args, host, states, dhs, emb_c)
    if "profiles" in args.show and states:
        plot_profiles(args, host, states)


if __name__ == "__main__":
    main()
