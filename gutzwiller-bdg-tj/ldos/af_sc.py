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


def embed_map(host, perts, omega, eta, Nk, c, half, sub, cells=5, spin_diff=False):
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
                    sgn = -1.0 if (spin_diff and spin == "dn") else 1.0
                    for k in perts:
                        out[k][ir, sel] += sgn * (-np.einsum("ks,kl,ls->s", W, Ts[k], W).imag / np.pi)

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
    low = k[rho[k] < args.gap_thresh * rho.max()]
    if len(low) == 0:
        print("\nBOUND-STATE TEST: the clean host has no empty gap region at this doping (host LDOS never below 1% of its peak); skipped")
        return
    lo, hi = om[low.min()], om[low.max()]
    print(f"\nBOUND-STATE TEST  gap region (clean host LDOS < {args.gap_thresh * 100:g}% of its peak): [{lo:.3f}, {hi:.3f}]t")
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


def mag_map(host, emb_c, dh, half, sub):
    corr = emb_c.delta_corr(dh)
    M = corr["up"] - corr["dn"]
    M = 0.5 * (M + M.T)
    n = (2 * half + 1) * sub
    xs_ = np.linspace(-half - 0.5 + 0.5 / sub, half + 0.5 - 0.5 / sub, n)
    X, Y = np.meshgrid(xs_, xs_)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    W = wannier_envelope(P[:, None, 0] - emb_c.pos[None, :, 0], P[:, None, 1] - emb_c.pos[None, :, 1], gamma=af.GAMMA)
    val = np.einsum("pi,ij,pj->p", W, M, W)
    return val.reshape(n, n), [-half - 0.5, half + 0.5, -half - 0.5, half + 0.5]


# ====================================================================== plots
def _ax_off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plot_stack(args, host, states, dhs, emb_c):
    import matplotlib.pyplot as plt
    Vs = sorted(args.Vs)
    wmax = 1.0 if args.wmax is None else args.wmax
    wmin = -wmax if args.wmin is None else args.wmin
    omegas = np.linspace(wmin, wmax, args.n_omega)
    tip = tuple(args.tip)
    C = emb_c.c
    perts = {("fr", V): frozen_delta_h(V, emb_c.pos) for V in Vs}
    perts.update({("sc", V): dhs[V] for V in states})
    rho, rho_cl = embed_spectra(host, perts, tip, omegas, args.eta, args.Nk, C)
    have = sorted(states)
    side = set(args.side)
    maps, smaps, oms, ext = {}, {}, {}, None
    if "map" in side or "spin" in side:
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
            if "map" in side:
                maps[V], ext = embed_map(host, mp, oms[V], args.eta, args.Nk, C, args.half, args.sub)
            if "spin" in side:
                smaps[V], ext = embed_map(host, mp, oms[V], args.eta, args.Nk, C, args.half, args.sub, spin_diff=True)
            print(f"   map V={V:g}: omega = {oms[V]:+.3f}t", flush=True)
    mmaps = {}
    if "mag" in side:
        for V in Vs:
            mm = {"fr": mag_map(host, emb_c, perts[("fr", V)], args.half, args.sub)[0]}
            if V in states:
                mm["sc"] = mag_map(host, emb_c, dhs[V], args.half, args.sub)[0]
            mmaps[V] = mm
        ext_mag = [-args.half - 0.5, args.half + 0.5, -args.half - 0.5, args.half + 0.5]
    obs = {V: observables(host, states[V]["st"]) for V in states}
    cols = []
    if "map" in side:
        cols += ["fmap", "smap"]
    if "spin" in side:
        cols += ["fspin", "sspin"]
    if "mag" in side:
        cols += ["fmag", "smag"]
    cols += ["spec"]
    cols += [c for c in ("dm", "dd") if c in side]
    wr = {"fmap": 1.3, "smap": 1.3, "fspin": 1.3, "sspin": 1.3, "fmag": 1.3, "smag": 1.3, "spec": 3.4, "dm": 1.3, "dd": 1.3}
    nrow = len(Vs)
    rh, sw = args.row_h, args.spec_w
    nsq = len([c for c in cols if c != "spec"])
    fig_w, fig_h = nsq * rh + sw + 1.0, nrow * rh + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    wr = {c: (sw if c == "spec" else rh) for c in cols}
    gs = fig.add_gridspec(nrow, len(cols), width_ratios=[wr[c] for c in cols], hspace=0.0, wspace=args.wspace,
                          left=0.25 / fig_w, right=1 - 0.15 / fig_w, top=1 - 0.55 / fig_h, bottom=0.45 / fig_h)
    ymax = 1.08 * max(max(rho[("fr", V)].max() for V in Vs), max([rho[("sc", V)].max() for V in have] + [0]), rho_cl.max())
    ispec = cols.index("spec")
    titles = {"fmap": r"froz: $\Delta\rho(r)$", "smap": r"SC $\Delta\rho(r)$",
              "fmag": r"froz: $\Delta s_z(r)$", "smag": r"SC $\Delta s_z(r)$",
              "fspin": r"froz: $\Delta(\rho_\uparrow-\rho_\downarrow)$", "sspin": r"SC $\Delta(\rho_\uparrow-\rho_\downarrow)$",
              "dm": r"$\Delta m_{stag}$ per site", "dd": r"$\Delta\delta$ per site"}
    for row in range(nrow):
        V = Vs[nrow - 1 - row]
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
                    ax.plot(omegas, rho_cl, "r:", lw=1.0, label="clean (V=0)")
                    ax.plot(omegas, rho[("fr", V)], color="tab:blue", lw=1.0, label="frozen host")
                    if V in states:
                        ax.plot(omegas, rho[("sc", V)], "k", lw=1.3, label="self-consistent")
                    lab = f"V={V:g}t" + ("" if V in states else "  no self-consistent state")
                    ax.text(0.99, 0.85, lab, transform=ax.transAxes, ha="right", fontsize=9)
                    if oms:
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
            if V is None or (c in ("smap", "sspin", "smag", "dm", "dd") and V not in states):
                ax.axis("off")
                continue
            if c in ("fmag", "smag"):
                img = mmaps[V]["fr" if c == "fmag" else "sc"]
                v = max(np.abs(mmaps[V][k]).max() for k in mmaps[V])
                ax.imshow(img, extent=ext_mag, origin="lower", cmap="PRGn", vmin=-v, vmax=v)
                ax.plot(0, 0, "k+", ms=8)
                ax.plot(args.tip[0], args.tip[1], "gx", ms=5)
                _ax_off(ax)
                ax.text(0.97, 0.03, f"max {v:.3f}", transform=ax.transAxes, ha="right", fontsize=6)
            elif c in ("fmap", "smap", "fspin", "sspin"):
                src = maps if c in ("fmap", "smap") else smaps
                img = src[V]["fr" if c in ("fmap", "fspin") else "sc"]
                v = max(np.abs(src[V][k]).max() for k in src[V])
                ax.imshow(img, extent=ext, origin="lower", cmap="bwr", vmin=-v, vmax=v)
                ax.plot(0, 0, "k+", ms=8)
                ax.plot(args.tip[0], args.tip[1], "gx", ms=5)
                _ax_off(ax)
                if c in ("fmap", "fspin"):
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
    fig.savefig(fn, dpi=140, bbox_inches="tight", pad_inches=0.1)
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
# ====================================================================== spin-resolved tips
SPIN_TIPS = {"impurity (0,0)": (0.0, 0.0), "nn (1,0)": (1.0, 0.0), "nnn (1,1)": (1.0, 1.0), "bond (0.5,0)": (0.5, 0.0)}


def tip_orbitals(mode, tips):
    """tip name -> (positions, weights K, sublattice).  bare: the orbital centred at the tip (one site, a bond or a plaquette,
    symmetric combination); wannier: the d-wave filter of the real tip."""
    out = {}
    for name, (x, y) in tips.items():
        if mode == "wannier":
            out[name] = neighborhood((x, y))
            continue
        xs_ = [int(np.floor(x))] if float(x).is_integer() else [int(np.floor(x)), int(np.floor(x)) + 1]
        ys_ = [int(np.floor(y))] if float(y).is_integer() else [int(np.floor(y)), int(np.floor(y)) + 1]
        pos = np.array([(a, b) for a in xs_ for b in ys_])
        out[name] = (pos, np.full(len(pos), 1.0 / np.sqrt(len(pos))), (pos[:, 0] + pos[:, 1]) % 2)
    return out


def spin_spectra(host, perts, tipvecs, omegas, eta, Nk, c):
    """rho_sigma(tip, omega) for sigma = up, dn; keys: 'clean' and every key of perts (dict of cluster dH)."""
    prm = af.host_tb_params(host)
    C_pos, C_a = cluster(c)
    eye = np.eye(len(C_pos))
    out = {k: {sp: {t: np.zeros(len(omegas)) for t in tipvecs} for sp in ("up", "dn")} for k in ["clean"] + list(perts)}
    for n, om in enumerate(omegas):
        for sp in ("up", "dn"):
            gAA, gBB, gAB = af.host_green_real(prm, sp, om + 1j * eta, Nk)
            G0CC = g0_pairs(C_pos, C_a, C_pos, C_a, gAA, gBB, gAB, Nk)
            Ts = {k: np.linalg.solve(eye - d[sp] @ G0CC, d[sp]) for k, d in perts.items()}
            for t, (pos, K, a) in tipvecs.items():
                host_term = K @ g0_pairs(pos, a, pos, a, gAA, gBB, gAB, Nk) @ K
                w = g0_pairs(pos, a, C_pos, C_a, gAA, gBB, gAB, Nk).T @ K
                out["clean"][sp][t][n] = -host_term.imag / np.pi
                for k, T in Ts.items():
                    out[k][sp][t][n] = -(host_term + w @ T @ w).imag / np.pi
    return out


def spin_site_resolved(host, perts, omegas, eta, Nk, c):
    """bare site-resolved: out['clean'][sp] = rho0_sigma(R, omega); out[k][sp] = Delta rho_sigma(R, omega); shape (n_omega, nC)."""
    prm = af.host_tb_params(host)
    C_pos, C_a = cluster(c)
    nC = len(C_pos)
    eye = np.eye(nC)
    out = {k: {sp: np.zeros((len(omegas), nC)) for sp in ("up", "dn")} for k in ["clean"] + list(perts)}
    for n, om in enumerate(omegas):
        for sp in ("up", "dn"):
            gAA, gBB, gAB = af.host_green_real(prm, sp, om + 1j * eta, Nk)
            G = g0_pairs(C_pos, C_a, C_pos, C_a, gAA, gBB, gAB, Nk)
            out["clean"][sp][n] = -np.diag(G).imag / np.pi
            for k, d in perts.items():
                T = np.linalg.solve(eye - d[sp] @ G, d[sp])
                out[k][sp][n] = -((G @ T) * G.T).sum(axis=1).imag / np.pi
    return out


def _window_integral(om, y, baseline):
    if baseline:
        y = y - (y[0] + (y[-1] - y[0]) * (om - om[0]) / (om[-1] - om[0]))
    return float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(y, om))


def analyse_peaks(args, om, sr, keys, labels, eps):
    """peaks of the total change of the DOS; weight per spin and sublattice in the peak window; M_u, M_st."""
    from scipy.signal import find_peaks
    A = eps > 0
    peaks = {}
    print(f"\nSPIN CONTENT OF THE PEAKS  (bare sites of the cluster, window +-{args.peak_w:g}t around the peak, "
          f"baseline {'not ' if args.no_baseline else ''}subtracted)")
    print("  W = integral of Delta rho_sigma summed over the cluster sites, split into sublattice A (impurity) and B")
    print("  M_u = W_up - W_dn (uniform, ferro-like);  M_st = (W_up^A - W_up^B) - (W_dn^A - W_dn^B) (staggered, AF-like)")
    print(f"{'':>4}{'':>4} {'V':>5} {'peak':>7} | {'W_up^A':>8} {'W_up^B':>8} {'W_dn^A':>8} {'W_dn^B':>8} | {'M_u/W':>7} {'M_st/W':>7}")
    for k in keys:
        tot = (sr[k]["up"] + sr[k]["dn"]).sum(1)
        pk, pr = find_peaks(tot, prominence=args.peak_prom * np.abs(tot).max())
        pk = sorted(pk[np.argsort(-pr["prominences"])[:args.n_peaks]])
        peaks[k] = [float(om[i]) for i in pk]
        for i in pk:
            sel = np.where(np.abs(om - om[i]) <= args.peak_w)[0]
            o = om[sel]
            q = {(sp, s): _window_integral(o, sr[k][sp][sel][:, m].sum(1), not args.no_baseline)
                 for sp in ("up", "dn") for s, m in (("A", A), ("B", ~A))}
            W = sum(q.values())
            Mu = q[("up", "A")] + q[("up", "B")] - q[("dn", "A")] - q[("dn", "B")]
            Ms = (q[("up", "A")] - q[("up", "B")]) - (q[("dn", "A")] - q[("dn", "B")])
            print(f"{labels[k][0]:>4}{'':>4} {labels[k][1]:5.2f} {om[i]:7.3f} | {q[('up', 'A')]:8.4f} {q[('up', 'B')]:8.4f} "
                  f"{q[('dn', 'A')]:8.4f} {q[('dn', 'B')]:8.4f} | {Mu / W:+7.2f} {Ms / W:+7.2f}")
    return peaks


def _cl_img(vec, c, half):
    n = 2 * c + 1
    h = min(half, c)
    return vec.reshape(n, n).T[c - h:c + h + 1, c - h:c + h + 1], h


def _arrow_panel(ax, X, Y, val, h, cm, title=None, arrow=0.75, acol="k"):
    """one dot per lattice site coloured by val in [-1,1], arrow up/down with length ~ val; ring: black = sublattice A, grey = B."""
    ok = np.isfinite(val)
    A = (X + Y) % 2 == 0
    ax.scatter(X[~ok], Y[~ok], s=8, c="0.8", zorder=1)
    ax.scatter(X[ok], Y[ok], c=val[ok], cmap=cm, vmin=-1, vmax=1, s=120, zorder=2,
               edgecolors=np.where(A[ok], "k", "0.6"), linewidths=np.where(A[ok], 1.2, 0.6))
    ax.quiver(X[ok], Y[ok], np.zeros(ok.sum()), arrow * val[ok], angles="xy", scale_units="xy", scale=1, pivot="mid",
              width=0.014, headwidth=3.2, headlength=3.8, headaxislength=3.2, color=acol, zorder=3)
    ax.plot(0, 0, "r+", ms=9, mew=1.5, zorder=4)
    ax.set_xlim(-h - .6, h + .6)
    ax.set_ylim(-h - .6, h + .6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def run_spin_tips(args, host, states, dhs, emb_c):
    import matplotlib.pyplot as plt
    Vs = [V for V in (args.spin_Vs if args.spin_Vs else args.Vs) if V > 0 and V in dhs]
    if not Vs:
        print("spin_tips: no self-consistent state among the requested V")
        return
    c = emb_c.c
    wmax = 1.0 if args.wmax is None else args.wmax
    wmin = -wmax if args.wmin is None else args.wmin
    stamp = time.strftime('%Y%m%d-%H%M%S')
    tipvecs = tip_orbitals(args.spin_mode, SPIN_TIPS)
    perts, labels = {}, {}
    for V in Vs:
        perts[f"fr{V:g}"], labels[f"fr{V:g}"] = frozen_delta_h(V, emb_c.pos), ("froz", V)
        perts[f"sc{V:g}"], labels[f"sc{V:g}"] = dhs[V], ("SC", V)
    eps = 1.0 - 2.0 * emb_c.sub
    print(f"\nspin_tips: tips = {list(SPIN_TIPS)}, mode = {args.spin_mode}, cluster half-width {c}")
    t0 = time.time()
    om = np.linspace(wmin, wmax, args.n_omega)
    sp = spin_spectra(host, perts, tipvecs, om, args.eta, args.Nk, c)
    print(f"  tip spectra done ({time.time() - t0:.0f}s)")
    pw = list(args.p_omegas) if args.p_omegas else []
    lo2, hi2 = min([args.pmin] + [w - 0.02 for w in pw]), max([args.pmax] + [w + 0.02 for w in pw])
    om2 = np.arange(lo2, hi2 + 1e-9, args.d_om_sp)
    sr = spin_site_resolved(host, perts, om2, args.eta, args.Nk, c)
    print(f"  site-resolved spectra done ({time.time() - t0:.0f}s)")

    # ---- background of the clean host
    m0 = host["m_i"] * sd.neel_sign
    print(f"\nBACKGROUND (clean host, delta={args.delta:g}): staggered static moment (-1)^(x+y) m_i = {m0.mean():+.4f} on every site "
          f"-> {'antiferromagnet' if abs(m0.mean()) > 1e-3 else 'no static order'}; neel_sign on the impurity site = {sd.neel_sign[SITE]:+.0f} "
          f"(epsilon = +1 on the impurity sublattice A)")
    for k0 in (args.pmin, 0.5 * (args.pmin + args.pmax), args.pmax):
        i0 = int(np.argmin(np.abs(om2 - k0)))
        rs = sr["clean"]["up"][i0] - sr["clean"]["dn"][i0]
        rt = sr["clean"]["up"][i0] + sr["clean"]["dn"][i0]
        print(f"   omega={om2[i0]:+.3f}: clean rho_s = rho_up - rho_dn: <rho_s>/<rho> = {rs.mean() / rt.mean():+.3f} (uniform part), "
              f"<eps rho_s>/<rho> = {(eps * rs).mean() / rt.mean():+.3f} (staggered part)")

    keys = [f"sc{V:g}" for V in Vs] + [f"fr{V:g}" for V in Vs]
    peaks = analyse_peaks(args, om2, sr, keys, labels, eps)

    names = list(SPIN_TIPS)
    rows = Vs[::-1]
    pos_txt = {k: f"{(k.split('(')[1].rstrip(')'))}" for k in names}

    def frame(nr, nc, w, h, **kw):
        return plt.subplots(nr, nc, figsize=(w * nc, h * nr), squeeze=False, **kw)

    # ---- fig 1: rho_up, rho_dn
    fig, axs = frame(len(rows), len(names), 3.0, 1.7, sharex=True)
    for r, V in enumerate(rows):
        for j, t in enumerate(names):
            ax = axs[r, j]
            ax.plot(om, sp["clean"]["up"][t], color="salmon", ls=":", lw=1)
            ax.plot(om, sp["clean"]["dn"][t], color="lightskyblue", ls=":", lw=1)
            ax.plot(om, sp[f"sc{V:g}"]["up"][t], color="tab:red", lw=1.3, label=r"$\rho_\uparrow$ SC")
            ax.plot(om, sp[f"sc{V:g}"]["dn"][t], color="tab:blue", lw=1.3, label=r"$\rho_\downarrow$ SC")
            for pe in peaks.get(f"sc{V:g}", []):
                ax.axvline(pe, color="0.6", lw=0.6, ls="--")
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"V={V:g}t", fontsize=8)
            if r == 0:
                ax.set_title(t, fontsize=9)
    axs[0, 0].legend(fontsize=6, loc="upper left")
    for ax in axs[-1]:
        ax.set_xlabel(r"bias $\omega/t$")
    fig.suptitle(rf"spin-resolved LDOS ({args.spin_mode} tip), $\delta={args.delta:g}$, $T={args.T:g}$, $\eta={args.eta:g}$; dotted: clean host", fontsize=9)
    fig.tight_layout()
    fn = os.path.join(_DATA_DIR, f"afsc_spin_rho_{args.spin_mode}_d{args.delta:.3f}_{stamp}.png")
    fig.savefig(fn, dpi=140)
    print("saved", fn)

    # ---- fig 2: Delta rho_s ; fig 3: polarisation
    for what in ("ds", "P"):
        fig, axs = frame(len(rows), len(names), 3.0, 1.7, sharex=True)
        for r, V in enumerate(rows):
            for j, t in enumerate(names):
                ax = axs[r, j]
                s0 = sp["clean"]["up"][t] - sp["clean"]["dn"][t]
                t0_ = sp["clean"]["up"][t] + sp["clean"]["dn"][t]
                for key, st_ in ((f"sc{V:g}", dict(color="k", lw=1.3, label="self-consistent")), (f"fr{V:g}", dict(color="tab:blue", lw=1.0, ls="--", label="frozen"))):
                    s = sp[key]["up"][t] - sp[key]["dn"][t]
                    tt = sp[key]["up"][t] + sp[key]["dn"][t]
                    if what == "ds":
                        ax.plot(om, s - s0, **st_)
                    else:
                        ax.plot(om, np.where(tt > 0.03 * t0_.max(), s / np.maximum(tt, 1e-12), np.nan), **st_)
                if what == "P":
                    ax.plot(om, np.where(t0_ > 0.03 * t0_.max(), s0 / np.maximum(t0_, 1e-12), np.nan), color="tab:red", ls=":", lw=1, label="clean")
                    ax.set_ylim(-1, 1)
                ax.axhline(0, color="0.7", lw=0.5)
                for pe in peaks.get(f"sc{V:g}", []):
                    ax.axvline(pe, color="0.6", lw=0.6, ls="--")
                if j == 0:
                    ax.set_ylabel(f"V={V:g}t", fontsize=8)
                if r == 0:
                    ax.set_title(t, fontsize=9)
        axs[0, 0].legend(fontsize=6, loc="upper left")
        for ax in axs[-1]:
            ax.set_xlabel(r"bias $\omega/t$")
        ttl = r"$\Delta\rho_s=(\rho_\uparrow-\rho_\downarrow)_{\rm imp}-(\rho_\uparrow-\rho_\downarrow)_{\rm clean}$" if what == "ds" \
            else r"polarisation $P=(\rho_\uparrow-\rho_\downarrow)/(\rho_\uparrow+\rho_\downarrow)$ (masked where the DOS is tiny)"
        fig.suptitle(ttl + rf" ({args.spin_mode} tip), $\delta={args.delta:g}$", fontsize=9)
        fig.tight_layout()
        fn = os.path.join(_DATA_DIR, f"afsc_spin_{what}_{args.spin_mode}_d{args.delta:.3f}_{stamp}.png")
        fig.savefig(fn, dpi=140)
        print("saved", fn)

    # ---- fig 4: site-resolved maps at the strongest peak of each row
    cols = ["eps m (state)", r"clean $\rho_s$", r"$\Delta\rho_\uparrow$", r"$\Delta\rho_\downarrow$", r"$\Delta\rho_s$", r"$\epsilon\,\Delta\rho_s$"]
    fig, axs = frame(len(rows), len(cols), 1.9, 1.9)
    for r, V in enumerate(rows):
        key = f"sc{V:g}"
        if args.omega is not None:
            wb = args.omega
        elif peaks.get(key):
            tot = (sr[key]["up"] + sr[key]["dn"]).sum(1)
            wb = max(peaks[key], key=lambda w: tot[int(np.argmin(np.abs(om2 - w)))])
        else:
            wb = None
        for j in range(len(cols)):
            ax = axs[r, j]
            if wb is None:
                ax.axis("off")
                continue
            i0 = int(np.argmin(np.abs(om2 - wb)))
            du, dd = sr[key]["up"][i0], sr[key]["dn"][i0]
            if j == 0:
                o = observables(host, states[V]["st"])
                o["epsm"] = states[V]["st"]["m_i"] * sd.neel_sign
                img = site_image(o, "epsm", min(args.half, c))
                h = min(args.half, c)
                cm = "viridis"
                v0, v1 = np.nanmin(img), np.nanmax(img)
            else:
                vec = {1: sr["clean"]["up"][i0] - sr["clean"]["dn"][i0], 2: du, 3: dd, 4: du - dd, 5: eps * (du - dd)}[j]
                img, h = _cl_img(vec, c, args.half)
                v1 = np.abs(img).max()
                v0, cm = -v1, "PRGn"
            ax.imshow(img, extent=[-h - .5, h + .5] * 2, origin="lower", cmap=cm, vmin=v0, vmax=v1)
            ax.plot(0, 0, "k+", ms=6)
            _ax_off(ax)
            ax.text(0.03, 0.95, f"max {v1:.3g}", transform=ax.transAxes, va="top", fontsize=6)
            if j == 0:
                ax.text(0.03, 0.05, rf"V={V:g}t, $\omega_b$={wb:+.3f}", transform=ax.transAxes, fontsize=7)
            if r == 0:
                ax.set_title(cols[j], fontsize=8)
    fig.suptitle(rf"bare site-resolved spin maps at the strongest peak, $\delta={args.delta:g}$; eps = +1 on the impurity sublattice; "
                 r"PRGn: green $>0$, purple $<0$, scale symmetric per panel", fontsize=8)
    fig.tight_layout()
    fn = os.path.join(_DATA_DIR, f"afsc_spin_maps_d{args.delta:.3f}_{stamp}.png")
    fig.savefig(fn, dpi=140)
    print("saved", fn)

    # ---- fig 5: polarisation map P(R, omega) = rho_s / rho at the peak energies (bare sites)
    prows = []
    for V in rows:
        ws = pw if pw else ([args.omega] if args.omega is not None else peaks.get(f"sc{V:g}", []))
        prows += [(V, w) for w in ws]
    if prows:
        pcols = [r"$\rho(R)$ SC", r"$P$ clean", r"$P$ frozen", r"$P$ SC", r"$\Delta P$ (SC $-$ clean)"]
        cmP = plt.get_cmap("PRGn").copy()
        cmP.set_bad("0.85")
        spots = [("(0,0)", 0, 0), ("(1,0)", 1, 0), ("(1,1)", 1, 1), ("(2,0)", 2, 0)]
        idx = {nm: int(np.where((emb_c.pos[:, 0] == x) & (emb_c.pos[:, 1] == y))[0][0]) for nm, x, y in spots}
        blk = (np.abs(emb_c.pos[:, 0]) <= 1) & (np.abs(emb_c.pos[:, 1]) <= 1)
        print(f"\nPOLARISATION P = (rho_up - rho_dn)/(rho_up + rho_dn) at the peak energies (bare sites; masked where rho < {args.p_mask:g} x mean clean DOS)")
        print(f"{'V':>5} {'omega':>6} {'host':>6} | " + " ".join(f"{nm:>7}" for nm in idx) + " | 3x3 block")
        store = []
        saved = []
        fig, axs = frame(len(prows), len(pcols), 1.9, 1.9)
        for r, (V, wb) in enumerate(prows):
            i0 = int(np.argmin(np.abs(om2 - wb)))
            u0, d0 = sr["clean"]["up"][i0], sr["clean"]["dn"][i0]
            thr = args.p_mask * (u0 + d0).mean()
            res = {}
            for nm, kk in (("clean", None), ("froz", f"fr{V:g}"), ("SC", f"sc{V:g}")):
                u = u0 + (sr[kk]["up"][i0] if kk else 0.0)
                d = d0 + (sr[kk]["dn"][i0] if kk else 0.0)
                t = u + d
                res[nm] = (np.where(t > thr, (u - d) / np.maximum(t, 1e-12), np.nan), t, (u - d), u, d)
                Pb = (u - d)[blk].sum() / max(t[blk].sum(), 1e-12)
                print(f"{V:5.2f} {om2[i0]:6.3f} {nm:>6} | " + " ".join(f"{res[nm][0][i]:+7.2f}" for i in idx.values()) + f" | {Pb:+.2f}")
            dP = res["SC"][0] - res["clean"][0]
            store.append((V, om2[i0], res))
            rowimgs = []
            panels = [(res["SC"][1], "viridis", 0, None), (res["clean"][0], cmP, -1, 1), (res["froz"][0], cmP, -1, 1), (res["SC"][0], cmP, -1, 1), (dP, cmP, None, None)]
            for j, (vec, cm, lo, hi) in enumerate(panels):
                ax = axs[r, j]
                img, h = _cl_img(vec, c, args.half)
                rowimgs.append(img)
                if j == 0:
                    lo, hi = 0, np.nanmax(img)
                elif j == 4:
                    hi = np.nanmax(np.abs(img)) if np.isfinite(img).any() else 1
                    lo = -hi
                ax.imshow(img, extent=[-h - .5, h + .5] * 2, origin="lower", cmap=cm, vmin=lo, vmax=hi)
                ax.plot(0, 0, "k+", ms=6)
                _ax_off(ax)
                if j in (0, 4):
                    ax.text(0.03, 0.95, f"max {hi:.3g}", transform=ax.transAxes, va="top", fontsize=6)
                if j == 0:
                    ax.text(0.03, 0.05, rf"V={V:g}t, $\omega$={om2[i0]:+.3f}", transform=ax.transAxes, fontsize=7)
                if r == 0:
                    ax.set_title(pcols[j], fontsize=8)
            saved.append(dict(V=float(V), w=float(om2[i0]), h=int(h), imgs=rowimgs))
        fig.suptitle(rf"polarisation map $P(R,\omega)$ on the bare sites, $\delta={args.delta:g}$; PRGn: green $=+1$ ($\uparrow$), purple $=-1$ ($\downarrow$), "
                     "grey: DOS too small", fontsize=8)
        fig.tight_layout()
        fn = os.path.join(_DATA_DIR, f"afsc_spin_Pmap_d{args.delta:.3f}_{stamp}.png")
        fig.savefig(fn, dpi=140)
        print("saved", fn)
        fn = os.path.join(_DATA_DIR, f"afsc_pmap_rows_d{args.delta:.3f}.pkl")
        with open(fn, "wb") as f:
            pickle.dump(dict(delta=args.delta, rows=saved), f)
        print("saved rows for the combined figure:", fn)

        # ---- fig 6: the same on the lattice: one dot + arrow per site (order: staggered arrows = AF, uniform = FM)
        h = min(args.p_half, c)
        X, Y = emb_c.pos[:, 0], emb_c.pos[:, 1]
        win = (np.abs(X) <= h) & (np.abs(Y) <= h)
        m0 = abs((host["m_i"] * sd.neel_sign).mean())
        fig, axs = frame(len(store), 4, 2.5, 2.5)
        for r, (V, wv, res) in enumerate(store):
            o = observables(host, states[V]["st"])
            sel = (np.abs(o["dx"]) <= h) & (np.abs(o["dy"]) <= h)
            _arrow_panel(axs[r, 0], o["dx"][sel], o["dy"][sel], np.asarray(states[V]["st"]["m_i"])[sel] / m0, h, cmP,
                         r"static moment $m_i/m_0$" if r == 0 else None, acol="w")
            for j, nm in enumerate(("clean", "froz", "SC")):
                _arrow_panel(axs[r, j + 1], X[win], Y[win], res[nm][0][win], h, cmP, rf"$P$ {nm}" if r == 0 else None)
            axs[r, 0].set_ylabel(rf"V={V:g}t, $\omega$={wv:+.3f}", fontsize=8)
        fig.suptitle(rf"order on the lattice, $\delta={args.delta:g}$: dot colour and arrow = polarisation (green/up $=+1$); "
                     r"black ring: sublattice A (impurity at red +), grey ring: B; first column: static moment of the state", fontsize=8)
        fig.tight_layout()
        fn = os.path.join(_DATA_DIR, f"afsc_spin_Porder_d{args.delta:.3f}_{stamp}.png")
        fig.savefig(fn, dpi=140)
        print("saved", fn)


def run_spin_stack(args, host, dhs, emb_c):
    """stack in V (like the density/magnetisation stacks): P maps (frozen, self-consistent) + spin-resolved spectra, clean only in the V=0 row"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    Vs_all = sorted(set(args.Vs))
    Vs = [V for V in Vs_all if V == 0 or V in dhs]
    miss = [V for V in Vs_all if V not in Vs]
    if miss:
        print("spin_stack: no self-consistent state for V =", miss, "(skipped)")
    Vp = [V for V in Vs if V > 0]
    if not Vp:
        print("spin_stack: no V > 0 available")
        return
    c = emb_c.c
    full = {"imp": "impurity (0,0)", "nn": "nn (1,0)", "nnn": "nnn (1,1)", "bond": "bond (0.5,0)"}
    tips = {full[k]: SPIN_TIPS[full[k]] for k in args.stack_tips}
    tipvecs = tip_orbitals(args.spin_mode, tips)
    perts, labels = {}, {}
    for V in Vp:
        perts[f"fr{V:g}"] = frozen_delta_h(V, emb_c.pos)
        perts[f"sc{V:g}"], labels[f"sc{V:g}"] = dhs[V], ("SC", V)
    eps = 1.0 - 2.0 * emb_c.sub
    wmax = 1.0 if args.wmax is None else args.wmax
    wmin = -wmax if args.wmin is None else args.wmin
    ref = args.p_omegas[0] if args.p_omegas else args.p_track
    lo2 = args.pmin if ref is None else min(args.pmin, ref - 0.06)
    hi2 = args.pmax if ref is None else max(args.pmax, ref + 0.06)
    t0 = time.time()
    om = np.linspace(wmin, wmax, args.n_omega)
    sp = spin_spectra(host, perts, tipvecs, om, args.eta, args.Nk, c)
    om2 = np.arange(lo2, hi2 + 1e-9, args.d_om_sp)
    sr = spin_site_resolved(host, perts, om2, args.eta, args.Nk, c)
    print(f"spin_stack: spectra and maps computed ({time.time() - t0:.0f}s)")
    peaks = analyse_peaks(args, om2, sr, [f"sc{V:g}" for V in Vp], labels, eps)

    def energy(V):
        if args.p_omegas:
            return args.p_omegas[0]
        pk = peaks.get(f"sc{V:g}", [])
        if args.p_track is not None:
            near = [w for w in pk if abs(w - args.p_track) <= args.p_tol]
            return min(near, key=lambda w: abs(w - args.p_track)) if near else args.p_track
        if pk:
            tot = (sr[f"sc{V:g}"]["up"] + sr[f"sc{V:g}"]["dn"]).sum(1)
            return max(pk, key=lambda w: tot[int(np.argmin(np.abs(om2 - w)))])
        return 0.5 * (args.pmin + args.pmax)

    wrow = {V: energy(V) for V in Vp}
    wrow[0] = ref if ref is not None else wrow[min(Vp)]
    cmP = plt.get_cmap("PRGn").copy()
    cmP.set_bad("0.85")
    rows = [V for V in Vs[::-1]]
    nr, nt = len(rows), len(tips)
    ymax = {t: 1.05 * max(max(sp[k][s][t].max() for s in ("up", "dn")) for k in sp) for t in tips}
    fig = plt.figure(figsize=(3 * 1.55 + nt * 2.7 + 0.5, 1.55 * nr))
    gs = GridSpec(nr, 3 + nt, figure=fig, width_ratios=[1, 1, 1] + [1.8] * nt, wspace=0.14, hspace=0.1)
    print(f"\nmap energies per row: " + ", ".join(f"V={V:g}: {wrow[V]:+.3f}" for V in rows))
    for r, V in enumerate(rows):
        i0 = int(np.argmin(np.abs(om2 - wrow[V])))
        u0, d0 = sr["clean"]["up"][i0], sr["clean"]["dn"][i0]
        thr = args.p_mask * (u0 + d0).mean()

        def Pof(du, dd):
            u, d = u0 + du, d0 + dd
            t = u + d
            return np.where(t > thr, (u - d) / np.maximum(t, 1e-12), np.nan), t

        if V == 0:
            Pc, tc = Pof(0.0, 0.0)
            panels = [(tc, "viridis", 0, None), (Pc, cmP, -1, 1), (Pc, cmP, -1, 1)]
        else:
            Pf, tf = Pof(sr[f"fr{V:g}"]["up"][i0], sr[f"fr{V:g}"]["dn"][i0])
            Ps, ts = Pof(sr[f"sc{V:g}"]["up"][i0], sr[f"sc{V:g}"]["dn"][i0])
            panels = [(ts, "viridis", 0, None), (Pf, cmP, -1, 1), (Ps, cmP, -1, 1)]
        for j, (vec, cm, lo, hi) in enumerate(panels):
            ax = fig.add_subplot(gs[r, j])
            img, h = _cl_img(vec, c, args.half)
            if j == 0:
                hi = np.nanmax(img)
            ax.imshow(img, extent=[-h - .5, h + .5] * 2, origin="lower", cmap=cm, vmin=lo, vmax=hi)
            ax.plot(0, 0, "k+", ms=6)
            _ax_off(ax)
            if j == 0:
                ax.text(0.03, 0.95, f"max {hi:.3g}", transform=ax.transAxes, va="top", fontsize=6, color="w")
                ax.text(0.03, 0.04, rf"$\omega$={om2[i0]:+.3f}", transform=ax.transAxes, fontsize=7, color="k",
                        bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.2))
            if r == 0:
                ax.set_title([r"$\rho(R)$ " + ("SC" if V else "clean"), r"$P$ frozen", r"$P$ SC"][j], fontsize=8)
        for k, t in enumerate(tips):
            ax = fig.add_subplot(gs[r, 3 + k])
            if V == 0:
                ax.plot(om, sp["clean"]["up"][t], color="tab:red", lw=1.2)
                ax.plot(om, sp["clean"]["dn"][t], color="tab:blue", lw=1.2)
            else:
                ax.plot(om, sp[f"fr{V:g}"]["up"][t], color="tab:red", lw=0.8, ls="--", alpha=0.7)
                ax.plot(om, sp[f"fr{V:g}"]["dn"][t], color="tab:blue", lw=0.8, ls="--", alpha=0.7)
                ax.plot(om, sp[f"sc{V:g}"]["up"][t], color="tab:red", lw=1.2)
                ax.plot(om, sp[f"sc{V:g}"]["dn"][t], color="tab:blue", lw=1.2)
            ax.axvline(om2[i0], color="0.45", lw=0.8, ls="--")
            ax.set_xlim(wmin, wmax)
            ax.set_ylim(0, ymax[t])
            ax.set_yticks([])
            ax.tick_params(labelbottom=(r == nr - 1), labelsize=7)
            if r == 0:
                ax.set_title(t, fontsize=8)
            if k == nt - 1:
                ax.text(0.97, 0.88, "clean" if V == 0 else f"V={V:g}t", transform=ax.transAxes, ha="right", fontsize=8)
            if r == nr - 1:
                ax.set_xlabel(r"bias $\omega/t$", fontsize=8)
    if r == 0 or True:
        from matplotlib.lines import Line2D
        fig.legend([Line2D([0], [0], color="tab:red"), Line2D([0], [0], color="tab:blue"),
                    Line2D([0], [0], color="0.3", ls="--", lw=0.8)], [r"$\rho_\uparrow$", r"$\rho_\downarrow$", "frozen (thin)"],
                   loc="lower left", bbox_to_anchor=(0.01, 0.0), fontsize=7, ncol=3, frameon=False)
    how = (f"peak near {args.p_track:g}" if args.p_track is not None and not args.p_omegas else
           (f"fixed energy {args.p_omegas[0]:g}" if args.p_omegas else "strongest peak of each row"))
    fig.suptitle(rf"polarisation maps $P(R,\omega)$ {args.spin_mode} sites, $\delta={args.delta:g}$, map energy {how}; ", fontsize=7.5, y=0.995)
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.01, right=0.995)
    tag = f"w{args.p_track:g}" if args.p_track is not None and not args.p_omegas else ("fixed" if args.p_omegas else "auto")
    fn = os.path.join(_DATA_DIR, f"afsc_spin_stack_d{args.delta:.3f}_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.png")
    fig.savefig(fn, dpi=150)
    print("saved", fn)


def combine_pmaps(args):
    """stack the P-map rows saved by earlier spin_tips runs (one pkl per delta) into one figure"""
    import matplotlib.pyplot as plt
    rows = []
    for d in args.combine_deltas:
        fn = os.path.join(_DATA_DIR, f"afsc_pmap_rows_d{d:.3f}.pkl")
        if not os.path.exists(fn):
            print("missing", fn, "-> run --show spin_tips for this delta first")
            continue
        with open(fn, "rb") as f:
            data = pickle.load(f)
        sel = [r for r in data["rows"] if args.combine_V is None or abs(r["V"] - args.combine_V) < 1e-9]
        rows += [(d, r) for r in sorted(sel, key=lambda r: (r["V"], r["w"]))]
    if not rows:
        return
    cm = plt.get_cmap("PRGn").copy()
    cm.set_bad("0.85")
    cols = [r"$\rho(R)$ SC", r"$P$ clean", r"$P$ frozen", r"$P$ SC", r"$\Delta P$ (SC $-$ clean)"]
    fig, axs = plt.subplots(len(rows), 5, figsize=(1.9 * 5, 1.9 * len(rows)), squeeze=False)
    for r, (d, row) in enumerate(rows):
        h = row["h"]
        for j in range(5):
            ax = axs[r, j]
            img = row["imgs"][j]
            if j == 0:
                lo, hi, cmap = 0, np.nanmax(img), "viridis"
            elif j == 4:
                hi = np.nanmax(np.abs(img)) if np.isfinite(img).any() else 1
                lo, cmap = -hi, cm
            else:
                lo, hi, cmap = -1, 1, cm
            ax.imshow(img, extent=[-h - .5, h + .5] * 2, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            ax.plot(0, 0, "k+", ms=6)
            _ax_off(ax)
            if j in (0, 4):
                ax.text(0.03, 0.95, f"max {hi:.3g}", transform=ax.transAxes, va="top", fontsize=6)
            if j == 0:
                ax.text(0.03, 0.05, rf"$\delta$={d:g}, V={row['V']:g}t, $\omega$={row['w']:+.3f}", transform=ax.transAxes, fontsize=7, color="k", bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.5))
            if r == 0:
                ax.set_title(cols[j], fontsize=8)
    fig.suptitle(r"polarisation map $P(R,\omega)$ on the bare sites; PRGn: green $=+1$ ($\uparrow$), purple $=-1$ ($\downarrow$), grey: DOS too small", fontsize=8)
    fig.tight_layout()
    fn = os.path.join(_DATA_DIR, f"afsc_spin_Pmap_combined_{time.strftime('%Y%m%d-%H%M%S')}.png")
    fig.savefig(fn, dpi=150)
    print("saved", fn)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("physics")
    g.add_argument("--delta", type=float, default=0.10, help="hole doping")
    g.add_argument("--T", type=float, default=0.02)
    g.add_argument("--Vs", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0, 2.0], help="impurity potentials units of t")
    g.add_argument("--method", choices=["embed", "finite"], default="embed",
                   help="embed: self-consistent cluster inside the infinite clean host; "
                        "finite: full 24x24 lattice, Newton-Krylov")
    g = ap.add_argument_group("what to draw")
    g.add_argument("--show", nargs="+", default=["stack", "summary"], choices=["stack", "summary", "poles", "zoom", "profiles", "spin_tips", "spin_stack", "pmap_combine"],
                   help="spin_tips: spin-resolved tips and magnetic character of the peaks; stack: spectra stacked in V; summary: table; poles: bound-state test; zoom: gap-region panels; profiles: |dm|,|ddelta| vs distance")
    g.add_argument("--side", nargs="*", default=[], choices=["map", "spin", "mag", "dm", "dd"],
                   help="extra panels beside every row of the stack: map = real-space STM difference maps (frozen and self-consistent) on the left; spin = the same maps for the spin difference rho_up - rho_down (spin-polarized STM); mag = energy-integrated tip-filtered spin density change over all occupied states; "
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
    g.add_argument("--row_h", type=float, default=1.25, help="stack: height of one row in inches (map panels are squares of this size)")
    g.add_argument("--spec_w", type=float, default=6.0, help="stack: width of the spectrum column in inches")
    g.add_argument("--wspace", type=float, default=0.03, help="stack: horizontal gap between columns (fraction of a panel)")
    g = ap.add_argument_group("spin_tips")
    g.add_argument("--spin_mode", choices=["bare", "wannier"], default="bare", help="bare: orbital centred at the tip (site/bond), wannier: d-wave filtered tip")
    g.add_argument("--spin_Vs", type=float, nargs="*", default=None, help="V rows for spin_tips (default: --Vs)")
    g.add_argument("--peak_w", type=float, default=0.03, help="half-width of the energy window around a peak")
    g.add_argument("--n_peaks", type=int, default=2, help="number of peaks analysed per state")
    g.add_argument("--peak_prom", type=float, default=0.1, help="peak prominence relative to the largest total DOS change")
    g.add_argument("--d_om_sp", type=float, default=0.004, help="energy step of the site-resolved calculation (window pmin..pmax)")
    g.add_argument("--p_half", type=int, default=4, help="half-width (lattice constants) of the lattice order view")
    g.add_argument("--stack_tips", nargs="+", choices=["imp", "nn", "nnn", "bond"], default=["imp", "nn"], help="spin_stack: tips of the spectra panels")
    g.add_argument("--p_track", type=float, default=None, help="spin_stack: follow the peak closest to this energy in every row")
    g.add_argument("--p_tol", type=float, default=0.05, help="spin_stack: max distance of the followed peak from --p_track")
    g.add_argument("--p_omegas", type=float, nargs="*", default=None, help="energies of the P-map rows (default: the strongest peaks)")
    g.add_argument("--combine_deltas", type=float, nargs="*", default=[0.05, 0.10, 0.15], help="pmap_combine: dopings to stack")
    g.add_argument("--combine_V", type=float, default=None, help="pmap_combine: use only this V")
    g.add_argument("--p_mask", type=float, default=0.1, help="polarisation map: mask sites with rho < p_mask x mean clean DOS")
    g.add_argument("--no_baseline", action="store_true", help="do not subtract a linear baseline in the peak windows")
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
    g.add_argument("--gap_thresh", type=float, default=0.01, help="poles: the gap region is where the clean host LDOS is below this fraction of its peak")
    g.add_argument("--lam_max", type=float, default=4.0, help="poles: largest scale factor for the margin")
    args = ap.parse_args()
    if "pmap_combine" in args.show:
        combine_pmaps(args)
        return
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
    emb_c = SelfConsistentEmbedding(host, c=cl, Nk=256, n_mats=100, build_g0=("mag" in args.side))
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
    if "spin_tips" in args.show and states:
        run_spin_tips(args, host, states, dhs, emb_c)
    if "spin_stack" in args.show and states:
        run_spin_stack(args, host, dhs, emb_c)


if __name__ == "__main__":
    main()
