import os
import sys
import time
import pickle
import argparse
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
for _p in (os.path.join(_PROJECT_ROOT, "bdg"), _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import solve_disordered as sd
from solve_disordered import N, Lx, Ly, neel_sign, nn_i, nnn_i

NAMES = ("delta_i", "m_i", "chi_up", "chi_dn", "chin_up", "chin_dn")
SIZES = (N, N, len(nn_i), len(nn_i), len(nnn_i), len(nnn_i))
OFFS = np.concatenate([[0], np.cumsum(SIZES)])
DIM = int(OFFS[-1])


def pack(s):
    return np.concatenate([np.asarray(s[k], float).ravel() for k in NAMES])


def unpack(x, base):
    s = dict(base)
    for k, a, b in zip(NAMES, OFFS[:-1], OFFS[1:]):
        s[k] = x[a:b].copy()
    s["Delta_nn"] = np.zeros(len(nn_i))
    return s


class Map:
    def __init__(self, base, delta, T):
        self.base, self.delta, self.T = base, delta, T
        self.V0 = np.zeros(N)
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        r = sd.solve_disordered(self.delta, self.V0, seed=unpack(x, self.base), max_iter=1, mix=1.0,
                                mix_fluct=1.0, mix_mu=1.0, warmup=0, T=self.T, pairing=False)
        return pack(r)


def ckpt_path(delta, T):
    return os.path.join(_DATA_DIR, f"af_jac_ckpt_d{delta:.3f}_T{T:g}.pkl")


def load_base(delta, T):
    path = os.path.join(_DATA_DIR, f"af_selfcons_final_clean_d{delta:.3f}_T{T:g}.pkl")
    with open(path, "rb") as f:
        res = pickle.load(f)
    keys = NAMES + ("Delta_nn", "mu")
    return {k: res[k] for k in keys}, res


def arnoldi(args):
    base, res = load_base(args.delta, args.T)
    x0 = pack(base)
    F = Map(base, args.delta, args.T)
    ck = ckpt_path(args.delta, args.T)
    if os.path.exists(ck):
        with open(ck, "rb") as f:
            st = pickle.load(f)
        Q, H, F0 = st["Q"], st["H"], st["F0"]
        print(f"resuming Arnoldi at step {len(Q) - 1}", flush=True)
    else:
        F0 = F(x0)
        print(f"|F(x0)-x0| = {np.abs(F0 - x0).max():.2e}  (dimension {DIM})", flush=True)
        rng = np.random.default_rng(args.seed)
        q = rng.standard_normal(DIM)
        Q = [q / np.linalg.norm(q)]
        H = np.zeros((args.steps + 1, args.steps))
    if H.shape[1] < args.steps:
        H2 = np.zeros((args.steps + 1, args.steps))
        H2[:H.shape[0], :H.shape[1]] = H
        H = H2
    t0 = time.time()
    j = len(Q) - 1
    while j < args.steps:
        w = (F(x0 + args.eps * Q[j]) - F0) / args.eps
        for _ in range(2):
            for i in range(j + 1):
                c = Q[i] @ w
                H[i, j] += c
                w = w - c * Q[i]
        H[j + 1, j] = np.linalg.norm(w)
        Q.append(w / H[j + 1, j])
        j += 1
        with open(ck, "wb") as f:
            pickle.dump(dict(Q=Q, H=H, F0=F0), f)
        ev = np.linalg.eigvals(H[:j, :j])
        top = ev[np.argsort(-ev.real)[:3]]
        print(f"  step {j:3d}  largest Re(J) so far: " + "  ".join(f"{z.real:+.4f}{z.imag:+.4f}j" for z in top)
              + f"   [{time.time() - t0:.0f}s]", flush=True)
        if args.budget and time.time() - t0 > args.budget:
            print("time budget reached: run the same command again to continue")
            return
    print("finished", flush=True)


def analyse(args):
    base, res = load_base(args.delta, args.T)
    with open(ckpt_path(args.delta, args.T), "rb") as f:
        st = pickle.load(f)
    Q, H = st["Q"], st["H"]
    m = len(Q) - 1
    Hm = H[:m, :m]
    ev, vec = np.linalg.eig(Hm)
    resid = np.abs(H[m, m - 1] * vec[-1, :])
    order = np.argsort(-ev.real)
    alpha = args.mix
    print(f"Arnoldi dimension {m}.  Eigenvalues J of the map F (stability: iteration factor 1+alpha(J-1), alpha=mix)")
    print(f"{'Re J':>9} {'Im J':>9} {'|J|':>8} {'ritz resid':>11} {'1+a(J-1) at mix=0.15':>22} {'at mix_fluct=0.03':>19}")
    for i in order[:8]:
        z = ev[i]
        print(f"{z.real:+9.4f} {z.imag:+9.4f} {abs(z):8.4f} {resid[i]:11.1e} {abs(1 + 0.15 * (z - 1)):22.4f} {abs(1 + 0.03 * (z - 1)):19.4f}")
    Qm = np.array(Q[:m]).T
    lead = [i for i in order[:3]]
    for i in lead:
        v = Qm @ vec[:, i]
        v = v.real if abs(ev[i].imag) < 1e-6 else v.real
        v = v / np.linalg.norm(v)
        parts = {k: float(np.linalg.norm(v[a:b])) ** 2 for k, a, b in zip(NAMES, OFFS[:-1], OFFS[1:])}
        print(f"\nmode with J = {ev[i].real:+.4f}{ev[i].imag:+.4f}j : weight " + ", ".join(f"{k}={w:.2f}" for k, w in parts.items()))
        for k, field in (("delta_i", v[OFFS[0]:OFFS[1]]), ("m_i*neel", v[OFFS[1]:OFFS[2]] * neel_sign)):
            img = field.reshape(Ly, Lx)
            F2 = np.fft.fft2(img)
            amp = np.abs(F2) ** 2
            amp /= max(amp.sum(), 1e-30)
            iy, ix = np.unravel_index(np.argmax(amp), amp.shape)
            qx = 2 * np.pi * (ix if ix <= Lx // 2 else ix - Lx) / Lx
            qy = 2 * np.pi * (iy if iy <= Ly // 2 else iy - Ly) / Ly
            print(f"   {k:9s}: dominant wavevector q=({qx / np.pi:+.2f}pi, {qy / np.pi:+.2f}pi), share of power {amp[iy, ix]:.2f}")
        if args.save_mode:
            np.save(os.path.join(_DATA_DIR, "af_jac_mode.npy"), v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Jacobian of the self-consistency map at the clean AF state (Arnoldi)")
    ap.add_argument("mode", choices=["arnoldi", "analyse", "linearity"])
    ap.add_argument("--delta", type=float, default=0.12)
    ap.add_argument("--T", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", type=float, default=0.15)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--save_mode", action="store_true")
    a = ap.parse_args()
    if a.mode == "arnoldi":
        arnoldi(a)
    elif a.mode == "analyse":
        analyse(a)
    else:
        base, res = load_base(a.delta, a.T)
        x0 = pack(base)
        F = Map(base, a.delta, a.T)
        F0 = F(x0)
        print(f"|F(x0)-x0| = {np.abs(F0 - x0).max():.2e}")
        rng = np.random.default_rng(1)
        v = rng.standard_normal(DIM)
        v /= np.linalg.norm(v)
        t = time.time()
        r1 = (F(x0 + a.eps * v) - F0) / a.eps
        print(f"one matvec: {time.time() - t:.1f}s")
        r2 = (F(x0 + 2 * a.eps * v) - F0) / (2 * a.eps)
        r3 = (F(x0 - a.eps * v) - F0) / (-a.eps)
        print(f"|Jv| = {np.linalg.norm(r1):.4f};  rel diff eps vs 2eps: {np.linalg.norm(r1 - r2) / np.linalg.norm(r1):.2e};"
              f"  forward vs backward: {np.linalg.norm(r1 - r3) / np.linalg.norm(r1):.2e}")
