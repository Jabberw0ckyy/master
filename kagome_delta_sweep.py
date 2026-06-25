import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

# ── Hamiltonian (same as before) ──────────────────────────────────────────────
def build_slab_H(kx, N, t=1.0, delta=0.0, ty=1.0, mu=0.0):
    H = np.zeros((3*N, 3*N), dtype=complex)
    def idx(ny, s): return 3*ny + s
    h_AB = -(t + delta) - (t - delta) * np.exp(-1j * kx)
    for ny in range(N):
        for s in range(3):
            H[idx(ny,s), idx(ny,s)] -= mu
        H[idx(ny,0), idx(ny,1)] += h_AB
        H[idx(ny,1), idx(ny,0)] += np.conj(h_AB)
        H[idx(ny,0), idx(ny,2)] += -ty
        H[idx(ny,2), idx(ny,0)] += -ty
        H[idx(ny,1), idx(ny,2)] += -ty
        H[idx(ny,2), idx(ny,1)] += -ty
        if ny >= 1:
            H[idx(ny,0),   idx(ny-1,2)] += -ty
            H[idx(ny-1,2), idx(ny,0)]   += -ty
            h_BC = -ty * np.exp(1j * kx)
            H[idx(ny,1),   idx(ny-1,2)] += h_BC
            H[idx(ny-1,2), idx(ny,1)]   += np.conj(h_BC)
    return H

# ── Fixed parameters ──────────────────────────────────────────────────────────
N      = 30
t      = 1.0
ty     = 1.0
mu     = 0.0
n_pts  = 250          
n_edge = 2

kx_all = np.concatenate([
    np.linspace(0,          2*np.pi/3, n_pts, endpoint=False),
    np.linspace(2*np.pi/3, np.pi,      n_pts, endpoint=False),
    np.linspace(np.pi,      4*np.pi/3, n_pts, endpoint=False),
    np.linspace(4*np.pi/3, 2*np.pi,    n_pts, endpoint=True),
])
Nk    = len(kx_all)
M_idx = 2 * n_pts

bot = list(range(3 * n_edge))
top = list(range(3 * (N - n_edge), 3 * N))

hs_x = [0, n_pts, 2*n_pts, 3*n_pts, 4*n_pts - 1]
hs_l = [r'$\bar{\Gamma}$', r'$\bar{K}$', r'$\bar{M}$', r"$\bar{K}'$", r'$\bar{\Gamma}$']

marker_cycle = [('v', 'royalblue'), ('^', 'crimson'), ('D', 'darkorange'),
                ('s', 'purple'),    ('o', 'green')]

# ── Sweep values ───────────────────────────────────────────────────────────────
delta_values = [0.0, 0.8, 1.0, 1.5]

def compute_for_delta(delta):
    all_evals       = np.zeros((Nk, 3*N))
    all_edge_weight = np.zeros((Nk, 3*N))
    for i, kx in enumerate(kx_all):
        H = build_slab_H(kx, N, t, delta, ty, mu)
        evals, evecs = np.linalg.eigh(H)
        all_evals[i]       = evals
        all_edge_weight[i] = np.sum(np.abs(evecs[bot + top, :])**2, axis=0)

    H_M              = build_slab_H(np.pi, N, t, delta, ty, mu)
    evals_M, evecs_M = np.linalg.eigh(H_M)
    edge_w_M         = np.sum(np.abs(evecs_M[bot + top, :])**2, axis=0)

    E_tamm = 2 * delta
    E_c    = 2 * np.sqrt(delta**2 + ty**2)
    analytic_Es = [E_tamm, -E_tamm, E_c, -E_c, 0.0]

    ldos_states = []
    for E_target in analytic_Es:
        candidates = [(i, abs(evals_M[i] - E_target), edge_w_M[i])
                      for i in range(len(evals_M))
                      if edge_w_M[i] > 0.50 and abs(evals_M[i] - E_target) < 0.15]
        if candidates:
            best = min(candidates, key=lambda x: x[1])
            if best[0] not in ldos_states:
                ldos_states.append(best[0])
    ldos_states = sorted(set(ldos_states))

    # global band extrema -> gap01 (between band index N..2N-1 sorted bulk groups)
    e_sorted_per_k = np.sort(all_evals, axis=1)
    band0_max = e_sorted_per_k[:, :N].max()
    band1_min = e_sorted_per_k[:, N:2*N].min()
    return dict(all_evals=all_evals, all_edge_weight=all_edge_weight,
                evals_M=evals_M, edge_w_M=edge_w_M,
                ldos_states=ldos_states, E_tamm=E_tamm, E_c=E_c,
                band0_max=band0_max, band1_min=band1_min)

results = {d: compute_for_delta(d) for d in delta_values}

# ── Figure 1: 2x2 grid of band structures across delta ────────────────────────
fig = plt.figure(figsize=(13, 9.5))
gs  = gridspec.GridSpec(2, 2, wspace=0.22, hspace=0.32,
                         left=0.06, right=0.97, top=0.91, bottom=0.07)

for panel_idx, delta in enumerate(delta_values):
    r = results[delta]
    ax = fig.add_subplot(gs[panel_idx // 2, panel_idx % 2])
    x = np.arange(Nk)

    for n in range(3*N):
        ew  = r['all_edge_weight'][:, n]
        en  = r['all_evals'][:, n]
        pts = np.array([x, en]).T.reshape(-1, 1, 2)
        segs    = np.concatenate([pts[:-1], pts[1:]], axis=1)
        ew_mid  = (ew[:-1] + ew[1:]) / 2
        colors  = plt.cm.Blues(0.15 + 0.85 * ew_mid)
        lc = LineCollection(segs, colors=colors, lw=0.8, zorder=2)
        ax.add_collection(lc)

    for n in range(3*N):
        mask = r['all_edge_weight'][:, n] > 0.65
        if mask.any():
            ax.plot(x[mask], r['all_evals'][mask, n],
                    color='royalblue', lw=1.4, zorder=5, solid_capstyle='round')

    ax.axvline(M_idx, color='red', lw=1.0, ls='--', zorder=8, alpha=0.7)

    # shade the indirect bulk gap (band0_max -> band1_min) — same style as
    # kagome_breathing.py: gold band, no width label
    gap_width = r['band1_min'] - r['band0_max']
    if gap_width > 1e-3:
        ax.axhspan(r['band0_max'], r['band1_min'],
                   color='gold', alpha=0.20, zorder=1)

    for k, i in enumerate(r['ldos_states']):
        E_i = r['evals_M'][i]
        mk, col = marker_cycle[k % len(marker_cycle)]
        ax.plot(M_idx, E_i, marker=mk, color=col, ms=8,
                zorder=10, ls='none', mec='black', mew=0.5)

    for xp in hs_x:
        ax.axvline(xp, color='gray', lw=0.5, ls='--', zorder=0)
    ax.set_xticks(hs_x)
    ax.set_xticklabels(hs_l, fontsize=11)
    ax.set_xlim(0, Nk - 1)
    ax.set_ylim(-4.5, 3.5)
    ax.set_ylabel(r'$E/t$', fontsize=12)
    ax.set_title(rf'$\delta = {delta}$    '
                 rf'($E_{{\rm Tamm}}={r["E_tamm"]:.2f},\ '
                 rf'E_C=\pm{r["E_c"]:.2f}$)', fontsize=11)
    panel_letter = ['(a)', '(b)', '(c)', '(d)'][panel_idx]
    ax.text(0.015, 0.97, panel_letter, transform=ax.transAxes,
            fontsize=12, va='top', fontweight='bold')

plt.show()

# ── Figure 2: continuous tracking of key energies vs delta ────────────────────
"""
delta_fine = np.linspace(0, 1.0, 60)
E_tamm_fine = 2 * delta_fine
E_c_fine    = 2 * np.sqrt(delta_fine**2 + ty**2)

# also track numerically: gap01 (band0-band1 separation) and flat band max
gap01_fine   = np.zeros_like(delta_fine)
flatband_max = np.zeros_like(delta_fine)
zero_mode_E  = np.zeros_like(delta_fine)

for j, delta in enumerate(delta_fine):
    all_evals = np.zeros((Nk, 3*N))
    for i, kx in enumerate(kx_all):
        H = build_slab_H(kx, N, t, delta, ty, mu)
        evals = np.linalg.eigvalsh(H)
        all_evals[i] = evals
    # crude bulk-band separation: use bottom third / middle third / top third
    # of the sorted spectrum at each k as proxies for band0/1/2 groups
    sorted_e = np.sort(all_evals, axis=1)
    band0 = sorted_e[:, :N]
    band1 = sorted_e[:, N:2*N]
    band2 = sorted_e[:, 2*N:]
    gap01_fine[j]   = band1.min() - band0.max()
    flatband_max[j] = band2.max()

    H_M = build_slab_H(np.pi, N, t, delta, ty, mu)
    evals_M, evecs_M = np.linalg.eigh(H_M)
    edge_w_M = np.sum(np.abs(evecs_M[bot + top, :])**2, axis=0)
    near_zero = [(abs(evals_M[i]), edge_w_M[i], evals_M[i]) for i in range(len(evals_M))
                 if edge_w_M[i] > 0.5 and abs(evals_M[i]) < 0.3]
    zero_mode_E[j] = min(near_zero, key=lambda x: x[0])[2] if near_zero else np.nan

fig2, axs = plt.subplots(1, 2, figsize=(11, 4.3))

ax1 = axs[0]
ax1.plot(delta_fine, E_tamm_fine, color='royalblue', lw=2, label=r'Tamm state $E=2\delta$ (analytic)')
ax1.plot(delta_fine, E_c_fine,    color='darkorange', lw=2, label=r'$C$-resonance $E=2\sqrt{\delta^2+t_y^2}$')
ax1.plot(delta_fine, zero_mode_E, color='green', lw=1.5, ls='--', marker='o', ms=3,
         label='zero mode (numerical)')
ax1.axhline(0, color='gray', lw=0.5)
ax1.set_xlabel(r'$\delta/t$', fontsize=12)
ax1.set_ylabel(r'$E/t$', fontsize=12)
ax1.set_title('Surface-state energies at $\\bar{M}$ vs $\\delta$', fontsize=11)
ax1.legend(fontsize=8.5, loc='upper left')
ax1.text(0.02, 0.97, '(a)', transform=ax1.transAxes, fontsize=12, va='top', fontweight='bold')

ax2 = axs[1]
ax2.plot(delta_fine, gap01_fine, color='crimson', lw=2, label=r'gap$_{01}$')
ax2.plot(delta_fine, flatband_max - 2*t, color='purple', lw=2, ls='--',
         label=r'flat-band dispersion (max $-\,2t$)')
ax2.axhline(0, color='gray', lw=0.5)
ax2.set_xlabel(r'$\delta/t$', fontsize=12)
ax2.set_ylabel(r'$\Delta E/t$', fontsize=12)
ax2.set_title('Bulk gap and flat-band dispersion vs $\\delta$', fontsize=11)
ax2.legend(fontsize=9, loc='upper left')
ax2.text(0.02, 0.97, '(b)', transform=ax2.transAxes, fontsize=12, va='top', fontweight='bold')

fig2.suptitle(r'Continuous evolution with $\delta$    ($t=t_y=1,\ \mu=0,\ N=30$)', fontsize=11)
fig2.tight_layout(rect=[0, 0, 1, 0.93])
"""