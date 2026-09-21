import numpy as np


def wannier_envelope(dx, dy, gamma=1.5, a=1.0):
    r_a = a * (gamma / 2.0) ** (1.0 / gamma)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    angular = (dx / a) ** 2 - (dy / a) ** 2
    return angular * np.exp(-(dist / r_a) ** gamma)


def wannier_weight(rx, ry, Rx, Ry, a=1.0, gamma=1.5):
    return wannier_envelope(rx - Rx, ry - Ry, gamma=gamma, a=a)


def plot_wannier(gamma=1.5, extent=3.0, n=300, a=1.0, save_path=None):
    import matplotlib.pyplot as plt

    r = np.linspace(-extent, extent, n)
    RX, RY = np.meshgrid(r, r)
    W = wannier_envelope(RX, RY, gamma=gamma, a=a)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    im = ax.imshow(W, extent=[-extent, extent, -extent, extent], origin="lower",
                    cmap="bwr_r", vmin=-np.abs(W).max(), vmax=np.abs(W).max())
    s = np.arange(-int(extent), int(extent) + 1)
    SX, SY = np.meshgrid(s, s)
    ax.scatter(SX, SY, color="black", s=8)
    ax.set_title(rf"Wannier function $\omega_{{R=0}}(\mathbf{{r}})$, $\gamma$={gamma}")
    ax.set_xlabel("r_x/a")
    ax.set_ylabel("r_y/a")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig
