"""
animate_rippled_disc.py
=======================

Animation for docs/rippled_disc.md: a flash from the lamppost sweeping across a
flat disc and across the rimmed disc of Starkey, Huang, Horne & Lin (2023), and
why the rim makes the delays longer.

Two columns (flat disc, rimmed disc), seen face-on, at one wavelength:

- top: the disc's cross-section (height exaggerated), coloured by its
  blackbody colour, with the flash's light front expanding from the lamppost;
- middle: the disc seen from above, glowing where the flash is being reprocessed
  at that moment (surface brightness of the response, one shared colour scale);
- bottom: each disc's response function filling in as time passes, with its
  mean delay.

The flat disc's response is dominated by its hot inner parts and is over within
about a day; the rim's inner face, tilted towards the lamppost, intercepts far
more of the flash than a flat disc at the same radius and lights up as a bright
ring at about 5 light-days, pulling the mean delay out.

    MPLBACKEND=Agg poetry run python scripts/animate_rippled_disc.py                    # the rim
    MPLBACKEND=Agg poetry run python scripts/animate_rippled_disc.py --model ripples    # ripples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from pycream2.grid_utils import graded_tau_grid  # noqa: E402
from pycream2.plotting import blackbody_to_colour  # noqa: E402
from pycream2.rippled_disc import (  # noqa: E402
    EXAMPLE_RIPPLES,
    NGC5548_RIM,
    mean_delay,
    response_elements,
    rippled_disc_response,
)

IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT = {"rim": IMAGES / "rippled_disc_animation.gif", "ripples": IMAGES / "rippled_disc_ripples_animation.gif"}
M_BH, LOG_MDOT, WAVELENGTH = 7e7, 0.0587, 9000.0  # viscous disc at 1500 K at 5 light-days
R_VIEW = 5.6  # light-days shown
FLASH_WIDTH = 0.12  # days: the flash's duration, for the glow


def disc_models(which):
    """The flat disc and the rimmed (``"rim"``) or rippled (``"ripples"``) one,
    with the same lamppost efficiency and outer radius, and the cross-section's
    height range (light-days)."""
    other = NGC5548_RIM if which == "rim" else EXAMPLE_RIPPLES
    name = "rimmed disc (Starkey et al. 2023)" if which == "rim" else "rippled disc"
    flat = dict(r_out=other["r_out"], lamppost_efficiency=other["lamppost_efficiency"])
    return {"flat disc": flat, name: dict(other)}, (0.22 if which == "rim" else 0.07)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--t-max", type=float, default=7.5, help="days")
    parser.add_argument("--model", choices=("rim", "ripples"), default="rim",
                        help="the disc shown beside the flat one (default: the NGC 5548 rim)")
    args = parser.parse_args()

    tau = jnp.asarray(graded_tau_grid(20.0, 800))
    t_np = np.asarray(tau)
    columns = []
    models, z_top = disc_models(args.model)
    for name, geom in models.items():
        e = response_elements(LOG_MDOT, WAVELENGTH, 0.0, M_BH, n_phi=96, full_circle=True, **geom)
        s = e["surface"]
        r = np.asarray(e["r"])
        dr = np.asarray(s["dr"])
        keep = r <= R_VIEW
        weight = np.asarray(e["weight"])[keep]
        delay = np.asarray(e["delay"])[keep]
        # Surface brightness of the response: weight per unit (projected) area.
        area = (r * dr)[keep][:, None] * (2 * np.pi / weight.shape[1])
        brightness = weight / area
        psi = np.asarray(rippled_disc_response(tau, LOG_MDOT, WAVELENGTH, 0.0, M_BH, **geom))
        columns.append(dict(
            name=name, r=r[keep], H=np.asarray(s["H"])[keep], T=np.asarray(s["T"])[keep],
            h_lp=float(s["h_lp"]), phi=np.asarray(e["phi"]), delay=delay, brightness=brightness,
            psi=psi, mean=mean_delay(t_np, psi),
        ))

    vmax = max(c["brightness"].max() for c in columns)
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    psi_max = max(c["psi"].max() for c in columns)

    fig = plt.figure(figsize=(10, 9.2))
    grid = fig.add_gridspec(3, 2, height_ratios=[0.8, 2.2, 1.2], hspace=0.35, wspace=0.12)
    title = fig.suptitle("", fontsize=12)
    artists = []
    for k, c in enumerate(columns):
        colour = "tab:grey" if k == 0 else "tab:red"
        # Cross-section, height exaggerated.
        ax = fig.add_subplot(grid[0, k])
        x = np.concatenate([-c["r"][::-1], c["r"]])
        z = np.concatenate([c["H"][::-1], c["H"]])
        temp = np.concatenate([c["T"][::-1], c["T"]])
        ax.set_facecolor("0.12")
        ax.fill_between(x, -0.01, z, color="0.35", lw=0)
        segments = np.stack([np.column_stack([x[:-1], z[:-1]]), np.column_stack([x[1:], z[1:]])], axis=1)
        ax.add_collection(LineCollection(segments, colors=[blackbody_to_colour(max(T, 1000.0)) for T in temp[:-1]],
                                         linewidths=3))
        ax.plot(0, c["h_lp"], marker="*", ms=13, color="gold", mec="k", mew=0.5, zorder=5)
        front, = ax.plot([], [], color="gold", lw=1.5, alpha=0.9, label="light front")
        ax.set_xlim(-R_VIEW, R_VIEW)
        ax.set_ylim(-0.01 * z_top / 0.22, z_top)
        ax.set_title(c["name"], fontsize=11)
        ax.set_ylabel("height (ld)" if k == 0 else "")
        if k == 1:
            ax.tick_params(labelleft=False)  # same scale as the left panel
        ax.text(0.01, 0.95, "cross-section (height exaggerated), coloured by temperature", transform=ax.transAxes,
                fontsize=7, va="top", color="0.85")
        # Face-on view of the reprocessed flash.
        axp = fig.add_subplot(grid[1, k], projection="polar")
        rr, pp = np.meshgrid(c["r"], c["phi"], indexing="ij")
        mesh = axp.pcolormesh(pp, rr, np.full_like(c["brightness"], np.nan), norm=norm, cmap="inferno",
                              shading="auto")
        axp.set_facecolor("black")
        axp.set_ylim(0, R_VIEW)
        axp.set_yticks([1, 3, 5])
        axp.set_yticklabels(["1", "3", "5 ld"], color="0.7", fontsize=7)
        axp.set_rlabel_position(100)
        axp.set_xticks([])
        axp.grid(color="0.4", alpha=0.6, lw=0.5)
        # Response function filling in.
        axr = fig.add_subplot(grid[2, k])
        axr.plot(t_np, c["psi"], color=colour, lw=1, alpha=0.35)
        fill = [axr.fill_between(t_np[:1], 0, c["psi"][:1], color=colour, alpha=0.6, lw=0)]
        now = axr.axvline(0.0, color="k", lw=0.8)
        axr.axvline(c["mean"], color=colour, ls="--", lw=1)
        axr.text(0.97, 0.6, f"mean delay {c['mean']:.1f} d", transform=axr.transAxes, ha="right", color=colour,
                 fontsize=10, bbox=dict(fc="white", ec="none", alpha=0.8))
        axr.set_xlim(0, args.t_max)
        axr.set_ylim(0, 1.05 * psi_max)
        axr.set_xlabel(r"time after the flash, $\tau$ (days)")
        axr.set_ylabel(rf"response $\psi(\tau)$ at {WAVELENGTH:.0f} Å" if k == 0 else "")
        axr.grid(alpha=0.6)
        artists.append(dict(front=front, mesh=mesh, fill=fill, now=now, axr=axr, axp=axp, colour=colour))
    cbar = fig.colorbar(artists[1]["mesh"], ax=[a["axp"] for a in artists], fraction=0.03, pad=0.02)
    cbar.set_label("reprocessed surface brightness (relative)", fontsize=9)

    theta = np.linspace(0, np.pi, 200)
    times = np.linspace(0.0, args.t_max, args.frames)

    def update(i):
        t = times[i]
        title.set_text(f"A flash from the lamppost, seen face-on at {WAVELENGTH:.0f} Å: t = {t:4.1f} days")
        for c, a in zip(columns, artists):
            a["front"].set_data(t * np.cos(theta), c["h_lp"] + t * np.sin(theta))
            glow = c["brightness"] * np.exp(-0.5 * ((t - c["delay"]) / FLASH_WIDTH) ** 2)
            a["mesh"].set_array(np.where(glow > norm.vmin, glow, np.nan).ravel())
            a["fill"][0].remove()
            upto = t_np <= t
            a["fill"][0] = a["axr"].fill_between(t_np[upto], 0, c["psi"][upto], color=a["colour"], alpha=0.6, lw=0)
            a["now"].set_xdata([t, t])
        return []

    ani = FuncAnimation(fig, update, frames=args.frames, blit=False)
    out = OUT[args.model]
    out.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
