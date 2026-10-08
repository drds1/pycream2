"""
plot_rippled_disc.py
====================

Figures and numbers for docs/rippled_disc.md (pycream2.rippled_disc; Starkey,
Huang, Horne & Lin 2023): a flat disc, the NGC 5548 rim and illustrative
ripples, all with a viscous disc at 1500 K at 5 light-days (M_BH = 7e7 Msun)
irradiated with eps_LP = 0.2.

- rippled_disc_geometry.png: height and temperature profiles;
- rippled_disc_responses.png: response functions at three wavelengths;
- rippled_disc_lags_sed.png: lag spectra and spectra;
- rippled_disc_synthetic.png and .json: light curves made with the rim, fitted
  with the rimmed model, a flat thin disc (Model 1) and a flat disc with a free
  temperature slope (Model 2). ``--skip-fits`` reuses the saved .json.

    MPLBACKEND=Agg poetry run python scripts/plot_rippled_disc.py
"""

from __future__ import annotations

import argparse
import functools
import json
import time
import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pycream2.model as model  # noqa: E402
from pycream2 import EchoFit  # noqa: E402
from pycream2.forward_model import thin_disk_response  # noqa: E402
from pycream2.grid_utils import graded_tau_grid  # noqa: E402
from pycream2.plotting import wavelength_to_colour  # noqa: E402
from pycream2.rippled_disc import (  # noqa: E402
    EXAMPLE_RIPPLES,
    NGC5548_RIM,
    disc_surface,
    mean_delay,
    rippled_disc_fnu,
    rippled_disc_response,
)
from pycream2.synthetic import generate_synthetic_dataset  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
M_BH, LOG_MDOT, INCLINATION = 7e7, 0.0587, 45.0
EPS = NGC5548_RIM["lamppost_efficiency"]
DISCS = {
    "flat": (dict(r_out=5.6, lamppost_efficiency=EPS), "0.4"),
    "rim (NGC 5548)": (dict(NGC5548_RIM), "tab:red"),
    "ripples": (dict(EXAMPLE_RIPPLES), "tab:blue"),
}
BANDS = {"w2": 1928.0, "u": 3543.0, "g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}
TAU = jnp.asarray(graded_tau_grid(30.0, 600))


def fig_geometry():
    fig, (ax_h, ax_t) = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    for name, (geom, colour) in DISCS.items():
        s = disc_surface(LOG_MDOT, M_BH, **geom)
        r, H, T, lit = (np.asarray(s[k]) for k in ("r", "H", "T", "lit"))
        ax_h.plot(r, H, color=colour, label=name)
        ax_t.plot(r, T, color=colour, lw=0.8)
        ax_t.plot(r, np.where(lit > 0, T, np.nan), color=colour, lw=1.8)
    viscous = disc_surface(LOG_MDOT, M_BH, r_out=5.6, lamppost_efficiency=0.0)
    ax_t.plot(np.asarray(viscous["r"]), np.asarray(viscous["T"]), "k:", lw=1, label="viscous only")
    ax_h.set_ylabel("height H (light-days)")
    ax_h.legend(frameon=False)
    ax_t.set_yscale("log")
    ax_t.set_ylim(800, 3e4)
    ax_t.set_xlim(0, 5.8)
    ax_t.set_xlabel("radius (light-days)")
    ax_t.set_ylabel("temperature (K)")
    ax_t.legend(frameon=False)
    ax_t.text(0.98, 0.95, "thick: lit by the lamppost; thin: in shadow", transform=ax_t.transAxes, ha="right",
              va="top", fontsize=8)
    for ax in (ax_h, ax_t):
        ax.grid(alpha=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "rippled_disc_geometry.png", dpi=150)
    plt.close(fig)


def fig_responses():
    t = np.asarray(TAU)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, lam in zip(axes, (1928.0, 4770.0, 9134.0)):
        for name, (geom, colour) in DISCS.items():
            psi = np.asarray(rippled_disc_response(TAU, LOG_MDOT, lam, INCLINATION, M_BH, **geom))
            ax.plot(t, psi / psi.max(), color=colour, label=name)
            ax.axvline(mean_delay(t, psi), color=colour, ls="--", lw=0.8)
        ax.set_xlim(0, 10)
        ax.set_title(f"{lam:.0f} Å", color=wavelength_to_colour(lam))
        ax.set_xlabel(r"delay $\tau$ (days)")
        ax.grid(alpha=0.6)
    axes[0].set_ylabel(r"$\psi(\tau)$ / peak")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Responses at i = {INCLINATION:.0f}° (dashed: mean delays)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "rippled_disc_responses.png", dpi=150)
    plt.close(fig)


def fig_lags_sed():
    t = np.asarray(TAU)
    lams = np.geomspace(1300.0, 10000.0, 14)
    lam_sed = np.geomspace(1000.0, 20000.0, 60)
    fig, (ax_l, ax_s) = plt.subplots(1, 2, figsize=(10, 3.8))
    lags = {}
    for name, (geom, colour) in DISCS.items():
        lags[name] = np.array([mean_delay(t, rippled_disc_response(TAU, LOG_MDOT, lam, INCLINATION, M_BH, **geom))
                               for lam in lams])
        ax_l.plot(lams, lags[name], "o-", ms=3, color=colour, label=name)
        fnu = rippled_disc_fnu(lam_sed, LOG_MDOT, INCLINATION, M_BH, dl_mpc=75.0, **geom)
        ax_s.loglog(lam_sed, fnu, color=colour, label=name)
    # tau ~ lambda**(4/3), normalised to the flat disc at 5000 A, for reference.
    ref = np.interp(5000.0, lams, lags["flat"]) * (lams / 5000.0) ** (4 / 3)
    ax_l.plot(lams, ref, "k:", lw=1, label=r"$\tau\propto\lambda^{4/3}$")
    ax_l.set_xlabel("wavelength (Å)")
    ax_l.set_ylabel("mean delay (days)")
    ax_l.legend(frameon=False, fontsize=8)
    ax_s.set_xlabel("wavelength (Å)")
    ax_s.set_ylabel(r"$f_\nu$ at 75 Mpc (mJy)")
    ax_s.set_ylim(2.0, 12.0)
    for ax in (ax_l, ax_s):
        ax.grid(alpha=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "rippled_disc_lags_sed.png", dpi=150)
    plt.close(fig)
    return {name: dict(zip([f"{x:.0f}" for x in lams], v.round(3).tolist())) for name, v in lags.items()}


def synthetic_fits():
    rim = functools.partial(rippled_disc_response, **NGC5548_RIM)
    data = generate_synthetic_dataset(bands=BANDS, M_BH=M_BH, log_mdot_true=LOG_MDOT, inclination_true=INCLINATION,
                                      t_span=250.0, n_obs_per_band=150, noise_level=0.02, tau_max=30.0, seed=3,
                                      response=rim)
    truth = {b: d["tau_mean"] for b, d in data["truth"]["bands"].items()}
    out = dict(truth=dict(log_mdot=LOG_MDOT, inclination=INCLINATION, mean_delays=truth), fits={})
    t = np.asarray(TAU)
    for label, response, slope in (("rimmed disc (true geometry)", rim, False),
                                   ("flat disc, Model 1", thin_disk_response, False),
                                   ("flat disc, Model 2", thin_disk_response, True)):
        model.response_function = response
        ef = EchoFit(M_BH=M_BH, fit_temperature_slope=slope)
        for name, d in data["bands"].items():
            ef.add_lightcurve(name, d["wavelength"], d["t"], d["y"], d["yerr"])
        ef.build_grid(tau_max=30.0)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ef.optimise(num_samples=300, num_restarts=2, method="laplace")
        s = ef.samples
        params = {k: [float(np.median(s[k])), float(np.std(s[k]))]
                  for k in ("log_mdot", "inclination", "temperature_slope") if k in s}
        best = {k: float(np.median(s[k])) for k in ("log_mdot", "inclination", "temperature_slope") if k in s}
        kw = {"viscous_slope": best["temperature_slope"]} if slope else {}
        delays = {b: mean_delay(t, response(TAU, best["log_mdot"], BANDS[b], best["inclination"], M_BH, **kw))
                  for b in BANDS}
        out["fits"][label] = dict(params=params, log_evidence=float(ef.log_evidence), mean_delays=delays,
                                  seconds=time.time() - t0)
        print(label, params, f"lnZ {ef.log_evidence:.1f}", flush=True)
    model.response_function = thin_disk_response
    return out


def fig_synthetic(result):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    lam = np.array(list(BANDS.values()))
    w2 = lambda d: np.array([d[b] - d["w2"] for b in BANDS])  # noqa: E731
    ax.plot(lam, w2(result["truth"]["mean_delays"]), "ko", label="truth (rimmed disc)", zorder=5)
    best = max(f["log_evidence"] for f in result["fits"].values())
    for (label, f), colour in zip(result["fits"].items(), ("tab:red", "0.4", "tab:green")):
        p = f["params"]
        extra = f", α = {p['temperature_slope'][0]:.2f}" if "temperature_slope" in p else ""
        ax.plot(lam, w2(f["mean_delays"]), "s-", ms=4, color=colour,
                label=f"{label}: log ṁ = {p['log_mdot'][0]:.2f}{extra}, Δln Z = {f['log_evidence'] - best:.0f}")
    ax.set_xlabel("wavelength (Å)")
    ax.set_ylabel("mean delay relative to w2 (days)")
    ax.set_title(f"Light curves from the rimmed disc (true log ṁ = {LOG_MDOT:.2f}, i = {INCLINATION:.0f}°)",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "rippled_disc_synthetic.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-fits", action="store_true", help="reuse rippled_disc_synthetic.json")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_geometry()
    fig_responses()
    lags = fig_lags_sed()
    print(json.dumps(lags, indent=1))
    path = OUT / "rippled_disc_synthetic.json"
    if args.skip_fits and path.exists():
        result = json.loads(path.read_text())
    else:
        result = synthetic_fits()
        path.write_text(json.dumps(result, indent=1))
    fig_synthetic(result)
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
