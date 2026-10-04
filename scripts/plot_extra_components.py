"""
Figures and numbers for docs/extra_components.md: the diffuse-continuum
(second reprocessor) and slow-background components, and a synthetic
recovery study showing the accretion-disc parameters are still recovered
with them switched on.

The study simulates light curves (pycream2.synthetic.generate_synthetic_dataset,
five SDSS-like bands, the default response) either without extra components
("plain") or with a diffuse continuum in the three reddest bands and slow
trends in two bands ("extras"), for several noise realisations, and fits each
with optimise() both without and with the components switched on. It writes
docs/images/extra_components_*.png and docs/images/extra_components_study.json.

Usage (about 30 minutes on a laptop; --replot redraws the recovery figure from
the saved results without refitting; --example-only refits and redraws only
the example-fit figure):
    MPLBACKEND=Agg poetry run python scripts/plot_extra_components.py [--replot | --example-only]
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pycream2 import EchoFit
from pycream2.forward_model import legendre_background_basis, lognormal_response, mix_diffuse_continuum, response_function
from pycream2.plotting import wavelength_to_colour
from pycream2.synthetic import generate_synthetic_dataset

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
M_BH = 1e8
LOG_MDOT, INCLINATION = 0.3, 35.0
BANDS = {"u": 3543.0, "g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}
# Diffuse continuum (fraction, median delay in days, width in dex) rising with
# wavelength, as Cackett, Zoghbi & Ulrich (2022) found; slow trends in u and z.
DCE = {"r": (0.2, 8.0, 0.25), "i": (0.3, 8.0, 0.25), "z": (0.4, 8.0, 0.25)}
BACKGROUND = {"u": [0.4, -0.2], "z": [-0.3, 0.15]}
SEEDS = [11, 12, 13, 14]
TAU_MAX = 60.0


def fig_responses():
    """The two response components and their mixture, for one band."""
    tau = jnp.linspace(0.0, 30.0, 3001)
    disc = response_function(tau, LOG_MDOT, BANDS["i"], INCLINATION, M_BH)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ax = axes[0]
    ax.plot(tau, disc, color="k", label="disc")
    for frac, ls in ((0.15, ":"), (0.3, "--"), (0.5, "-.")):
        ax.plot(tau, mix_diffuse_continuum(disc, tau, frac, 8.0, 0.25), ls=ls, color="tab:red", label=f"f = {frac}")
    ax.set_title("fraction f (median 8 d, 0.25 dex)")
    ax = axes[1]
    for median, ls in ((3.0, ":"), (8.0, "--"), (20.0, "-.")):
        ax.plot(tau, lognormal_response(tau, median, 0.25), ls=ls, color="tab:red", label=f"median {median:g} d")
    ax.set_title("diffuse component: median delay (0.25 dex)")
    ax = axes[2]
    for width, ls in ((0.1, ":"), (0.25, "--"), (0.5, "-.")):
        ax.plot(tau, lognormal_response(tau, 8.0, width), ls=ls, color="tab:red", label=f"{width:g} dex")
    ax.set_title("diffuse component: width (median 8 d)")
    for ax in axes:
        ax.set_xlabel("lag (days)")
        ax.grid(alpha=0.6)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("response (per day)")
    fig.tight_layout()
    fig.savefig(OUT / "extra_components_responses.png", dpi=150)
    plt.close(fig)


def fig_background(data):
    """A light curve with an injected slow background, and the basis."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [2, 1]})
    d, truth = data["bands"]["u"], data["truth"]["bands"]["u"]
    ax = axes[0]
    ax.errorbar(d["t"], d["y"], d["yerr"], fmt="o", ms=2.5, color=wavelength_to_colour(BANDS["u"]), alpha=0.7,
                label="u light curve")
    order = np.argsort(d["t"])
    ax.plot(d["t"][order], (truth["C_band"] + truth["background_curve"])[order], color="k", lw=1.5,
            label="offset + injected background")
    ax.set_xlabel("time (days)")
    ax.set_ylabel("flux")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.6)
    ax = axes[1]
    t = np.linspace(0.0, 250.0, 300)
    basis = np.asarray(legendre_background_basis(t, (0.0, 250.0), 3))
    for k in range(3):
        ax.plot(t, basis[:, k], label=f"P$_{k + 1}$")
    ax.set_xlabel("time (days)")
    ax.set_title("background basis (Legendre)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "extra_components_background.png", dpi=150)
    plt.close(fig)


def fit(data, extras: bool):
    ef = EchoFit(M_BH=M_BH)
    for n, d in data["bands"].items():
        ef.add_lightcurve(n, d["wavelength"], d["t"], d["y"], d["yerr"],
                          diffuse_continuum=extras and n in DCE, background_order=2 if extras and n in BACKGROUND else 0)
    ef.build_grid(tau_max=TAU_MAX)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ef.optimise(num_samples=300, num_restarts=2)
    s = ef.samples
    out = dict(seconds=time.perf_counter() - t0, log_evidence=ef.log_evidence,
               restarts_agreeing=ef.optimise_timings["restarts_agreeing"],
               log_mdot=[float(np.mean(s["log_mdot"])), float(np.std(s["log_mdot"]))],
               inclination=[float(np.mean(s["inclination"])), float(np.std(s["inclination"]))])
    if extras:
        out["dce"] = {n: {k: [float(np.mean(s[f"dce_{k}_{n}"])), float(np.std(s[f"dce_{k}_{n}"]))]
                          for k in ("fraction", "delay", "width")} for n in DCE}
        out["background"] = {n: [np.mean(s[f"bg_{n}"], axis=0).tolist(), np.std(s[f"bg_{n}"], axis=0).tolist()]
                             for n in BACKGROUND}
    return out, ef


def run_study():
    results = []
    example = None
    for seed in SEEDS:
        for scenario in ("plain", "extras"):
            kwargs = dict(diffuse_continuum=DCE, background=BACKGROUND) if scenario == "extras" else {}
            data = generate_synthetic_dataset(bands=BANDS, M_BH=M_BH, log_mdot_true=LOG_MDOT, inclination_true=INCLINATION,
                                              n_obs_per_band=120, t_span=250.0, noise_level=0.03, seed=seed, **kwargs)
            for model in ("without components", "with components"):
                out, ef = fit(data, extras=model == "with components")
                out.update(seed=seed, data=scenario, model=model)
                results.append(out)
                print(f"seed {seed} data {scenario:6s} fit {model:18s}: log_mdot {out['log_mdot'][0]:.3f}"
                      f"+/-{out['log_mdot'][1]:.3f}, i {out['inclination'][0]:.1f}+/-{out['inclination'][1]:.1f}, "
                      f"lnZ {out['log_evidence']:.1f}, {out['seconds']:.0f} s", flush=True)
                if example is None and scenario == "extras" and model == "with components":
                    example = (data, ef)
    return results, example


def fig_recovery(results):
    """Disc parameters (median and 1 sd) for every data set and model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    combos = [("plain", "without components"), ("plain", "with components"),
              ("extras", "without components"), ("extras", "with components")]
    colours = ["tab:blue", "tab:cyan", "tab:red", "tab:green"]
    labels = ["no extras in data, fitted without", "no extras in data, fitted with",
              "extras in data, fitted without", "extras in data, fitted with"]
    limits = {"log_mdot": (0.0, 1.0), "inclination": (0.0, 80.0)}
    for ax, key, truth, xl in ((axes[0], "log_mdot", LOG_MDOT, "log mdot"), (axes[1], "inclination", INCLINATION, "inclination (deg)")):
        lo, hi = limits[key]
        for j, ((data, model), colour, label) in enumerate(zip(combos, colours, labels)):
            rows = [r for r in results if r["data"] == data and r["model"] == model]
            labelled = False
            for k, r in enumerate(rows):
                y = j * (len(SEEDS) + 1) + k
                value, err = r[key]
                if lo <= value <= hi:
                    ax.errorbar(value, y, xerr=min(err, hi - lo), fmt="o", ms=4, color=colour,
                                label=label if not labelled and ax is axes[0] else None)
                    labelled = True
                else:  # off the plotted range: an arrow at the edge, labelled with the value
                    edge = hi if value > hi else lo
                    ax.annotate(f"{value:.2g}", xy=(edge, y), xytext=(edge - 0.12 * (hi - lo) * np.sign(value - edge), y),
                                color=colour, fontsize=7, va="center", ha="center",
                                arrowprops=dict(arrowstyle="->", color=colour))
        ax.axvline(truth, color="k", lw=1, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.6, axis="x")
    axes[0].set_yticks([])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT / "extra_components_recovery.png", dpi=150)
    plt.close(fig)


def fig_example(example):
    data, ef = example
    fig, _ = ef.plot_lightcurve_fits()
    fig.savefig(OUT / "extra_components_example_fit.png", dpi=110)
    plt.close(fig)


def main():
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    if "--replot" in sys.argv:  # redraw the recovery figure from the saved results only
        fig_recovery(json.loads((OUT / "extra_components_study.json").read_text())["results"])
        return
    if "--example-only" in sys.argv:  # refit and redraw only the example-fit figure (~1 minute)
        data = generate_synthetic_dataset(bands=BANDS, M_BH=M_BH, log_mdot_true=LOG_MDOT, inclination_true=INCLINATION,
                                          n_obs_per_band=120, t_span=250.0, noise_level=0.03, seed=SEEDS[0],
                                          diffuse_continuum=DCE, background=BACKGROUND)
        fig_example((data, fit(data, extras=True)[1]))
        return
    fig_responses()
    fig_background(generate_synthetic_dataset(bands=BANDS, M_BH=M_BH, log_mdot_true=LOG_MDOT,
                                              inclination_true=INCLINATION, n_obs_per_band=120, t_span=250.0,
                                              noise_level=0.03, seed=SEEDS[0], diffuse_continuum=DCE,
                                              background=BACKGROUND))
    results, example = run_study()
    fig_recovery(results)
    fig_example(example)
    (OUT / "extra_components_study.json").write_text(json.dumps(dict(
        truth=dict(log_mdot=LOG_MDOT, inclination=INCLINATION, dce=DCE, background=BACKGROUND),
        results=results), indent=1))


if __name__ == "__main__":
    main()
