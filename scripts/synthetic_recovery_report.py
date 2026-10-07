"""
Figures and summary tables for scripts/synthetic_recovery_grid.py's results.

Reads every ``<cache>/<config>/<method>.json`` (and the samples beside them)
and writes, into ``--figures``:
- ``example_lightcurves.png``: one seed's synthetic ugriz light curves;
- ``precision.png``: the posterior standard deviation of log(M Mdot) and of
  inclination against SNR, one panel per filter set, one line per cadence,
  NUTS and optimise() side by side (median over seeds);
- ``accuracy.png``: the rms error of the posterior mean over seeds, same layout;
- ``calibration.png``: the z-scores (posterior mean minus truth, over the
  posterior sd) of every fit, per method, against a unit Gaussian;
- ``optimise_vs_nuts.png``: optimise()'s posterior mean and sd against NUTS's,
  configuration by configuration;
- ``timing.png``: wall time per fit against the number of data points;
and ``summary.csv`` (one row per fit) and ``summary.json`` (aggregates quoted
in the report).

Usage (from the repository root):
    MPLBACKEND=Agg poetry run python scripts/synthetic_recovery_report.py \
        --cache experiments/recovery_grid/cache --figures experiments/recovery_grid/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_recovery_grid as grid  # noqa: E402
from pycream2.plotting import wavelength_to_colour  # noqa: E402

METHOD_STYLE = {"nuts": dict(ls="-", marker="o", label="NUTS"),
                "optimise": dict(ls="--", marker="s", label="optimise()", mfc="white")}
PARAMS = {"log_mdot": "log(M Mdot)", "inclination": "inclination (deg)"}
# The 2016 paper's results (Table 4, mean of its three chains' sd), at mean 1-day cadence.
PAPER_2016 = {("gi", 700): (0.03, 4.8), ("gi", 100): (0.18, 14.0), ("ugriz", 100): (0.09, 8.1)}


def cadence_colours(cadences):
    """One sequential hue, light to dark with finer sampling (darker = more data)."""
    cmap = plt.get_cmap("Blues")
    order = sorted(cadences, reverse=True)
    return {c: cmap(0.4 + 0.6 * k / max(len(order) - 1, 1)) for k, c in enumerate(order)}


def load(cache: Path) -> list[dict]:
    rows = []
    for f in sorted(cache.glob("*/*.json")):
        m = json.loads(f.read_text())
        row = {k: m[k] for k in ("band_set", "snr", "cadence", "seed", "method", "seconds", "n_obs")}
        for p in PARAMS:
            for k in ("mean", "sd", "median", "q16", "q84", "q025", "q975", "z"):
                row[f"{p}_{k}"] = m[p][k]
            row[f"{p}_truth"] = m[p]["truth"]
        row["divergences"] = m.get("divergences", "")
        row["min_ess"] = min(m["ess"].values()) if "ess" in m else ""
        row["restarts_agreeing"] = m.get("timings", {}).get("restarts_agreeing", "")
        row["n_warnings"] = len(m["warnings"])
        t = m.get("timings", {})
        for k in ("lbfgs_seconds", "hessian_and_newton_seconds", "draws_seconds"):
            row[k] = t.get(k, "")
        rows.append(row)
    return rows


def _scale(p):
    # Report log_mdot as log(M Mdot); the offset does not change sds or errors.
    return grid.LOG_MMDOT_OFFSET if p == "log_mdot" else 0.0


def _agg(rows, band_set, method, cadence, snr, key):
    v = [r[key] for r in rows if (r["band_set"], r["method"], r["cadence"], r["snr"]) == (band_set, method, cadence, snr)]
    return v


def figure_example(figures: Path, seed: int = 0) -> None:
    fine = grid.simulate("ugriz", 1e9, 0.1, seed)
    data = {snr: grid.simulate("ugriz", snr, 1.0, seed) for snr in (100,)}
    fig, axes = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
    for ax, name in zip(axes, "ugriz"):
        colour = wavelength_to_colour(grid.FILTERS[name][0])
        d = data[100][name]
        ax.errorbar(d["t"], d["y"], d["yerr"], fmt="o", ms=2.5, color="0.25", lw=0.8, zorder=1)
        ax.plot(fine[name]["t"], fine[name]["y"], color=colour, lw=1.5, zorder=3)
        ax.set_ylabel(f"{name}\n(mJy)")
        ax.grid(alpha=0.6)
    axes[-1].set_xlabel("time (days)")
    axes[0].set_title(f"Seed {seed}: noise-free light curves and SNR = 100 data at a mean 1-day cadence")
    fig.tight_layout()
    fig.savefig(figures / "example_lightcurves.png", dpi=150)
    plt.close(fig)


def figure_by_snr(rows, figures: Path, stat: str, fname: str, ylabel: str) -> None:
    band_sets = [b for b in grid.BAND_SETS if any(r["band_set"] == b for r in rows)]
    cadences = sorted({r["cadence"] for r in rows})
    snrs = sorted({r["snr"] for r in rows})
    colours = cadence_colours(cadences)
    fig, axes = plt.subplots(2, len(band_sets), figsize=(4.2 * len(band_sets), 7), sharey="row", squeeze=False)
    for col, band_set in enumerate(band_sets):
        for row_i, p in enumerate(PARAMS):
            ax = axes[row_i, col]
            for cadence in cadences:
                for method, style in METHOD_STYLE.items():
                    y = []
                    for snr in snrs:
                        if stat == "precision":
                            v = _agg(rows, band_set, method, cadence, snr, f"{p}_sd")
                            y.append(np.median(v) if v else np.nan)
                        else:
                            v = np.array(_agg(rows, band_set, method, cadence, snr, f"{p}_mean"))
                            y.append(np.sqrt(np.mean((v - rows[0][f"{p}_truth"]) ** 2)) if len(v) else np.nan)
                    ax.plot(snrs, y, color=colours[cadence], lw=2, ms=6, **{k: v for k, v in style.items()
                                                                          if k != "label"})
            if stat == "precision":
                for (b, snr), vals in PAPER_2016.items():
                    if b == band_set:
                        ax.plot(snr, vals[row_i], marker="*", ms=13, color="black", ls="none", zorder=5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.6, which="both")
            if row_i == 0:
                ax.set_title(f"{band_set}")
            else:
                ax.set_xlabel("signal-to-noise ratio per point")
            if col == 0:
                ax.set_ylabel(f"{ylabel}\n{PARAMS[p]}")
    handles = [plt.Line2D([], [], color=colours[c], lw=2, label=f"{c:g}-day cadence") for c in cadences]
    handles += [plt.Line2D([], [], color="0.3", lw=2, ms=6, **s) for s in METHOD_STYLE.values()]
    if stat == "precision":
        handles.append(plt.Line2D([], [], marker="*", ms=13, color="black", ls="none",
                                  label="Starkey et al. (2016), 1-day cadence"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(figures / fname, dpi=150)
    plt.close(fig)


def figure_calibration(rows, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    bins = np.linspace(-5, 5, 41)
    x = np.linspace(-5, 5, 200)
    for ax, p in zip(axes, PARAMS):
        for method, style in METHOD_STYLE.items():
            z = np.clip([r[f"{p}_z"] for r in rows if r["method"] == method], -4.99, 4.99)
            ax.hist(z, bins=bins, density=True, histtype="step", lw=2, ls=style["ls"],
                    color="#2a78d6" if method == "nuts" else "#eb6834", label=style["label"])
        ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color="0.4", lw=1, label="unit Gaussian")
        ax.set_xlabel(f"z = (posterior mean - truth) / sd, {PARAMS[p]}")
        ax.grid(alpha=0.6)
    axes[0].set_ylabel("density")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "calibration.png", dpi=150)
    plt.close(fig)


def paired(rows):
    """{config: {method: row}} for configurations fitted by both methods."""
    out = {}
    for r in rows:
        out.setdefault((r["band_set"], r["snr"], r["cadence"], r["seed"]), {})[r["method"]] = r
    return {k: v for k, v in out.items() if "nuts" in v and "optimise" in v}


def figure_optimise_vs_nuts(rows, figures: Path) -> None:
    pairs = paired(rows)
    snrs = sorted({k[1] for k in pairs})
    cmap = plt.get_cmap("Blues")
    colour = {s: cmap(0.35 + 0.65 * k / max(len(snrs) - 1, 1)) for k, s in enumerate(snrs)}
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
    for col, p in enumerate(PARAMS):
        off = _scale(p)
        for row_i, (k, what) in enumerate((("mean", "posterior mean"), ("sd", "posterior sd"))):
            ax = axes[row_i, col]
            for snr in snrs:
                v = np.array([(m["nuts"][f"{p}_{k}"], m["optimise"][f"{p}_{k}"]) for c, m in pairs.items()
                              if c[1] == snr]) + (off if k == "mean" else 0.0)
                ax.scatter(v[:, 0], v[:, 1], s=14, color=colour[snr], edgecolor="white", linewidth=0.4,
                           label=f"SNR {snr:g}", zorder=3)
            lo, hi = ax.get_xlim()[0], ax.get_xlim()[1]
            lo, hi = min(lo, ax.get_ylim()[0]), max(hi, ax.get_ylim()[1])
            ax.plot([lo, hi], [lo, hi], color="0.4", lw=1, zorder=1)
            if k == "sd":
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_xlabel(f"NUTS {what}")
            ax.set_ylabel(f"optimise() {what}")
            ax.set_title(PARAMS[p])
            ax.grid(alpha=0.6)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "optimise_vs_nuts.png", dpi=150)
    plt.close(fig)


GRID_VARIANTS = {"optimise_baseline": ("longest period = baseline (old default)", "#eb6834"),
                 "optimise_auto": ("2 (baseline + tau_max), new default", "#2a78d6"),
                 "optimise": ("400 days, 80 frequencies", "#1baf7a")}


def figure_frequency_grid(rows, figures: Path) -> None:
    """rms error over seeds and cadences against SNR, per filter set, for
    optimise() on each driver frequency grid."""
    variants = [m for m in GRID_VARIANTS if any(r["method"] == m for r in rows)]
    band_sets = [b for b in grid.BAND_SETS if any(r["band_set"] == b for r in rows)]
    snrs = sorted({r["snr"] for r in rows})
    fig, axes = plt.subplots(2, len(band_sets), figsize=(4.2 * len(band_sets), 7), sharey="row", squeeze=False)
    for col, band_set in enumerate(band_sets):
        for row_i, p in enumerate(PARAMS):
            ax = axes[row_i, col]
            for m in variants:
                y = []
                for snr in snrs:
                    v = np.array([r[f"{p}_mean"] - r[f"{p}_truth"] for r in rows
                                  if (r["method"], r["band_set"], r["snr"]) == (m, band_set, snr)])
                    y.append(np.sqrt(np.mean(v ** 2)) if len(v) else np.nan)
                ax.plot(snrs, y, color=GRID_VARIANTS[m][1], lw=2, marker="o", ms=6)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.6, which="both")
            if row_i == 0:
                ax.set_title(band_set)
            else:
                ax.set_xlabel("signal-to-noise ratio per point")
            if col == 0:
                ax.set_ylabel(f"rms error of posterior mean,\n{PARAMS[p]}")
    handles = [plt.Line2D([], [], color=GRID_VARIANTS[m][1], lw=2, marker="o", label=GRID_VARIANTS[m][0])
               for m in variants]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(figures / "frequency_grid.png", dpi=150)
    plt.close(fig)


def figure_timing(rows, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, style in METHOD_STYLE.items():
        v = np.array([(r["n_obs"], r["seconds"]) for r in rows if r["method"] == method])
        ax.scatter(v[:, 0], v[:, 1], s=14, marker=style["marker"], label=style["label"],
                   color="#2a78d6" if method == "nuts" else "#eb6834", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("data points (all filters)")
    ax.set_ylabel("wall time per fit (s)")
    ax.grid(alpha=0.6, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "timing.png", dpi=150)
    plt.close(fig)


def aggregates(rows) -> dict:
    methods = sorted({r["method"] for r in rows})
    out = {"n_fits": {m: sum(r["method"] == m for r in rows) for m in methods}}
    for method in methods:
        mr = [r for r in rows if r["method"] == method]
        for p in PARAMS:
            z = np.array([r[f"{p}_z"] for r in mr])
            in68 = np.mean([(r[f"{p}_q16"] <= r[f"{p}_truth"] <= r[f"{p}_q84"]) for r in mr])
            in95 = np.mean([(r[f"{p}_q025"] <= r[f"{p}_truth"] <= r[f"{p}_q975"]) for r in mr])
            out[f"{method}_{p}"] = dict(z_mean=float(z.mean()), z_sd=float(z.std()), coverage68=float(in68),
                                        coverage95=float(in95))
        out[f"{method}_median_seconds"] = float(np.median([r["seconds"] for r in mr]))
    pairs = paired(rows)
    for p in PARAMS:
        d = np.array([(m["optimise"][f"{p}_mean"] - m["nuts"][f"{p}_mean"]) / m["nuts"][f"{p}_sd"]
                      for m in pairs.values()])
        ratio = np.array([m["optimise"][f"{p}_sd"] / m["nuts"][f"{p}_sd"] for m in pairs.values()])
        out[f"paired_{p}"] = dict(mean_shift_in_nuts_sd_median=float(np.median(d)),
                                  mean_shift_in_nuts_sd_p90_abs=float(np.percentile(np.abs(d), 90)),
                                  sd_ratio_median=float(np.median(ratio)),
                                  sd_ratio_p10=float(np.percentile(ratio, 10)),
                                  sd_ratio_p90=float(np.percentile(ratio, 90)))
    nuts = [r for r in rows if r["method"] == "nuts"]
    out["nuts_fits_with_divergences"] = int(sum(r["divergences"] > 0 for r in nuts))
    out["nuts_min_ess_median"] = float(np.median([r["min_ess"] for r in nuts])) if nuts else None
    # Per (filter set, SNR, cadence): median sd and rms error over seeds, per method.
    table = []
    keys = sorted({(r["band_set"], r["snr"], r["cadence"]) for r in rows},
                  key=lambda k: (grid.BAND_SETS.index(k[0]), k[1], k[2]))
    for band_set, snr, cadence in keys:
        row = dict(band_set=band_set, snr=snr, cadence=cadence)
        for method in METHOD_STYLE:
            for p in PARAMS:
                sd = _agg(rows, band_set, method, cadence, snr, f"{p}_sd")
                mean = np.array(_agg(rows, band_set, method, cadence, snr, f"{p}_mean"))
                if len(sd):
                    row[f"{method}_{p}_sd"] = float(np.median(sd))
                    row[f"{method}_{p}_rms_error"] = float(np.sqrt(np.mean((mean - rows[0][f"{p}_truth"]) ** 2)))
                    row[f"{method}_{p}_n"] = len(sd)
        table.append(row)
    out["table"] = table
    # Run times: median wall time per fit for each method, filter set and
    # cadence (pooled over SNR and seeds), and optimise()'s phases.
    timing = []
    for method in sorted({r["method"] for r in rows}):
        for band_set in grid.BAND_SETS:
            for cadence in sorted({r["cadence"] for r in rows}):
                sel = [r for r in rows if (r["method"], r["band_set"], r["cadence"]) == (method, band_set, cadence)]
                if sel:
                    timing.append(dict(method=method, band_set=band_set, cadence=cadence, n_obs=sel[0]["n_obs"],
                                       n_fits=len(sel), median_seconds=float(np.median([r["seconds"] for r in sel])),
                                       max_seconds=float(np.max([r["seconds"] for r in sel]))))
    out["timing"] = timing
    opt = [r for r in rows if r["method"].startswith("optimise")]
    out["optimise_phases_median_seconds"] = {
        k: float(np.median([r[k] for r in opt])) for k in ("lbfgs_seconds", "hessian_and_newton_seconds",
                                                           "draws_seconds") if opt}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    rows = load(args.cache)
    if not rows:
        raise SystemExit(f"no results in {args.cache}")
    with open(args.figures / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.figures / "summary.json").write_text(json.dumps(aggregates(rows), indent=1))
    figure_example(args.figures)
    figure_by_snr(rows, args.figures, "precision", "precision.png", "posterior sd,")
    figure_by_snr(rows, args.figures, "accuracy", "accuracy.png", "rms error over seeds,")
    figure_calibration(rows, args.figures)
    if paired(rows):
        figure_optimise_vs_nuts(rows, args.figures)
    figure_frequency_grid(rows, args.figures)
    figure_timing(rows, args.figures)
    print(f"{len(rows)} fits; figures in {args.figures}")


if __name__ == "__main__":
    main()
