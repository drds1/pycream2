"""
reporting.py
============

Shared visual-report generation for a fitted ``EchoFit``: the same plots
and ``report.html`` used by ``scripts/smoke_test.py`` and by
``EchoFit.fit(title=...)``'s automatic per-run report. Kept separate from
``echofit.py`` so both call sites (and any future ones) share one
implementation instead of duplicating the HTML.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import numpy as np

# Sample sites that aren't single scalars per posterior draw (per-frequency
# driver coefficients, per-observation predictions) -- excluded from the
# scalar summary table.
_NON_SCALAR_PREFIXES = ("y_pred_", "S_raw", "C_raw")
_NON_SCALAR_NAMES = ("S", "C")


def _scalar_param_names(samples: dict):
    names = []
    for k, v in samples.items():
        if k in _NON_SCALAR_NAMES or k.startswith(_NON_SCALAR_PREFIXES):
            continue
        if np.asarray(v).ndim == 1:
            names.append(k)
    return sorted(names)


def _img_data_uri(path: Path) -> str:
    """Base64 ``data:`` URI for a PNG already saved at ``path``, so
    ``report.html`` can embed it inline instead of referencing it by
    relative filename -- the report is then a single self-contained file
    (openable and emailable on its own), not a folder of PNGs plus one HTML
    file that only renders correctly alongside them."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def generate_report(
    ef,
    out_dir,
    fit_seconds: Optional[float] = None,
    truth: Optional[dict] = None,
    title: Optional[str] = None,
) -> Path:
    """Render the standard plot set for a fitted ``ef`` and write
    ``report.html`` into ``out_dir``, alongside the individual PNGs (kept on
    disk for the smoke-test/checkpoint machinery that inspects them
    directly, e.g. ``tests/test_run_manager.py``). ``report.html`` itself
    embeds every image inline as a base64 ``data:`` URI rather than
    referencing those PNGs by relative path, so it's a single
    self-contained file: it can be emailed or copied elsewhere on its own
    and will still render correctly without the rest of ``out_dir``.

    Parameters
    ----------
    ef : EchoFit
        Already-fitted (``ef.samples`` populated).
    out_dir : path-like
        Directory to write into (created if missing).
    fit_seconds : float, optional
        Wall time of the fit, shown in the report header if given.
    truth : dict, optional
        Ground-truth parameter values (as in ``synthetic.generate_synthetic_dataset``'s
        ``data["truth"]``) to compare the posterior against -- only
        meaningful for synthetic data. Omit for real-data runs.
    title : str, optional
        Run title, shown in the report header.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_raw, _ = ef.plot_raw_lightcurves()
    fig_fits, _ = ef.plot_lightcurve_fits()
    fig_power, _ = ef.plot_power_spectrum()
    fig_diag, _ = ef.plot_mcmc_diagnostics()

    paths = {
        "raw": out_dir / "raw_lightcurves.png",
        "fits": out_dir / "lightcurve_fits.png",
        "power": out_dir / "power_spectrum.png",
        "diagnostics": out_dir / "mcmc_diagnostics.png",
    }
    import matplotlib.pyplot as plt

    fig_raw.savefig(paths["raw"], dpi=150, bbox_inches="tight")
    fig_fits.savefig(paths["fits"], dpi=150, bbox_inches="tight")
    fig_power.savefig(paths["power"], dpi=150, bbox_inches="tight")
    fig_diag.savefig(paths["diagnostics"], dpi=150, bbox_inches="tight")
    figs_to_close = [fig_raw, fig_fits, fig_power, fig_diag]

    # log_mdot/inclination only exist if at least one band used
    # lag_mode="physical" (CLAUDE.md decision #7) -- skip the corner plot
    # rather than error for a free-lag-only fit.
    if {"log_mdot", "inclination"} <= set(ef.samples):
        true_values = {"log_mdot": _truth_value_for("log_mdot", truth), "inclination": _truth_value_for("inclination", truth)}
        true_values = {k: v for k, v in true_values.items() if v is not None} or None
        fig_corner, _ = ef.plot_corner(true_values=true_values)
        paths["corner"] = out_dir / "corner.png"
        fig_corner.savefig(paths["corner"], dpi=150, bbox_inches="tight")
        figs_to_close.append(fig_corner)

    band_param_names = [f"{p}_{name}" for name in ef.bands for p in ("S", "C")]
    true_values = {name: _truth_value_for(name, truth) for name in band_param_names}
    true_values = {k: v for k, v in true_values.items() if v is not None} or None
    fig_bands, _ = ef.plot_corner_bands(true_values=true_values)
    paths["corner_bands"] = out_dir / "corner_bands.png"
    fig_bands.savefig(paths["corner_bands"], dpi=150, bbox_inches="tight")
    figs_to_close.append(fig_bands)

    if any(d["lag_mode"] == "free" for d in ef.bands.values()):
        free_lag_names = [f"tau_{name}" for name, d in ef.bands.items() if d["lag_mode"] == "free"]
        true_values = {name: _truth_value_for(name, truth) for name in free_lag_names}
        true_values = {k: v for k, v in true_values.items() if v is not None} or None
        fig_free_lag, _ = ef.plot_corner_free_lag(true_values=true_values)
        paths["corner_free_lag"] = out_dir / "corner_free_lag.png"
        fig_free_lag.savefig(paths["corner_free_lag"], dpi=150, bbox_inches="tight")
        figs_to_close.append(fig_free_lag)

    # Only after EchoFit.optimise(); a later .fit() on the same object leaves
    # optimise_restarts set, so also require that no sampler has run since.
    if getattr(ef, "optimise_restarts", None) and not ef.extra_fields:
        fig_restarts, _ = ef.plot_optimise_restarts()
        paths["restarts"] = out_dir / "optimise_restarts.png"
        fig_restarts.savefig(paths["restarts"], dpi=150, bbox_inches="tight")
        figs_to_close.append(fig_restarts)

    # Only after EchoFit.nested_laplace() with log_mdot and inclination gridded,
    # and no sampler run since.
    nl = getattr(ef, "nested_laplace_result", None)
    if nl and list(nl["names"])[:2] == ["log_mdot", "cos_inclination"] and not ef.extra_fields:
        fig_land, _ = ef.plot_landscape()
        paths["landscape"] = out_dir / "landscape.png"
        fig_land.savefig(paths["landscape"], dpi=150, bbox_inches="tight")
        figs_to_close.append(fig_land)

    fig_fourier = ef.plot_fourier_correlation()[0]
    paths["fourier_correlation"] = out_dir / "fourier_correlation.png"
    fig_fourier.savefig(paths["fourier_correlation"], dpi=150, bbox_inches="tight")
    figs_to_close.append(fig_fourier)

    # potential_energy is only absent for a checkpoint resumed from before
    # this feature existed (CLAUDE.md) -- skip rather than error.
    if "potential_energy" in ef.extra_fields:
        checkpoint_every = (ef._fit_config or {}).get("checkpoint_every")
        fig_bof, _ = ef.plot_bof(checkpoint_every=checkpoint_every)
        paths["bof"] = out_dir / "bof.png"
        fig_bof.savefig(paths["bof"], dpi=150, bbox_inches="tight")
        figs_to_close.append(fig_bof)

    # Only when the disc SED/distance analysis has run (EchoFit(sed_analysis=True)
    # or ef.disc_sed_analysis()).
    if getattr(ef, "disc_sed", None) is not None:
        fig_sed, _ = ef.plot_disc_sed()
        paths["disc_sed"] = out_dir / "disc_sed.png"
        fig_sed.savefig(paths["disc_sed"], dpi=150, bbox_inches="tight")
        figs_to_close.append(fig_sed)

    for fig in figs_to_close:
        plt.close(fig)

    diverging = np.asarray(ef.extra_fields.get("diverging", []))
    n_div = int(diverging.sum()) if diverging.size else 0
    n_total = len(diverging)

    data_uris = {key: _img_data_uri(p) for key, p in paths.items()}

    report_path = out_dir / "report.html"
    report_path.write_text(
        _report_html(ef, paths, data_uris, fit_seconds, n_div, n_total, truth, title)
    )
    return report_path


def _disc_sed_section_html(ef, paths, data_uris) -> str:
    s = ef.disc_sed["summary"]

    def pm(q, fmt="{:.3g}"):
        lo, med, hi = q
        return f"{fmt.format(med)} (+{fmt.format(hi - med)}/&minus;{fmt.format(med - lo)})"

    rows = [("luminosity distance D_L (Mpc)", pm(s["dl_mpc"], "{:.0f}")),
            ("H_0 (km/s/Mpc)", pm(s["h0"], "{:.1f}")),
            ("estimator", s["distance_method"]),
            ("H_0 from the host band alone", pm(s["h0_host_band"], "{:.1f}")),
            ("H_0 from the flux-flux decomposition", pm(s["h0_flux_flux"], "{:.1f}")),
            ("temperature slope alpha (fit)", pm(s["alpha"], "{:.2f}")),
            ("alpha implied by the variable SED", pm(s["alpha_var"], "{:.2f}")),
            ("T_1 at 1 light-day (K)", pm(s["t1_kelvin"], "{:.3g}")),
            ("inclination (deg)", pm(s["inclination"], "{:.1f}"))]
    if s["fit_intrinsic_ebv"]:
        rows.append(("intrinsic E(B-V) (mag)", pm(s["ebv_intrinsic"], "{:.3f}")))
    table = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows)
    band_rows = "".join(
        f"<tr><td>{n}</td><td>{b['lam_rest']:.0f}</td><td>{b['variable_mjy'][1]:.3g}</td>"
        f"<td>{b['predicted_variable_mjy'][1]:.3g}</td><td>{b['disc_mean_mjy'][1]:.3g}</td>"
        f"<td>{b['constant_mjy'][1]:.3g}</td><td>{b['model_disc_mjy'][1]:.3g}</td><td>{b['dl_mpc'][1]:.0f}</td></tr>"
        for n, b in s["bands"].items())
    warning = ""
    if s["zero_point_below_faint_state"] < 0.95:
        warning = ("<p><b>Warning:</b> in some draws the disc zero point lies inside the observed range of the "
                   "driver, so the host band has a negative constant component; the host subtraction is "
                   "unreliable.</p>")
    return f"""
<h2>Disc SED, luminosity distance and H<sub>0</sub></h2>
<p>The delays fix the disc's temperature profile in light-days, so the model disc's flux depends only on its
distance (Cackett, Horne &amp; Winkler 2007): comparing it with the observed disc flux gives D<sub>L</sub>, and
with z = {s['redshift']:.4g}, H<sub>0</sub> (flat &Lambda;CDM, &Omega;<sub>m</sub> = {s['omega_m']:.2f}).
The {s['host_band']} band is assumed to have no host light: with the default estimator its whole mean flux
is disc, which gives D<sub>L</sub>, and every other band's host is its mean flux minus the model disc. The
flux-flux estimator instead puts the disc's zero point where the {s['host_band']} model flux vanishes and fits
all bands; it assumes the variable SED has the mean disc's shape, which the bluer lamppost response does not,
and so places D<sub>L</sub> too far (see docs/disc_sed.md). Fluxes are in mJy, corrected for Galactic
E(B&minus;V) =
{s['ebv_galactic']:.3f}{" and a fitted intrinsic E(B&minus;V)" if s['fit_intrinsic_ebv'] else ""}.
Median (16th/84th percentile offsets) over {s['n_draws']} posterior draws. An implausible H<sub>0</sub> is a
warning that the disc fit is absorbing something else (e.g. slow variability: try background_order).</p>
<table>{table}</table>
<table><tr><th>band</th><th>rest &lambda; (&Aring;)</th><th>variable (mJy)</th><th>predicted variable</th>
<th>disc, mean state</th><th>constant (host)</th><th>model disc at D<sub>L</sub></th>
<th>D<sub>L</sub> from band (Mpc)</th></tr>{band_rows}</table>
{warning}<img src="{data_uris['disc_sed']}" alt="{paths['disc_sed'].name}">
"""


def _truth_value_for(name: str, truth: Optional[dict]):
    """Look up ``name``'s ground-truth value from a
    ``synthetic.generate_*_dataset``-style ``truth`` dict, or None if there
    isn't one (real data, or a name truth has nothing to say about).
    Shared by the summary table and the corner plots' true-value crosshairs
    so the two don't drift apart."""
    if truth is None:
        return None
    if name in truth:
        return truth[name]
    if name in ("S_driver", "C_driver"):
        return truth.get("driver", {}).get(name)
    if name.startswith("tau_"):
        return truth.get("bands", {}).get(name[len("tau_"):], {}).get("tau")
    if name.startswith(("S_", "C_")):
        prefix, band = name.split("_", 1)
        return truth.get("bands", {}).get(band, {}).get(f"{prefix}_band")
    return None


def _summary_table_html(ef, truth: Optional[dict]) -> str:
    rows = []
    for name in _scalar_param_names(ef.samples):
        s = np.asarray(ef.samples[name])
        truth_val = _truth_value_for(name, truth)
        truth_cell = f"{truth_val:.4g}" if truth_val is not None else "&mdash;"
        rows.append(
            f"<tr><td>{name}</td><td>{s.mean():.4g}</td><td>{s.std():.4g}</td>"
            f"<td>{truth_cell}</td></tr>"
        )
    return (
        "<table><tr><th>parameter</th><th>posterior mean</th>"
        "<th>posterior std</th><th>truth</th></tr>" + "".join(rows) + "</table>"
    )


def _report_html(ef, paths, data_uris, fit_seconds, n_div, n_total, truth, title) -> str:
    header_bits = []
    if title:
        header_bits.append(f"run: <b>{title}</b>")
    if fit_seconds is not None:
        header_bits.append(f"fit wall time: <b>{fit_seconds:.1f}s</b>")
    if n_total == 0 and getattr(ef, "optimum", None) is not None:
        # EchoFit.optimise(): no sampler ran, so a "0/0 divergences" line would
        # suggest a clean MCMC run that never happened.
        header_bits.append("method: <b>direct solve (optimise(): L-BFGS + Laplace approximation, no MCMC)</b>")
    else:
        header_bits.append(f"divergent transitions: <b>{n_div}/{n_total}</b>")
    header_line = " &nbsp;|&nbsp; ".join(header_bits)

    corner_sections = []
    if "corner" in paths:
        corner_sections.append(f"""
<h2>Posterior corner plot: disk parameters</h2>
<p>log_mdot / inclination, coloured per chain. Chains that land in visibly
different places here are the same thing a Gelman-Rubin R-hat check would
flag, made visible -- see CLAUDE.md's note on why a single chain isn't
sufficient evidence of convergence.</p>
<img src="{data_uris['corner']}" alt="{paths['corner'].name}">
""")
    if "corner_bands" in paths:
        corner_sections.append(f"""
<h2>Posterior corner plot: band offset/stretch parameters</h2>
<p>S_band / C_band for every band -- the linear scale and offset absorbing
each band's own flux calibration, not physically meaningful on their own
but worth checking for the same reason as the disk corner plot above:
chains disagreeing here means the fit hasn't converged.</p>
<img src="{data_uris['corner_bands']}" alt="{paths['corner_bands'].name}">
""")
    if "corner_free_lag" in paths:
        corner_sections.append(f"""
<h2>Posterior corner plot: free-lag (top-hat) centroids</h2>
<p>tau_band for every lag_mode="free" band -- the independently inferred
lag each such band's top-hat response is centred on (see
CLAUDE.md decision #7 on why these need a driver light curve to be
identifiable at all).</p>
<img src="{data_uris['corner_free_lag']}" alt="{paths['corner_free_lag'].name}">
""")
    if "fourier_correlation" in paths:
        corner_sections.append(f"""
<h2>Driver Fourier coefficient correlation</h2>
<p>Posterior correlation matrix of the driver's S/C Fourier coefficients
(pooled across chains). Mostly-diagonal (near zero off-diagonal) is what
the non-centred DRW prior parameterisation assumes; strong off-diagonal
structure would be worth a closer look.</p>
<img src="{data_uris['fourier_correlation']}" alt="{paths['fourier_correlation'].name}">
""")
    corner_section = "".join(corner_sections)

    restarts_section = ""
    if "restarts" in paths:
        rows = []
        for r in ef.optimise_restarts:
            v = r["values"]
            cells = [
                str(r["index"]), f"{r['start_offset_in_sd']:.3g}", str(r["lbfgs_evaluations"]),
                str(r["newton_iterations"]), f"{2.0 * r['delta_potential']:.3g}",
                f"{r['max_offset_in_sd']:.3g}", "yes" if r["agrees"] else "<b>no</b>",
            ] + [f"{v[p]:.4g}" if p in v else "&mdash;" for p in ("log_mdot", "inclination")]
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        n_agree = sum(r["agrees"] for r in ef.optimise_restarts)
        restarts_section = f"""
<h2>Direct-solve reproducibility (multi-start)</h2>
<p>The optimiser's counterpart of running several MCMC chains from different
starting points: optimise() ran {len(ef.optimise_restarts)} times, from the
data-anchored starting point and from random perturbations of it, and polished
each to its own optimum. <b>{n_agree} of {len(ef.optimise_restarts)} agree</b>
with the best, meaning every parameter lies within 0.5 Laplace posterior
standard deviations of it. "Start offset" is how far each start was from the
first, on the same scale; "&Delta;BOF" is how far above the best each optimum
sits in Badness of Fit (2 &times; potential).</p>
<table><tr><th>restart</th><th>start offset (sd)</th><th>L-BFGS evaluations</th>
<th>Newton steps</th><th>&Delta;BOF</th><th>max offset from best (sd)</th>
<th>agrees</th><th>log_mdot</th><th>inclination</th></tr>{"".join(rows)}</table>
<img src="{data_uris['restarts']}" alt="{paths['restarts'].name}">
"""

    landscape_section = ""
    if "landscape" in paths:
        landscape_section = f"""
<h2>Badness-of-Fit landscape (nested Laplace)</h2>
<p>The marginal posterior of log_mdot and inclination from nested_laplace(),
shown as &Delta;BOF = &minus;2 &Delta;log posterior: at every grid point the
other parameters are optimised and integrated. Left, the scout grid over the
prior range (every basin the search saw); right, the final grid, with
contours enclosing 68, 95 and 99.7 per cent. Importance-sampling check:
Pareto k&#770; = {ef.nested_laplace_result['k_hat']:.2f} (reliable below 0.7), effective
sample size {ef.nested_laplace_result['ess']:.0f}.</p>
<img src="{data_uris['landscape']}" alt="{paths['landscape'].name}">
"""

    bof_section = ""
    if "bof" in paths:
        bof_section = f"""
<h2>Badness of Fit vs. sample</h2>
<p>2 x NUTS potential energy, one line per chain -- exactly the Badness-of-Fit
that Starkey, Horne &amp; Villforth (2016, MNRAS 456, 1960) eq. 12 defines, up
to an additive constant. Should decrease during warm-up then flatten out
once the chain has converged; vertical grey lines (if shown) mark checkpoint
boundaries.</p>
<img src="{data_uris['bof']}" alt="{paths['bof'].name}">
"""

    disc_sed_section = _disc_sed_section_html(ef, paths, data_uris) if "disc_sed" in paths else ""

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>pycream2 report{f' -- {title}' if title else ''}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
img {{ max-width: 100%; display: block; margin: 1rem 0; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
td, th {{ border: 1px solid #ddd; padding: 4px 10px; text-align: left; }}
code {{ background: #f2f2f2; padding: 1px 4px; }}
</style></head>
<body>
<h1>pycream2 report</h1>
<p>{header_line}</p>

<h2>Posterior summary</h2>
{_summary_table_html(ef, truth)}
{restarts_section}{landscape_section}
<h2>Raw light curves</h2>
<img src="{data_uris['raw']}" alt="{paths['raw'].name}">

<h2>Posterior-predictive fit + response function</h2>
<p>Shaded bands are 68%/95% credible intervals; black points are the data. The
top panel is the inferred driving light curve (time-aligned with the bands
below it), extended a bit before/after the data -- the credible band should
widen roughly like t^(1/2) outside the data before saturating, since the
driver is a DRW. The right-hand panels are the inferred response function
&psi;(&tau;) per band.</p>
<img src="{data_uris['fits']}" alt="{paths['fits'].name}">

<h2>Driver power spectrum</h2>
<p>Posterior P(&omega;) = (S<sup>2</sup>+C<sup>2</sup>)/(2&Delta;&omega;) per
frequency (black squares) vs. the fitted DRW Lorentzian from posterior
sigma_drw/tau_drw draws (blue dashed) and a plain &omega;<sup>-2</sup>
random-walk reference (red dotted). These should roughly track each other --
if the posterior power spectrum diverges from the Lorentzian shape a lot,
that's worth a closer look.</p>
<img src="{data_uris['power']}" alt="{paths['power'].name}">

<h2>MCMC trace diagnostics</h2>
<p>Traces should look like noisy horizontal bands (well-mixed), not
slow drifts or a chain stuck at one value.</p>
<img src="{data_uris['diagnostics']}" alt="{paths['diagnostics'].name}">
{corner_section}{bof_section}{disc_sed_section}</body></html>
"""
