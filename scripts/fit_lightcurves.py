"""
fit_lightcurves.py
===================

Command-line entry point for fitting your own (real or otherwise
already-saved) light curves, without writing any Python. Wraps the same
``EchoFit`` API used throughout the README/notebook -- see there for the
underlying model.

Each light curve file is a plain text file, whitespace- or comma-separated,
three columns ``t y yerr`` (one observation per line; ``#``-prefixed lines
are ignored as comments/headers). Time ``t`` should be in days, on a
consistent zero-point across every band (and the driver, if given).

Usage
-----
Managed run (checkpointed, resumable, writes to outputs/<title>/run_.../ --
see README's "Fitting your own light curves"; forces --num-chains 1):

    python scripts/fit_lightcurves.py \\
        --title ngc_5548 --m-bh 1e8 \\
        --band g 4770 data/g_band.txt \\
        --band i 7625 data/i_band.txt \\
        --num-warmup 1000 --num-samples 2000 \\
        --checkpoint-every 200 --report-every 1000

Resume an interrupted managed run (reuses its original settings; only
--title and --output-dir matter here, everything else is ignored):

    python scripts/fit_lightcurves.py --title ngc_5548 --resume

Quick in-memory run with multiple chains for R-hat/ESS diagnostics (no
--title -- nothing is checkpointed, but a one-off report.html is still
written to --output-dir):

    python scripts/fit_lightcurves.py \\
        --m-bh 1e8 --band g 4770 data/g_band.txt --band i 7625 data/i_band.txt \\
        --num-warmup 1000 --num-samples 1000 --num-chains 4 --chain-method vectorized \\
        --output-dir diagnostic_run

Free-lag band (e.g. an emission line with no assumed physical lag law)
anchored by a driver light curve (see CLAUDE.md decision #7 on why the
driver is required for this to be identifiable):

    python scripts/fit_lightcurves.py \\
        --title ngc_5548_lines --m-bh 1e8 \\
        --band continuum 5100 data/continuum.txt \\
        --free-lag-band halpha 6563 data/halpha.txt \\
        --driver data/continuum.txt \\
        --num-warmup 1000 --num-samples 2000

See scripts/run_example_fit.sh for a fully worked, runnable example
(including generating example data first).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycream2.echofit import EchoFit
from pycream2 import reporting


def _check_distinct_files(bands):
    """Refuse two bands that resolve to the same file, e.g. "r_band.txt" and
    "R_band.txt" on a case-insensitive filesystem: the fit would otherwise
    silently use one light curve twice, at two different wavelengths."""
    seen = {}
    for name, _, path in bands:
        key = Path(path).resolve()
        for other_name, other_key in seen.items():
            if key == other_key or (key.exists() and other_key.exists() and key.samefile(other_key)):
                raise SystemExit(f"Bands '{other_name}' and '{name}' both read the same file ({path}).")
        seen[name] = key


def _load_lightcurve(path: str):
    t, y, yerr = np.loadtxt(path, comments="#", unpack=True)
    return np.atleast_1d(t), np.atleast_1d(y), np.atleast_1d(yerr)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--band", nargs=3, action="append", default=[], metavar=("NAME", "WAVELENGTH", "PATH"),
        help="A lag_mode=\"physical\" band (repeatable): name, wavelength in Angstrom, "
             "path to its t/y/yerr text file.",
    )
    parser.add_argument(
        "--free-lag-band", nargs=3, action="append", default=[], metavar=("NAME", "WAVELENGTH", "PATH"),
        help="A lag_mode=\"free\" band (repeatable) -- an independently inferred lag, "
             "not tied to the others through M_BH/log_mdot. Needs --driver to be "
             "identifiable (see CLAUDE.md decision #7).",
    )
    parser.add_argument(
        "--driver", default=None, metavar="PATH",
        help="Optional direct (zero-lag) observation of the driving light curve -- "
             "required to anchor any --free-lag-band.",
    )
    parser.add_argument("--m-bh", type=float, default=None, help="Fixed black hole mass, solar masses.")
    parser.add_argument(
        "--fit-error-model", action="store_true",
        help="Fit an error rescale factor and extra jitter for every band (and the driver), "
             "instead of trusting the quoted errors exactly. Recommended for real data: on "
             "NGC 5548 all 13 bands only fit cleanly with it on.",
    )
    parser.add_argument(
        "--diffuse-continuum", nargs="*", default=None, metavar="BAND",
        help="Give these bands (every --band/--free-lag-band if none are named) a second, "
             "log-normal reprocessor, e.g. diffuse continuum from the broad-line region: three "
             "extra parameters each (dce_fraction_, dce_delay_, dce_width_). Off by default; "
             "see docs/extra_components.md.",
    )
    parser.add_argument(
        "--background-order", type=int, default=0,
        help="Add a slowly varying background of this many Legendre terms to every light curve "
             "(and the driver), for variability unrelated to reverberation. 0 (default) is off; "
             "see docs/extra_components.md.",
    )

    parser.add_argument("--title", default=None, help="Run name -- enables checkpointed/resumable output management.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest run under --title instead of starting a new fit.")
    parser.add_argument("--output-dir", default=None, help="Output root directory (default: $PYCREAM2_OUTPUT_DIR or ./outputs).")

    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=1, help="Ignored (forced to 1) when --title is given -- see CLAUDE.md decision #6.")
    parser.add_argument("--chain-method", default="parallel", choices=("parallel", "vectorized", "sequential"))
    parser.add_argument("--max-tree-depth", type=int, default=None)
    parser.add_argument(
        "--diagonal-mass", action="store_true",
        help=(
            "Use a diagonal NUTS mass matrix instead of the default dense one -- see "
            "CLAUDE.md decisions #17/#21 (dense measured ~17x more effective samples "
            "per second on this model). A dense matrix needs a reasonable --num-warmup "
            "(the default 1000 is fine); check the report's divergence count."
        ),
    )
    parser.add_argument("--dense-mass", action="store_true", help="No-op, kept for old command files: dense is now the default.")
    parser.add_argument(
        "--drw-prior", action="store_true",
        help=(
            "Use the original damped random walk (DRW) driver prior (tau_drw "
            "inferred alongside sigma_drw) instead of the default pure "
            "random-walk (RW) power-law prior (sigma_drw only, no tau_drw "
            "site) -- see CLAUDE.md decision #19."
        ),
    )
    parser.add_argument(
        "--marginalise-linear", action="store_true",
        help=(
            "Integrate the driver's Fourier coefficients and every band's offset "
            "out analytically, so NUTS only samples the ~10 nonlinear parameters "
            "-- see CLAUDE.md decision #21. Same posterior; the linear parameters "
            "are drawn exactly afterwards, so every plot still works."
        ),
    )
    parser.add_argument(
        "--optimise", action="store_true",
        help=(
            "Direct solve instead of MCMC: EchoFit.optimise() (L-BFGS on the "
            "linear-marginalised posterior + a Laplace Gaussian around the peak), "
            "about half a 500-sample NUTS run's time; see CLAUDE.md decision #21. Fine for a "
            "single well-constrained peak; use MCMC for free-lag or multimodal fits. "
            "Ignores the MCMC-only options and --title's checkpointing."
        ),
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Only used with --title.")
    parser.add_argument("--report-every", type=int, default=None, help="Only used with --title -- refresh report.html every this many new samples.")
    parser.add_argument("--no-progress-bar", action="store_true")

    parser.add_argument("--n-freq", type=int, default=None, help="Driver Fourier frequencies (default: build_grid()'s own default).")
    parser.add_argument("--n-tau", type=int, default=None, help="Lag grid points (default: build_grid()'s own default).")
    parser.add_argument("--tau-max", type=float, default=None, help="Maximum lag, days (default: half the observed time baseline).")

    return parser.parse_args()


def main():
    args = _parse_args()

    if args.resume:
        if not args.title:
            raise SystemExit("--resume requires --title.")
        ef = EchoFit.resume(args.title, output_dir=args.output_dir)
        ef.fit(progress_bar=not args.no_progress_bar)
        print(f"Resumed and finished '{args.title}' -> {ef.run_dir}")
        return

    if not args.band and not args.free_lag_band:
        raise SystemExit("Add at least one --band or --free-lag-band.")
    _check_distinct_files(args.band + args.free_lag_band)

    ef = EchoFit(
        M_BH=args.m_bh, title=args.title, output_dir=args.output_dir, drw_prior=args.drw_prior,
        marginalise_linear=args.marginalise_linear,
    )
    # --diffuse-continuum with no band names means every band.
    band_names = [b[0] for b in args.band] + [b[0] for b in args.free_lag_band]
    dce_bands = set() if args.diffuse_continuum is None else set(args.diffuse_continuum or band_names)
    unknown = dce_bands - set(band_names)
    if unknown:
        raise SystemExit(f"--diffuse-continuum names unknown band(s): {sorted(unknown)}")
    extras = dict(fit_error_model=args.fit_error_model, background_order=args.background_order)
    for name, wavelength, path in args.band:
        t, y, yerr = _load_lightcurve(path)
        ef.add_lightcurve(name, wavelength=float(wavelength), t=t, y=y, yerr=yerr,
                          diffuse_continuum=name in dce_bands, **extras)
    for name, wavelength, path in args.free_lag_band:
        t, y, yerr = _load_lightcurve(path)
        ef.add_lightcurve(name, wavelength=float(wavelength), t=t, y=y, yerr=yerr, lag_mode="free",
                          diffuse_continuum=name in dce_bands, **extras)
    if args.driver:
        t, y, yerr = _load_lightcurve(args.driver)
        ef.add_driver_lightcurve(t=t, y=y, yerr=yerr, **extras)

    build_grid_kwargs = {}
    if args.n_freq is not None:
        build_grid_kwargs["n_freq"] = args.n_freq
    if args.n_tau is not None:
        build_grid_kwargs["n_tau"] = args.n_tau
    if args.tau_max is not None:
        build_grid_kwargs["tau_max"] = args.tau_max
    ef.build_grid(**build_grid_kwargs)

    if args.optimise:
        ef.optimise(num_samples=args.num_samples, rng_seed=args.rng_seed)
        out_dir = Path(args.output_dir or "fit_output")
        report_path = reporting.generate_report(ef, out_dir, title=args.title)
        print(f"Done (direct solve) -> {report_path}")
        return

    ef.fit(
        rng_seed=args.rng_seed,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method=args.chain_method,
        max_tree_depth=args.max_tree_depth,
        dense_mass=not args.diagonal_mass,
        checkpoint_every=args.checkpoint_every,
        report_every=args.report_every,
        progress_bar=not args.no_progress_bar,
    )

    if args.title:
        print(f"Done -> {ef.run_dir}")
    else:
        # No --title: nothing was written to disk automatically (the
        # in-memory fit path, CLAUDE.md decision #6) -- write a one-off
        # report here instead so a plain multi-chain diagnostic run still
        # produces something to look at.
        out_dir = Path(args.output_dir or "fit_output")
        report_path = reporting.generate_report(ef, out_dir)
        print(f"Done -> {report_path}")


if __name__ == "__main__":
    main()
