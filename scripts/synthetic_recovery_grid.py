"""
Injection-recovery grid on synthetic light curves, in the style of Starkey,
Horne & Villforth (2016, MNRAS 456, 1960; the CREAM paper, section 4-5): how
well are the disc's accretion rate (``log_mdot``, equivalently M Mdot) and
inclination recovered as the photometric signal-to-noise ratio, the mean
sampling cadence and the set of filters change, and does the direct solve
(``EchoFit.optimise()``) give the same answers as NUTS (``EchoFit.fit()``)?

Each configuration is (filter set, SNR, mean cadence, seed). Following the
2016 paper:
- the truth is M_BH = 1e8 Msun, M Mdot = 1e8 Msun^2/yr (``log_mdot`` = 1.953,
  see ``TRUE_LOG_MDOT``), inclination 30 degrees, temperature slope 3/4;
- the driver is a random walk (sigma = 1 after one day) simulated on a
  0.05-day grid, independently of the fit's Fourier-series driver, and each
  echo is a brute-force numerical convolution with ``thin_disk_response``
  (the response the fit also uses, which is what we test recovery of);
- the campaign lasts 100 days; each filter is sampled at its own jittered
  times with the given mean cadence;
- mean fluxes follow F_nu ~ lambda^(-1/3) from AB = 17 in g, each filter's
  variability rms is the paper's Table 2 value, and the noise is Gaussian with
  sigma = F/SNR (the paper's eq. 21, without the lunar-phase modulation).
A seed fixes the driver, the sampling and the noise, so different
configurations with the same seed share the same driver (common random
numbers, as the paper reused one driver across its tests).

Each fit writes ``<out>/<config>/<method>.json`` (posterior summaries, timing,
diagnostics) and ``<method>_samples.npz`` (the log_mdot and inclination
draws); existing results are skipped, so an interrupted run resumes.

Usage (from the repository root):
    MPLBACKEND=Agg poetry run python scripts/synthetic_recovery_grid.py --out experiments/recovery_grid/cache
    # a subset, or one shard of the full grid (for parallel workers):
    ... --band-sets gi --snrs 100 --cadences 1 --seeds 0 --methods optimise
    ... --shard 2/8
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np

# The dimensionless log_mdot in pycream2 maps onto M Mdot through the disc's
# temperature at one light-day, T_1 = (3 G M Mdot / (8 pi sigma (1 ld)^3))^(1/4)
# (forward_model.disk_t1_kelvin). For M_BH = 1e8 that gives
# log10(M Mdot / Msun^2 yr^-1) = log_mdot + LOG_MMDOT_OFFSET.
_G, _MSUN, _YR, _SIGMA_SB, _LD = 6.674e-8, 1.989e33, 3.156e7, 5.6704e-5, 2.998e10 * 86400.0
_T1_MMDOT_1E8 = (3 * _G * 1e8 * _MSUN ** 2 / _YR / (8 * np.pi * _SIGMA_SB * _LD ** 3)) ** 0.25
_T1_LOG_MDOT_0 = 5795.545829  # disk_t1_kelvin(0.0, 1e8), Kelvin
LOG_MMDOT_OFFSET = 8.0 - 4.0 * np.log10(_T1_MMDOT_1E8 / _T1_LOG_MDOT_0)

M_BH = 1.0e8
TRUE_LOG_MDOT = 8.0 - LOG_MMDOT_OFFSET  # M Mdot = 1e8 Msun^2/yr, as the 2016 paper
TRUE_INCLINATION = 30.0

# Starkey, Horne & Villforth (2016) Table 2: wavelength (A), mean AB, rms (mag).
FILTERS = {"u": (3540.0, 16.89, 0.098), "g": (4770.0, 17.00, 0.085), "r": (6215.0, 17.10, 0.076),
           "i": (7545.0, 17.17, 0.069), "z": (8700.0, 17.22, 0.065)}
BAND_SETS = ["gi", "gri", "ugriz"]
SNRS = [30, 100, 300, 1000]
CADENCES = [0.5, 1.0, 2.0, 4.0]  # mean days between points, per filter
SEEDS = list(range(5))
# Two more optimise() runs on other frequency grids: "optimise_auto" uses
# build_grid()'s default, 2 (baseline + tau_max) = ~260 days here, with 60
# frequencies; "optimise_baseline" uses the baseline as the longest period (the
# default until October 2026), to measure the bias a longer period removes.
METHODS = ["optimise", "nuts", "optimise_auto", "optimise_baseline"]

T_SPAN = 100.0  # days
DT_SIM = 0.05  # days, simulation grid
TAU_MAX = 30.0  # days, the fit's lag grid; the responses are ~0 well before this
# The driver's longest Fourier period and number of frequencies for the main
# fits (optimise and nuts). A longest period of only the 100-day baseline
# cannot represent a random walk's longer trends, and the fit absorbed them
# into long, inclined responses (log_mdot ~2.8 too high on one seed); see the
# build_grid() docstring. These fits were run before build_grid()'s default
# became 2 (baseline + tau_max), and use a longer period still.
PERIOD_MAX = 400.0  # days, ~3 x (T_SPAN + TAU_MAX)
N_FREQ = 80
JITTER = 0.3  # each time is i * cadence + U(-JITTER, JITTER) * cadence
OPTIMISE = dict(num_samples=1000, num_restarts=4)
# NUTS samples the linear-marginalised model (marginalise_linear=True; the same
# posterior for log_mdot and inclination). The default, sampled model was
# unusable here: warmup collapsed the step size to ~1e-5 with every iteration
# at the 1023-step ceiling (~20 s per iteration on g and i alone), whereas the
# marginalised model needs ~7 steps per sample.
NUTS = dict(num_warmup=500, num_samples=500)


def config_label(band_set: str, snr: float, cadence: float, seed: int) -> str:
    return f"{band_set}_snr{snr:g}_dt{cadence:g}_s{seed}"


def _response():
    from pycream2.forward_model import thin_disk_response

    return thin_disk_response


def simulate(band_set: str, snr: float, cadence: float, seed: int) -> dict:
    """Synthetic light curves (mJy) for one configuration; see the module docstring."""
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    # The driver, a random walk with unit standard deviation after one day,
    # depends only on the seed, so every configuration of a seed shares it.
    t_grid = np.arange(-TAU_MAX - 5.0, T_SPAN + 1.0, DT_SIM)
    x = np.cumsum(rng.normal(scale=np.sqrt(DT_SIM), size=len(t_grid)))
    tau = np.arange(0.0, TAU_MAX, DT_SIM)
    in_window = (t_grid >= 0.0) & (t_grid <= T_SPAN)
    bands = {}
    for name in "ugriz":  # always draw every filter, so a filter's data do not depend on the set
        wl, ab, rms_mag = FILTERS[name]
        brng = np.random.default_rng([seed, ord(name), int(round(1000 * cadence))])  # shared across SNRs
        psi = np.asarray(_response()(jnp.asarray(tau), TRUE_LOG_MDOT, wl, TRUE_INCLINATION, M_BH))
        echo = np.convolve(x, psi / psi.sum())[: len(t_grid)]  # echo(t) = sum psi(tau) x(t - tau)
        mean_mjy = 3631e3 * 10 ** (-0.4 * FILTERS["g"][1]) * (wl / FILTERS["g"][0]) ** (-1.0 / 3.0)
        rms_frac = rms_mag * np.log(10.0) / 2.5
        w = echo[in_window]
        f_grid = mean_mjy * (1.0 + rms_frac * (echo - w.mean()) / w.std())
        n = int(round(T_SPAN / cadence))
        t = np.sort(np.clip((np.arange(n) + 0.5 + brng.uniform(-JITTER, JITTER, n)) * cadence, 0.0, T_SPAN))
        f_true = np.interp(t, t_grid, f_grid)
        yerr = f_true / snr
        if name in band_set:
            bands[name] = dict(t=t, y=f_true + yerr * brng.normal(size=n), yerr=yerr, wavelength=wl)
    return bands


def make_echofit(bands: dict, frequency_grid: str = "main", marginalise_linear: bool = False):
    """``frequency_grid``: "main" (PERIOD_MAX, N_FREQ), "auto" (build_grid()'s
    default) or "baseline" (longest period = the data's baseline)."""
    import pycream2.model as model
    from pycream2 import EchoFit

    model.response_function = _response()
    ef = EchoFit(M_BH=M_BH, marginalise_linear=marginalise_linear)
    for name, d in bands.items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    if frequency_grid == "auto":
        ef.build_grid(tau_max=TAU_MAX)
    elif frequency_grid == "baseline":
        t = np.concatenate([d["t"] for d in bands.values()])
        ef.build_grid(tau_max=TAU_MAX, period_max=t.max() - t.min())
    else:
        ef.build_grid(tau_max=TAU_MAX, period_max=PERIOD_MAX, n_freq=N_FREQ)
    return ef


def _summary(draws: np.ndarray, truth: float) -> dict:
    q = np.percentile(draws, [2.5, 16, 50, 84, 97.5])
    return dict(mean=float(np.mean(draws)), sd=float(np.std(draws)), q025=q[0], q16=q[1], median=q[2],
                q84=q[3], q975=q[4], truth=truth, z=float((np.mean(draws) - truth) / np.std(draws)))


def run_one(band_set: str, snr: float, cadence: float, seed: int, method: str, out: Path) -> None:
    label = config_label(band_set, snr, cadence, seed)
    dest = out / label
    if (dest / f"{method}.json").exists():
        return
    bands = simulate(band_set, snr, cadence, seed)
    grid = method.split("_")[1] if "_" in method else "main"
    ef = make_echofit(bands, frequency_grid=grid, marginalise_linear=(method == "nuts"))
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method.startswith("optimise"):
            ef.optimise(rng_seed=seed, **OPTIMISE)
        else:
            ef.fit(rng_seed=seed, num_chains=1, progress_bar=False, generate_report=False, **NUTS)
    seconds = time.perf_counter() - t0
    s = {k: np.asarray(ef.samples[k], dtype=np.float64) for k in ("log_mdot", "inclination")}
    meta = dict(band_set=band_set, snr=snr, cadence=cadence, seed=seed, method=method, seconds=seconds,
                n_obs=int(sum(len(d["t"]) for d in bands.values())),
                log_mdot=_summary(s["log_mdot"], TRUE_LOG_MDOT),
                inclination=_summary(s["inclination"], TRUE_INCLINATION),
                warnings=sorted({str(w.message).splitlines()[0][:200] for w in caught}))
    if method.startswith("optimise"):
        meta.update(timings=ef.optimise_timings, potential=float(ef.optimise_result.fun),
                    optimum={k: float(ef.optimum[k]) for k in ("log_mdot", "inclination")})
    else:
        from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

        meta.update(divergences=int(np.sum(ef.extra_fields["diverging"])),
                    ess={k: float(effective_sample_size(v[None])) for k, v in s.items()},
                    split_rhat={k: float(split_gelman_rubin(v[None])) for k, v in s.items()})
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / f"{method}_samples.npz", **s)
    (dest / f"{method}.json").write_text(json.dumps(meta, indent=1, default=float))
    lm, inc = meta["log_mdot"], meta["inclination"]
    print(f"{label} {method}: {seconds:.0f}s  log_mdot {lm['mean']:.3f}+/-{lm['sd']:.3f} (true {TRUE_LOG_MDOT:.3f})"
          f"  i {inc['mean']:.1f}+/-{inc['sd']:.1f} (true {TRUE_INCLINATION:g})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output directory, one folder per configuration")
    parser.add_argument("--band-sets", nargs="+", default=BAND_SETS)
    parser.add_argument("--snrs", nargs="+", type=float, default=SNRS)
    parser.add_argument("--cadences", nargs="+", type=float, default=CADENCES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--shard", default="0/1", help="K/N: run every N-th fit starting at the K-th")
    args = parser.parse_args()
    k, n = (int(v) for v in args.shard.split("/"))
    todo = list(itertools.product(args.seeds, args.band_sets, args.snrs, args.cadences, args.methods))[k::n]
    for seed, band_set, snr, cadence, method in todo:
        try:
            run_one(band_set, snr, cadence, seed, method, args.out)
        except Exception as exc:  # keep going: one failed fit should not lose the rest of a long batch
            print(f"FAILED {config_label(band_set, snr, cadence, seed)} {method}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
