"""
synthetic.py
============

Minimal synthetic-data generator used for the demo notebook and tests: draws
a DRW driver on a dense Fourier grid, echoes it into several bands with the
true forward model, and adds Gaussian noise -- so the resulting light curves
have the correct lag ordering with wavelength (longer wavelength => longer
lag) by construction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .forward_model import (
    response_function, tophat_response_free, transfer_coeffs, compute_echo, driver_at,
    legendre_background_basis, mix_diffuse_continuum,
)
from .grid_utils import estimate_dt_min, graded_tau_grid

# np.trapz was removed in NumPy 2.x in favour of np.trapezoid; this matches
# the jnp.trapezoid/jnp.trapz fallback already used throughout forward_model.py.
_np_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def make_frequency_grid(n_freq: int, t_span: float, dt_min: float) -> np.ndarray:
    """Log-spaced angular-frequency grid from the light-curve baseline up to
    (roughly) the Nyquist frequency set by the finest sampling."""
    w_min = 2.0 * np.pi / t_span
    w_max = np.pi / dt_min
    return np.geomspace(w_min, w_max, n_freq)


def _sample_uniform_excluding_gaps(
    rng: np.random.Generator,
    t_span: float,
    gaps: Sequence[Tuple[float, float]],
    size: int,
) -> np.ndarray:
    """Uniformly sample ``size`` times in ``[0, t_span)``, skipping the given
    ``(gap_start, gap_duration)`` windows -- e.g. a seasonal/weather/downtime
    gap in an observing campaign. Density stays uniform over the remaining
    (observable) time, it's just zero inside the gaps.
    """
    allowed = []  # list of (start, end) intervals actually observable
    cursor = 0.0
    for start, duration in sorted(gaps, key=lambda g: g[0]):
        start = float(np.clip(start, 0.0, t_span))
        end = float(np.clip(start + duration, start, t_span))
        if start > cursor:
            allowed.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < t_span:
        allowed.append((cursor, t_span))
    if not allowed:
        raise ValueError("gaps cover the entire time span; no observable time remains")

    lengths = np.array([b - a for a, b in allowed])
    edges = np.concatenate([[0.0], np.cumsum(lengths)])
    u = rng.uniform(0.0, edges[-1], size=size)
    interval_idx = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, len(allowed) - 1)
    starts = np.array([a for a, _ in allowed])[interval_idx]
    return starts + (u - edges[interval_idx])


def generate_synthetic_dataset(
    bands: Optional[Dict[str, float]] = None,
    M_BH: float = 1.0e8,
    log_mdot_true: float = 0.2,
    inclination_true: float = 35.0,
    sigma_drw_true: float = 0.3,
    tau_drw_true: float = 30.0,
    t_span: float = 200.0,
    n_obs_per_band: int = 60,
    n_freq: int = 60,
    n_tau: int = 400,
    tau_max: float = 60.0,
    noise_level: float = 0.02,
    gaps: Optional[Sequence[Tuple[float, float]]] = None,
    seed: int = 0,
    diffuse_continuum: Optional[Dict[str, Tuple[float, float, float]]] = None,
    background: Optional[Dict[str, Sequence[float]]] = None,
    response=None,
) -> dict:
    """Generate a synthetic multi-band reverberation-mapping dataset.

    Parameters
    ----------
    bands : dict, optional
        Mapping ``band_name -> wavelength_angstrom``. Defaults to five SDSS-
        like bands spanning u through z, which guarantees a range of lags.
    M_BH, log_mdot_true, inclination_true, sigma_drw_true, tau_drw_true :
        Ground-truth physical parameters used to generate the data.
    t_span : float
        Total light-curve baseline, days.
    n_obs_per_band : int
        Number of (irregular) observation epochs per band, before removing
        any that fall in ``gaps``.
    n_freq, n_tau, tau_max : int, int, float
        Resolution of the driver Fourier grid and the lag grid used to
        build the ground-truth response functions.
    noise_level : float
        Fractional Gaussian noise added to each band (relative to that
        band's echo amplitude).
    gaps : sequence of (start, duration), optional
        Observing gaps (days) applied to every band identically -- e.g.
        ``[(50, 14), (150, 21)]`` for a 2-week gap starting day 50 and a
        3-week gap starting day 150 (weather/downtime/seasonal-style
        campaign structure). ``n_obs_per_band`` observations are still drawn
        uniformly at random, just excluding these windows, so the same
        total point count is spread more densely over the remaining time.
    seed : int
        RNG seed.
    diffuse_continuum : dict, optional
        ``band_name -> (fraction, median_delay_days, width_dex)``: mix a
        log-normal diffuse-continuum response into that band's response, as
        ``EchoFit.add_lightcurve(..., diffuse_continuum=True)`` models it
        (``forward_model.mix_diffuse_continuum``). Bands not listed have none.
    background : dict, optional
        ``band_name -> [c_1, .., c_K]``: add a slow background
        ``sum_k c_k P_k(t)``, Legendre polynomials over the whole campaign's
        time span, exactly the basis ``EchoFit.add_lightcurve(...,
        background_order=K)`` fits (``forward_model.legendre_background_basis``),
        so the coefficients are directly comparable with fitted ``bg_{band}``.
    response : callable, optional
        The ground-truth response function, with the response-function contract
        of :mod:`pycream2.model` (``(tau_grid, log_mdot, wavelength, inclination,
        M_BH)``, e.g. a ``functools.partial`` of
        :func:`pycream2.forward_model.thin_disk_response` or
        :func:`pycream2.rippled_disc.rippled_disc_response`). Default: the
        skew-normal :func:`pycream2.forward_model.response_function`.

    Returns
    -------
    data : dict
        ``{"bands": {name: {"t", "y", "yerr", "wavelength"}}, "truth": {...},
        "freqs": array, "tau_grid": array}``. The driver's longest period is
        ``t_span``, shorter than ``EchoFit.build_grid()``'s default
        (``2 (baseline + tau_max)``); to fit on the truth's own basis, pass
        ``period_max=2 * np.pi / data["freqs"].min()``. With ``diffuse_continuum`` or
        ``background``, each band's truth also has ``"diffuse_continuum"``
        and/or ``"background"`` (its coefficients) and ``"background_curve"``
        (at the band's observation times).
    """
    diffuse_continuum = diffuse_continuum or {}
    background = background or {}
    rng = np.random.default_rng(seed)

    if bands is None:
        bands = {"u": 3543.0, "g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}

    sorted_bands = sorted(bands.items(), key=lambda kv: kv[1])
    # sample every band's observation times up front so the frequency grid
    # below is derived from the *actual* (irregular) cadence -- via the same
    # estimator EchoFit.build_grid() uses -- rather than a prior guess. This
    # keeps the ground-truth driver and the fitting basis on the same grid.
    if gaps:
        t_by_band = {
            name: np.sort(_sample_uniform_excluding_gaps(rng, t_span, gaps, n_obs_per_band))
            for name, _ in sorted_bands
        }
    else:
        t_by_band = {
            name: np.sort(rng.uniform(0.0, t_span, size=n_obs_per_band))
            for name, _ in sorted_bands
        }

    dt_min = estimate_dt_min(t_by_band.values(), t_span=t_span)
    freqs = make_frequency_grid(n_freq, t_span, dt_min)
    tau_grid = graded_tau_grid(tau_max, n_tau)  # matches EchoFit.build_grid's default grading

    # -- draw a DRW driver realisation on the fixed Fourier grid ---------
    dw = np.gradient(freqs)
    power = sigma_drw_true ** 2 * tau_drw_true / (1.0 + (freqs * tau_drw_true) ** 2)
    amp_scale = np.sqrt(power * np.clip(dw, 1e-8, None))
    S_true = rng.normal(0.0, amp_scale)
    C_true = rng.normal(0.0, amp_scale)

    all_t = np.concatenate(list(t_by_band.values()))
    t_range = (float(all_t.min()), float(all_t.max()))

    out_bands = {}
    per_band_truth = {}
    for name, wavelength in sorted_bands:
        t = t_by_band[name]

        psi = np.asarray(
            (response or response_function)(
                tau_grid,
                log_mdot=log_mdot_true,
                wavelength=wavelength,
                inclination=inclination_true,
                M_BH=M_BH,
            )
        )
        if name in diffuse_continuum:
            psi = np.asarray(mix_diffuse_continuum(psi, tau_grid, *diffuse_continuum[name]))
        A, B = transfer_coeffs(tau_grid, psi, freqs)
        echo = np.asarray(compute_echo(S_true, C_true, freqs, np.asarray(A), np.asarray(B), t))

        S_band_true = rng.uniform(0.8, 1.5)
        C_band_true = rng.uniform(-0.5, 0.5)
        y_clean = S_band_true * echo + C_band_true
        background_curve = None
        if name in background:
            coeffs = np.asarray(background[name], dtype=float)
            background_curve = np.asarray(legendre_background_basis(t, t_range, len(coeffs))) @ coeffs
            y_clean = y_clean + background_curve

        yerr = np.full_like(y_clean, noise_level * (np.std(y_clean) + 1e-3))
        y = y_clean + rng.normal(0.0, yerr)

        out_bands[name] = {
            "t": t,
            "y": y,
            "yerr": yerr,
            "wavelength": wavelength,
        }
        per_band_truth[name] = {
            "S_band": S_band_true,
            "C_band": C_band_true,
            "tau_mean": float(_np_trapz(tau_grid * psi, tau_grid)),
        }
        if name in diffuse_continuum:
            per_band_truth[name]["diffuse_continuum"] = tuple(diffuse_continuum[name])
        if background_curve is not None:
            per_band_truth[name]["background"] = np.asarray(background[name], dtype=float)
            per_band_truth[name]["background_curve"] = background_curve

    truth = {
        "M_BH": M_BH,
        "log_mdot": log_mdot_true,
        "inclination": inclination_true,
        "sigma_drw": sigma_drw_true,
        "tau_drw": tau_drw_true,
        "S": S_true,
        "C": C_true,
        "bands": per_band_truth,
    }

    return {"bands": out_bands, "truth": truth, "freqs": freqs, "tau_grid": tau_grid, "gaps": gaps}


def generate_free_lag_dataset(
    lines: Optional[Dict[str, float]] = None,
    sigma_drw_true: float = 0.3,
    tau_drw_true: float = 30.0,
    t_span: float = 200.0,
    n_obs_per_line: int = 40,
    n_obs_driver: int = 80,
    include_driver: bool = True,
    n_freq: int = 40,
    n_tau: int = 300,
    tau_max: float = 60.0,
    noise_level: float = 0.05,
    seed: int = 0,
) -> dict:
    """Synthetic dataset with independent, not-physically-tied per-band lags
    (e.g. emission lines), and optionally a driver light curve -- the
    "free"-lag-mode counterpart to :func:`generate_synthetic_dataset`, for
    testing/demonstrating the identifiability point in ``model.py``'s module
    docstring: a driver light curve is essentially required to pin down
    these lags' absolute scale.

    Parameters not listed below mirror :func:`generate_synthetic_dataset`.

    Parameters
    ----------
    lines : dict, optional
        Mapping ``line_name -> true_lag_days``. Defaults to three lines at
        8, 15, and 22 days.
    include_driver : bool
        Whether to also generate a direct (zero-lag) driver light curve.
        Set ``False`` to generate a deliberately-unidentifiable dataset
        (see the module docstring / ``tests/test_response_function_swap.py``
        for what "unidentifiable" looks like in practice here).
    n_obs_driver : int
        Observation count for the driver light curve (only used if
        ``include_driver``).

    Returns
    -------
    data : dict
        ``{"bands": {name: {"t", "y", "yerr", "wavelength"}}, "driver":
        {"t", "y", "yerr"} or None, "truth": {...}, "freqs": array,
        "tau_grid": array}``. Each band's ``"wavelength"`` is just an
        arbitrary increasing placeholder (for plot ordering/colour only --
        it plays no role in a free-lag band's model).
    """
    rng = np.random.default_rng(seed)

    if lines is None:
        lines = {"line_a": 8.0, "line_b": 15.0, "line_c": 22.0}

    t_by_series = {}
    if include_driver:
        t_by_series["__driver__"] = np.sort(rng.uniform(0.0, t_span, size=n_obs_driver))
    for name in lines:
        t_by_series[name] = np.sort(rng.uniform(0.0, t_span, size=n_obs_per_line))

    dt_min = estimate_dt_min(t_by_series.values(), t_span=t_span)
    freqs = make_frequency_grid(n_freq, t_span, dt_min)
    tau_grid = graded_tau_grid(tau_max, n_tau)  # matches EchoFit.build_grid's default grading

    dw = np.gradient(freqs)
    power = sigma_drw_true ** 2 * tau_drw_true / (1.0 + (freqs * tau_drw_true) ** 2)
    amp_scale = np.sqrt(power * np.clip(dw, 1e-8, None))
    S_true = rng.normal(0.0, amp_scale)
    C_true = rng.normal(0.0, amp_scale)

    out_bands = {}
    per_band_truth = {}
    for i, (name, tau_true) in enumerate(sorted(lines.items(), key=lambda kv: kv[1])):
        t = t_by_series[name]
        psi = np.asarray(tophat_response_free(tau_grid, tau_mean=tau_true))
        A, B = transfer_coeffs(tau_grid, psi, freqs)
        echo = np.asarray(compute_echo(S_true, C_true, freqs, np.asarray(A), np.asarray(B), t))

        S_band_true = rng.uniform(0.8, 1.5)
        C_band_true = rng.uniform(-0.5, 0.5)
        y_clean = S_band_true * echo + C_band_true

        yerr = np.full_like(y_clean, noise_level * (np.std(y_clean) + 1e-3))
        y = y_clean + rng.normal(0.0, yerr)

        out_bands[name] = {
            "t": t, "y": y, "yerr": yerr,
            "wavelength": 4000.0 + 500.0 * i,  # placeholder, for plot ordering only
        }
        per_band_truth[name] = {"S_band": S_band_true, "C_band": C_band_true, "tau": tau_true}

    driver_out, driver_truth = None, None
    if include_driver:
        t_d = t_by_series["__driver__"]
        X_d = np.asarray(driver_at(S_true, C_true, freqs, t_d))
        S_driver_true = rng.uniform(0.8, 1.5)
        C_driver_true = rng.uniform(-0.5, 0.5)
        y_d_clean = S_driver_true * X_d + C_driver_true
        yerr_d = np.full_like(y_d_clean, noise_level * (np.std(y_d_clean) + 1e-3))
        y_d = y_d_clean + rng.normal(0.0, yerr_d)
        driver_out = {"t": t_d, "y": y_d, "yerr": yerr_d}
        driver_truth = {"S_driver": S_driver_true, "C_driver": C_driver_true}

    truth = {
        "sigma_drw": sigma_drw_true, "tau_drw": tau_drw_true,
        "S": S_true, "C": C_true, "bands": per_band_truth, "driver": driver_truth,
    }
    return {"bands": out_bands, "driver": driver_out, "truth": truth, "freqs": freqs, "tau_grid": tau_grid}


def with_disc_fluxes(
    data: dict, redshift: float, h0: float = 70.0, omega_m: float = 0.3, variable_fraction: float = 0.15,
    host_mjy: Optional[Dict[str, float]] = None, ebv_galactic: float = 0.0, flux_unit: str = "mJy",
    flux_scale: float = 1.0, variable_sed: str = "response", n_fine: int = 1500,
) -> dict:
    """Give a :func:`generate_synthetic_dataset` dataset absolute fluxes, for
    trying the disc SED, distance and ``H_0`` analysis (:mod:`pycream2.disc_sed`)
    on data whose answer is known.

    Each band becomes ``host + disc``: the model disc of the dataset's own
    ``log_mdot``/``inclination`` (``T ~ r^-3/4``, ``T_1`` from
    :func:`~pycream2.forward_model.disk_t1_kelvin`) at the luminosity distance
    that ``h0`` gives at ``redshift``. Its variable part, per standard
    deviation of the driver, is ``variable_fraction`` of the bluest band's mean
    disc flux there, and follows across the bands either the spectrum of the
    disc's response to the lamppost (``variable_sed="response"``, the default
    and the physical case) or the mean disc spectrum (``"mean"``: a constant
    fractional amplitude, exactly the flux-flux analyses' assumption, so the
    host subtraction is then exact). The light
    curves are then dimmed by Galactic extinction and expressed in
    ``flux_unit`` (``"mJy"``, or ``"f_lambda"`` in units of ``flux_scale``
    erg/s/cm^2/A). Wavelengths are rest-frame. ``host_mjy`` gives each band's
    host flux (default none). The truth is added as ``data["truth"]["disc_sed"]``.
    """
    from . import disc_sed
    from .forward_model import disk_t1_kelvin, driver_at

    truth = data["truth"]
    host_mjy = host_mjy or {}
    t_all = np.concatenate([d["t"] for d in data["bands"].values()])
    t_fine = np.linspace(t_all.min(), t_all.max(), n_fine)
    x = np.asarray(driver_at(np.asarray(truth["S"]), np.asarray(truth["C"]), data["freqs"], t_fine))
    x_mean, x_sd = float(x.mean()), float(x.std())
    if variable_sed not in ("response", "mean"):
        raise ValueError(f"variable_sed must be 'response' or 'mean', got {variable_sed!r}")
    t1 = float(disk_t1_kelvin(truth["log_mdot"], truth["M_BH"]))
    dl_mpc = float(disc_sed.h0_from_dl(1.0, redshift, omega_m) / h0)
    names = sorted(data["bands"], key=lambda n: data["bands"][n]["wavelength"])
    lam = np.array([float(data["bands"][n]["wavelength"]) for n in names])
    discs = disc_sed.disc_fnu_at_1cm(lam, redshift, t1, 0.75, truth["inclination"], truth["M_BH"]) / (
        dl_mpc * disc_sed.MPC_CM) ** 2
    shape = (disc_sed.response_sed_shape(lam, t1, 0.75, truth["M_BH"]) if variable_sed == "response" else discs)
    variable = dict(zip(names, discs[0] * variable_fraction * shape / shape[0]))  # mJy per driver sd
    discs = dict(zip(names, discs))
    out = {"bands": {}, **{k: v for k, v in data.items() if k != "bands"}}
    disc_truth = {}
    for name, d in data["bands"].items():
        lam_rest = float(d["wavelength"])
        lam_obs = lam_rest * (1 + redshift)
        disc = float(discs[name])
        bt = truth["bands"][name]
        echo = (np.asarray(d["y"]) - bt["C_band"]) / bt["S_band"]  # the response-smoothed driver (+ noise)
        gain = float(variable[name]) / x_sd
        y_mjy = host_mjy.get(name, 0.0) + disc + gain * (echo - x_mean)
        dim = 10 ** (-0.4 * disc_sed.ccm_a_lambda(lam_obs, ebv_galactic)) / disc_sed.to_mjy([lam_obs], flux_unit, flux_scale)[0]
        out["bands"][name] = dict(d, y=y_mjy * dim, yerr=np.asarray(d["yerr"]) / bt["S_band"] * gain * dim)
        disc_truth[name] = dict(disc_mean_mjy=disc, host_mjy=host_mjy.get(name, 0.0))
    out["truth"] = dict(truth, disc_sed=dict(redshift=redshift, h0=h0, omega_m=omega_m, dl_mpc=dl_mpc, t1_kelvin=t1,
                                             ebv_galactic=ebv_galactic, flux_unit=flux_unit, flux_scale=flux_scale,
                                             variable_sed=variable_sed, bands=disc_truth))
    return out
