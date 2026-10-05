"""
plotting.py
===========

Plotting entry points, matching the EchoFit API:

* ``plot_raw_lightcurves``   -- one panel per band, ordered by wavelength.
* ``plot_lightcurve_fits``   -- posterior predictive light curves (68%/95%
                                 credible bands) with the corresponding
                                 response function psi(tau) alongside each
                                 band.
* ``plot_power_spectrum``    -- posterior driver power spectrum vs. the
                                 fitted DRW prior.
* ``plot_mcmc_diagnostics``  -- trace plots per (scalar) parameter.
* ``plot_corner``            -- pairwise joint posteriors + marginals,
                                 coloured per chain.
* ``plot_fourier_correlation`` -- posterior correlation matrix of the
                                 driver's Fourier coefficients (a corner
                                 plot doesn't scale to n_freq often being
                                 in the tens; a heatmap does).
* ``plot_bof``                -- Badness-of-Fit (Starkey+2016's BOF, here
                                 ``2 * potential_energy``) per sample,
                                 coloured per chain -- should decrease then
                                 flatten as the chain converges.
* ``wavelength_to_colour``    -- real-world-ish colour for a given
                                 wavelength (X-ray black, UV violet, the
                                 visible range its actual spectral colour,
                                 IR+ reddish), used for every band's colour
                                 in the plots above instead of an arbitrary
                                 per-plot palette; also reused directly by
                                 ``scripts/make_fit_animation.py``.

All functions take plain numpy-able arrays / dicts so they have no
dependency on JAX or NumPyro themselves.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def _visible_wavelength_to_rgb(wavelength_nm: float):
    """Approximate real-world colour of visible light at ``wavelength_nm``
    (380-780 nm), via the standard piecewise-linear spectrum-to-RGB
    approximation (as used e.g. in Dan Bruton's "colour science" reference
    implementation) -- violet at the blue end, through green and yellow, to
    red at the long end, with an intensity taper near the edges of vision.
    """
    w = wavelength_nm
    if w < 440.0:
        r, g, b = -(w - 440.0) / (440.0 - 380.0), 0.0, 1.0
    elif w < 490.0:
        r, g, b = 0.0, (w - 440.0) / (490.0 - 440.0), 1.0
    elif w < 510.0:
        r, g, b = 0.0, 1.0, -(w - 510.0) / (510.0 - 490.0)
    elif w < 580.0:
        r, g, b = (w - 510.0) / (580.0 - 510.0), 1.0, 0.0
    elif w < 645.0:
        r, g, b = 1.0, -(w - 645.0) / (645.0 - 580.0), 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0

    if w < 420.0:
        factor = 0.3 + 0.7 * (w - 380.0) / (420.0 - 380.0)
    elif w < 701.0:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780.0 - w) / (780.0 - 700.0)
    factor = np.clip(factor, 0.0, 1.0)

    gamma = 0.8
    adjust = lambda c: 0.0 if c <= 0.0 else (c * factor) ** gamma
    return (adjust(r), adjust(g), adjust(b))


def wavelength_to_colour(wavelength_angstrom: float):
    """Real-world-ish colour for a light curve at ``wavelength_angstrom``,
    rather than an arbitrary per-plot palette colour -- X-ray driving light
    curves (< 100 A) black, UV (100-3800 A) a strong violet, the visible
    range (3800-7500 A) its actual approximate spectral colour, and IR and
    beyond (> 7500 A) a reddish colour, per the physical picture of the
    lamppost model (a short-wavelength driver reprocessed into progressively
    redder, more slowly varying bands further out in the disk)."""
    if wavelength_angstrom < 100.0:
        return "black"
    if wavelength_angstrom < 3800.0:
        return "darkviolet"
    if wavelength_angstrom <= 7500.0:
        return _visible_wavelength_to_rgb(wavelength_angstrom / 10.0)
    return "firebrick"


def _cie_colour_matching(wavelength_nm):
    """CIE 1931 2-degree colour-matching functions x̄, ȳ, z̄ via the
    multi-lobe Gaussian fit of Wyman, Sloan & Shirley (2013, JCGT 2, 1),
    accurate to well within plotting needs and avoiding a tabulated
    dataset dependency."""
    lam = np.asarray(wavelength_nm, dtype=float)

    def g(mu, s1, s2):
        s = np.where(lam < mu, s1, s2)
        return np.exp(-0.5 * ((lam - mu) / s) ** 2)

    x = 1.056 * g(599.8, 37.9, 31.0) + 0.362 * g(442.0, 16.0, 26.7) - 0.065 * g(501.1, 20.4, 26.2)
    y = 0.821 * g(568.8, 46.9, 40.5) + 0.286 * g(530.9, 16.3, 31.1)
    z = 1.217 * g(437.0, 11.8, 36.0) + 0.681 * g(459.0, 26.0, 13.8)
    return x, y, z


def blackbody_to_colour(temperature_kelvin):
    """The sRGB colour a blackbody at ``temperature_kelvin`` appears to the
    eye: the Planck spectrum integrated against the CIE 1931 colour-matching
    functions, converted XYZ -> linear sRGB (D65), normalised so the
    brightest channel is 1 (chromaticity only, not luminosity), then
    gamma-encoded. Hot (>~10^4 K) is blue-white, ~6500 K white, cooler
    orange then deep red, i.e. what a real accretion disk's colour does
    with radius.

    Returns
    -------
    array, shape ``temperature_kelvin.shape + (3,)``, values in [0, 1].
    """
    # Below ~500 K a blackbody's visible spectrum is vanishingly faint and its
    # colour has saturated at deep red anyway; the floor keeps the Planck
    # integral finite (no underflow to 0/0) for arbitrarily cold input.
    T = np.clip(np.atleast_1d(np.asarray(temperature_kelvin, dtype=float)), 500.0, None)
    lam_nm = np.linspace(380.0, 780.0, 201)
    xbar, ybar, zbar = _cie_colour_matching(lam_nm)
    lam_m = lam_nm * 1e-9
    hc_over_k = 1.438777e-2  # m K
    # Planck B_lambda up to a constant (the normalisation below removes it).
    planck = lam_m[None, :] ** -5 / np.expm1(hc_over_k / (lam_m[None, :] * T.reshape(-1, 1)))
    xyz = np.stack([planck @ xbar, planck @ ybar, planck @ zbar], axis=-1)
    xyz_to_rgb = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ])
    rgb = np.clip(xyz @ xyz_to_rgb.T, 0.0, None)
    rgb /= rgb.max(axis=-1, keepdims=True)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1.0 / 2.4) - 0.055)
    return np.clip(rgb, 0.0, 1.0).reshape(np.shape(temperature_kelvin) + (3,))


def _band_colours(bands: Dict[str, dict]):
    """Consistent, physically-motivated colour per band (see
    ``wavelength_to_colour``), ordered by wavelength."""
    ordered = sorted(bands.items(), key=lambda kv: kv[1]["wavelength"])
    colours = {name: wavelength_to_colour(d["wavelength"]) for name, d in ordered}
    return ordered, colours


def plot_raw_lightcurves(bands: Dict[str, dict], driver: Optional[dict] = None, figsize_per_panel=(7, 1.8)):
    """One panel per band (no overlay), ordered shortest -> longest wavelength,
    plus a driver panel first if given.

    Parameters
    ----------
    bands : dict
        ``{band_name: {"t", "y", "yerr", "wavelength"}}``.
    driver : dict, optional
        ``{"t", "y", "yerr"}`` for a light curve directly observing the
        driver (see ``EchoFit.add_driver_lightcurve``), plotted in its own
        panel first, ahead of the wavelength-ordered bands.
    """
    ordered, colours = _band_colours(bands)
    n = len(ordered) + (1 if driver is not None else 0)
    fig, axes = plt.subplots(
        n, 1, figsize=(figsize_per_panel[0], figsize_per_panel[1] * n), sharex=True
    )
    if n == 1:
        axes = [axes]

    if driver is not None:
        ax = axes[0]
        ax.errorbar(
            driver["t"], driver["y"], yerr=driver["yerr"], fmt="o", ms=4, color="0.2",
            ecolor="0.2", alpha=0.85, capsize=3, elinewidth=1.2, capthick=1.2,
        )
        ax.set_ylabel("driver")
        ax.grid(alpha=0.6)
        axes = axes[1:]

    for ax, (name, d) in zip(axes, ordered):
        ax.errorbar(
            d["t"], d["y"], yerr=d["yerr"], fmt="o", ms=4, color=colours[name],
            ecolor=colours[name], alpha=0.85, capsize=3, elinewidth=1.2, capthick=1.2,
        )
        ax.set_ylabel(f"{name}\n({d['wavelength']:.0f} \u00c5)")
        ax.grid(alpha=0.6)

    axes[-1].set_xlabel("time [days]")
    fig.suptitle("Raw multi-band light curves")
    fig.tight_layout()
    return fig, fig.axes


def response_function_xlim(tau_grid, psi_list, coverage: float = 0.99, padding: float = 1.2) -> float:
    """A shared x-axis limit for every psi(tau) panel, based on where the
    *widest* response has decayed to negligible area -- not the full
    extent of ``tau_grid``, which ``EchoFit.build_grid`` deliberately sets
    as a generous ceiling (comfortably exceeding any lag you'd expect),
    not a claim that every band's response is actually that wide. Used by
    both ``plot_lightcurve_fits`` (below) and ``scripts/make_fit_animation.py``,
    so the README animation, the standard report/smoke-test output, and
    every other plot built on this all use the same mechanism to set this
    limit, rather than each picking (or hardcoding) their own.

    For each ``psi`` in ``psi_list`` (already area-normalised to 1 on
    ``tau_grid``, per the response-function contract, CLAUDE.md decision
    #5), finds the smallest ``tau`` where its cumulative area (CDF)
    reaches ``coverage`` (default 99%), then returns ``padding`` times the
    largest such ``tau`` across every band -- so every band's response
    panel shares one scale, and that scale is set by whichever band's
    response actually reaches furthest, not by ``tau_grid``'s own
    (typically much larger) upper bound.

    Parameters
    ----------
    tau_grid : (n_tau,) array
    psi_list : sequence of (n_tau,) array
        One response per band -- pass a robust representative (e.g. the
        mean across posterior draws) rather than a pointwise median if
        individual draws' peaks might land on different grid points; a
        pointwise median of several narrow, laterally-jittering peaks can
        itself look artificially flat.
    """
    tau_grid = np.asarray(tau_grid)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    widest = 0.0
    for psi in psi_list:
        psi = np.asarray(psi)
        area = trapz(psi, tau_grid)
        if area <= 0:
            continue
        segment_area = (psi[1:] + psi[:-1]) / 2.0 * np.diff(tau_grid)
        cdf = np.concatenate([[0.0], np.cumsum(segment_area)]) / area
        idx = min(int(np.searchsorted(cdf, coverage)), len(tau_grid) - 1)
        widest = max(widest, float(tau_grid[idx]))
    if widest <= 0:
        return float(tau_grid[-1])  # degenerate fallback: every psi was ~0 everywhere
    return min(padding * widest, float(tau_grid[-1]))


def plot_lightcurve_fits(
    bands: Dict[str, dict],
    t_fine: np.ndarray,
    y_pred_samples: Dict[str, np.ndarray],
    tau_grid: np.ndarray,
    psi_samples: Dict[str, np.ndarray],
    driver_samples: Optional[np.ndarray] = None,
    driver_points: Optional[tuple] = None,
    driver_sigma_eff: Optional[np.ndarray] = None,
    sigma_eff_by_band: Optional[Dict[str, np.ndarray]] = None,
    c_band_samples: Optional[Dict[str, np.ndarray]] = None,
    baseline_samples: Optional[Dict[str, np.ndarray]] = None,
    figsize_per_row=(10, 2.2),
    driver_row_height=1.8,
):
    """Posterior-predictive light curves with 68%/95% credible bands, plus
    the corresponding response function psi(tau) next to each band.

    Parameters
    ----------
    bands : dict
        Observed data, ``{band_name: {"t", "y", "yerr", "wavelength"}}``.
    t_fine : (n_fine,) array
        Dense time grid the posterior predictive was evaluated on.
    y_pred_samples : dict
        ``{band_name: array of shape (n_samples, n_fine)}`` posterior
        predictive draws of the light curve, evaluated at ``t_fine``.
    tau_grid : (n_tau,) array
        Lag grid the response functions live on.
    psi_samples : dict
        ``{band_name: array of shape (n_samples, n_tau)}`` posterior draws
        of psi(tau) for that band.
    driver_samples : (n_samples, n_fine) array, optional
        Posterior draws of the raw driver X(t) (see
        ``forward_model.driver_at``), evaluated at the same ``t_fine`` as
        ``y_pred_samples``. If given, plotted as an extra panel above the
        per-band rows, sharing the same time axis so lags between the
        driver and each band's echo are visually alignable.
    driver_points : (t, X_implied, yerr_implied) tuple, optional
        A registered driver light curve's own data, back-transformed
        through the posterior-mean ``S_driver``/``C_driver`` into the same
        units as ``driver_samples`` (i.e. ``(y - C_driver) / S_driver``) so
        it can be overlaid on the driver panel as a direct cross-check that
        the inferred X(t) actually tracks what was observed.
    driver_sigma_eff : (n_driver_points,) array, optional
        The driver's own effective error (``sqrt((sigma_scale*yerr)**2 +
        sigma_jitter**2)``, already back-transformed into the same units as
        ``driver_points``), drawn as a lighter, wider errorbar overlaid on
        top of ``driver_points``' own error bars when its light curve was
        registered with ``fit_error_model=True``.
    sigma_eff_by_band : dict, optional
        ``{band_name: array of shape (n_obs,)}`` per-band effective error
        (same formula as ``driver_sigma_eff``, in the band's own native
        units), drawn the same way for any band fitted with
        ``fit_error_model=True``. A band absent from this dict is drawn with
        only its own quoted ``yerr``.
    c_band_samples : dict, optional
        ``{band_name: array of shape (n_samples,)}`` posterior draws of that
        band's constant offset ``C_{band}`` (the additive term in ``y_pred =
        S_band*echo + C_band``). Drawn as a dashed horizontal line at the
        posterior median plus 68%/95% credible spans across the whole
        panel, in the same wavelength-appropriate colour as that band's own
        fit, so the offset the model is anchoring the echo to is visible
        alongside it.
    baseline_samples : dict, optional
        ``{band_name: array of shape (n_samples, len(t_fine))}``: for a band
        with a slow background (``background_order > 0``), posterior draws of
        its whole non-reverberating part, offset plus background, on
        ``t_fine``. Drawn in place of that band's flat offset line, as a
        dashed median curve with 68%/95% credible bands.
    """
    ordered, colours = _band_colours(bands)
    n = len(ordered)
    has_driver = driver_samples is not None
    n_rows = n + (1 if has_driver else 0)
    height_ratios = ([driver_row_height] if has_driver else []) + [figsize_per_row[1]] * n
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(figsize_per_row[0], sum(height_ratios)),
        gridspec_kw={"width_ratios": [3, 1], "height_ratios": height_ratios},
    )
    if n_rows == 1:
        axes = axes[None, :]

    if has_driver:
        ax_drv, ax_drv_unused = axes[0, 0], axes[0, 1]
        lo95, lo68, med, hi68, hi95 = np.percentile(driver_samples, [2.5, 16, 50, 84, 97.5], axis=0)
        ax_drv.fill_between(t_fine, lo95, hi95, color="0.5", alpha=0.15, zorder=1)
        ax_drv.fill_between(t_fine, lo68, hi68, color="0.5", alpha=0.35, zorder=2)
        if driver_points is not None:
            t_d, X_d, yerr_d = driver_points
            ax_drv.errorbar(
                t_d, X_d, yerr=yerr_d, fmt="o", ms=4, color="C3",
                ecolor="C3", alpha=0.7, capsize=3, elinewidth=1.2, capthick=1.2,
                zorder=3, label="driver data (back-transformed)",
            )
            if driver_sigma_eff is not None:
                ax_drv.errorbar(
                    t_d, X_d, yerr=driver_sigma_eff, fmt="none",
                    ecolor="C3", alpha=0.35, capsize=3, elinewidth=2.0, capthick=1.2,
                    zorder=4, label="effective error (fit_error_model)",
                )
        ax_drv.plot(t_fine, med, color="black", lw=1.5, zorder=5)
        if driver_points is not None:
            ax_drv.legend(fontsize=7, loc="upper right")
        ax_drv.set_ylabel("driver\nX(t)")
        ax_drv.set_title("Inferred driving light curve", fontsize=10)
        ax_drv.grid(alpha=0.6)
        ax_drv.sharex(axes[1, 0])
        ax_drv_unused.axis("off")

    psi_axes = []
    psi_means = []
    for row, (name, d) in enumerate(ordered):
        ax_lc, ax_psi = axes[row + has_driver, 0], axes[row + has_driver, 1]
        colour = colours[name]

        preds = y_pred_samples[name]
        lo95, lo68, med, hi68, hi95 = np.percentile(preds, [2.5, 16, 50, 84, 97.5], axis=0)

        ax_lc.fill_between(t_fine, lo95, hi95, color=colour, alpha=0.15, label="95% CI", zorder=1)
        ax_lc.fill_between(t_fine, lo68, hi68, color=colour, alpha=0.35, label="68% CI", zorder=2)
        c_samples = None if c_band_samples is None else c_band_samples.get(name)
        baseline = None if baseline_samples is None else baseline_samples.get(name)
        if baseline is not None:
            b_lo95, b_lo68, b_med, b_hi68, b_hi95 = np.percentile(baseline, [2.5, 16, 50, 84, 97.5], axis=0)
            ax_lc.fill_between(t_fine, b_lo95, b_hi95, color=colour, alpha=0.15, lw=0, zorder=0)
            ax_lc.fill_between(t_fine, b_lo68, b_hi68, color=colour, alpha=0.35, lw=0, zorder=0)
            ax_lc.plot(t_fine, b_med, color=colour, lw=1.0, ls="--", zorder=2.5, label="offset + background")
        elif c_samples is not None:
            c_lo95, c_lo68, c_med, c_hi68, c_hi95 = np.percentile(c_samples, [2.5, 16, 50, 84, 97.5])
            ax_lc.axhspan(c_lo95, c_hi95, color=colour, alpha=0.15, zorder=0)
            ax_lc.axhspan(c_lo68, c_hi68, color=colour, alpha=0.35, zorder=0)
            ax_lc.axhline(
                c_med, color=colour, lw=1.0, ls="--", zorder=2.5,
                label="constant offset (C_band)" if row == 0 else None,
            )
        ax_lc.errorbar(
            d["t"], d["y"], yerr=d["yerr"], fmt="o", ms=4, color="k",
            ecolor="k", alpha=0.7, capsize=3, elinewidth=1.2, capthick=1.2, zorder=3,
        )
        sigma_eff = None if sigma_eff_by_band is None else sigma_eff_by_band.get(name)
        if sigma_eff is not None:
            ax_lc.errorbar(
                d["t"], d["y"], yerr=sigma_eff, fmt="none",
                ecolor=colour, alpha=0.35, capsize=3, elinewidth=2.0, capthick=1.2,
                zorder=4, label="effective error (fit_error_model)" if row == 0 else None,
            )
        ax_lc.plot(t_fine, med, color=colour, lw=1.5, zorder=5)
        ax_lc.set_ylabel(f"{name}\n({d['wavelength']:.0f} \u00c5)")
        ax_lc.grid(alpha=0.6)
        if row == 0 or baseline is not None:  # label a background wherever one is drawn
            ax_lc.legend(fontsize=8, ncol=2, loc="upper right")

        psis = psi_samples[name]
        plo95, plo68, pmed, phi68, phi95 = np.percentile(psis, [2.5, 16, 50, 84, 97.5], axis=0)
        ax_psi.fill_between(tau_grid, plo95, phi95, color=colour, alpha=0.15)
        ax_psi.fill_between(tau_grid, plo68, phi68, color=colour, alpha=0.35)
        ax_psi.plot(tau_grid, pmed, color=colour, lw=1.5)
        ax_psi.set_ylabel(r"$\psi(\tau)$")
        ax_psi.grid(alpha=0.6)
        psi_axes.append(ax_psi)
        psi_means.append(np.asarray(psis).mean(axis=0))

    xlim = response_function_xlim(tau_grid, psi_means)
    for ax_psi in psi_axes:
        ax_psi.set_xlim(0.0, xlim)

    axes[-1, 0].set_xlabel("time [days]")
    axes[-1, 1].set_xlabel(r"$\tau$ [days]")
    fig.suptitle("Posterior-predictive light curves and response functions")
    fig.tight_layout()
    return fig, axes


def plot_power_spectrum(
    freqs: np.ndarray,
    S_samples: np.ndarray,
    C_samples: np.ndarray,
    sigma_drw_samples: np.ndarray,
    tau_drw_samples: Optional[np.ndarray] = None,
    figsize=(7, 5),
):
    """Posterior driver power spectrum vs. the prior it was drawn under,
    displayed in cycles/day (``f = w / (2*pi)``) throughout rather than
    the angular frequency ``w`` the model itself is built on internally
    (``EchoFit.freqs``, ``model.drw_prior_scale``/``model.rw_prior_scale``
    -- the Fourier series' own ``sin(w*t)``/``cos(w*t)`` needs angular
    frequency, so that internal representation is unchanged; this is a
    display-only conversion) -- cycles/day is just a more directly
    interpretable unit (a value of 1 means "one cycle per day").

    Converting a power *spectral density* between frequency variables
    needs its own Jacobian factor, not just relabelling the axis: a
    density has to satisfy ``P_f(f) df = P_w(w) dw`` (the actual power in
    a bin can't change just from renaming the variable it's a density
    over), so ``P_f(f) = P_w(w) * dw/df = 2*pi * P_w(w)``. In practice
    this comes out to the same empirical-periodogram formula either way
    (``(S**2 + C**2) / (2 * delta_k)``), just with the local bin spacing
    ``delta_k`` computed on the converted (cycles/day) grid instead of the
    original (rad/day) one -- the ``2*pi`` factors cancel algebraically.
    The fitted prior curves (Lorentzian/power-law) don't get that
    cancellation for free, since they're evaluated from a closed-form
    formula rather than a grid spacing, so those are computed from the
    original angular ``freqs`` as before and then explicitly multiplied
    by ``2*pi``.

    The empirical periodogram-style estimate per posterior draw is
    ``P(f_k) = (S_k**2 + C_k**2) / (2 * df_k)``, where ``df_k`` is the
    local frequency-grid spacing in cycles/day -- matching how
    ``model.drw_prior_scale``/``model.rw_prior_scale`` set ``Var(S_k) =
    Var(C_k) = power(w_k) * dw_k`` in the first place (converted, per
    above), so this is directly comparable to the fitted ``power(f)``
    curve overlaid from the same posterior draws. Since ``freqs`` is
    log-spaced (geomspace), skipping the ``df_k`` normalisation would
    flatten the apparent log-log slope purely from the growing bin width
    at high frequency -- not a real physical effect.

    ``tau_drw_samples`` given (``drw_prior=True`` fits): the fitted curve is
    the DRW's Lorentzian, flat for ``f << 1/(2*pi*tau_drw)``, falling off
    as ``f**-2`` for ``f >> 1/(2*pi*tau_drw)``. ``tau_drw_samples`` omitted
    (the default ``drw_prior=False`` fits): the fitted curve is the pure
    random-walk power law, ``f**-2`` everywhere -- see ``model.py``'s
    "random-walk prior" note. Either way, a dotted reference line at the
    ``-2`` slope is overlaid so the high-frequency (DRW) or everywhere
    (RW) asymptote is easy to eyeball.

    Parameters
    ----------
    freqs : (n_freq,) array
        Driver angular frequency grid (rad/day) -- ``EchoFit.freqs``,
        converted to cycles/day internally for display only.
    S_samples, C_samples : (n_samples, n_freq) array
        Posterior draws of the driver's sine/cosine Fourier coefficients.
    sigma_drw_samples : (n_samples,) array
        Posterior draws of the driver amplitude hyperparameter.
    tau_drw_samples : (n_samples,) array, optional
        Posterior draws of the DRW damping timescale. Omit for a
        ``drw_prior=False`` (random-walk) fit, which has no such site.
    """
    f_cycles = freqs / (2.0 * np.pi)
    df = np.clip(np.gradient(f_cycles), 1e-8, None)
    P_samples = (S_samples ** 2 + C_samples ** 2) / (2.0 * df[None, :])
    plo95, plo68, pmed, phi68, phi95 = np.percentile(P_samples, [2.5, 16, 50, 84, 97.5], axis=0)

    if tau_drw_samples is not None:
        fit_power_omega = (
            sigma_drw_samples[:, None] ** 2 * tau_drw_samples[:, None]
            / (1.0 + (freqs[None, :] * tau_drw_samples[:, None]) ** 2)
        )
        fit_label = "fitted DRW prior (Lorentzian)"
    else:
        fit_power_omega = sigma_drw_samples[:, None] ** 2 / freqs[None, :] ** 2
        fit_label = "fitted RW prior (power law)"
    fit_power = fit_power_omega * 2.0 * np.pi  # P_f(f) = 2*pi * P_w(w) -- see docstring
    flo95, flo68, fmed, fhi68, fhi95 = np.percentile(fit_power, [2.5, 16, 50, 84, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(f_cycles, plo95, phi95, color="0.6", alpha=0.15, label="posterior P(f) 95% CI")
    ax.fill_between(f_cycles, plo68, phi68, color="0.6", alpha=0.3, label="posterior P(f) 68% CI")
    ax.plot(
        f_cycles, pmed, color="k", lw=1.0, marker="s", markersize=3,
        label="posterior P(f) median (per frequency)",
    )

    ax.fill_between(f_cycles, flo95, fhi95, color="C0", alpha=0.12)
    ax.plot(f_cycles, fmed, color="C0", lw=1.5, ls="--", label=fit_label)

    f_ref = np.sqrt(f_cycles[0] * f_cycles[-1])
    P_ref = np.interp(f_ref, f_cycles, pmed)
    f_line = f_cycles[f_cycles >= f_ref]
    ax.plot(
        f_line, P_ref * (f_line / f_ref) ** -2, color="C3", lw=1.2, ls=":",
        label=r"$f^{-2}$ (random-walk asymptote)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$f$ [cycles/day]")
    ax.set_ylabel(r"$P(f)$")
    ax.grid(alpha=0.6, which="both")
    ax.legend(fontsize=8)
    ax.set_title("Driver power spectrum: posterior vs. fitted prior")
    fig.tight_layout()
    return fig, ax


def plot_mcmc_diagnostics(
    samples: Dict[str, np.ndarray],
    param_names: Optional[Sequence[str]] = None,
    max_params: int = 12,
):
    """Trace plots per scalar parameter.

    Parameters
    ----------
    samples : dict
        ``mcmc.get_samples(group_by_chain=True)`` output, i.e.
        ``{name: array of shape (n_chains, n_samples, ...)}``. Non-scalar
        (vector) sites -- e.g. the per-frequency driver coefficients ``S``,
        ``C`` -- are summarised by their mean across the vector dimension
        so the figure stays readable.
    param_names : sequence of str, optional
        Restrict to these parameter names; defaults to all scalar-friendly
        sites in ``samples``, capped at ``max_params``.
    """
    if param_names is None:
        param_names = list(samples.keys())[:max_params]

    n = len(param_names)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, param_names):
        arr = np.asarray(samples[name])
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1).mean(axis=-1)
        n_chains = arr.shape[0]
        for c in range(n_chains):
            ax.plot(arr[c], lw=0.7, alpha=0.8, label=f"chain {c}" if len(param_names) == 1 else None)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(alpha=0.6)

    axes[-1].set_xlabel("sample")
    fig.suptitle("MCMC trace diagnostics")
    fig.tight_layout()
    return fig, axes


def plot_corner(
    samples: Dict[str, np.ndarray],
    param_names: Sequence[str] = ("log_mdot", "inclination"),
    true_values: Optional[Dict[str, float]] = None,
    figsize=None,
):
    """Corner plot (pairwise joint posteriors + 1-D marginals) coloured per
    chain, in the style of Starkey, Horne & Villforth (2016, MNRAS 456,
    1960; arXiv:1511.06162) Figure 6 -- overlaid chains distinguished by
    colour rather than pooled into one, so this doubles as a convergence
    check: independently-initialised chains that land in visibly different
    places are exactly what a Gelman-Rubin R-hat check would also catch
    (see CLAUDE.md's rough-edges note on `lag_mode="free"` recovery, which
    needed exactly this kind of multi-chain scrutiny), but seeing *where*
    they disagree is easier from a picture than from a single number.

    Parameters
    ----------
    samples : dict
        ``mcmc.get_samples(group_by_chain=True)`` output, i.e.
        ``{name: array of shape (n_chains, n_samples)}`` (scalar sites
        only -- pick vector sites like ``S``/``C`` apart first if you want
        one of their components here).
    param_names : sequence of str
        Which parameters to plot, in order (also sets the grid size).
        Defaults to ``("log_mdot", "inclination")``, the two parameters
        shared across every physical-mode band and the ones Starkey+2016's
        own Figure 6 uses -- only present if at least one band used
        ``lag_mode="physical"`` (CLAUDE.md decision #7); raises a clear
        ``KeyError`` (not a confusing plotting failure) if a requested name
        isn't in ``samples``.
    true_values : dict, optional
        ``{param_name: value}`` to mark with crosshair lines across the
        relevant panels (as in Starkey+2016 Figure 6) -- only meaningful
        for synthetic data with a known answer; omit for real fits.
    """
    missing = [name for name in param_names if name not in samples]
    if missing:
        raise KeyError(
            f"plot_corner: {missing} not found in samples -- available scalar "
            f"sites: {sorted(samples.keys())}"
        )

    n = len(param_names)
    if figsize is None:
        figsize = (2.4 * n, 2.4 * n)
    fig, axes = plt.subplots(n, n, figsize=figsize)
    if n == 1:
        axes = np.array([[axes]])

    arrays = {name: np.asarray(samples[name]) for name in param_names}
    n_chains = arrays[param_names[0]].shape[0]
    colours = [f"C{c}" for c in range(n_chains)]

    for i, name_i in enumerate(param_names):
        for j, name_j in enumerate(param_names):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                for c in range(n_chains):
                    ax.hist(
                        arrays[name_i][c], bins=30, histtype="step",
                        color=colours[c], density=True,
                    )
                if true_values and name_i in true_values:
                    ax.axvline(true_values[name_i], color="k", lw=1)
            else:
                for c in range(n_chains):
                    ax.scatter(
                        arrays[name_j][c], arrays[name_i][c],
                        s=4, alpha=0.3, color=colours[c], linewidths=0,
                    )
                if true_values and name_j in true_values:
                    ax.axvline(true_values[name_j], color="k", lw=1)
                if true_values and name_i in true_values:
                    ax.axhline(true_values[name_i], color="k", lw=1)

            if i == n - 1:
                ax.set_xlabel(name_j, fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(name_i, fontsize=9)
            elif j == 0 and i == 0:
                ax.set_yticklabels([])
            elif i != j:
                ax.set_yticklabels([])

    handles = [plt.Line2D([0], [0], color=colours[c], lw=2, label=f"chain {c}") for c in range(n_chains)]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Posterior corner plot")
    fig.tight_layout()
    return fig, axes


def plot_fourier_correlation(S_samples: np.ndarray, C_samples: np.ndarray, freqs: np.ndarray, figsize=(9, 4)):
    """Posterior correlation matrix of the driver's Fourier coefficients,
    ``S`` and ``C`` shown separately -- the scalable stand-in for a
    ``plot_corner`` on ``S``/``C``, which are a single vector-valued site
    each (from ``numpyro.plate("freq", n_freq)`` in ``model.py``, not
    ``n_freq`` individually-named scalar sites) with ``n_freq`` often in
    the tens, making an ``n_freq x n_freq`` scatter-matrix corner plot both
    unreadable and slow to render.

    A correlation heatmap answers the same underlying question a corner
    plot would -- are these coefficients behaving as the (non-centred)
    DRW prior assumes, roughly independent given ``sigma_drw``/``tau_drw``,
    or is there a problematic posterior correlation between frequencies --
    at a size that doesn't grow past one readable image regardless of
    ``n_freq``. Chains are pooled here (unlike ``plot_corner``'s per-chain
    colouring), since a correlation matrix is already a summary over
    samples; check chain agreement via ``plot_corner`` or
    ``plot_mcmc_diagnostics`` instead.

    Parameters
    ----------
    S_samples, C_samples : array, shape (n_samples, n_freq) or (n_chains, n_samples, n_freq)
        Posterior draws of the driver's sine/cosine Fourier coefficients
        (``ef.samples["S"]``/``["C"]``, or the ``_samples_by_chain``
        equivalent -- either is pooled into ``(n_total_samples, n_freq)``
        before computing the correlation matrix).
    freqs : (n_freq,) array
        Driver angular frequency grid (rad/day) -- ``EchoFit.freqs``,
        converted to cycles/day for axis labelling only (display, not the
        internal representation -- see ``plot_power_spectrum``'s
        docstring for why).

    Notes
    -----
    A frequency whose posterior samples have ~zero variance (the DRW prior
    can suppress the highest frequencies enough for this to happen, and a
    too-short warmup can leave a coefficient stuck near its initial value)
    would make a raw correlation coefficient a 0/0 division -- caught here
    and shown as exactly 0 (no defined correlation) rather than left as
    ``numpy``'s ``nan``, which would otherwise leave unexplained gaps in
    the heatmap and raise a ``RuntimeWarning``.
    """
    S = np.asarray(S_samples).reshape(-1, np.asarray(S_samples).shape[-1])
    C = np.asarray(C_samples).reshape(-1, np.asarray(C_samples).shape[-1])

    with np.errstate(invalid="ignore", divide="ignore"):
        corr_S = np.nan_to_num(np.corrcoef(S, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
        corr_C = np.nan_to_num(np.corrcoef(C, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)

    f_cycles = freqs / (2.0 * np.pi)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    tick_idx = np.linspace(0, len(f_cycles) - 1, min(6, len(f_cycles))).astype(int)
    for ax, corr, label in zip(axes, (corr_S, corr_C), ("S (sine)", "C (cosine)")):
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([f"{f_cycles[k]:.2g}" for k in tick_idx], rotation=90, fontsize=7)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([f"{f_cycles[k]:.2g}" for k in tick_idx], fontsize=7)
        ax.set_xlabel(r"$f$ [cycles/day]", fontsize=8)
        ax.set_title(f"{label} correlation ({len(freqs)} frequencies)", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="posterior correlation")
    fig.suptitle("Driver Fourier coefficient correlation")
    return fig, axes


def plot_bof(potential_energy: np.ndarray, checkpoint_every: Optional[int] = None, figsize=(8, 4)):
    """Badness-of-Fit trace, one line per chain -- the same diagnostic
    Starkey, Horne & Villforth (2016, MNRAS 456, 1960; arXiv:1511.06162)'s
    eq. 12 defines, ``BOF = chi**2 + sum(ln(sigma_i**2)) - 2*ln(P(Theta)) +
    const``, which is exactly ``2 * potential_energy`` up to that additive
    constant: NUTS's ``potential_energy`` is already
    ``-log(likelihood * prior)`` (a standard Gaussian log-likelihood's
    ``-2*ln`` is ``chi**2 + sum(ln(sigma_i**2)) + const``, matching eq. 12
    directly once doubled), so no separate BOF bookkeeping is needed --
    NUTS already computes this value on every step to decide whether to
    accept a proposal, it just isn't returned unless asked for (see
    ``inference.run_mcmc``/``run_mcmc_chunked``, which now request
    ``extra_fields=("potential_energy",)``).

    Should decrease (noisily) then flatten once a chain has converged;
    a chain that's still trending down at the end of the run hasn't
    finished burning in, and multiple chains whose BOF settles at visibly
    different levels is the same non-convergence signal
    ``plot_corner``/Gelman-Rubin R-hat would also flag.

    Parameters
    ----------
    potential_energy : array, shape (n_samples,) or (n_chains, n_samples)
        ``ef.extra_fields["potential_energy"]`` (single-chain/pooled) or
        ``ef.mcmc.get_extra_fields(group_by_chain=True)["potential_energy"]``
        (multi-chain, one line per chain).
    checkpoint_every : int, optional
        If given, draws a light vertical line at each checkpoint boundary
        (only meaningful for an ``EchoFit(title=...)`` checkpointed run,
        which is always single-chain -- see CLAUDE.md decision #6).
    """
    potential_energy = np.asarray(potential_energy)
    if potential_energy.ndim == 1:
        potential_energy = potential_energy[None, :]
    bof = 2.0 * potential_energy

    fig, ax = plt.subplots(figsize=figsize)
    n_chains, n_samples = bof.shape
    for c in range(n_chains):
        ax.plot(bof[c], lw=0.8, alpha=0.85, label=f"chain {c}" if n_chains > 1 else None)
    if checkpoint_every:
        for x in range(checkpoint_every, n_samples, checkpoint_every):
            ax.axvline(x, color="0.7", lw=0.6, zorder=0)
    ax.set_xlabel("sample")
    ax.set_ylabel("BOF (2 x potential energy)")
    ax.grid(alpha=0.6)
    if n_chains > 1:
        ax.legend(fontsize=8)
    ax.set_title("Badness of Fit vs. sample")
    fig.tight_layout()
    return fig, ax


def plot_optimise_restarts(
    restarts: Sequence[dict], samples: Dict[str, np.ndarray], param_names: Optional[Sequence[str]] = None,
    agreement_sd: float = 0.5, figsize=None,
):
    """Multi-start reproducibility of ``EchoFit.optimise()``: the direct-solve
    counterpart of overlaying several MCMC chains. The left panel shows how
    far above the best optimum each restart's own (Newton-polished) optimum
    sits in Badness of Fit (2 x potential); each other panel shows one
    parameter's Laplace posterior (histogram) with a line at every restart's
    optimum. Restarts that agree (every parameter within ``agreement_sd``
    posterior standard deviations of the best) are blue, others red.

    Parameters
    ----------
    restarts : list of dict
        ``ef.optimise_restarts``.
    samples : dict
        ``ef.samples`` (the Laplace posterior draws).
    param_names : sequence of str, optional
        Parameters to show; defaults to whichever of ``log_mdot``,
        ``inclination`` and ``sigma_drw`` exist.
    """
    if param_names is None:
        param_names = [p for p in ("log_mdot", "inclination", "sigma_drw") if p in samples]
    n = 1 + len(param_names)
    fig, axes = plt.subplots(1, n, figsize=figsize or (3.4 * n, 3.2))
    axes = np.atleast_1d(axes)
    colours = ["tab:blue" if r["agrees"] else "tab:red" for r in restarts]

    ax = axes[0]
    idx = [r["index"] for r in restarts]
    dbof = [2.0 * r["delta_potential"] for r in restarts]
    ax.bar(idx, np.nan_to_num(dbof, nan=0.0), color=colours)
    ax.set_xlabel("restart")
    ax.set_ylabel(r"$\Delta$BOF above best optimum")
    ax.set_xticks(idx)
    ax.grid(alpha=0.6, axis="y")

    for ax, name in zip(axes[1:], param_names):
        ax.hist(np.asarray(samples[name]), bins=40, color="0.8", density=True)
        for r, c in zip(restarts, colours):
            if name in r["values"]:
                ax.axvline(r["values"][name], color=c, lw=1.2, alpha=0.8)
        ax.set_xlabel(name)
        ax.set_yticks([])
        ax.grid(alpha=0.6)
    n_agree = sum(r["agrees"] for r in restarts)
    fig.suptitle(f"{n_agree} of {len(restarts)} restarts agree (within {agreement_sd:g} posterior sd)")
    fig.tight_layout()
    return fig, axes


def plot_disc_sed(result: dict, bands: Dict[str, dict], figsize=(10, 7.5)):
    """The disc SED, distance and H_0 analysis (:func:`pycream2.disc_sed.disc_sed_analysis`):
    flux-flux diagram, variable SED against the disc's predicted response
    spectrum, mean disc SED against the model disc at the fitted distance, and
    the H_0 posterior. Fluxes in mJy, corrected for Galactic extinction."""
    d, s = result["draws"], result["summary"]
    names, lam = d["names"], d["lam_rest"]
    colours = {n: wavelength_to_colour(bands[n]["wavelength"]) for n in names}
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Flux-flux: each band's flux against the driver, the fitted line, and the
    # disc's zero point X_0 (where the host band's model flux vanishes).
    ax = axes[0, 0]
    x_med = np.median(d["x"], axis=0)
    x0 = float(np.median(d["x0"]))
    grid = np.linspace(min(x0, x_med.min()), x_med.max(), 50)
    for k, n in enumerate(names):
        t = np.asarray(bands[n]["t"])
        ax.plot(np.interp(t, d["t_fine"], x_med), np.asarray(bands[n]["y"]) * d["conv"][k], ".", ms=2,
                color=colours[n], alpha=0.5)
        ax.plot(grid, (np.median(d["C"][k]) + np.median(d["S"][k]) * grid) * d["conv"][k], color=colours[n],
                lw=1.0, label=n)
    ax.axvline(x0, color="k", ls="--", lw=0.8, label=r"disc zero point $X_0$")
    ax.axvline(float(np.median(d["xmean"])), color="0.5", ls=":", lw=0.8, label=r"mean state $\langle X\rangle$")
    ax.set_xlabel(r"driver $X(t)$")
    ax.set_ylabel(r"$f_\nu$ (mJy)")
    ax.set_title("flux-flux decomposition")
    ax.legend(frameon=False, fontsize=6, ncol=2)

    def band_errorbar(ax, arr, **kw):
        lo, med, hi = np.percentile(arr, [16, 50, 84], axis=1)
        ax.errorbar(lam, med, [med - lo, hi - med], **kw)

    ax = axes[0, 1]
    band_errorbar(ax, d["variable"], fmt="o", ms=4, color="k", label="observed bright $-$ faint")
    ax.plot(lam, np.median(d["predicted_variable"], axis=1), "s--", ms=3, color="tab:red", lw=0.9,
            label="disc response, fitted $T(r)$")
    a_var, a_fit = s["alpha_var"], s["alpha"]
    ax.set_title(rf"variable SED: $\alpha_{{\rm var}}$ = {a_var[1]:.2f}, $\alpha_{{\rm fit}}$ = {a_fit[1]:.2f}")

    ax = axes[1, 0]
    band_errorbar(ax, d["total_mean"], fmt="o", ms=4, color="k", label="observed mean")
    ax.plot(lam, np.median(d["model_disc"], axis=1), "s--", ms=3, color="tab:blue", lw=0.9,
            label=rf"model disc at $D_L$ = {s['dl_mpc'][1]:.0f} Mpc")
    if d["distance_method"] == "flux_flux":
        band_errorbar(ax, d["disc_mean"], fmt="D", ms=3, color="tab:blue", mfc="white", label="flux-flux disc")
    host = np.median(d["constant"], axis=1)
    ax.plot(lam[host > 0], host[host > 0], "^:", ms=4, color="tab:green", lw=0.8, label="host (mean $-$ disc)")
    ax.set_title(f"mean SED: disc + host ({s['host_band']} has no host)")
    from matplotlib.ticker import FuncFormatter

    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plain)
            axis.set_minor_formatter(plain)
        ax.set_xlabel(r"rest wavelength (Å)")
        ax.set_ylabel(r"$f_\nu$ (mJy)")
        ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    h0 = d["h0"][np.isfinite(d["h0"])]
    ax.hist(h0, bins=40, color="0.6")
    lo, med, hi = s["h0"]
    ax.axvline(med, color="k", lw=1.0)
    ax.axvspan(lo, hi, color="0.85", zorder=0)
    ax.axvline(70.0, color="tab:green", ls="--", lw=0.9, label=r"70 km s$^{-1}$ Mpc$^{-1}$")
    ax.set_xlabel(r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)")
    ax.set_title(rf"$H_0$ = {med:.0f}$^{{+{hi - med:.0f}}}_{{-{med - lo:.0f}}}$, "
                 rf"$D_L$ = {s['dl_mpc'][1]:.0f} Mpc (z = {s['redshift']:.4g})")
    ax.legend(frameon=False, fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=0.6)
    fig.tight_layout()
    return fig, axes
