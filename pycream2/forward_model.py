"""
forward_model.py
=================

Deterministic pieces of the reverberation-mapping forward model:

1. ``lag_scaling``       -- physically motivated mean-lag scaling law.
2. ``response_function`` -- causal, positive transfer function psi(tau).
3. ``transfer_coeffs``   -- Fourier cosine/sine transform of psi on a fixed
                            frequency grid (used to convolve analytically).
4. ``compute_echo``      -- evaluate the convolution of a Fourier-series
                            driver with psi at arbitrary observation times.

Design notes
------------
The driver is represented as a truncated Fourier series::

    X(t) = sum_k [ S_k sin(w_k t) + C_k cos(w_k t) ]

Because the driver is *exactly* a sum of sinusoids, the convolution with the
response function has a closed form in terms of the response function's
Fourier transform evaluated at each driver frequency w_k:

    A_k = int psi(tau) cos(w_k tau) dtau
    B_k = int psi(tau) sin(w_k tau) dtau

    int psi(tau) X(t - tau) dtau
        = sum_k [ sin(w_k t) (S_k A_k + C_k B_k)
                  + cos(w_k t) (C_k A_k - S_k B_k) ]

This avoids ever looping over observation times and tau jointly (an
O(n_obs * n_tau) operation); instead the expensive tau-integral is done once
per frequency (O(n_freq * n_tau)) to get (A_k, B_k), and then the echo at any
set of times is a single O(n_obs * n_freq) matrix contraction. Everything is
written with ``jax.numpy`` so it vectorises and is differentiable / jittable
end to end, which is what NumPyro's NUTS sampler needs.

The response function is intentionally isolated in ``response_function`` so
it can be swapped for a different parametric family later without touching
``compute_echo`` or the NumPyro model.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf as _erf

# ----------------------------------------------------------------------
# Pivot / calibration constants for the lag-scaling law.
# These set the overall day-scale and are not inferred.
# ----------------------------------------------------------------------
TAU0_DAYS = 1.0          # normalisation: lag (days) at the pivot point below
M_BH_PIVOT = 1.0e8       # pivot black hole mass, solar masses
MDOT_PIVOT = 1.0         # pivot dimensionless accretion rate (10**log_mdot)
LAMBDA_PIVOT = 5000.0    # pivot wavelength, Angstrom



# Default width of thin_disk_response's smoothing, a Gaussian in ln(tau)
# (5 per cent in delay at every delay). Causal by construction, unlike the
# Gaussian in tau (smoothing_frac/smoothing_days) it replaced as the default in
# September 2026, which spread the near-vertical onset of the response across
# tau = 0; see docs/cream_response_comparison.md.
DEFAULT_SMOOTHING_LOG = 0.05

def lag_scaling(log_mdot, wavelength, M_BH):
    """Mean disk-reprocessing lag, in days.

    Follows the standard lamppost/thin-disk scaling for the radius at which
    the disk temperature matches a given observing wavelength,
    R(lambda) ~ M_BH^(2/3) Mdot^(1/3) lambda^(4/3), converted to a light
    travel time. Black hole mass is a *fixed input*, not inferred.

    Parameters
    ----------
    log_mdot : array_like
        log10 of the (dimensionless) mass accretion rate. Inferred.
    wavelength : array_like
        Observing wavelength in Angstrom (rest frame).
    M_BH : float
        Black hole mass in solar masses. Fixed, not inferred.

    Returns
    -------
    tau_mean : array_like
        Mean lag in days, same broadcast shape as (log_mdot, wavelength).
    """
    mdot = 10.0 ** log_mdot
    tau_mean = (
        TAU0_DAYS
        * (M_BH / M_BH_PIVOT) ** (2.0 / 3.0)
        * (mdot / MDOT_PIVOT) ** (1.0 / 3.0)
        * (wavelength / LAMBDA_PIVOT) ** (4.0 / 3.0)
    )
    return tau_mean


def _skew_normal_pdf(x, loc, scale, alpha):
    """Standard skew-normal pdf (unnormalised area beyond truncation)."""
    z = (x - loc) / scale
    phi = jnp.exp(-0.5 * z ** 2) / jnp.sqrt(2.0 * jnp.pi)
    Phi = 0.5 * (1.0 + _erf(alpha * z / jnp.sqrt(2.0)))
    return (2.0 / scale) * phi * Phi


def response_function(
    tau_grid,
    log_mdot,
    wavelength,
    inclination,
    M_BH,
    width_frac: float = 0.35,
    skew_offset: float = 1.0,
    skew_incl_slope: float = 4.0,
):
    """Causal, positive transfer function psi(tau, lambda; theta).

    A skew-normal response, truncated to tau >= 0 and re-normalised on the
    supplied ``tau_grid`` so that ``trapz(psi, tau_grid) == 1``. The mean lag
    is set purely by :func:`lag_scaling` (accretion rate, wavelength, fixed
    mass). Inclination controls only the *skewness*, not the mean lag, per
    the physical requirement that inclination reshapes the iso-delay surface
    without changing the mean reprocessing radius.

    Parameters
    ----------
    tau_grid : array_like, shape (n_tau,)
        Non-negative lag grid (days) the response is evaluated / normalised
        on.
    log_mdot, wavelength, inclination : array_like
        Physical parameters. ``inclination`` in degrees, 0 = face-on.
    M_BH : float
        Fixed black hole mass (solar masses).
    width_frac : float
        Response width as a fraction of the mean lag.
    skew_offset, skew_incl_slope : float
        alpha = skew_offset + skew_incl_slope * sin(inclination); this keeps
        the response symmetric-ish face-on and increasingly skewed at high
        inclination, without touching the mean lag.

    Returns
    -------
    psi : array_like, shape (n_tau,)
        Normalised response evaluated at ``tau_grid``, zero for tau < 0.
    """
    tau_mean = lag_scaling(log_mdot, wavelength, M_BH)
    scale = jnp.clip(width_frac * tau_mean, 1e-3, None)
    alpha = skew_offset + skew_incl_slope * jnp.sin(jnp.deg2rad(inclination))

    raw = _skew_normal_pdf(tau_grid, loc=tau_mean, scale=scale, alpha=alpha)
    raw = jnp.where(tau_grid >= 0.0, raw, 0.0)

    area = jnp.trapezoid(raw, tau_grid) if hasattr(jnp, "trapezoid") else jnp.trapz(raw, tau_grid)
    psi = raw / jnp.clip(area, 1e-12, None)
    return psi


def tophat_response_free(tau_grid, tau_mean, width_frac: float = 0.3, edge_softness_frac: float = 0.15):
    """Causal top-hat-*like* response centred directly on ``tau_mean``, with
    smoothed (not hard) edges -- see "Why not a literal top-hat" below.

    Unlike :func:`response_function`, ``tau_mean`` is taken as given rather
    than derived from ``lag_scaling(log_mdot, wavelength, M_BH)`` -- this is
    the response used for a band in "free" lag mode (see
    ``model.reverberation_model``'s ``bands[...]["lag_mode"]``), where the
    lag itself is an independently inferred parameter rather than tied to
    the other bands through the shared thin-disk scaling law. That
    independence is exactly what makes a *single* such band's lag
    unidentifiable from its light curve alone (a global shift of the driver
    and an equal shift of the lag leave the data unchanged -- see
    CLAUDE.md); a fit using this response for any band should also register
    a driver light curve (:meth:`~pycream2.echofit.EchoFit.add_driver_lightcurve`)
    to anchor the absolute lag scale, or tie multiple free-lag bands
    together some other way.

    Why not a literal top-hat: a hard ``jnp.where(|tau - tau_mean| <=
    half_width, 1, 0)`` box has **exactly zero gradient** with respect to
    ``tau_mean`` everywhere except the measure-zero edge (autodiff doesn't
    backprop through a comparison's operands) -- since ``tau_mean`` is a
    ``numpyro.sample`` site here, NUTS would see zero gradient signal and
    be unable to move it at all (confirmed directly: ``jax.grad`` of the
    hard version w.r.t. ``tau_mean`` is identically ``0.0``). This uses a
    sigmoid-smoothed edge instead -- still a box in the limit
    ``edge_softness_frac -> 0``, but differentiable everywhere.

    Parameters
    ----------
    tau_grid : array_like, shape (n_tau,)
    tau_mean : array_like
        The lag (days), e.g. a ``numpyro.sample`` site rather than a fixed
        constant.
    width_frac : float
        Half-width as a fraction of ``tau_mean`` (fixed, not inferred --
        matches ``response_function``'s ``width_frac`` convention). Note
        this width-scales-with-the-lag convention means the shift
        degeneracy above is not perfectly exact when ``tau_mean`` itself
        changes (the width changes slightly with it, unlike a true
        ``psi'(tau) = psi(tau + Delta)`` shift, which preserves width) --
        see ``tests/test_shift_degeneracy.py`` for the width-independent
        version used to verify the degeneracy claim exactly.
    edge_softness_frac : float
        Edge transition width as a fraction of the half-width (fixed, not
        inferred). Smaller is closer to a true top-hat but with a narrower
        region of usable gradient for NUTS to find the edges through.

    Returns
    -------
    psi : array_like, shape (n_tau,)
        Normalised response evaluated at ``tau_grid``, ~zero well outside
        ``[tau_mean - half_width, tau_mean + half_width]`` and for tau < 0.
    """
    half_width = jnp.clip(width_frac * tau_mean, 1e-3, None)
    edge_softness = jnp.clip(edge_softness_frac * half_width, 1e-3, None)
    left_edge = tau_mean - half_width
    right_edge = tau_mean + half_width
    rising = jax.nn.sigmoid((tau_grid - left_edge) / edge_softness)
    falling = jax.nn.sigmoid((right_edge - tau_grid) / edge_softness)
    raw = rising * falling
    raw = jnp.where(tau_grid >= 0.0, raw, 0.0)

    trapz = jnp.trapezoid if hasattr(jnp, "trapezoid") else jnp.trapz
    area = trapz(raw, tau_grid)
    psi = raw / jnp.clip(area, 1e-12, None)
    return psi


def lognormal_response(tau_grid, median, width_dex):
    """Causal, area-normalised log-normal response: the delay distribution of
    an extended reprocessor such as the broad-line region's diffuse
    continuum emission (Cackett, Zoghbi & Ulrich 2022), with ``ln tau``
    Gaussian about ``ln(median)``.

    Parameters
    ----------
    tau_grid : array_like, shape (n_tau,)
        Lag grid (days); zero for ``tau <= 0``.
    median : float
        Median delay (days), ``exp(M)`` in Cackett et al.'s notation.
    width_dex : float
        Standard deviation of ``log10 tau`` (dex), ``S / ln 10`` in theirs.
        The mean delay is ``median * exp((width_dex ln 10)**2 / 2)``.

    The width is in dex rather than days so that it describes the reprocessor's
    fractional extent, which is what a broad-line region's geometry sets.
    Differentiable in both parameters (no hard edges; see CLAUDE.md decision
    #7's gradient trap).
    """
    sigma_ln = jnp.clip(width_dex, 1e-3, None) * jnp.log(10.0)
    positive = tau_grid > 0.0
    log_tau = jnp.log(jnp.where(positive, tau_grid, 1.0))
    z = (log_tau - jnp.log(jnp.clip(median, 1e-6, None))) / sigma_ln
    raw = jnp.where(positive, jnp.exp(-0.5 * z ** 2) / jnp.where(positive, tau_grid, 1.0), 0.0)
    trapz = jnp.trapezoid if hasattr(jnp, "trapezoid") else jnp.trapz
    area = trapz(raw, tau_grid)
    return raw / jnp.clip(area, 1e-12, None)


def mix_diffuse_continuum(psi, tau_grid, fraction, median, width_dex):
    """A band's response with a diffuse-continuum component:
    ``(1 - fraction) * psi + fraction * lognormal_response(...)``. Both parts
    are area-normalised, so ``fraction`` is the diffuse component's share of
    the band's integrated response (its integrated response is ``fraction``
    times the band's gain)."""
    return (1.0 - fraction) * psi + fraction * lognormal_response(tau_grid, median, width_dex)


def legendre_background_basis(t, t_range, order):
    """Slowly varying background basis: Legendre polynomials ``P_1 .. P_order``
    of time mapped to ``[-1, 1]`` over ``t_range = (t_start, t_end)`` (the whole
    campaign, shared by every light curve), shape ``(len(t), order)``. ``P_0``
    is left out: the constant is each light curve's own offset ``C``.
    Legendre rather than plain powers keeps the columns close to orthogonal
    over the campaign, so the coefficients are well conditioned."""
    t = jnp.asarray(t)
    x = 2.0 * (t - t_range[0]) / (t_range[1] - t_range[0]) - 1.0
    cols, p_prev, p_cur = [], jnp.ones_like(x), x
    for k in range(1, order + 1):
        cols.append(p_cur)
        # Bonnet's recursion: (k + 1) P_{k+1} = (2k + 1) x P_k - k P_{k-1}.
        p_prev, p_cur = p_cur, ((2 * k + 1) * x * p_cur - k * p_prev) / (k + 1)
    return jnp.stack(cols, axis=1) if cols else jnp.zeros((x.shape[0], 0))


# ----------------------------------------------------------------------
# Physical constants for thin_disk_response. Only used to convert the
# black hole mass into an inner (ISCO) radius -- the overall lag scale
# still comes from lag_scaling, see thin_disk_response's docstring.
# ----------------------------------------------------------------------
_G = 6.674e-11                              # m^3 kg^-1 s^-2
_C_LIGHT = 2.99792458e8                     # m / s
_M_SUN = 1.98855e30                         # kg
_LIGHT_DAY_M = _C_LIGHT * 86400.0           # metres per light-day
_HC_OVER_K_ANGSTROM = 1.4387773538277204e8  # hc / k_B, in Angstrom * Kelvin
_WIEN_X_PEAK = 4.965114231744276            # solution of x = 5(1 - e^-x)
_WIEN_B_ANGSTROM_KELVIN = _HC_OVER_K_ANGSTROM / _WIEN_X_PEAK


def _schwarzschild_radius_light_days(M_BH):
    """One Schwarzschild radius, R_s = 2GM/c^2, in light-days."""
    r_g_m = _G * (M_BH * _M_SUN) / _C_LIGHT ** 2
    return 2.0 * r_g_m / _LIGHT_DAY_M


def disk_t1_kelvin(log_mdot, M_BH):
    """Temperature (Kelvin) of the disk's power law at one light-day, ``T_1``,
    for a given ``log_mdot``.

    ``lag_scaling`` puts the radius where the disk temperature equals the
    Wien temperature ``b / wavelength`` at ``tau_ref(wavelength)``, which
    scales as ``wavelength**(4/3)``; for ``T = T_1 r**(-3/4)`` that makes
    ``T_1 = (b / wavelength) * tau_ref**(3/4)`` the same at every wavelength.
    This is the one place ``log_mdot`` becomes a temperature, so every
    temperature slope (``viscous_slope``) shares the same ``T_1`` for the
    same ``log_mdot``.
    """
    tau_ref = lag_scaling(log_mdot, LAMBDA_PIVOT, M_BH)
    return _WIEN_B_ANGSTROM_KELVIN / LAMBDA_PIVOT * tau_ref ** 0.75


def wien_radius(log_mdot, wavelength, M_BH, viscous_slope=0.75):
    """Radius (light-days) where ``T_1 r**(-viscous_slope)`` equals the Wien
    temperature ``b / wavelength``: ``lag_scaling``'s ``tau_ref`` exactly at
    the standard slope 0.75, and scaling as ``wavelength**(1/viscous_slope)``
    in general."""
    return (disk_t1_kelvin(log_mdot, M_BH) * wavelength / _WIEN_B_ANGSTROM_KELVIN) ** (1.0 / viscous_slope)


def _viscous_t4_shape(r_safe, r_in, viscous_slope):
    """Viscous-disk ``T**4(r) / T_1**4``: the power law ``r**(-4 slope)``
    (``r`` in light-days) times the zero-torque inner-boundary factor
    ``1 - sqrt(r_in / r)``. The one formula shared by
    :func:`disk_temperature_profile` and :func:`thin_disk_response`'s viscous
    term, so it exists in exactly one place.

    Takes ``r_safe`` already clipped by the caller rather than clipping
    internally, since the two callers need genuinely different clipping
    strategies at ``r <= r_in`` -- ``thin_disk_response`` needs a smooth,
    differentiable-in-inclination cutoff (its ``r_star`` depends on a sampled
    parameter, handled via a separate sigmoid mask applied to its result)
    where a hard clip would reintroduce the zero-gradient trap documented on
    that mask; :func:`disk_temperature_profile` doesn't sample ``r``, so a
    plain hard clip at ``r_in`` is fine and simpler there.

    Before September 2026 this was normalised at ``lag_scaling``'s
    ``tau_ref(wavelength)`` for every slope, which for any slope other than
    0.75 made the disk's temperature at a fixed radius depend on the
    observing wavelength.
    """
    return r_safe ** (-4.0 * viscous_slope) * (1.0 - jnp.sqrt(r_in / r_safe))


def disk_temperature_profile(r, log_mdot, wavelength, M_BH, viscous_slope: float = 0.75):
    """Axisymmetric Shakura-Sunyaev viscous disk temperature (Kelvin) as a
    function of radius ``r`` (light-days):
    ``T**4 = T_1**4 r**(-4 viscous_slope) (1 - sqrt(r_in / r))``, with
    ``T_1`` from :func:`disk_t1_kelvin`.

    Shares its actual formula with ``thin_disk_response``'s viscous term
    via :func:`_viscous_t4_shape` (only the clipping-at-the-ISCO and
    irradiation-mixing steps around it differ, per-caller -- see that
    function's docstring), rather than the two duplicating it, since
    ``thin_disk_response`` evaluates on a 2-D ``(tau, phi)`` grid via the
    disk's light-travel-time surface, not a plain radius -- this is for
    callers that just want "the temperature at radius r" directly, e.g. a
    disk visualisation (see ``scripts/make_fit_animation.py``), independent
    of any particular response-function evaluation. Ignores the optional
    lamppost-irradiation term (``thin_disk_response``'s
    ``include_irradiation``), which is azimuthally asymmetric and not
    meaningful for a purely radius-dependent profile.

    Parameters
    ----------
    r : array_like
        Radius (light-days), any shape. Clipped at the ISCO (3
        Schwarzschild radii) from below -- there's no disk material inside
        it, so no temperature to report.
    log_mdot, M_BH, viscous_slope
        As in ``thin_disk_response``.
    wavelength : float
        Kept for backward compatibility; the profile no longer depends on it
        (the disk's temperature is a property of the disk, not of the band
        it's observed in).
    """
    r_in = 3.0 * _schwarzschild_radius_light_days(M_BH)
    r_safe = jnp.clip(r, r_in, None)
    t4_shape = _viscous_t4_shape(r_safe, r_in, viscous_slope)
    return disk_t1_kelvin(log_mdot, M_BH) * jnp.clip(t4_shape, 1e-30, None) ** 0.25


def thin_disk_response(
    tau_grid,
    log_mdot,
    wavelength,
    inclination,
    M_BH,
    viscous_slope: float = 0.75,
    include_irradiation: bool = False,
    irradiation_slope: float = 0.75,
    irradiation_weight: float = 0.5,
    lamppost_height_rs: float = 3.0,
    n_phi: int = 200,
    smoothing_frac: float | None = None,
    smoothing_days: float | None = None,
    smoothing_log: float = DEFAULT_SMOOTHING_LOG,
    delay_lamppost_height: bool = True,
):
    """Causal, area-normalised transfer function from thin-disk reprocessing
    theory -- an analytic reduction of the 2-D (radius, azimuth) disk
    integral to a single 1-D integral over azimuth, exact for any tau_grid
    point (no radial grid, no truncated domain).

    Physics follows Starkey, Horne & Villforth (2016, MNRAS 456, 1960;
    arXiv:1511.06162, the CREAM paper), which in turn cites Cackett, Horne &
    Winkler (2007) for the response function derivation, and the author's
    PhD-era Fortran CREAM code (``pycecream``'s ``cream_f90.f90``,
    ``tfbx``/``tr4visc``/``tr4irad``) for the original numerical (Monte
    Carlo) implementation this replaces:

    * Shakura-Sunyaev viscous + lamppost-irradiation temperature profile
      (Starkey+2016 eq. 2), ``T**4(r) = 3GM*Mdot/(8 pi sigma r**3) * (1 -
      sqrt(r_in/r))  +  L_b*(1-a)*h_x / (4 pi sigma x**3)``, ``x = sqrt(r**2
      + h_x**2)``, with ``r_in`` the innermost stable circular orbit
      (3 Schwarzschild radii for a non-spinning black hole) and ``h_x`` the
      lamppost height above the disk plane (3 Schwarzschild radii by
      default, Starkey+2016's own illustrative value).
    * The disk light-travel-time delay surface, including the lamppost's
      height: ``tau(r, phi) = sqrt(r**2 + h_x**2) + h_x cos(inclination) +
      r cos(phi) sin(inclination)``, relative to the direct ray. Starkey+2016
      eq. 5 and CREAM's ``tfbx`` (and this code before September 2026) use
      ``h_x = 0`` here, ``tau = r (1 + cos(phi) sin(inclination))``; the
      height only enters them through the irradiation. With it, nothing
      responds before the shortest lamppost-disk-observer path, and the mean
      delay is ``<d> + h_x cos(inclination)`` rather than exactly
      inclination-independent (decision #4), a shift of ~h_x ~ 0.01 d for
      NGC 5548. This is what gives the response a genuine,
      inclination-driven skew and a causal onset, rather than
      :func:`response_function`'s ad-hoc skew-normal shape.
    * A response weight ``dB_lambda/dT * dT/dL_x``: the Planck-function
      derivative, ``dB/dT ~ X**2 * e**X / (e**X - 1)**2`` with
      ``X = hc / (k * lambda * T(r))``, times the temperature change a small
      change in the lamppost luminosity causes. The lamppost deposits
      ``d(T**4) ~ dL_x * h_x / x**3`` per unit area (``x = sqrt(r**2 +
      h_x**2)``, Starkey+2016 eq. 2's irradiation term), so
      ``dT/dL_x ~ h_x / (x**3 T**3)``, and the full weight is
      ``X**5 * e**X / (e**X - 1)**2 * h_x / x**3``. The ``h_x / x**3``
      dilution was missing before September 2026: without it the outer
      disk responded far too strongly, mean delays came out ~1.9 times the
      Wien radius, and the temperature implied by a fit's delays was ~3 times
      the model's own ``T_1``.

    The analytic reduction (written here for ``h_x = 0`` in the delay; with
    the height the delta function's roots at fixed ``phi`` are those of a
    quadratic in ``r``, up to two on the near side, each weighted by
    ``r / |dtau/dr|`` -- see the comment in the code): the full disk integral is
    ``psi_raw(tau) = int_0^{2pi} int weight(r) delta(tau - tau(r,phi)) r dr dphi``.
    At fixed ``phi``, ``tau(r, phi)`` is *linear* in ``r`` (unlike at fixed
    ``r``, where it is two-valued in ``phi``), so the delta function has a
    single root, ``r*(phi, tau) = tau / (1 + cos(phi) sin(inclination))``,
    with Jacobian ``d(tau)/dr = 1 + cos(phi) sin(inclination)`` -- collapsing
    the radial integral exactly (no truncation, no radial grid) and leaving

    ``psi_raw(tau) = tau * int_0^{2pi} weight(r*(phi, tau)) / (1 + cos(phi)
    sin(inclination))**2 dphi``,

    a plain 1-D integral over a *fixed* ``phi`` grid for every ``tau_grid``
    point, evaluated with a uniform-grid Riemann sum (``phi`` is periodic,
    so this is spectrally accurate -- no ``trapz`` edge correction needed).
    This replaces the earlier two-radial-grid, Gaussian-kernel-deposit
    implementation entirely: that approach needed careful control of a
    radial domain cutoff and a smoothing bandwidth to avoid visible
    quadrature artefacts (worst exactly at high inclination, where the
    naive radial domain a face-on response needs blows up by ~2-3 orders of
    magnitude -- see git history / CLAUDE.md for that whole saga), none of
    which this needs, since there is no radial grid to under-resolve.
    Faster too: removing the radial dimension drops the cost from
    ``O(n_r * n_phi * n_tau)`` to ``O(n_phi * n_tau)``, with a smaller
    ``n_phi`` sufficient since it's now exact rather than an approximate
    deposit (confirmed: this and the old grid-based version agree away from
    the old version's known artefacts).

    One thing this exact reduction does *not* remove: the Fortran's own
    Gaussian smoothing, applied not as clean-up of a numerical artefact but
    as part of the physical model itself -- each disk element's response is
    deposited not as a single delta-function contribution at its own exact
    delay, but spread with a Gaussian of width ``sig_gaus`` (``tfbx``, set
    once, globally, to the output tau grid's own bin spacing, ``dtau``)
    across several neighbouring tau bins (``tfbx`` lines ~12604-12714: the
    nested radius/azimuth loop computes each grid point's ``taulag`` and
    ``resp``, then spreads ``resp`` across ``iblo:ibhi`` with Gaussian
    weights ``exp(-((taulag-taubin)/sig_gaus)**2 / 2)``). Confirmed present
    in both of the Fortran's response-function subroutines (``tfbx``:
    ``sig_gaus = dtau``; the older ``tfb``: ``sig_g = dtau/2``). Since
    Gaussian smoothing is linear and ``sig_gaus`` is a single constant for
    the whole disk (not varying grid point to grid point), smoothing each
    element's contribution individually and then summing is mathematically
    identical to computing the exact unsmoothed integral above and
    convolving *that* once with the same Gaussian -- not an approximation
    of the per-element version, just a cheaper way to compute the same
    result. ``smoothing_frac``/``smoothing_days`` (below) does exactly
    that, with a width chosen relative to the response's own characteristic
    lag rather than the Fortran's literal (and, for a fine ``tau_grid``,
    negligible) ``sig_gaus = dtau``. Passing ``smoothing_days=0.0`` (the
    behaviour before this parameter was added back) leaves a real,
    physically-meaningful feature unsmoothed: at high inclination the
    near-side tangent line contributes a genuine grazing-incidence
    enhancement right at tau=0 (confirmed a single, real local maximum
    there, not a numerical spike), which then decays through a distinctly
    faster initial rate than the disk's main-body contribution further out
    -- not two separate humps with a dip between them (checked directly:
    no interior local minimum), but an abrupt-enough change in decay rate
    to look like a separate feature (a "shoulder") rather than the single,
    smoothly-varying skewed peak the smoothed Starkey+2016 Figure 3 shows.

    Absolute normalisation: the temperature profile is
    ``T**4 = T_1**4 r**(-4 viscous_slope) (1 - sqrt(r_in / r))`` (``r`` in
    light-days), with ``T_1`` from :func:`disk_t1_kelvin`, which takes it
    from :func:`lag_scaling`'s Wien radius rather than an independently
    derived Eddington-ratio-to-accretion-rate conversion (which would give
    ``log_mdot`` a second, incompatible meaning depending which response
    function a band uses). ``T_1`` does not depend on ``viscous_slope``, so
    ``(log_mdot, viscous_slope)`` map one-to-one onto Starkey et al. (2017)'s
    ``(T_1, alpha)``. Only the disk's *inner* edge (the ISCO) and the
    lamppost height use real physical constants (G, c, M_sun), since that
    conversion needs no separate accretion-rate calibration.

    Parameters
    ----------
    tau_grid : array_like, shape (n_tau,)
    log_mdot, wavelength, inclination : array_like
        Same meaning and units as in :func:`response_function`.
    M_BH : float
        Fixed black hole mass (solar masses).
    viscous_slope : float or array_like
        Temperature-radius power-law index ``alpha`` in ``T ~ r**(-alpha)``
        (0.75 = standard thin disk, matching ``lag_scaling``'s
        ``wavelength**(4/3)``). Can be a sampled parameter
        (``EchoFit(fit_temperature_slope=True)``); delays then scale as
        ``wavelength**(1/alpha)``.
    include_irradiation : bool
        If True, mix in the lamppost-irradiation temperature component
        (``irradiation_slope``/``lamppost_height_rs``, weighted by
        ``irradiation_weight``) instead of a single-component viscous disk.
    irradiation_slope : float
        Only used if ``include_irradiation``; the irradiation term's own
        power-law index far from the lamppost (where ``T_irr**4 ~
        1/r**(4*irradiation_slope)``, matching eq. 3's ``r >> h_x`` limit
        at the standard 0.75).
    irradiation_weight : float
        Fraction of T**4 attributed to irradiation vs viscous heating at
        the characteristic radius, if ``include_irradiation``. A
        phenomenological mixing knob (not derived from a physical
        Eddington ratio/efficiency, for the same reason ``lag_scaling`` is
        reused rather than an independent Mdot calibration above).
    lamppost_height_rs : float
        Lamppost height above the disk plane, in Schwarzschild radii. Only
        affects the irradiation term's shape near ``r ~ h_x`` (it reduces
        to a pure ``1/r**3``-like power law for ``r >> h_x``); the delay
        surface itself does not depend on it, matching Starkey+2016 eq. 5.
    n_phi : int
        Azimuthal quadrature resolution (fixed, not inferred). Higher is
        more accurate and more expensive; unlike the old radial-grid
        implementation, this converges quickly since it's evaluating a
        smooth periodic integrand exactly, not depositing samples onto a
        histogram.
    smoothing_log : float
        Width, in ``ln tau``, of the default smoothing: a local Gaussian
        average in ``ln tau`` (a fixed fractional width at every delay, 5
        per cent by default, ``DEFAULT_SMOOTHING_LOG``). It represents the
        finite extent of the region responding at each delay, as CREAM's
        per-element Gaussian deposit did, but is causal by construction:
        nothing is moved below zero lag, so the response starts at zero.
        ``0.0`` gives the exact integral.
    smoothing_frac, smoothing_days : float, optional
        The older smoothing, a Gaussian in ``tau``, used instead of the
        log-normal one when either is given: ``smoothing_days`` is its width
        in days (CREAM used the lag-grid spacing), ``smoothing_frac`` its
        width as a fraction of the Wien radius (:func:`wien_radius`).
        ``smoothing_days=0.0`` gives the exact integral. It was the default
        until September 2026 (``smoothing_frac=0.4``, then ``0.1``), but a
        Gaussian in ``tau`` spreads the response's near-vertical onset across
        ``tau = 0``: the smoothed response then had its peak at zero lag for
        inclined disks, where the true response is zero, and its width (~0.5
        d at 5000 A for NGC 5548) blurred the near-peak shape that carries the
        inclination (``docs/cream_response_comparison.md``).

    delay_lamppost_height : bool
        Include the lamppost's height in the delay surface (default). ``False``
        gives CREAM's delay, ``r (1 + sin i cos phi)``, for comparisons; the
        height still enters the irradiation either way.

    Returns
    -------
    psi : array_like, shape (n_tau,)
        Normalised response evaluated at ``tau_grid``, zero for tau < 0.
    """
    r_wien = wien_radius(log_mdot, wavelength, M_BH, viscous_slope)
    rs = _schwarzschild_radius_light_days(M_BH)
    r_in = 3.0 * rs
    hx = lamppost_height_rs * rs
    # Height used in the delay; 0 reproduces CREAM's (and this code's before
    # September 2026) delay surface, r (1 + sin i cos phi), for comparisons.
    hd = hx if delay_lamppost_height else 0.0

    incl_rad = jnp.deg2rad(jnp.clip(inclination, 0.0, 89.9))
    sininc, cosinc = jnp.sin(incl_rad), jnp.cos(incl_rad)
    phi = (jnp.arange(n_phi) + 0.5) * (2.0 * jnp.pi / n_phi)
    dphi = 2.0 * jnp.pi / n_phi
    s_phi = (sininc * jnp.cos(phi))[None, :]  # (1, n_phi)
    one_m_s2 = jnp.clip(1.0 - s_phi ** 2, 1e-6, None)

    # Delay of the disk element at (r, phi), relative to the direct ray from
    # the lamppost at height hx: tau = d + hx cos(i) + r sin(i) cos(phi), with
    # d = sqrt(r**2 + hx**2) (CREAM, and this code before September 2026, used
    # hx = 0 in the delay: tau = r (1 + sin(i) cos(phi))). At fixed phi the
    # delta function in the disk integral picks out the radii with that delay:
    # the roots of (1 - s**2) r**2 + 2 A s r + hx**2 - A**2 = 0, with
    # A = tau - hx cos(i) and s = sin(i) cos(phi),
    #     r = (-A s +/- sqrt(A**2 - hx**2 (1 - s**2))) / (1 - s**2),
    # kept where r >= 0 and d = A - s r > 0 (squaring admits spurious roots).
    # On the far side (s >= 0) only "+" survives; on the near side the delay
    # first falls then rises with r, so a delay can come from two radii, and
    # each contributes w(r) r / |d tau/d r| with d tau/d r = r/d + s. For
    # hx -> 0 this is exactly the old tau w(r*) / (1 + s)**2.
    tau_pos = jnp.clip(tau_grid, 0.0, None)[:, None]  # (n_tau, 1)
    A = tau_pos - hd * cosinc
    disc = A ** 2 - hd ** 2 * one_m_s2
    sq = jnp.sqrt(jnp.where(disc > 0.0, disc, 1.0))
    t1_kelvin = disk_t1_kelvin(log_mdot, M_BH)
    r_peak = r_in * ((4.0 * viscous_slope + 0.5) / (4.0 * viscous_slope)) ** 2

    def contribution(r_root):
        valid = (disc > 0.0) & (r_root > 0.0) & (A - s_phi * r_root > 0.0)
        r_root = jnp.where(valid, r_root, r_in)
        # Evaluate the temperature no further in than the ISCO, where r**-3 and
        # the inner-boundary factor are finite; the sigmoid mask below switches
        # the region inside the temperature peak off. Clamping at ~0 instead
        # overflowed float32 there, and the overflow times the clamp's zero
        # gradient made every inclination gradient NaN.
        r_safe = jnp.where(r_root > r_in, r_root, r_in)
        # T**4(r) / T_1**4 -- shared with disk_temperature_profile, see
        # _viscous_t4_shape.
        t4_visc = _viscous_t4_shape(r_safe, r_in, viscous_slope)
        d_safe = jnp.sqrt(r_safe ** 2 + hx ** 2)
        if include_irradiation:
            # Irradiation's share of T**4 is irradiation_weight at the Wien radius.
            x_ref = jnp.sqrt(r_wien ** 2 + hx ** 2)
            t4_irad = (x_ref / d_safe) ** 3 * _viscous_t4_shape(r_wien, r_in, viscous_slope)
            t4_shape = irradiation_weight * t4_irad + (1.0 - irradiation_weight) * t4_visc
        else:
            t4_shape = t4_visc
        # Only take the fourth root where the disk is actually warm (the double
        # jnp.where keeps every derivative finite): clipping T**4 at a tiny floor
        # and rooting it overflowed float32 in the second derivative with respect
        # to viscous_slope (x**0.25's curvature at 1e-30), and the clip's zero
        # gradient times that overflow made the Hessian NaN. Cold points get
        # X = 50, i.e. no response.
        warm = t4_shape > 1e-20
        T = t1_kelvin * jnp.where(warm, t4_shape, 1.0) ** 0.25
        X = jnp.where(warm, jnp.clip(_HC_OVER_K_ANGSTROM / (wavelength * T), None, 50.0), 50.0)
        emX = jnp.exp(-X)
        # dB/dT * dT/dL_x: X**2 e**X/(e**X-1)**2 from the Planck derivative, X**3
        # (i.e. T**-3) and h_x/d**3 from d(T**4) ~ dL_x h_x/d**3. Written as
        # (r_wien/d)**3, the same up to a constant, to keep float32 well scaled.
        planck_deriv = X ** 5 * emX / (1.0 - emX) ** 2  # e^-X form cannot overflow
        dilution = (r_wien / d_safe) ** 3
        # Only the disk outside its temperature peak responds: a smooth (not
        # hard) cutoff, since the root depends on inclination, a sampled
        # parameter, and a hard jnp.where here would have the same
        # zero-gradient risk documented for tophat_response_free/CLAUDE.md
        # decision #7. Between the ISCO and the peak, r_pk = r_in ((4 slope +
        # 0.5) / (4 slope))**2 (1.36 r_in for the standard slope), the
        # zero-torque temperature rises from zero, so every band's Wien
        # temperature is crossed in an extremely thin ring (~1e-6 r_in wide in
        # the optical) right next to the lamppost, where the dilution is huge.
        # Its true contribution is negligible, but whenever a (tau, phi)
        # quadrature point landed inside it psi spiked, and the NGC 5548
        # potential had a pole at one inclination (54.144 degrees) that trapped
        # the optimiser. Before September 2026 the cutoff was at r_in.
        mask = jax.nn.sigmoid((r_root - r_peak) / jnp.clip(0.05 * r_in, 1e-6, None))
        # |d tau / d r| = |r / d + s| (d the delay's lamppost distance) vanishes
        # where the two near-side roots merge (an integrable caustic); the floor
        # keeps the quadrature finite there.
        d_delay = jnp.sqrt(r_root ** 2 + hd ** 2)
        jac = jnp.clip(jnp.abs(r_root / jnp.clip(d_delay, 1e-12, None) + s_phi), 1e-3, None)
        return jnp.where(valid, planck_deriv * dilution * mask * r_root / jac, 0.0)

    r_plus = (-A * s_phi + sq) / one_m_s2
    r_minus = (-A * s_phi - sq) / one_m_s2
    raw = jnp.sum(contribution(r_plus) + contribution(r_minus), axis=1) * dphi
    raw = jnp.where(tau_grid >= 0.0, raw, 0.0)
    return _smooth_response(raw, tau_grid, r_wien, smoothing_frac, smoothing_days, smoothing_log)


def _smooth_response(raw, tau_grid, r_scale, smoothing_frac, smoothing_days, smoothing_log):
    """Smooth and area-normalise a causal response on ``tau_grid``.

    By default (``smoothing_frac`` and ``smoothing_days`` both ``None``) the
    smoothing is a Gaussian in ``ln tau`` of width ``smoothing_log``
    (:func:`_log_smooth`): causal by construction, so no response is moved
    below zero lag. Giving ``smoothing_days`` (days) or ``smoothing_frac``
    (times ``r_scale``) selects the older Gaussian in ``tau`` instead, which
    spreads the response's near-vertical onset across ``tau = 0``; ``0`` in
    any of the three switches smoothing off.
    """
    if smoothing_days is not None or smoothing_frac is not None:
        sigma = smoothing_days if smoothing_days is not None else smoothing_frac * r_scale
        raw = jax.lax.cond(
            sigma > 0.0,
            lambda raw: _gaussian_smooth(raw, tau_grid, sigma),
            lambda raw: raw,
            raw,
        )
    elif smoothing_log > 0.0:
        raw = _log_smooth(raw, tau_grid, smoothing_log)
    # Gaussian smoothing in tau spreads weight across tau=0 from the tau>=0
    # side, so the causal mask has to be re-applied after it, not just before.
    raw = jnp.where(tau_grid >= 0.0, raw, 0.0)
    trapz = jnp.trapezoid if hasattr(jnp, "trapezoid") else jnp.trapz
    area = trapz(raw, tau_grid)
    return raw / jnp.clip(area, 1e-12, None)


def _log_smooth(values, tau_grid, sigma_log):
    """Local average of ``values`` in ``ln tau`` with a Gaussian kernel of
    width ``sigma_log``, over the positive part of ``tau_grid`` only: causal by
    construction (points at ``tau <= 0`` neither contribute nor receive
    anything), and a fixed fractional width, so the sharp onset of a
    high-inclination response is smoothed on its own (short) time-scale and
    the long tail on its (long) one."""
    positive = tau_grid > 0.0
    log_tau = jnp.log(jnp.where(positive, tau_grid, 1.0))
    # Trapezoid weights in ln tau, so an uneven (e.g. graded) grid is handled.
    gaps = jnp.where(positive[1:] & positive[:-1], jnp.diff(log_tau), 0.0)
    weights = 0.5 * (jnp.concatenate([gaps, jnp.zeros(1)]) + jnp.concatenate([jnp.zeros(1), gaps]))
    weights = jnp.where(positive, jnp.clip(weights, 1e-12, None), 0.0)
    kernel = jnp.exp(-0.5 * ((log_tau[:, None] - log_tau[None, :]) / sigma_log) ** 2) * weights[None, :]
    smoothed = (kernel @ values) / jnp.clip(jnp.sum(kernel, axis=1), 1e-30, None)
    return jnp.where(positive, smoothed, 0.0)


def _gaussian_smooth(values, tau_grid, sigma):
    """Convolve ``values`` (on ``tau_grid``) with a Gaussian of width
    ``sigma``, mathematically equivalent to depositing each of many
    individual point contributions with that same Gaussian spread and
    summing (see :func:`thin_disk_response`'s docstring) -- linear in
    ``values``, so it commutes with wherever those contributions actually
    came from."""
    kernel = jnp.exp(-0.5 * ((tau_grid[:, None] - tau_grid[None, :]) / sigma) ** 2)
    kernel = kernel / jnp.clip(jnp.sum(kernel, axis=1, keepdims=True), 1e-12, None)
    return kernel @ values


class ThinDiskResponseTable(NamedTuple):
    """Precomputed :func:`thin_disk_response` templates across inclination,
    for :func:`thin_disk_response_from_table` / :func:`build_thin_disk_response_fast`
    -- see those for why this exists."""

    M_BH: float
    incl_grid: jnp.ndarray       # (n_incl,) degrees
    u_grid: jnp.ndarray          # (n_u,) days, dimensionless-lag axis at the reference mdot/wavelength
    templates: jnp.ndarray       # (n_incl, n_u)
    tau_ref_reference: float     # lag_scaling(reference_log_mdot, reference_wavelength, M_BH)


def build_thin_disk_response_table(
    M_BH,
    incl_grid=None,
    reference_log_mdot: float = 0.0,
    reference_wavelength: float = 5000.0,
    u_max_factor: float = 15.0,
    n_u: int = 1200,
    n_phi: int = 400,
    **thin_disk_kwargs,
) -> ThinDiskResponseTable:
    """Precompute a lookup table of :func:`thin_disk_response` shapes across
    inclination, at high quadrature resolution, once -- so a fit can look
    the response up (:func:`thin_disk_response_from_table`) instead of
    re-running the full O(n_phi * n_tau) disk integral on every NUTS
    step.

    This is a JAX-differentiable version of the precompute-and-interpolate
    trick used in the author's PhD-era CREAM Fortran code: precompute the
    response at a fixed accretion rate across an inclination grid once at
    the start of a run, then get any other inclination by interpolating the
    table, and any other accretion rate (or wavelength) by *stretching* the
    lag axis according to the ``mdot**(1/3)`` (and ``wavelength**(4/3)``)
    scaling :func:`lag_scaling` already uses. Section 5 of
    ``docs/thin_disk_response.md`` explains why that stretch is only
    approximate (a fixed absolute inner radius means the disk isn't
    *exactly* self-similar under it), and it is the same approximation the
    original Fortran made, not something new introduced here.

    The returned table is a plain, fixed set of arrays, meant to be built
    *before* a fit (M_BH is fixed input anyway, per CLAUDE.md decision #3)
    and passed to :func:`build_thin_disk_response_fast` to get an
    interpolation-based response function ready to assign to
    ``pycream2.model.response_function``.

    Parameters
    ----------
    M_BH : float
        Fixed black hole mass, matching the run this table is for.
    incl_grid : array_like, optional
        Inclinations (degrees) to precompute templates at. Defaults to
        every 2.5 degrees from 0 to 90 (37 templates).
    reference_log_mdot, reference_wavelength : float
        The one (log_mdot, wavelength) pair templates are actually computed
        at; every other (log_mdot, wavelength) is reached by stretching.
    u_max_factor : float
        The template's own lag axis spans ``[-0.05, u_max_factor] *
        tau_ref_reference``. Must be generous enough that, after stretching,
        it still covers whatever part of the real ``tau_grid`` matters for
        the (log_mdot, wavelength) values a fit actually visits -- lookups
        outside this range return 0 (see :func:`thin_disk_response_from_table`),
        which is safe but silently loses accuracy in the tail if too small.
    n_u : int
        Resolution of the template's own lag axis.
    n_phi : int
        Passed through to :func:`thin_disk_response` for the (one-off, so
        affordably high-resolution) template computation.
    **thin_disk_kwargs
        Any other :func:`thin_disk_response` keyword (``viscous_slope``,
        ``include_irradiation``, ...), applied identically to every
        inclination in the table.
    """
    if incl_grid is None:
        incl_grid = jnp.arange(0.0, 90.001, 2.5)
    else:
        incl_grid = jnp.asarray(incl_grid)

    tau_ref_reference = float(lag_scaling(reference_log_mdot, reference_wavelength, M_BH))
    u_grid = jnp.linspace(-0.05 * tau_ref_reference, u_max_factor * tau_ref_reference, n_u)

    # Templates are built *unsmoothed* (smoothing_days=0.0), even though
    # thin_disk_response smooths by default -- smoothing has to happen
    # *after* stretching a template onto a query's own tau_ref, using that
    # query's own smoothing width, or the reference point's smoothing
    # bandwidth gets stretched along with everything else and the
    # approximation gets substantially worse (confirmed directly: this
    # dropped the near-reference case from ~13% peak error back to <1%).
    # thin_disk_response_from_table applies the equivalent smoothing once,
    # after stretching.
    thin_disk_kwargs.setdefault("smoothing_days", 0.0)
    templates = jnp.stack([
        thin_disk_response(
            u_grid, reference_log_mdot, reference_wavelength, float(incl), M_BH,
            n_phi=n_phi, **thin_disk_kwargs,
        )
        for incl in incl_grid
    ])

    return ThinDiskResponseTable(
        M_BH=M_BH, incl_grid=incl_grid, u_grid=u_grid,
        templates=templates, tau_ref_reference=tau_ref_reference,
    )


def thin_disk_response_from_table(
    table: ThinDiskResponseTable, tau_grid, log_mdot, wavelength, inclination,
    smoothing_frac: float | None = None, smoothing_days: float | None = None,
    smoothing_log: float = DEFAULT_SMOOTHING_LOG,
):
    """Fast, interpolated stand-in for :func:`thin_disk_response`, using a
    precomputed :class:`ThinDiskResponseTable` (see
    :func:`build_thin_disk_response_table`) instead of recomputing the disk
    integral. Two cheap lookups replace it:

    1. Interpolate the template family over ``inclination`` (linear,
       differentiable in ``inclination`` almost everywhere, same as any
       other lookup-table use in JAX) to get one dimensionless template for
       this exact inclination, still on the table's own ``u_grid``.
    2. Stretch that template onto the real ``tau_grid`` by
       ``s = lag_scaling(log_mdot, wavelength, M_BH) / table.tau_ref_reference``
       -- i.e. evaluate it at ``tau_grid / s`` and divide by ``s`` to keep
       the area normalised to 1 under that change of variables -- which is
       exactly the "stretch to a different mdot" trick from
       :func:`build_thin_disk_response_table`'s docstring.

    Both steps are ``jnp.interp`` against a fixed table, so this is `O(n_u +
    n_tau)` per call rather than :func:`thin_disk_response`'s `O(n_phi *
    n_tau)` disk integral -- the whole point for use inside NUTS. Queries
    landing outside the table's ``u_grid`` (an accretion rate/wavelength
    combination far from what the table was built for) return 0 rather than
    extrapolating, which is safe but a sign the table needs a larger
    ``u_max_factor`` or a reference point closer to where the fit actually
    lives.

    Gaussian smoothing (``smoothing_frac``/``smoothing_days``, same
    meaning and default as :func:`thin_disk_response`'s) is applied *after*
    stretching, at this query's own ``tau_ref`` -- not baked into the
    table's templates (which :func:`build_thin_disk_response_table` builds
    unsmoothed). Smoothing at the reference point and then stretching that
    already-smoothed shape would stretch the smoothing bandwidth along
    with everything else, which is a substantially worse approximation
    (confirmed directly: this dropped the near-reference-point case from
    ~13% peak error back to <1%, matching how close the two functions are
    without smoothing at all).
    """
    tau_ref = lag_scaling(log_mdot, wavelength, table.M_BH)
    stretch = jnp.clip(tau_ref / table.tau_ref_reference, 1e-6, None)

    psi_u = jax.vmap(lambda column: jnp.interp(inclination, table.incl_grid, column))(table.templates.T)

    u_query = tau_grid / stretch
    raw = jnp.interp(u_query, table.u_grid, psi_u, left=0.0, right=0.0) / stretch
    raw = jnp.where(tau_grid >= 0.0, raw, 0.0)
    return _smooth_response(raw, tau_grid, tau_ref, smoothing_frac, smoothing_days, smoothing_log)


def build_thin_disk_response_fast(
    M_BH, smoothing_frac: float | None = None, smoothing_days: float | None = None,
    smoothing_log: float = DEFAULT_SMOOTHING_LOG, **table_kwargs,
) -> Callable:
    """Build and return a ready-to-use, interpolation-based response
    function matching the standard ``(tau_grid, log_mdot, wavelength,
    inclination, M_BH, ...)`` contract (CLAUDE.md decision #5) -- the
    one-call convenience wrapper around :func:`build_thin_disk_response_table`
    + :func:`thin_disk_response_from_table`::

        import pycream2.model as model
        from pycream2.forward_model import build_thin_disk_response_fast

        model.response_function = build_thin_disk_response_fast(M_BH=1e8)

    ``smoothing_frac``/``smoothing_days`` (same meaning as
    :func:`thin_disk_response`'s) control the query-time smoothing applied
    by :func:`thin_disk_response_from_table` -- kept separate from
    ``**table_kwargs`` (passed to :func:`build_thin_disk_response_table`
    for the one-off template build, which always builds unsmoothed
    templates regardless, per that function's own docstring) so smoothing
    is applied once, correctly, at each query's own scale.

    The underlying table (useful for inspecting/plotting what got
    precomputed) is attached as ``.table`` on the returned function.
    """
    table = build_thin_disk_response_table(M_BH, **table_kwargs)

    def _response(tau_grid, log_mdot, wavelength, inclination, M_BH=None, **kwargs):
        return thin_disk_response_from_table(
            table, tau_grid, log_mdot, wavelength, inclination,
            smoothing_frac=smoothing_frac, smoothing_days=smoothing_days, smoothing_log=smoothing_log,
        )

    _response.table = table
    return _response


def _filon_weights(tau_grid, freqs, xp):
    """Weights ``(Wc, Ws)``, shape ``(n_freq, n_tau)``, such that
    ``Wc @ psi`` and ``Ws @ psi`` are the exact cosine and sine transforms of
    the piecewise-linear interpolant of ``psi`` on ``tau_grid`` (Filon-type
    quadrature). Written for either NumPy or ``jax.numpy`` (``xp``).

    Plain trapezoid weights ``cos(w tau_j) * dtau_j`` alias badly once
    ``w * dtau`` is not small: on NGC 5548 the driver reached w = 11 rad/day
    while the graded lag grid's tail spacing was 0.3 day (3.3 radians per
    step), so the high-frequency transfer coefficients were mostly aliasing
    noise that jumped as the parameters reshaped psi. The optimiser stalled
    on the resulting jagged potential (restarts stopped along the
    inclination ridge with gradients ~100x the optimum's), and NUTS would
    see the same roughness. Integrating each segment's linear hat functions
    against ``e^{i w tau}`` exactly removes the aliasing for any ``w``; for
    ``w * dtau -> 0`` it reduces to the trapezoid rule.

    Per segment ``[a, b]``, ``h = b - a``, ``theta = w h``, with
    ``g0 = int_0^1 e^{i theta s} ds`` and ``g1 = int_0^1 s e^{i theta s} ds``:
    the left node gets ``h e^{i w a} (g0 - g1)`` and the right node
    ``h e^{i w a} g1``; real parts weight the cosine transform and imaginary
    parts the sine transform. Below ``theta = 0.05`` a Taylor series replaces
    the closed forms, which cancel badly there.
    """
    a = tau_grid[:-1][None, :]
    h = (tau_grid[1:] - tau_grid[:-1])[None, :]
    w = freqs[:, None]
    theta = w * h
    small = theta < 0.05
    ts = xp.where(small, 1.0, theta)  # safe denominator for the unused branch
    sin_t, cos_t = xp.sin(ts), xp.cos(ts)
    g0r = xp.where(small, 1.0 - theta ** 2 / 6.0 + theta ** 4 / 120.0, sin_t / ts)
    g0i = xp.where(small, theta / 2.0 - theta ** 3 / 24.0, (1.0 - cos_t) / ts)
    g1r = xp.where(small, 0.5 - theta ** 2 / 8.0 + theta ** 4 / 144.0, sin_t / ts + (cos_t - 1.0) / ts ** 2)
    g1i = xp.where(small, theta / 3.0 - theta ** 3 / 30.0, -cos_t / ts + sin_t / ts ** 2)
    pr, pi = xp.cos(w * a), xp.sin(w * a)  # e^{i w a}

    def times_phase(gr, gi):
        return h * (pr * gr - pi * gi), h * (pr * gi + pi * gr)

    left_c, left_s = times_phase(g0r - g1r, g0i - g1i)
    right_c, right_s = times_phase(g1r, g1i)
    zeros = xp.zeros((freqs.shape[0], 1), dtype=left_c.dtype)
    Wc = xp.concatenate([left_c, zeros], axis=1) + xp.concatenate([zeros, right_c], axis=1)
    Ws = xp.concatenate([left_s, zeros], axis=1) + xp.concatenate([zeros, right_s], axis=1)
    return Wc, Ws


def transfer_matrices(tau_grid, freqs):
    """Precompute the fixed quadrature-weighted cosine/sine matrices that
    turn :func:`transfer_coeffs` into two matrix-vector products.

    ``tau_grid`` and ``freqs`` are both fixed for the whole fit (built once
    in ``EchoFit.build_grid``), so ``cos(w_k tau_j)``/``sin(w_k tau_j)`` and
    the trapezoid weights never change between NUTS steps; only ``psi``
    does. Recomputing the ``(n_freq, n_tau)`` trig matrices on every
    gradient evaluation, per band, was measured to be roughly half of the
    whole potential-plus-gradient cost (``jax.jit``, 5 bands, default
    ``n_freq=60``/``n_tau=400``); doing it once here instead cut that
    evaluation from ~8.4ms to ~1.7ms, with an identical result. Computed in
    float64 NumPy, then cast, so the precomputed matrices are if anything
    slightly more accurate than the on-the-fly float32 version.

    This is the Fourier-space counterpart of the author's CREAM Fortran
    code's ``itaumax`` early cut-off on its real-space convolution
    lookback: there, most of the cost was the ``(n_t, n_tau)`` lookback
    loop, so skipping lags where ``psi`` had decayed to zero paid off. Here
    the lag grid only enters via these matrices, so precomputing them
    removes almost all of the lag-dependent cost at once, and a cut-off
    would have nothing left to save (and could only be static anyway:
    ``jax.jit`` needs fixed array shapes, so a ``psi``-dependent cut-off
    would still multiply the same number of zeros).

    Parameters
    ----------
    tau_grid : array_like, shape (n_tau,)
    freqs : array_like, shape (n_freq,)
        Must be concrete (not traced) arrays.

    Returns
    -------
    Wc, Ws : jnp.ndarray, shape (n_freq, n_tau)
        ``A = Wc @ psi`` and ``B = Ws @ psi`` reproduce
        :func:`transfer_coeffs`'s integrals (Filon quadrature, see
        :func:`_filon_weights`).
    """
    Wc, Ws = _filon_weights(np.asarray(tau_grid, dtype=np.float64), np.asarray(freqs, dtype=np.float64), np)
    dtype = jnp.asarray(tau_grid).dtype
    return jnp.asarray(Wc, dtype=dtype), jnp.asarray(Ws, dtype=dtype)


def fourier_basis(freqs, t):
    """Precompute ``sin(w_k t_i)``/``cos(w_k t_i)`` for fixed times ``t``,
    the matrices :func:`compute_echo`/:func:`driver_at` would otherwise
    rebuild on every call. Observation times and ``freqs`` are both fixed
    for a whole fit, so this is done once per light curve, for the same
    reason as :func:`transfer_matrices`.

    Returns
    -------
    sin_wt, cos_wt : jnp.ndarray, shape (n_obs, n_freq)
    """
    wt = np.asarray(freqs, dtype=np.float64)[None, :] * np.asarray(t, dtype=np.float64)[:, None]
    dtype = jnp.asarray(freqs).dtype
    return jnp.asarray(np.sin(wt), dtype=dtype), jnp.asarray(np.cos(wt), dtype=dtype)


def transfer_coeffs(tau_grid, psi, freqs, matrices=None):
    """Fourier cosine/sine transform of psi at each driver frequency.

    A_k = int psi(tau) cos(w_k tau) dtau
    B_k = int psi(tau) sin(w_k tau) dtau

    Integrated exactly for the piecewise-linear interpolant of psi (Filon
    quadrature, :func:`_filon_weights`), so high frequencies don't alias on a
    coarse lag grid.

    Parameters
    ----------
    tau_grid : array_like, shape (n_tau,)
    psi : array_like, shape (n_tau,)
    freqs : array_like, shape (n_freq,)
        Angular frequencies w_k of the driver's Fourier basis.
    matrices : tuple, optional
        ``(Wc, Ws)`` from :func:`transfer_matrices` for this same
        ``tau_grid``/``freqs``. When given, the trig matrices aren't
        rebuilt (much faster inside a fit); otherwise computed on the fly.

    Returns
    -------
    A, B : array_like, shape (n_freq,)
    """
    if matrices is None:
        matrices = _filon_weights(jnp.asarray(tau_grid), jnp.asarray(freqs), jnp)
    Wc, Ws = matrices
    return Wc @ psi, Ws @ psi


def compute_echo(S, C, freqs, A, B, t_obs, basis=None):
    """Evaluate int psi(tau) X(t_obs - tau) dtau for a Fourier-series driver.

    Parameters
    ----------
    S, C : array_like, shape (n_freq,)
        Driver sine/cosine amplitudes.
    freqs : array_like, shape (n_freq,)
        Driver angular frequencies w_k.
    A, B : array_like, shape (n_freq,)
        Response transfer coefficients from :func:`transfer_coeffs`.
    t_obs : array_like, shape (n_obs,)
        Observation times at which to evaluate the echo.
    basis : tuple, optional
        ``(sin_wt, cos_wt)`` from :func:`fourier_basis` for this same
        ``freqs``/``t_obs``; computed on the fly if omitted.

    Returns
    -------
    echo : array_like, shape (n_obs,)
    """
    if basis is None:
        wt = freqs[None, :] * t_obs[:, None]       # (n_obs, n_freq)
        basis = (jnp.sin(wt), jnp.cos(wt))
    sin_wt, cos_wt = basis
    sin_coef = S * A + C * B                        # (n_freq,)
    cos_coef = C * A - S * B                        # (n_freq,)
    echo = sin_wt @ sin_coef + cos_wt @ cos_coef
    return echo


def driver_at(S, C, freqs, t, basis=None):
    """Evaluate the raw driver X(t) = sum_k S_k sin(w_k t) + C_k cos(w_k t).

    Convenience function for plotting / diagnostics (not used in the echo
    convolution itself, which uses the closed-form ``compute_echo``).
    ``basis`` is an optional precomputed :func:`fourier_basis` for ``t``.
    """
    if basis is None:
        wt = freqs[None, :] * t[:, None]
        basis = (jnp.sin(wt), jnp.cos(wt))
    sin_wt, cos_wt = basis
    return sin_wt @ S + cos_wt @ C
