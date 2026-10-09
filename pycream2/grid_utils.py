"""
grid_utils.py
=============

Shared logic for picking the driver Fourier series' finest resolvable
timescale from a set of (possibly irregularly sampled) observation times.

Used by both ``EchoFit.build_grid`` (fitting) and ``synthetic.py``
(ground-truth generation) so the two always agree on what frequency grid
a given set of observation times implies -- see CLAUDE.md's "Known rough
edges" note on why this needs to be robust rather than the raw minimum gap.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def estimate_dt_min(
    t_arrays: Iterable[np.ndarray],
    percentile: float = 5.0,
    floor_frac: float = 1e-3,
    t_span: Optional[float] = None,
) -> float:
    """Robust estimate of the shortest timescale the driver should resolve.

    Sets the driver frequency grid's upper bound (``w_max = pi / dt_min``).
    Uses a low percentile of consecutive observation gaps rather than the
    strict minimum: for irregularly/randomly sampled data the single
    tightest gap between any two points can be pathologically small (its
    expectation shrinks much faster than the typical spacing as more points
    are added), which would otherwise blow up ``w_max`` and force the model
    to try to resolve spurious high-frequency structure.

    Parameters
    ----------
    t_arrays : iterable of array_like
        One array of observation times per band (need not be pre-sorted).
    percentile : float
        Percentile (0-100) of consecutive gaps to use, e.g. 5 = 5th
        percentile. Lower is closer to (and more sensitive like) the raw
        minimum; higher is more conservative (coarser resolution).
    floor_frac : float
        Floor on the returned ``dt_min`` as a fraction of ``t_span``, to
        avoid a degenerate (near-zero) estimate when very few points are
        available.
    t_span : float, optional
        Total observed time baseline (days), used for ``floor_frac`` and as
        the fallback when fewer than two total points are available.

    Returns
    -------
    dt_min : float
    """
    # A plain np.concatenate([]) raises rather than returning a size-0
    # array, so the "no bands have 2+ points" fallback below has to be
    # checked on this list directly, before concatenating -- found via a
    # coverage report showing that fallback as unreachable dead code (it
    # crashed here first, for every t_arrays that should have hit it).
    diffs = [np.diff(np.sort(np.asarray(t, float))) for t in t_arrays if len(t) > 1]
    if not diffs:
        dt_min = t_span if t_span is not None else 1.0
    else:
        dt_min = float(np.percentile(np.concatenate(diffs), percentile))
    if t_span is not None:
        dt_min = max(dt_min, floor_frac * t_span)
    return dt_min


def hybrid_frequency_grid(period_max: float, f_max: float, log_step: float) -> np.ndarray:
    """Driver angular frequencies, linear at low frequency and logarithmic above.

    Starts at ``w_1 = 2 pi / period_max`` and steps by
    ``max(2 pi / period_max, log_step * w)`` up to ``2 pi f_max``: harmonics
    of the longest period (CREAM's grid, Starkey et al. 2016) while
    ``log_step * w`` is smaller than their spacing, then a constant fractional
    step. A window of length ``period_max`` holds about one independent pair of
    Fourier coefficients per ``1 / period_max`` of bandwidth, so linear spacing
    is the natural basis; finer low-frequency spacing only adds near-duplicate
    modes. At high frequency the echoes are smoothed by the response and don't
    resolve individual harmonics: there a log grid at ``log_step = 0.03`` gave
    the same posteriors as linear spacing on the CREAM paper's synthetic tests,
    with far fewer modes (linear spacing to 2 cycles/day needs
    ``2 period_max`` modes, ~1700 for an NGC 5548-length campaign).

    Parameters
    ----------
    period_max : float
        Longest driver period (days).
    f_max : float
        Highest driver frequency, cycles per day.
    log_step : float
        Fractional frequency step in the logarithmic part.
    """
    if period_max <= 0 or f_max <= 0 or log_step <= 0:
        raise ValueError("period_max, f_max and log_step must all be positive")
    dw = 2.0 * np.pi / period_max
    w_max = 2.0 * np.pi * f_max
    freqs = [dw]
    while True:
        w = freqs[-1] + max(dw, log_step * freqs[-1])
        if w > w_max * (1.0 + 1e-9):
            break
        freqs.append(w)
    return np.asarray(freqs)


def graded_tau_grid(tau_max: float, n_tau: int, power: float = 3.0) -> np.ndarray:
    """A lag grid concentrated near ``tau=0`` rather than uniformly spaced:
    ``tau_grid = tau_max * linspace(0, 1, n_tau) ** power``.

    A response function's own width scales with its mean lag
    (``width_frac * tau_mean``, see ``forward_model.response_function``/
    ``thin_disk_response``), so a uniform grid spends most of its points
    resolving the large-tau tail, typically wider but also usually not
    where any real response actually lives. Confirmed directly on a real
    dataset (NGC 5548 AGN STORM, M_BH ~3.2e7 Msun): a uniform ``tau_grid``
    gave a band with a ~0.08-day mean lag a response width only ~10% of
    one grid spacing there, numerically indistinguishable from zero once
    normalised (``response_function``'s ``trapz``-based area, and
    ``transfer_coeffs``'s Fourier integral, both under-resolve a peak
    narrower than the local grid spacing) -- the light curve prediction
    for that band came out as a flat line, not an echo. The *same* grid
    only barely resolved a band with a ~1.2-day lag. Grading the grid
    towards ``tau=0`` fixed every band in that test, including the
    longest-lag one (from ~1.6 to ~10.6 grid points across its own
    response width) -- most of the uniform grid's resolution was being
    spent on a near-``tau_max`` region no band's actual response occupied.

    ``power=3`` (the default) is a deliberately simple, general-purpose
    choice, not tuned to one specific campaign: higher concentrates
    resolution more strongly near ``tau=0``, at the cost of coarser
    resolution near ``tau_max`` -- only relevant for a ``lag_mode="free"``
    band whose fitted ``tau_{band}`` lands out there, since a physical-mode
    response's width shrinks together with its own mean lag wherever that
    lands. ``power=1`` recovers the original uniform grid.
    """
    u = np.linspace(0.0, 1.0, n_tau)
    return tau_max * u ** power


def check_tau_grid_resolution(
    tau_grid: np.ndarray,
    bands: dict,
    M_BH: Optional[float],
    representative_log_mdot: float = -2.0,
    width_frac: float = 0.35,
    min_points_across_width: float = 2.0,
) -> list:
    """Check whether ``tau_grid`` resolves each ``lag_mode="physical"``
    band's expected response width -- returns a list of human-readable
    warning strings (one per band at risk), rather than raising or warning
    itself, so callers can decide what to do with them (``EchoFit.build_grid``
    turns them into ``warnings.warn`` calls).

    ``graded_tau_grid``'s default grading fixes this for realistic cases
    (see its own docstring), but it's a matter of degree, not an absolute
    guarantee: a short enough wavelength, small enough ``M_BH``, or small
    enough ``n_tau`` can still under-resolve a response -- confirmed
    directly once already (NGC 5548's shortest-wavelength UV continuum
    band, see CLAUDE.md decision #20), silently, as a response and light
    curve prediction that came out as a flat line rather than an error.
    This check exists so that happens as an explicit warning instead, as
    early as ``build_grid()`` rather than only visible after a full fit.

    The real fitted ``log_mdot`` isn't known until after fitting, so this
    checks at ``representative_log_mdot`` instead -- deliberately a
    conservative, plausible-worst-case value (well below the prior's own
    centre, since a smaller accretion rate implies a smaller, harder-to-
    resolve mean lag), not the prior's central value, so this is more
    likely to over-warn than to miss a real risk.
    """
    from .forward_model import lag_scaling

    tau_grid = np.asarray(tau_grid)
    messages = []
    if M_BH is None:
        return messages
    for name, d in bands.items():
        if d.get("lag_mode", "physical") != "physical":
            continue
        tau_mean = float(lag_scaling(representative_log_mdot, d["wavelength"], M_BH))
        width = width_frac * tau_mean
        idx = max(min(int(np.searchsorted(tau_grid, tau_mean)), len(tau_grid) - 1), 1)
        local_dtau = float(tau_grid[idx] - tau_grid[idx - 1])
        if local_dtau <= 0:
            continue
        points_across = width / local_dtau
        if points_across < min_points_across_width:
            messages.append(
                f"band {name!r} (wavelength {d['wavelength']:.0f} A): its expected response "
                f"width (~{width:.4f} days at a conservative log_mdot={representative_log_mdot}) "
                f"is only ~{points_across:.1f} tau_grid points wide there -- its response/light "
                f"curve prediction may come out as a flat line. Try a larger n_tau, a higher "
                f"tau_grid_power, or an explicit smaller tau_max in build_grid()."
            )
    return messages
