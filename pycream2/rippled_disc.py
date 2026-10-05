"""
Rimmed and rippled accretion discs, after Starkey, Huang, Horne & Lin (2023,
MNRAS 519, 2754; arXiv:2212.01379).

A flat thin disc reprocesses too little of the lamppost's light at large radii
to produce the long optical delays seen in AGN (the disc-size problem). That
paper gives the disc a thickness profile ``H(r)``: a surface tilted towards the
lamppost intercepts more of its light, so it is hotter and responds more
strongly than a flat disc at the same radius. Two profiles are considered:

- a **rim**: a steep power law, ``H = H_1 (r/r_1)**beta`` with ``beta`` of order
  100, ending at ``r_out = r_1`` (about 5 light-days for NGC 5548, near the dust
  sublimation radius); its inner face is heated from about 1500 K to 6000 K;
- **ripples**: ``H = H_1(r) [1 + A(r) cos(k r)]``, ``A(r) = A_w (r/r_1)**-beta_w``,
  whose crests are irradiated and whose troughs lie in their shadow.

This module implements the paper's model:

- temperature (its eq. 3), in units of ``T_1**4`` (``forward_model.disk_t1_kelvin``,
  the viscous temperature at one light-day):
  ``T**4 = T_1**4 r**-3 [(1 - sqrt(r_in/r)) + (4/3) eps_LP (r/r_g) f(r)]``,
  with ``r_g = GM/c**2`` and the covering factor (its eq. 16, in general form)
  ``f(r) = (r H' - H + H_LP) / r``, the irradiation term only where the surface
  is lit. By default (``exact_irradiation=True``) the irradiating flux is the
  exact ``cos(incidence) / d**2`` per unit surface area, which replaces
  ``f / r**2`` by ``r f / (d**3 sqrt(1 + H'**2))`` (see :func:`disc_surface`);
- delay (its eq. 11), ``c tau = sqrt(dh**2 + r**2) + dh cos(i) - r cos(phi) sin(i)``,
  ``dh = H_LP - H(r)``, azimuth ``phi = 0`` towards the observer;
- response (its eq. 12): each surface element contributes
  ``dB_nu/dT dT/dL_LP`` times its solid angle on the observer's sky, with
  ``dT/dL_LP ~ f / (r**2 T**3)``.

Differences from the paper worth knowing: the inner radius is pycream2's
``3 R_s`` (the paper uses the ISCO for a spinning black hole); a surface's
projection towards the observer is ``cos(i) - H' sin(i) cos(phi)`` (zero when it
faces away), but occultation of one part of the disc by another (a rim hiding
the inner disc at high inclination) is not modelled.

The response is a two-dimensional (radius, azimuth) quadrature, each element's
contribution deposited onto the lag grid with linear weights and the result
smoothed in ``ln tau`` (``forward_model._smooth_response``), so it stays
differentiable in ``log_mdot`` and ``inclination``. The disc's geometry is fixed
(plain Python floats), not sampled: it sets the radial grid and the shadows,
which are therefore computed once, outside the gradient.

Use it in a fit through the usual swap point::

    import functools
    import pycream2.model as model
    from pycream2.rippled_disc import rippled_disc_response, NGC5548_RIM
    model.response_function = functools.partial(rippled_disc_response, **NGC5548_RIM)
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np

from .forward_model import (
    DEFAULT_SMOOTHING_LOG,
    _HC_OVER_K_ANGSTROM,
    _schwarzschild_radius_light_days,
    _smooth_response,
    disk_t1_kelvin,
    wien_radius,
)

# The steep rim of Starkey et al. (2023, section 5.2) for NGC 5548: H/r = 0.03 at
# r_out = 5 light-days, beta > 100. lamppost_efficiency (eps_LP) is the paper's
# free irradiation strength; 0.2 is illustrative: with a viscous disc at 1500 K at
# 5 light-days (T_1 = 5000 K) it heats the rim face to about 6000 K, as in the paper.
NGC5548_RIM = dict(height=0.15, r_ref=5.0, beta=100.0, r_out=5.0, lamppost_efficiency=0.2)
# Illustrative ripples (not fitted): a thin disc, H/r = 0.006, whose ripples, one light-day
# long, grow outwards (A ~ r**3) to 60 per cent of its height at 5 light-days (and stay
# below 100 per cent, so H > 0, out to r_out = 5.6).
EXAMPLE_RIPPLES = dict(height=0.03, r_ref=5.0, beta=1.0, ripple_amplitude=0.6, ripple_wavelength=1.0,
                       ripple_slope=-3.0, r_out=5.6, lamppost_efficiency=0.2)

_LIGHT_DAY_CM = 2.59020684e15
_MPC_CM = 3.0857e24
_H_CGS, _C_CGS, _K_CGS = 6.62607e-27, 2.99792458e10, 1.380649e-16


def disc_height(r, height: float = 0.0, r_ref: float = 1.0, beta: float = 1.0, ripple_amplitude: float = 0.0,
                ripple_wavelength: float = 1.0, ripple_slope: float = 0.0):
    """Disc surface height ``H(r)`` (light-days; ``r`` in light-days).

    ``H_1(r) = height (r / r_ref)**beta`` (Starkey et al. 2023 eq. 15) times the
    ripple ``1 + A(r) cos(2 pi r / ripple_wavelength)``, with
    ``A(r) = ripple_amplitude (r / r_ref)**-ripple_slope`` (their eq. 22, with
    ``k = 2 pi / ripple_wavelength``). ``height = 0`` is a flat disc. For a rim
    (``beta`` ~ 100) keep ``r`` <= ``r_ref``: beyond it ``(r/r_ref)**beta`` grows
    without bound.
    """
    base = height * (r / r_ref) ** beta
    amp = ripple_amplitude * (r / r_ref) ** (-ripple_slope)
    return base * (1.0 + amp * jnp.cos(2.0 * jnp.pi * r / ripple_wavelength))


def _height_and_slope(r, height, r_ref, beta, ripple_amplitude, ripple_wavelength, ripple_slope):
    """:func:`disc_height` and its radial derivative in plain numpy (float64), for
    the static geometry (computed outside any trace)."""
    base = height * (r / r_ref) ** beta
    amp = ripple_amplitude * (r / r_ref) ** (-ripple_slope)
    k = 2.0 * np.pi / ripple_wavelength
    wave = 1.0 + amp * np.cos(k * r)
    d_wave = -ripple_slope * amp / r * np.cos(k * r) - amp * k * np.sin(k * r)
    return base * wave, beta * base / r * wave + base * d_wave


def _radial_grid(r_in, r_out, n_r, r_ref, beta, ripple_amplitude, ripple_wavelength):
    """Static radial grid (numpy, light-days) and its quadrature widths: geometric
    from just outside ``r_in`` to ``r_out``, refined across a steep rim (whose
    face is only ``~r_ref / beta`` wide) and to at least 24 points per ripple."""
    r = [np.geomspace(r_in * 1.0001, r_out, n_r)]
    if beta >= 10.0 and r_ref <= r_out:
        lo = max(r_in * 1.0001, r_ref * (1.0 - 12.0 / beta))
        r.append(np.linspace(lo, min(r_ref, r_out), 240))
    if ripple_amplitude > 0.0:
        step = ripple_wavelength / 24.0
        lo = max(r_in * 1.0001, ripple_wavelength / 4.0)
        n = int(min((r_out - lo) / step, 6000))
        if n > 1:
            r.append(np.linspace(lo, r_out, n))
    r = np.unique(np.concatenate(r))
    edges = np.concatenate([[r[0]], 0.5 * (r[1:] + r[:-1]), [r[-1]]])
    return r, np.diff(edges)


def disc_surface(log_mdot, M_BH, height: float = 0.0, r_ref: float = 1.0, beta: float = 1.0,
                 ripple_amplitude: float = 0.0, ripple_wavelength: float = 1.0, ripple_slope: float = 0.0,
                 r_out: float = 100.0, lamppost_height_rs: float = 3.0, lamppost_efficiency: float = 1.0,
                 exact_irradiation: bool = True, n_r: int = 400):
    """The disc's surface on a radial grid: a dict of ``r``, ``dr`` (quadrature
    widths), ``H``, ``dH`` (``dH/dr``), the covering factor ``f``, ``lit`` (1 where
    the lamppost sees the surface, 0 in shadow), ``h_lp`` (light-days), ``irr``
    (the lamppost's flux per unit surface area, up to a constant) and the
    temperature ``T`` (Kelvin, viscous plus irradiation where lit).

    ``exact_irradiation=True`` (the default) takes the flux on the surface as
    ``L cos(incidence) / d**2`` per unit *surface* area, ``r f / (d**3 sqrt(1 +
    H'**2))`` with ``d`` the true lamppost distance. ``False`` uses the paper's
    eq. 3 as written, ``f / r**2``: the small-angle form per unit disc *plane*
    area. The two agree for a gently sloped surface far from the lamppost; on a
    steep face the paper's form is larger by ``sqrt(1 + H'**2)`` (about 3 on
    the NGC 5548 rim) and diverges for a vertical wall, and within a few
    lamppost heights of the centre it ignores the lamppost's height.

    A point is lit when its elevation seen from the lamppost,
    ``(H - H_LP) / r``, is rising with radius (``f > 0``: the surface faces the
    lamppost) and is at least the largest elevation at any smaller radius (no
    inner crest or rim is in the way). The grid and the shadows depend only on
    the geometry, which is fixed, so they are computed outside the gradient.
    """
    rs = float(_schwarzschild_radius_light_days(M_BH))
    r_in, r_g, h_lp = 3.0 * rs, 0.5 * rs, lamppost_height_rs * rs
    r_np, dr_np = _radial_grid(r_in, r_out, n_r, r_ref, beta, ripple_amplitude, ripple_wavelength)
    geom = dict(height=height, r_ref=r_ref, beta=beta, ripple_amplitude=ripple_amplitude,
                ripple_wavelength=ripple_wavelength, ripple_slope=ripple_slope)
    r64 = r_np.astype(np.float64)
    h_np, dh_np = _height_and_slope(r64, **geom)
    f_np = (r64 * dh_np - h_np + h_lp) / r64
    elevation = (h_np - h_lp) / r64
    inner_max = np.concatenate([[-np.inf], np.maximum.accumulate(elevation)[:-1]])
    lit_np = ((f_np > 0.0) & (elevation >= inner_max - 1e-9 * np.abs(elevation))).astype(np.float64)
    if exact_irradiation:
        d = np.sqrt(r64 ** 2 + (h_lp - h_np) ** 2)
        irr_np = r64 * np.clip(f_np, 0.0, None) / (d ** 3 * np.sqrt(1.0 + dh_np ** 2))
    else:
        irr_np = np.clip(f_np, 0.0, None) / r64 ** 2
    irr_np = lit_np * irr_np

    r, f, lit, irr = jnp.asarray(r_np), jnp.asarray(f_np), jnp.asarray(lit_np), jnp.asarray(irr_np)
    # T**4 / T_1**4 = r**-3 (1 - sqrt(r_in/r)) + (4/3) eps (1/r_g) * irr; with the
    # paper's irr = f/r**2 this is its eq. 3 exactly.
    t4 = r ** -3.0 * (1.0 - jnp.sqrt(r_in / r)) + (4.0 / 3.0) * lamppost_efficiency / r_g * irr
    warm = t4 > 1e-20
    T = disk_t1_kelvin(log_mdot, M_BH) * jnp.where(warm, t4, 1.0) ** 0.25
    T = jnp.where(warm, T, 0.0)
    return dict(r=r, dr=jnp.asarray(dr_np), H=jnp.asarray(h_np), dH=jnp.asarray(dh_np), f=f, lit=lit,
                irr=irr, h_lp=h_lp, r_in=r_in, T=T)


def response_elements(log_mdot, wavelength, inclination, M_BH, n_phi: int = 180, full_circle: bool = False,
                      **geometry):
    """The disc's surface elements on a (radius, azimuth) grid: ``r``, ``phi``
    (0 towards the observer), each element's ``delay`` (days, Starkey et al. 2023
    eq. 11) and its ``weight`` in the response at ``wavelength`` (eq. 12, up to a
    constant: ``dB_nu/dT dT/dL_LP`` times its projected area), both
    ``(n_r, n_phi)``, and the ``surface`` from :func:`disc_surface`, whose
    keywords ``geometry`` takes. :func:`rippled_disc_response` deposits these on
    a lag grid; they are also what the animation in ``docs/rippled_disc.md``
    draws."""
    s = disc_surface(log_mdot, M_BH, **geometry)
    r, dr, H, dH, irr, T = s["r"], s["dr"], s["H"], s["dH"], s["irr"], s["T"]
    incl = jnp.deg2rad(jnp.clip(inclination, 0.0, 89.9))
    sini, cosi = jnp.sin(incl), jnp.cos(incl)
    # Symmetric in phi -> -phi, so [0, pi] is enough for the response (phi = 0
    # towards the observer); full_circle gives [0, 2 pi), for pictures.
    span = 2.0 * jnp.pi if full_circle else jnp.pi
    phi = (jnp.arange(n_phi) + 0.5) * (span / n_phi)
    cosphi = jnp.cos(phi)[None, :]

    warm = T > 0.0
    X = jnp.where(warm, jnp.clip(_HC_OVER_K_ANGSTROM / (wavelength * jnp.where(warm, T, 1.0)), None, 50.0), 50.0)
    emX = jnp.exp(-X)
    # dB/dT T**-3 (up to constants), as in thin_disk_response; times dT/dL, which is
    # the lamppost's flux per unit surface area over T**3 (the T**-3 is already in
    # the first factor), zero in shadow.
    planck_deriv = X ** 5 * emX / (1.0 - emX) ** 2
    radial = planck_deriv * irr * r * dr  # r dr: the area element (projection below)
    # The responding disc starts at the viscous temperature peak, 1.36 r_in, as in
    # thin_disk_response (CLAUDE.md decision #22): inside it the zero-torque
    # temperature rises from zero, and without irradiation a thin ring there,
    # right next to the lamppost, would dominate the response.
    radial = jnp.where(r >= s["r_in"] * (3.5 / 3.0) ** 2, radial, 0.0)
    # Projected area towards the observer: n.o sqrt(1 + H'**2) = cos(i) - H' sin(i) cos(phi).
    proj = jnp.clip(cosi - dH[:, None] * sini * cosphi, 0.0, None)
    weight = radial[:, None] * proj

    dh = s["h_lp"] - H
    delay = (jnp.sqrt(dh ** 2 + r ** 2) + dh * cosi)[:, None] - (r[:, None] * sini) * cosphi
    return dict(r=r, phi=phi, delay=delay, weight=weight, surface=s)


def rippled_disc_response(
    tau_grid,
    log_mdot,
    wavelength,
    inclination,
    M_BH,
    height: float = 0.0,
    r_ref: float = 1.0,
    beta: float = 1.0,
    ripple_amplitude: float = 0.0,
    ripple_wavelength: float = 1.0,
    ripple_slope: float = 0.0,
    r_out: float = 100.0,
    lamppost_height_rs: float = 3.0,
    lamppost_efficiency: float = 1.0,
    exact_irradiation: bool = True,
    n_r: int = 400,
    n_phi: int = 180,
    smoothing_frac: Optional[float] = None,
    smoothing_days: Optional[float] = None,
    smoothing_log: float = DEFAULT_SMOOTHING_LOG,
):
    """Causal, area-normalised response of a rimmed or rippled disc on
    ``tau_grid`` (days): the swappable response-function contract of
    :mod:`pycream2.model` (decision #5 in CLAUDE.md).

    ``height``, ``r_ref``, ``beta``, ``ripple_*`` set ``H(r)``
    (:func:`disc_height`, light-days); ``r_out`` is the disc's outer edge
    (light-days; for a rim, ``r_out = r_ref``); ``lamppost_height_rs`` is the
    lamppost's height in Schwarzschild radii and ``lamppost_efficiency`` its
    reprocessed luminosity in units of ``Mdot c**2`` (the paper's
    ``eps_LP = L_LP (1 - albedo) / (Mdot c**2)``), with ``Mdot`` the one that
    gives this ``log_mdot``'s ``T_1``. All of these are fixed geometry, not
    sampled parameters. ``exact_irradiation`` is as in :func:`disc_surface`.
    ``n_r``/``n_phi`` set the quadrature (the radial grid is
    refined automatically across a rim and ripples); the smoothing arguments are
    as in :func:`pycream2.forward_model.thin_disk_response`.

    ``tau_grid`` must extend past the longest delay that matters (about
    ``r_out (1 + sin i)``): light arriving later is dropped before normalising.
    """
    e = response_elements(log_mdot, wavelength, inclination, M_BH, n_phi=n_phi, height=height, r_ref=r_ref,
                          beta=beta, ripple_amplitude=ripple_amplitude, ripple_wavelength=ripple_wavelength,
                          ripple_slope=ripple_slope, r_out=r_out, lamppost_height_rs=lamppost_height_rs,
                          lamppost_efficiency=lamppost_efficiency, exact_irradiation=exact_irradiation, n_r=n_r)
    delay, weight = e["delay"].reshape(-1), e["weight"].reshape(-1)

    # Deposit each element on the lag grid with linear (cloud-in-cell) weights,
    # then turn masses into a density with the grid's cell widths.
    tau_grid = jnp.asarray(tau_grid)
    n_tau = tau_grid.shape[0]
    idx = jnp.clip(jnp.searchsorted(tau_grid, delay) - 1, 0, n_tau - 2)
    lo, hi = tau_grid[idx], tau_grid[idx + 1]
    frac = jnp.clip((delay - lo) / jnp.clip(hi - lo, 1e-12, None), 0.0, 1.0)
    inside = (delay >= tau_grid[0]) & (delay <= tau_grid[-1])
    w = jnp.where(inside, weight, 0.0)
    mass = jnp.zeros(n_tau).at[idx].add(w * (1.0 - frac)).at[idx + 1].add(w * frac)
    gaps = jnp.diff(tau_grid)
    cell = 0.5 * (jnp.concatenate([gaps, jnp.zeros(1)]) + jnp.concatenate([jnp.zeros(1), gaps]))
    raw = mass / jnp.clip(cell, 1e-12, None)
    raw = jnp.where(tau_grid >= 0.0, raw, 0.0)
    r_scale = wien_radius(log_mdot, wavelength, M_BH)
    return _smooth_response(raw, tau_grid, r_scale, smoothing_frac, smoothing_days, smoothing_log)


def rippled_disc_fnu(lam_rest_angstrom, log_mdot, inclination, M_BH, dl_mpc: float, redshift: float = 0.0,
                     n_phi: int = 90, **geometry):
    """Observed flux density (mJy) of a rimmed or rippled disc at luminosity
    distance ``dl_mpc`` (Starkey et al. 2023 eq. 9, with each surface element's
    projection towards the observer, ``cos(i) - H' sin(i) cos(phi)``, in place of
    ``cos(i)``): ``(1 + z) / D_L**2 int B_nu(nu_rest, T) dA_projected``.
    ``geometry`` takes :func:`disc_surface`'s keywords. Plain numpy (float64), for
    spectra and figures; not differentiable."""
    s = disc_surface(log_mdot, M_BH, **geometry)
    r, dr = np.asarray(s["r"], dtype=float), np.asarray(s["dr"], dtype=float)
    dH, T = np.asarray(s["dH"], dtype=float), np.asarray(s["T"], dtype=float)
    incl = np.deg2rad(float(inclination))
    phi = (np.arange(n_phi) + 0.5) * (np.pi / n_phi)
    proj = np.clip(np.cos(incl) - dH[:, None] * np.sin(incl) * np.cos(phi)[None, :], 0.0, None)
    area = 2.0 * (proj * (np.pi / n_phi)).sum(axis=1) * r * dr * _LIGHT_DAY_CM ** 2  # (n_r,), cm^2
    nu = _C_CGS / (np.asarray(lam_rest_angstrom, dtype=float)[:, None] * 1e-8)
    x = _H_CGS * nu / (_K_CGS * np.clip(T, 1e-3, None)[None, :])
    bnu = np.where(T[None, :] > 0, 2 * _H_CGS * nu ** 3 / _C_CGS ** 2 / np.expm1(np.clip(x, 1e-6, 700)), 0.0)
    return (1 + redshift) * (bnu * area[None, :]).sum(axis=1) / (dl_mpc * _MPC_CM) ** 2 * 1e26


def mean_delay(tau_grid, psi):
    """Mean of a response on its lag grid (trapezoid rule)."""
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    tau, psi = np.asarray(tau_grid, dtype=float), np.asarray(psi, dtype=float)
    return float(trapz(tau * psi, tau) / trapz(psi, tau))
