"""
The disc's spectrum, its luminosity distance and the Hubble constant from a
fitted :class:`~pycream2.echofit.EchoFit`: the test of Cackett, Horne & Winkler
(2007, MNRAS 380, 669), who found H_0 = 44 +/- 5 km/s/Mpc from 14 AGN discs.

The delays fix the disc's temperature profile in light-days,
``T(r) = T_1 r^-alpha (1 - sqrt(r_in/r))^(1/4)`` (``T_1`` from ``log_mdot`` and
``M_BH``, :func:`~pycream2.forward_model.disk_t1_kelvin`), and the fit gives its
inclination. Such a disc seen from a luminosity distance ``D_L`` has

    f_nu(nu_obs) = (1 + z) cos(i) / D_L^2  int B_nu(nu_obs (1 + z), T(r)) 2 pi r dr,

so the observed disc flux gives ``D_L``, and with the redshift, ``H_0``. The
analysis, per posterior draw:

1. **Flux-flux decomposition.** Each band's model light curve is
   ``C + S X(t)``. The constant part ``C`` includes the host galaxy (and any
   other non-varying light). As in flux-flux analyses, the disc's zero point
   ``X_0`` is where the bluest band's (``host_band``'s) model flux vanishes,
   i.e. that band is assumed to have no constant component. The disc's flux at
   the mean state is then ``S (<X> - X_0)``, and the constant component
   ``C + S X_0``. With a diffuse continuum (``diffuse_continuum=True``), only
   the disc's share ``1 - dce_fraction`` of ``S`` counts.
2. **Dust.** Fluxes are converted to mJy and corrected for Galactic extinction
   (Cardelli, Clayton & Mathis 1989, R_V = 3.1, at the observed wavelength).
   Optionally an intrinsic ``E(B-V)`` (same law, at the rest wavelength) is
   fitted together with the distance, as Cackett et al. (2007) did.
3. **Variable SED.** The bright-minus-faint flux, ``S (X_98 - X_2)``, is
   compared with the spectrum of the disc's response to the lamppost,
   ``int dB_nu/dT T^-3 h/x^3 2 pi r dr`` (its shape only: the lamppost's
   luminosity change is not known), and a power law ``nu^p`` fitted to it gives
   ``alpha_var = 1/(p + 1)``, the temperature slope that the variable SED alone
   implies (for ``delta T^4 ~ h/x^3``).
4. **Distance.** Two estimators, both reported:

   - ``distance_method="host_band"`` (default): the host band's whole mean
     flux is disc (the one assumption of the method), so it alone gives
     ``D_L``; every other band's host flux is then its mean flux minus the
     model disc at that distance (a check: it should be positive and
     galaxy-like).
   - ``distance_method="flux_flux"``: ``D_L`` (and the intrinsic ``E(B-V)``,
     if fitted) by least squares in log flux over every band's flux-flux disc
     flux; per-band distances test the SED's shape. The flux-flux
     decomposition assumes the variable SED has the mean disc's shape. The
     disc's response to the lamppost is bluer than the disc itself, so this
     leaves disc light in the redder bands' "host" and biases ``D_L`` high:
     by about 10% in ``H_0`` in the worked example of ``docs/disc_sed.md``.

   ``H_0`` follows in flat Lambda-CDM.

The band wavelengths are taken to be rest-frame, as the disc physics needs
(:meth:`EchoFit.add_lightcurve`); observed wavelengths are ``(1 + z)`` times
them. Only bands with a wavelength enter (not a driver light curve).

Caveats worth stating with any result: the zero point assumes the bluest band
has no constant component; a slow, non-reverberating trend left unmodelled is
absorbed into long responses and makes the disc look hotter and so the
distance larger (fit it with ``background_order``); and the absolute
temperature scale depends on the response weighting (pycream2's lamppost
geometry).
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .forward_model import disk_t1_kelvin, driver_at

FLUX_UNITS = ("mJy", "f_lambda")
LIGHT_DAY_CM = 2.59020684e15
MPC_CM = 3.0857e24
C_KMS = 2.99792458e5
_H, _C, _K = 6.62607e-27, 2.99792458e10, 1.380649e-16
_G, _MSUN = 6.674e-8, 1.98855e33
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def ccm_a_lambda(lam_angstrom, ebv: float, r_v: float = 3.1):
    """Extinction ``A_lambda`` (mag) of Cardelli, Clayton & Mathis (1989), for
    wavelengths 1000 A to 3.3 micron (infrared, optical, UV and far-UV branches)."""
    x = 1e4 / np.asarray(lam_angstrom, dtype=float)
    a, b = np.zeros_like(x), np.zeros_like(x)
    ir = (x >= 0.3) & (x < 1.1)
    a[ir], b[ir] = 0.574 * x[ir] ** 1.61, -0.527 * x[ir] ** 1.61
    op = (x >= 1.1) & (x < 3.3)
    y = x[op] - 1.82
    a[op] = (1 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 + 0.72085 * y ** 4 + 0.01979 * y ** 5
             - 0.77530 * y ** 6 + 0.32999 * y ** 7)
    b[op] = (1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 - 5.38434 * y ** 4 - 0.62251 * y ** 5
             + 5.30260 * y ** 6 - 2.09002 * y ** 7)
    uv = (x >= 3.3) & (x < 8.0)
    xu = x[uv]
    fa = np.where(xu >= 5.9, -0.04473 * (xu - 5.9) ** 2 - 0.009779 * (xu - 5.9) ** 3, 0.0)
    fb = np.where(xu >= 5.9, 0.2130 * (xu - 5.9) ** 2 + 0.1207 * (xu - 5.9) ** 3, 0.0)
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + fa
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + fb
    fu = (x >= 8.0) & (x <= 10.0)
    xf = x[fu] - 8.0
    a[fu] = -1.073 - 0.628 * xf + 0.137 * xf ** 2 - 0.070 * xf ** 3
    b[fu] = 13.670 + 4.257 * xf - 0.420 * xf ** 2 + 0.374 * xf ** 3
    return r_v * ebv * (a + b / r_v)


def to_mjy(lam_obs_angstrom, flux_unit: str, flux_scale: float = 1.0):
    """Factor converting a light curve's flux to mJy: ``flux_unit="mJy"`` (times
    ``flux_scale``), or ``"f_lambda"`` with ``flux_scale`` its unit in
    erg/s/cm^2/A (e.g. 1e-15)."""
    if flux_unit not in FLUX_UNITS:
        raise ValueError(f"flux_unit must be one of {FLUX_UNITS}, got {flux_unit!r}")
    lam = np.asarray(lam_obs_angstrom, dtype=float)
    if flux_unit == "mJy":
        return np.full_like(lam, float(flux_scale))
    return float(flux_scale) * lam ** 2 / (_C * 1e8) * 1e26  # f_nu = f_lambda lambda^2 / c


def _disc_grid(t1_k, alpha, m_bh, r_out_ld=1000.0, n_r=4000):
    rs_ld = 2 * _G * m_bh * _MSUN / _C ** 2 / LIGHT_DAY_CM
    r_in = 3 * rs_ld
    r = np.logspace(np.log10(r_in * 1.0001), np.log10(r_out_ld), n_r)
    temp = np.float64(t1_k) * r ** (-np.float64(alpha)) * (1 - np.sqrt(r_in / r)) ** 0.25
    return r, rs_ld, temp


def disc_fnu_at_1cm(lam_rest_angstrom, redshift, t1_k, alpha, incl_deg, m_bh):
    """Observed flux density (mJy) of the model disc at ``D_L = 1 cm``:
    ``(1 + z) cos(i) int B_nu(nu_rest, T(r)) 2 pi r dr``, per wavelength."""
    r, _, temp = _disc_grid(t1_k, alpha, m_bh)
    nu = _C / (np.asarray(lam_rest_angstrom, dtype=float)[:, None] * 1e-8)
    x = np.clip(_H * nu / (_K * temp[None, :]), 1e-6, 700)
    bnu = 2 * _H * nu ** 3 / _C ** 2 / np.expm1(x)
    r_cm = r * LIGHT_DAY_CM
    integral = _trapezoid(bnu * 2 * np.pi * r_cm[None, :], r_cm, axis=1)
    return (1 + redshift) * np.cos(np.deg2rad(np.float64(incl_deg))) * integral * 1e26


def response_sed_shape(lam_rest_angstrom, t1_k, alpha, m_bh, lamppost_height_rs=3.0):
    """Spectrum of the disc's response to the lamppost, ``int dB_nu/dT T^-3
    h/x^3 2 pi r dr`` (arbitrary units), per wavelength: the variable SED's
    predicted shape for the fitted temperature profile."""
    r, rs_ld, temp = _disc_grid(t1_k, alpha, m_bh)
    h = lamppost_height_rs * rs_ld
    nu = _C / (np.asarray(lam_rest_angstrom, dtype=float)[:, None] * 1e-8)
    x = np.clip(_H * nu / (_K * temp[None, :]), 1e-6, 700)
    db_dt = 2 * _H * nu ** 3 / _C ** 2 * x / temp[None, :] * np.exp(-x) / (-np.expm1(-x)) ** 2
    weight = temp[None, :] ** -3 * h / (r ** 2 + h ** 2) ** 1.5
    return _trapezoid(db_dt * weight * 2 * np.pi * r[None, :], r, axis=1)


def h0_from_dl(dl_mpc, redshift, omega_m: float = 0.3):
    """``H_0`` (km/s/Mpc) giving luminosity distance ``dl_mpc`` at ``redshift``
    in flat Lambda-CDM."""
    z = np.linspace(0.0, redshift, 2001)
    comoving = _trapezoid(1.0 / np.sqrt(omega_m * (1 + z) ** 3 + 1 - omega_m), z)
    return (1 + redshift) * C_KMS * comoving / np.asarray(dl_mpc, dtype=float)


def _q(a, axis=-1):
    return [float(v) for v in np.nanpercentile(a, [16, 50, 84], axis=axis)]


def disc_sed_analysis(
    ef, redshift: float, flux_unit: str = "mJy", flux_scale: float = 1.0, ebv_galactic: float = 0.0,
    fit_intrinsic_ebv: bool = False, omega_m: float = 0.3, host_band: Optional[str] = None,
    lamppost_height_rs: float = 3.0, n_draws: int = 300, seed: int = 0, distance_method: str = "host_band",
) -> Dict:
    """Run the analysis described in the module docstring on a fitted ``ef``.

    Returns a dict: ``"summary"`` (JSON-serialisable: 16/50/84th percentiles of
    ``D_L`` (Mpc), ``H_0``, the intrinsic ``E(B-V)``, ``alpha_var``, ``T_1``,
    ``alpha``, the inclination, and per band the variable, disc, total and
    constant fluxes (mJy, dereddened) and the per-band distance) and
    ``"draws"`` (per-draw arrays, for :func:`pycream2.plotting.plot_disc_sed`).
    """
    if distance_method not in ("host_band", "flux_flux"):
        raise ValueError(f"distance_method must be 'host_band' or 'flux_flux', got {distance_method!r}")
    if fit_intrinsic_ebv and distance_method != "flux_flux":
        raise ValueError("fit_intrinsic_ebv needs several bands' disc fluxes: use distance_method='flux_flux'.")
    s = ef.samples
    if s is None:
        raise RuntimeError("disc_sed_analysis: fit or optimise first.")
    if "log_mdot" not in s:
        raise ValueError("disc_sed_analysis needs a disc model: at least one lag_mode='physical' band.")
    if "S" not in s or "C" not in s:
        raise ValueError("disc_sed_analysis needs the driver's Fourier coefficients in the samples.")
    names = [n for n in ef.bands]
    lam_rest = np.array([ef.bands[n]["wavelength"] for n in names], dtype=float)
    order = np.argsort(lam_rest)
    names, lam_rest = [names[k] for k in order], lam_rest[order]
    lam_obs = lam_rest * (1 + redshift)
    host_band = host_band or names[0]
    if host_band not in names:
        raise ValueError(f"host_band {host_band!r} is not a registered band")
    k_host = names.index(host_band)

    n = len(np.asarray(s["log_mdot"]))
    idx = np.random.default_rng(seed).choice(n, min(n_draws, n), replace=False)
    t_all = np.concatenate([np.asarray(ef.bands[b]["t"]) for b in names])
    t_fine = jnp.linspace(float(t_all.min()), float(t_all.max()), 1500)
    x = np.asarray(jax.vmap(lambda a, b: driver_at(a, b, ef.freqs, t_fine))(
        jnp.asarray(np.asarray(s["S"])[idx]), jnp.asarray(np.asarray(s["C"])[idx])), dtype=np.float64)
    S = np.array([np.asarray(s[f"S_{b}"])[idx] for b in names], dtype=np.float64)  # (band, draw)
    C = np.array([np.asarray(s[f"C_{b}"])[idx] for b in names], dtype=np.float64)
    disc_share = np.array([1.0 - np.asarray(s[f"dce_fraction_{b}"])[idx] if f"dce_fraction_{b}" in s
                           else np.ones(len(idx)) for b in names])

    unit_mjy = to_mjy(lam_obs, flux_unit, flux_scale)[:, None]
    deredden = 10 ** (0.4 * ccm_a_lambda(lam_obs, ebv_galactic))[:, None]
    conv = unit_mjy * deredden
    x2, x98, xmean = np.percentile(x, 2, axis=1), np.percentile(x, 98, axis=1), x.mean(axis=1)
    x0 = -C[k_host] / S[k_host]
    variable = S * (x98 - x2) * conv
    disc_mean = disc_share * S * (xmean - x0) * conv
    total_mean = (C + S * xmean) * conv
    constant = (C + S * x0) * conv

    t1 = np.asarray(disk_t1_kelvin(jnp.asarray(np.asarray(s["log_mdot"])[idx]), ef.M_BH), dtype=np.float64)
    alpha = (np.asarray(s["temperature_slope"])[idx] if "temperature_slope" in s
             else np.full(len(idx), 0.75))
    incl = np.asarray(s["inclination"])[idx] if "inclination" in s else np.full(len(idx), 0.0)
    unit = np.array([disc_fnu_at_1cm(lam_rest, redshift, a, b, c, ef.M_BH)
                     for a, b, c in zip(t1, alpha, incl)]).T  # (band, draw), mJy at 1 cm
    shape = np.array([response_sed_shape(lam_rest, a, b, ef.M_BH, lamppost_height_rs)
                      for a, b in zip(t1, alpha)]).T
    k_int = ccm_a_lambda(lam_rest, 1.0) / 1.0  # A_lambda per unit intrinsic E(B-V), at rest wavelength

    # log10 f_disc = log10 unit - 2 log10 D - 0.4 E k: least squares per draw.
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.log10(unit) - np.log10(disc_mean)
    if fit_intrinsic_ebv:
        design = np.column_stack([np.full(len(names), 2.0), 0.4 * k_int])  # y = 2 log10 D + 0.4 E k
        sol = np.linalg.lstsq(design, np.nan_to_num(y, nan=0.0), rcond=None)[0]  # (2, draw)
        log_d, ebv_int = sol[0], sol[1]
    else:
        log_d, ebv_int = np.nanmean(y, axis=0) / 2.0, np.zeros(len(idx))
    dl_flux_flux = 10 ** log_d / MPC_CM
    with np.errstate(invalid="ignore"):
        dl_band = np.sqrt(unit * 10 ** (-0.4 * k_int[:, None] * ebv_int[None, :]) / disc_mean) / MPC_CM
        dl_host_band = np.sqrt(unit[k_host] / (total_mean[k_host] * disc_share[k_host])) / MPC_CM
    dl_mpc = dl_host_band if distance_method == "host_band" else dl_flux_flux
    model_disc = unit * 10 ** (-0.4 * k_int[:, None] * ebv_int[None, :]) / (dl_mpc * MPC_CM) ** 2
    if np.mean(np.all(disc_mean > 0, axis=0)) < 0.5:
        warnings.warn(
            "disc_sed_analysis: the disc's mean flux is not positive in every band for most draws: the zero "
            f"point (where {host_band}'s model flux vanishes) lies above the mean state. Are the light curves "
            "absolute fluxes (not mean-subtracted or normalised)? Distances from such bands are NaN.")
    if distance_method == "host_band":
        # The disc at that distance; the rest of each band's mean flux is host
        # (and, with a diffuse continuum, its share of the reverberating light,
        # taken to be the same fraction of the mean as of the variable flux).
        disc_mean = model_disc
        constant = total_mean - model_disc / disc_share
        negative = [b for k, b in enumerate(names) if np.mean(constant[k] < 0) > 0.5]
        if negative:
            warnings.warn(
                f"disc_sed_analysis: the model disc is brighter than the observed mean flux in {negative}, "
                "i.e. a negative host: the fitted disc's spectrum is too red for the data.")
    h0 = h0_from_dl(dl_mpc, redshift, omega_m)

    nu = _C / (lam_obs * 1e-8)
    p = np.array([np.polyfit(np.log(nu), np.log(np.abs(v)), 1)[0] for v in variable.T])
    reddened_shape = shape * 10 ** (-0.4 * k_int[:, None] * ebv_int[None, :])
    scale = np.exp(np.mean(np.log(np.abs(variable)) - np.log(reddened_shape), axis=0))
    predicted_variable = reddened_shape * scale

    summary = dict(
        redshift=float(redshift), flux_unit=flux_unit, flux_scale=float(flux_scale),
        ebv_galactic=float(ebv_galactic), fit_intrinsic_ebv=bool(fit_intrinsic_ebv), omega_m=float(omega_m),
        host_band=host_band, n_draws=int(len(idx)),
        distance_method=distance_method, dl_mpc=_q(dl_mpc), h0=_q(h0), ebv_intrinsic=_q(ebv_int),
        dl_mpc_host_band=_q(dl_host_band), h0_host_band=_q(h0_from_dl(dl_host_band, redshift, omega_m)),
        dl_mpc_flux_flux=_q(dl_flux_flux), h0_flux_flux=_q(h0_from_dl(dl_flux_flux, redshift, omega_m)),
        alpha_var=_q(1.0 / (p + 1.0)), variable_slope_p=_q(p),
        t1_kelvin=_q(t1), alpha=_q(alpha), inclination=_q(incl),
        zero_point_below_faint_state=float(np.mean(x0 < x2)),
        bands={b: dict(lam_rest=float(lam_rest[k]), lam_obs=float(lam_obs[k]),
                       variable_mjy=_q(variable[k]), predicted_variable_mjy=_q(predicted_variable[k]),
                       disc_mean_mjy=_q(disc_mean[k]), total_mean_mjy=_q(total_mean[k]),
                       constant_mjy=_q(constant[k]), model_disc_mjy=_q(model_disc[k]), dl_mpc=_q(dl_band[k]))
               for k, b in enumerate(names)},
    )
    draws = dict(names=names, lam_rest=lam_rest, x=x, t_fine=np.asarray(t_fine), x0=x0, xmean=xmean, x2=x2,
                 x98=x98, S=S, C=C, conv=conv[:, 0], variable=variable, predicted_variable=predicted_variable,
                 disc_mean=disc_mean, model_disc=model_disc, total_mean=total_mean, constant=constant,
                 dl_mpc=dl_mpc, h0=h0, ebv_intrinsic=ebv_int, distance_method=distance_method)
    return dict(summary=summary, draws=draws)
