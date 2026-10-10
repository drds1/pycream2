"""
echofit.py
==========

``EchoFit`` is the main user-facing class: collect light curves, build the
shared frequency/lag grids, run NUTS, and plot the results.

Optionally manages on-disk run outputs -- pass ``title=`` to enable it.
``.fit()`` then writes the light curve data, config, periodic checkpoints,
final posterior (as an ArviZ netCDF), and a visual report.html to
``<output_root>/<title>/run_<timestamp>/``. An interrupted fit can be
picked back up with ``EchoFit.resume(title)``. See ``run_manager.py`` for
the on-disk layout and output-directory resolution rules. Without
``title``, EchoFit behaves exactly as a purely in-memory fit -- nothing is
written to disk.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp

from . import model as _model
from .model import reverberation_model
from .inference import run_mcmc, run_mcmc_chunked
from .forward_model import (
    transfer_coeffs, compute_echo, driver_at, tophat_response_free, transfer_matrices, fourier_basis,
    legendre_background_basis, mix_diffuse_continuum,
)
from .grid_utils import estimate_dt_min, graded_tau_grid, check_tau_grid_resolution, hybrid_frequency_grid
from .nested_laplace import fd_hessian as _fd_hessian, lbfgs_sd as _lbfgs_sd
from . import disc_sed
from . import plotting
from . import reporting
from . import run_manager

_UNSET = object()

# build_grid()'s default longest driver period, as a multiple of the window the
# driver must cover (baseline + tau_max); see its period_max docstring.
PERIOD_MAX_FACTOR = 2.0

# build_grid()'s default highest driver frequency (cycles per day) and the
# fractional step of the grid's logarithmic part; see its f_max docstring.
F_MAX_CYCLES_PER_DAY = 2.0
FREQ_LOG_STEP = 0.03

# add_lightcurve()/add_driver_lightcurve()'s default background order: a linear
# trend in every light curve, for trends longer than the driver's longest period.
DEFAULT_BACKGROUND_ORDER = 1


# optimise()'s multi-start check: a restart "agrees" with the best optimum if
# every parameter lands within this many Laplace posterior standard deviations
# of it, i.e. well inside the posterior's own width.
RESTART_AGREEMENT_SD = 0.5

# optimise()'s Laplace curvature check: an exact Hessian eigenvalue is replaced
# by the potential's finite-difference curvature one posterior standard
# deviation along its eigenvector when it is not positive or differs from it by
# more than this factor. Not tighter: a first version replaced anything off by
# 50%, which also caught merely anharmonic directions (e.g. a jitter near
# zero, 2-3x steeper over one sd than at the peak), where the pointwise
# Hessian is the better Gaussian (PSIS k-hat 0.49 against 0.74 on NGC 5548).
# The ripple artefacts this exists for are off by far more (negative, -1000,
# 1e6 against ~1).
CURVATURE_RATIO_LIMIT = 4.0

# optimise()'s objective returns this (with a zero gradient) wherever the
# potential or its gradient is not finite, so L-BFGS's line search backtracks.
_NONFINITE_POTENTIAL = 1e10


def _posterior_scale_curvature(batch_potential, z, f0, eigvals, eigvecs, min_step=0.05, n_iter=3):
    """Curvature of the potential along each Hessian eigenvector, measured over
    one posterior standard deviation rather than infinitesimally.

    The Laplace approximation needs the posterior's curvature on the scale of
    its own width. The exact Hessian gives it at a point, which small-scale
    structure in the potential can dominate: on NGC 5548-cadence thin-disc
    fits, U has ripples ~1e-4 deep in inclination (and ~1e-3 in float32), which
    gave the Hessian a negative curvature (or -1000, or +1e6) where the
    finite-difference curvature over the posterior's width was +1.0. The
    optimiser then reported steps of hundreds of standard deviations, and the
    Laplace posterior came out absurdly broad or narrow in that direction.

    For each eigenvector ``v`` the curvature is the central difference
    ``c = (U(z+hv) + U(z-hv) - 2U(z)) / h**2``, with the step iterated to one
    standard deviation of the curvature it measures, ``h = 1/sqrt(c)``,
    starting from ``max(1/sqrt(|lam|), min_step)`` and capped at 1
    unconstrained unit. Starting at ``min_step`` or wider means a spurious
    spike in ``lam`` can't shrink the probe inside the feature that caused it;
    a genuinely stiff direction just converges back down to its own width.
    ``lam`` is replaced by ``c`` when ``c > 0`` and ``lam <= 0`` or the two
    differ by more than a factor ``CURVATURE_RATIO_LIMIT``. Along a
    quadratic, or merely anharmonic, direction nothing changes. Returns the corrected
    eigenvalues and how many were replaced.
    """
    n = len(eigvals)
    h = np.clip(np.maximum(1.0 / np.sqrt(np.maximum(np.abs(eigvals), 1e-12)), min_step), 1e-4, 1.0)
    c = np.asarray(eigvals, dtype=np.float64)
    for _ in range(n_iter):
        points = np.concatenate([z[None] + h[:, None] * eigvecs.T, z[None] - h[:, None] * eigvecs.T])
        u = np.asarray(batch_potential(points), dtype=np.float64)
        c = (u[:n] + u[n:] - 2.0 * f0) / h ** 2
        h_next = np.where(np.isfinite(c) & (c > 0), np.clip(1.0 / np.sqrt(np.maximum(c, 1e-12)), 1e-4, 1.0), h)
        if np.allclose(h_next, h, rtol=0.1):
            break
        h = h_next
    ratio = c / np.where(eigvals > 0, eigvals, np.nan)
    off = (eigvals <= 0) | ~((ratio <= CURVATURE_RATIO_LIMIT) & (ratio >= 1.0 / CURVATURE_RATIO_LIMIT))
    replace = np.isfinite(c) & (c > 0) & off
    return np.where(replace, c, eigvals), int(replace.sum())


def _merge_dicts(dicts) -> dict:
    """Concatenate (along axis 0) a list of ``{name: array}`` dicts, skipping
    any that are empty (e.g. "no prior checkpoint to merge in")."""
    dicts = [d for d in dicts if d]
    if not dicts:
        return {}
    return {k: np.concatenate([d[k] for d in dicts], axis=0) for k in dicts[0]}


def _initialize_marginal_model(rng_seed, kwargs, init_strategy):
    """NumPyro's ``initialize_model`` for the marginal model, without its
    eager gradient check (``validate_grad=False``): run eagerly, that gradient
    compiled ~280 small one-off operations (~20 s on a 5-parameter problem).
    The solvers evaluate the compiled gradient at the start straight away and
    handle a non-finite one."""
    from numpyro.infer.util import initialize_model

    return initialize_model(jax.random.PRNGKey(rng_seed), reverberation_model, model_kwargs=kwargs,
                            init_strategy=init_strategy, validate_grad=False)


class _LinearDraws:
    """Makes a direct solve's exact linear draws once, on first request (see
    ``EchoFit._set_lazy_linear_draws``), shared by both lazy sample dicts."""

    def __init__(self, ef, nonlinear_by_chain, rng_seed, sites):
        self._ef, self._nonlinear, self._seed, self.sites = ef, nonlinear_by_chain, rng_seed, set(sites)
        self._drawn = None

    def get(self) -> dict:
        if self._drawn is None:
            ef = self._ef
            was_marginal = ef.marginalise_linear
            ef.marginalise_linear = True
            try:
                self._drawn = ef._add_linear_draws(self._nonlinear, self._seed)
            finally:
                ef.marginalise_linear = was_marginal
        return self._drawn


class _LazySamples(dict):
    """A posterior-sample dict whose linear sites are drawn on first use.
    ``d[name]`` and ``name in d`` for a parameter already present cost
    nothing; reading a pending linear site, or anything that walks the whole
    dict (iteration, ``keys``/``values``/``items``, ``len``, ``copy``, ``dict(d)``,
    ``np.savez(**d)``), draws them first."""

    def __init__(self, data, resolver, by_chain):
        super().__init__(data)
        self._resolver, self._by_chain = resolver, by_chain

    def _fill(self):
        if self._resolver is not None:
            resolver, self._resolver = self._resolver, None
            drawn = resolver.get()
            dict.update(self, {k: (v if self._by_chain else v[0]) for k, v in drawn.items()
                               if not dict.__contains__(self, k)})

    def __getitem__(self, key):
        if not dict.__contains__(self, key) and self._resolver is not None and key in self._resolver.sites:
            self._fill()
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        return dict.__contains__(self, key) or (self._resolver is not None and key in self._resolver.sites)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __iter__(self):
        self._fill()
        return dict.__iter__(self)

    def __len__(self):
        self._fill()
        return dict.__len__(self)

    def keys(self):
        self._fill()
        return dict.keys(self)

    def values(self):
        self._fill()
        return dict.values(self)

    def items(self):
        self._fill()
        return dict.items(self)

    def copy(self):
        self._fill()
        return dict(dict.items(self))

    def __repr__(self):
        self._fill()
        return dict.__repr__(self)

    def __reduce__(self):
        self._fill()
        return (dict, (dict(dict.items(self)),))


class EchoFit:
    """Bayesian AGN reverberation-mapping fit for multi-band light curves.

    Parameters
    ----------
    M_BH : float, optional
        Fixed black hole mass, solar masses. Never inferred. Required only
        if at least one band is added with ``lag_mode="physical"`` (the
        default for ``add_lightcurve``) -- a purely free-lag fit (see
        ``add_lightcurve``'s ``lag_mode`` and ``add_driver_lightcurve``)
        doesn't use it and may leave it as ``None``.
    title : str, optional
        A name for this fit (e.g. an AGN name like ``"ngc_5548"``). If
        given, ``.fit()`` writes its outputs to
        ``<output_root>/<title>/run_<timestamp>/`` -- see
        :func:`~pycream2.run_manager.resolve_output_root`. If omitted,
        nothing is written to disk (the original, fully in-memory
        behaviour).
    output_dir : str, optional
        Override the output root directory. Only relevant when ``title``
        is given. Otherwise resolved from the ``PYCREAM2_OUTPUT_DIR``
        environment variable, or ``./outputs`` if that's unset too.
    drw_prior : bool, optional
        ``False`` (default): the driver's Fourier coefficients follow a
        pure random-walk (RW) prior (a plain ``1/w**2`` power law, only
        ``sigma_drw`` inferred, no ``tau_drw`` site). ``True``: the
        original damped random walk (DRW) prior (a Lorentzian, ``tau_drw``
        inferred too) -- see ``model.py``'s "random-walk prior" note for
        why RW is the default and why this isn't the same as fixing
        ``tau_drw`` to a large constant.
    fixed_params : dict, optional
        ``{site_name: value}`` to hold any of the model's scalar sites
        (``sigma_drw``, ``tau_drw`` if ``drw_prior=True``, ``log_mdot``,
        ``inclination``, ``S_driver``, ``C_driver``, ``S_{band}``,
        ``C_{band}``, or a free-lag band's ``tau_{band}``) fixed instead
        of inferring it -- e.g. ``fixed_params={"inclination": 0.0}`` to
        assume a face-on disk. The same mechanism as ``M_BH`` (always
        fixed, decision #3), generalised to any parameter -- see
        ``model.py``'s "fixed-parameter"
        docstring note. Validated against the actually-registered
        bands/driver in ``.fit()`` (a key that could never be a real site
        given the current setup raises, to catch typos).
    fit_temperature_slope : bool, optional
        ``False`` (default): the disk's temperature slope is fixed at
        ``T ~ r**(-3/4)``. ``True``: infer it as ``temperature_slope``
        (``alpha`` in ``T ~ r**(-alpha)``, prior ``Uniform(0.5, 2.5)``,
        ``model.TEMPERATURE_SLOPE_PRIOR``), the
        free-slope counterpart of Starkey et al. (2017)'s Model 2. Needs a
        response function that accepts ``viscous_slope`` (e.g.
        ``forward_model.thin_disk_response``; the default skew-normal does
        not). Delays then scale as ``wavelength**(1/alpha)``.
    redshift : float, optional
        The AGN's redshift. Needed only for the disc SED, distance and
        ``H_0`` analysis (``sed_analysis``, :meth:`disc_sed_analysis`); band
        wavelengths are taken to be rest-frame, as the disc physics needs.
    flux_unit : str, optional
        The light curves' flux unit, for the SED analysis: ``"mJy"``
        (default) or ``"f_lambda"``, with ``flux_scale`` the unit in
        erg/s/cm^2/A (e.g. ``1e-15``). For ``"mJy"``, ``flux_scale``
        multiplies the fluxes (default 1).
    flux_scale : float, optional
        See ``flux_unit``.
    ebv_galactic : float, optional
        Galactic ``E(B-V)`` towards the AGN (e.g. Schlafly & Finkbeiner
        2011), corrected for in the SED analysis. Default 0.
    sed_analysis : bool, optional
        ``True``: run :meth:`disc_sed_analysis` automatically at the end of
        every ``.fit()`` and ``.optimise()`` (needs ``redshift``), and include
        it in the report. Default ``False``. A failure there only warns, so it
        never costs a finished fit.
    sed_options : dict, optional
        Further keyword arguments for
        :func:`pycream2.disc_sed.disc_sed_analysis` (``fit_intrinsic_ebv``,
        ``omega_m``, ``host_band``, ``lamppost_height_rs``, ``n_draws``,
        ``include_irradiation``, ``irradiation_weight``; pass
        ``include_irradiation=False`` if the fit used a viscous-only response).

    Examples
    --------
    >>> ef = EchoFit(M_BH=1e8, title="ngc_5548")
    >>> ef.add_lightcurve("g", wavelength=4770.0, t=t_g, y=y_g, yerr=yerr_g)
    >>> ef.add_lightcurve("i", wavelength=7625.0, t=t_i, y=y_i, yerr=yerr_i)
    >>> ef.build_grid()
    >>> ef.fit(num_warmup=1000, num_samples=1000)  # writes outputs/ngc_5548/run_.../
    >>> ef.plot_lightcurve_fits()

    If that fit is interrupted (killed, crashed, ...), resume it with::

    >>> ef = EchoFit.resume("ngc_5548")
    >>> ef.fit()  # continues from the last checkpoint, same settings as before

    Assume a face-on disk (fixed inclination) to make the remaining
    parameters easier to solve for::

    >>> ef = EchoFit(M_BH=1e8, fixed_params={"inclination": 0.0})
    """

    def __init__(
        self, M_BH: Optional[float] = None, title: Optional[str] = None, output_dir: Optional[str] = None,
        fixed_params: Optional[Dict[str, float]] = None, drw_prior: bool = False,
        marginalise_linear: bool = False, fit_temperature_slope: bool = False,
        redshift: Optional[float] = None, flux_unit: str = "mJy", flux_scale: float = 1.0,
        ebv_galactic: float = 0.0, sed_analysis: bool = False, sed_options: Optional[dict] = None,
    ):
        self.M_BH = float(M_BH) if M_BH is not None else None
        self.title = title
        self.fixed_params: Dict[str, float] = dict(fixed_params) if fixed_params else {}
        self.drw_prior = bool(drw_prior)
        self.marginalise_linear = bool(marginalise_linear)
        self.fit_temperature_slope = bool(fit_temperature_slope)
        if flux_unit not in disc_sed.FLUX_UNITS:
            raise ValueError(f"flux_unit must be one of {disc_sed.FLUX_UNITS}, got {flux_unit!r}")
        if sed_analysis and redshift is None:
            raise ValueError("sed_analysis=True needs the redshift (EchoFit(..., redshift=...)).")
        self.redshift = float(redshift) if redshift is not None else None
        self.flux_unit, self.flux_scale = flux_unit, float(flux_scale)
        self.ebv_galactic = float(ebv_galactic)
        self.sed_analysis = bool(sed_analysis)
        self.sed_options: dict = dict(sed_options) if sed_options else {}
        self.disc_sed: Optional[dict] = None
        self.bands: Dict[str, dict] = {}
        self.driver_data: Optional[dict] = None
        self.freqs: Optional[np.ndarray] = None
        self.tau_grid: Optional[np.ndarray] = None
        self.mcmc = None
        self.optimum: Optional[dict] = None
        self.optimise_restarts: Optional[list] = None
        self.log_evidence: Optional[float] = None
        self.nested_laplace_result: Optional[dict] = None
        self._laplace_peak: Optional[dict] = None  # optimise(method="laplace")'s peak, for plot_landscape
        self.nested_laplace_timings: Optional[dict] = None
        self.samples: Optional[dict] = None
        self.extra_fields: dict = {}
        self._extra_fields_by_chain: dict = {}

        self._output_root = run_manager.resolve_output_root(output_dir) if title else None
        self.run_dir: Optional[Path] = None
        self._samples_by_chain: Optional[dict] = None
        self._resume_state = None
        self._fit_config: Optional[dict] = None

    # ------------------------------------------------------------------
    def add_lightcurve(
        self, name: str, wavelength: float, t, y, yerr, lag_mode: str = "physical",
        fit_error_model: bool = False, diffuse_continuum: bool = False,
        background_order: int = DEFAULT_BACKGROUND_ORDER,
    ):
        """Register a single band's (possibly irregularly sampled) light curve.

        Parameters
        ----------
        lag_mode : str
            ``"physical"`` (default): mean lag comes from
            ``lag_scaling(log_mdot, wavelength, M_BH)``, tied to every other
            physical-mode band through the shared ``log_mdot`` -- this is
            what breaks the driver's absolute-lag degeneracy across 2+ such
            bands (see the model.py module docstring). ``"free"``: an
            independently inferred ``tau_{name}``, e.g. for emission-line
            reverberation mapping where each line's lag isn't tied to the
            others by any shared physical parameter -- a fit with any
            ``"free"`` band needs a driver light curve
            (:meth:`add_driver_lightcurve`) to be identifiable; ``.fit()``
            warns if one isn't registered.
        fit_error_model : bool
            Off by default (this band's ``yerr`` is used exactly as given,
            today's behaviour). If ``True``, adds two nuisance parameters
            for this band, ``sigma_scale_{name}`` (multiplicative) and
            ``sigma_jitter_{name}`` (additive), so ``yerr`` is treated as
            only approximately right rather than exact -- see model.py's
            "error-model" docstring note and CLAUDE.md decision #18. Turn
            it on per light curve for any band whose quoted errors you
            don't fully trust; leave it off (e.g. for synthetic data, where
            ``yerr`` is correct by construction) to keep the likelihood
            exactly as before. Fix either nuisance parameter to a known
            value with ``fixed_params={"sigma_scale_{name}": 1.0}`` etc. if
            you want the model structure on but one of the two pinned.
        diffuse_continuum : bool
            Off by default. If ``True``, this band's response gets a second,
            extended reprocessor with a log-normal delay distribution, e.g.
            diffuse continuum emission from the broad-line region (Cackett,
            Zoghbi & Ulrich 2022): ``psi -> (1 - f) psi + f psi_LN``, with three
            new parameters, ``dce_fraction_{name}`` (``f``, its share of the
            band's integrated response), ``dce_delay_{name}`` (median delay,
            days) and ``dce_width_{name}`` (rms width, dex). See
            ``docs/extra_components.md``.
        background_order : int
            A slowly varying background, Legendre polynomials ``P_1 .. P_K`` in
            time over the whole campaign with coefficients ``bg_{name}`` (a
            length-``K`` vector), on top of the constant offset ``C_{name}``,
            for variability unrelated to reverberation, such as a slowly
            changing host or narrow-line contribution or a long-term trend (cf.
            detrending, Welsh 1999). Linear, so ``optimise()`` and
            ``marginalise_linear=True`` integrate it out exactly. Defaults to
            ``1`` (a linear trend, ``DEFAULT_BACKGROUND_ORDER``): a red-noise
            driver has trends longer than the driver's longest period, which
            otherwise leak into the response (on the CREAM paper's synthetic
            random-walk tests they pulled ``log_mdot`` towards high values at
            SNR 100; on data without such trends the term changed nothing).
            ``0`` keeps ``C_{name}`` as the only non-reverberating part (the
            default until October 2026); see ``docs/extra_components.md``.
        """
        if lag_mode not in ("physical", "free"):
            raise ValueError(f"lag_mode must be 'physical' or 'free', got {lag_mode!r}")
        if int(background_order) < 0:
            raise ValueError(f"background_order must be >= 0, got {background_order!r}")
        t, y, yerr = np.asarray(t, float), np.asarray(y, float), np.asarray(yerr, float)
        order = np.argsort(t)
        self.bands[name] = {
            "t": t[order],
            "y": y[order],
            "yerr": yerr[order],
            "wavelength": float(wavelength),
            "lag_mode": lag_mode,
            "fit_error_model": bool(fit_error_model),
            "diffuse_continuum": bool(diffuse_continuum),
            "background_order": int(background_order),
        }
        return self

    # ------------------------------------------------------------------
    def add_driver_lightcurve(self, t, y, yerr, fit_error_model: bool = False,
                              background_order: int = DEFAULT_BACKGROUND_ORDER):
        """Register a light curve that directly (zero-lag) observes the
        driver itself -- e.g. an X-ray/lamppost continuum, or a directly
        monitored AGN continuum anchoring an emission-line fit. Modelled as
        ``y(t) = S_driver * X(t) + C_driver`` (own flux scale/offset, no
        convolution) rather than an echo of ``X(t)``.

        Optional for a purely ``lag_mode="physical"`` fit with 2+ bands at
        different wavelengths (already identifiable via the shared
        ``log_mdot``/thin-disk scaling), but it's the only thing that
        anchors the absolute lag origin for any ``lag_mode="free"`` band --
        see ``add_lightcurve``'s ``lag_mode``.

        fit_error_model : bool
            Same meaning as ``add_lightcurve``'s: off by default, turns on
            ``sigma_scale_driver``/``sigma_jitter_driver`` if ``True``.
        background_order : int
            Same meaning, and default, as ``add_lightcurve``'s: ``K > 0`` adds
            a slow Legendre background with coefficients ``bg_driver``.
        """
        if int(background_order) < 0:
            raise ValueError(f"background_order must be >= 0, got {background_order!r}")
        t, y, yerr = np.asarray(t, float), np.asarray(y, float), np.asarray(yerr, float)
        order = np.argsort(t)
        self.driver_data = {
            "t": t[order], "y": y[order], "yerr": yerr[order],
            "fit_error_model": bool(fit_error_model),
            "background_order": int(background_order),
        }
        return self

    # ------------------------------------------------------------------
    def build_grid(
        self,
        n_freq: Optional[int] = None,
        n_tau: int = 400,
        tau_max: Optional[float] = None,
        dt_min: Optional[float] = None,
        tau_grid_power: float = 3.0,
        period_max: Optional[float] = None,
        f_max: float = F_MAX_CYCLES_PER_DAY,
        log_step: float = FREQ_LOG_STEP,
    ):
        """Build the shared driver-frequency grid and lag grid from the
        currently registered light curves.

        Parameters
        ----------
        n_freq : int, optional
            Asks for the pre-October-2026 grid instead of the default one:
            ``n_freq`` log-spaced frequencies from ``2 pi / period_max`` to
            ``pi / dt_min`` (``dt_min`` estimated from the cadence if not
            given). Useful for small, fast fits (tests, demos). Leave unset for
            the default grid, whose size follows from ``period_max``, ``f_max``
            and ``log_step``: harmonics of ``period_max`` at low frequency, then
            log spacing (:func:`~pycream2.grid_utils.hybrid_frequency_grid`).
        n_tau : int
            Number of points on the lag grid used to evaluate/integrate psi.
        tau_max : float, optional
            Maximum lag to consider (days). Defaults to half the observed
            time baseline, which is a generous ceiling for reprocessing
            lags relative to typical monitoring campaigns.
        tau_grid_power : float
            Grades ``tau_grid`` towards ``tau=0`` (``tau_max * linspace(0,
            1, n_tau) ** tau_grid_power``) instead of uniform spacing, so
            short-wavelength/short-lag bands' narrow response functions
            stay resolved -- see :func:`~pycream2.grid_utils.graded_tau_grid`
            for why this matters (confirmed directly: an under-resolved
            response silently normalises to a flat, near-zero echo, not an
            error). ``1.0`` recovers the original uniform grid.
        dt_min : float, optional
            Finest timescale (days) the driver's Fourier series should
            resolve; if given, sets the upper bound ``w_max = pi / dt_min`` in
            place of ``f_max``. With ``n_freq`` and no ``dt_min``, the old
            grid's upper bound is estimated from the observation gaps
            (5th percentile, :func:`~pycream2.grid_utils.estimate_dt_min`).
        f_max : float
            Highest driver frequency, cycles per day (default 2, i.e. a
            0.5-day period). An absolute default rather than one from the
            cadence: the echoes carry the driver's variability down to the
            sharpest feature of the response, its rise near zero lag, and at
            high SNR the data resolve it. The earlier cadence-based default (a
            1.6-day period for daily sampling) under-fitted the CREAM paper's
            high-SNR synthetic data (log posterior up to ~530 lower) and
            biased SNR-100 fits towards high inclination and accretion rate.
            Raise it for black holes much lighter than ~1e7 Msun, whose
            responses rise within hours.
        log_step : float
            Fractional frequency step of the grid's logarithmic part (default
            0.03). Smaller steps add modes at high frequency.
        period_max : float, optional
            Longest driver period (days); sets the frequency grid's lower
            bound ``w_min = 2 pi / period_max``. Defaults to twice the
            window the driver must cover, ``2 (baseline + tau_max)``: the
            echo at the first observation depends on the driver up to
            ``tau_max`` earlier. A red-noise driver (a random walk
            especially) has strong trends on timescales longer than the
            campaign, and a longest period of only the baseline (the
            default until October 2026) cannot represent them: on synthetic
            random-walk data the fit then biased ``log_mdot`` and
            ``inclination`` high, absorbing the trends into long responses
            (``scripts/synthetic_recovery_grid.py``). The fit at the truth
            improved up to about twice the window and was flat beyond it.
        """
        if not self.bands:
            raise ValueError("Add at least one light curve before build_grid().")

        all_t_arrays = [d["t"] for d in self.bands.values()]
        if self.driver_data is not None:
            all_t_arrays.append(self.driver_data["t"])
        all_t = np.concatenate(all_t_arrays)
        t_span = all_t.max() - all_t.min()

        if tau_max is None:
            tau_max = 0.5 * t_span
        if period_max is None:
            period_max = PERIOD_MAX_FACTOR * (t_span + tau_max)
        if n_freq is not None:
            if dt_min is None:
                dt_min = estimate_dt_min(all_t_arrays, t_span=t_span)
            self.freqs = jnp.asarray(np.geomspace(2.0 * np.pi / period_max, np.pi / dt_min, n_freq))
        else:
            if dt_min is not None:
                f_max = 0.5 / dt_min  # w_max = pi / dt_min
            self.freqs = jnp.asarray(hybrid_frequency_grid(period_max, f_max, log_step))

        self.tau_grid = jnp.asarray(graded_tau_grid(tau_max, n_tau, power=tau_grid_power))
        for message in check_tau_grid_resolution(self.tau_grid, self.bands, self.M_BH):
            warnings.warn(f"build_grid(): {message}")
        return self

    # ------------------------------------------------------------------
    def _background_t_range(self):
        """The campaign's time span, over which every light curve's slow
        background basis is defined (so ``P_k`` means the same thing for all
        of them, and in the plots)."""
        all_t = np.concatenate([d["t"] for d in self.bands.values()]
                               + ([self.driver_data["t"]] if self.driver_data is not None else []))
        return float(all_t.min()), float(all_t.max())

    def _background_basis(self, d, t=None):
        """``(n, K)`` Legendre background basis for light curve ``d`` at times
        ``t`` (default: its own), or ``None`` if it has no background."""
        order = d.get("background_order", 0)
        if not order:
            return None
        return legendre_background_basis(d["t"] if t is None else t, self._background_t_range(), order)

    def _model_kwargs(self):
        bands_jax = {
            name: {
                "t": jnp.asarray(d["t"]),
                "y": jnp.asarray(d["y"]),
                "yerr": jnp.asarray(d["yerr"]),
                "wavelength": d["wavelength"],
                "lag_mode": d["lag_mode"],
                "fit_error_model": d.get("fit_error_model", False),
                "diffuse_continuum": d.get("diffuse_continuum", False),
                "background_basis": self._background_basis(d),
                "basis": fourier_basis(self.freqs, d["t"]),
            }
            for name, d in self.bands.items()
        }
        driver_jax = None
        if self.driver_data is not None:
            driver_jax = {
                "t": jnp.asarray(self.driver_data["t"]),
                "y": jnp.asarray(self.driver_data["y"]),
                "yerr": jnp.asarray(self.driver_data["yerr"]),
                "fit_error_model": self.driver_data.get("fit_error_model", False),
                "background_basis": self._background_basis(self.driver_data),
                "basis": fourier_basis(self.freqs, self.driver_data["t"]),
            }
        return dict(
            freqs=self.freqs, tau_grid=self.tau_grid, M_BH=self.M_BH,
            bands=bands_jax, driver=driver_jax,
            sigma_drw_prior_scale=self._sigma_drw_prior_scale(),
            fixed_params=self.fixed_params,
            drw_prior=self.drw_prior,
            transfer_mats=transfer_matrices(self.tau_grid, self.freqs),
            marginalise_linear=self.marginalise_linear,
            fit_temperature_slope=self.fit_temperature_slope,
        )

    def _sigma_drw_prior_scale(self) -> float:
        """Data-anchored scale for ``sigma_drw``'s ``HalfNormal`` prior (see
        ``model.py``'s "driver amplitude" docstring note).

        A fixed prior scale, unrelated to the actual light curves' units,
        only weakly regularises the exact driver-amplitude/per-band-gain
        rescaling degeneracy every fit has -- anchoring it to the data
        instead is the same fix, by the same mechanism, as the author's
        PhD-era CREAM Fortran code's optional prior on its power-spectrum
        normalisation ``P0``. Prefers a registered driver light curve's own
        std (the most direct available observation of the driver, if one
        was registered via ``add_driver_lightcurve``); otherwise the largest
        std across the registered bands, since the least-reprocessed band is
        the closest available proxy for the driver's own amplitude (a
        reprocessed echo is usually damped relative to what drives it, not
        amplified).
        """
        if self.driver_data is not None:
            scale = float(np.std(self.driver_data["y"]))
        else:
            scale = max(float(np.std(d["y"])) for d in self.bands.values())
        return max(scale, 1e-3)

    def _valid_fixed_param_names(self) -> set:
        """Every scalar site ``fixed_params`` could actually pin, given the
        bands/driver currently registered -- used to catch typos (a key
        that could never be a real site) before spending time on a fit."""
        valid = {"sigma_drw"}
        if self.drw_prior:
            valid.add("tau_drw")
        if self.driver_data is not None:
            valid |= {"S_driver", "C_driver"}
            if self.driver_data.get("fit_error_model", False):
                valid |= {"sigma_scale_driver", "sigma_jitter_driver"}
        if any(d["lag_mode"] == "physical" for d in self.bands.values()):
            valid |= {"log_mdot", "inclination"}
            if self.fit_temperature_slope:
                valid.add("temperature_slope")
        for name, d in self.bands.items():
            valid.add(f"S_{name}")
            valid.add(f"C_{name}")
            if d["lag_mode"] == "free":
                valid.add(f"tau_{name}")
            if d.get("fit_error_model", False):
                valid.add(f"sigma_scale_{name}")
                valid.add(f"sigma_jitter_{name}")
            if d.get("diffuse_continuum", False):
                valid |= {f"dce_fraction_{name}", f"dce_delay_{name}", f"dce_width_{name}"}
        return valid

    def _init_strategy(self, num_chains: int = 1):
        """Data-anchored starting guesses for each band's ``S_{band}``/
        ``C_{band}`` (NUTS's own initial point, not the prior) -- the same
        idea as the author's PhD-era CREAM Fortran code's own
        initialisation (``stretch = rms(data)/rms1``, ``offset =
        med(data)``, see ``cream_f90.f90``), via NumPyro's
        ``init_to_value``. Only covers non-fixed sites; ``init_to_value``
        defers anything else (including any site ``fixed_params`` already
        pins, which never reaches this dict) to NumPyro's own default
        (``init_to_uniform``). ``C_band``'s guess is the band's own mean;
        ``S_band``'s is its std relative to ``_sigma_drw_prior_scale``,
        the same data-derived reference scale the driver's own amplitude
        prior is anchored to (decision #13), so the two stay consistent
        with each other.

        ``num_chains > 1`` disables this entirely (returns ``None``, NumPyro's
        own ``init_to_uniform`` default), on purpose: ``init_to_value`` gives
        every chain the exact same starting point, which is fine (even
        helpful) for a single chain but actively defeats multi-chain
        Gelman-Rubin R-hat convergence checking -- confirmed directly, not
        theoretically: with it applied to all 4 chains of
        ``tests/test_free_lag_mode.py::test_free_lag_recovery_with_driver_anchor``'s
        vectorized-chain recovery check, R-hat on the free-lag ``tau_{band}``
        sites blew up to ~1000 (chains no longer independently initialised,
        so genuinely landing in different modes stopped being visible as
        "chains disagree" the way it needs to be); with ``num_chains=1``
        (this method's default), R-hat was ~1.0 as expected. See CLAUDE.md's
        rough-edges note on why independently-initialised chains matter for
        this model's free-lag multimodality risk in the first place.
        """
        if num_chains != 1:
            return None
        from numpyro.infer import init_to_value

        sigma_drw_scale = self._sigma_drw_prior_scale()
        values = {}
        for name, d in self.bands.items():
            if f"C_{name}" not in self.fixed_params:
                values[f"C_{name}"] = float(np.mean(d["y"]))
            if f"S_{name}" not in self.fixed_params:
                values[f"S_{name}"] = max(float(np.std(d["y"])) / sigma_drw_scale, 1e-3)
        return init_to_value(values=values)

    def _fit_init_strategy(self, num_chains: int, init_from_optimum: bool):
        if not init_from_optimum:
            return self._init_strategy(num_chains)
        if getattr(self, "optimum", None) is None:
            raise ValueError("fit(init_from_optimum=True) needs .optimise() to have been run first.")
        if num_chains != 1:
            raise ValueError(
                "fit(init_from_optimum=True) is single-chain only: identical starts "
                "defeat multi-chain R-hat checks (see _init_strategy's docstring)."
            )
        from numpyro.infer import init_to_value

        return init_to_value(values=self._optimum_init_values())

    def _validate_before_fit(self):
        has_physical = any(d["lag_mode"] == "physical" for d in self.bands.values())
        has_free = any(d["lag_mode"] == "free" for d in self.bands.values())
        if has_physical and self.M_BH is None:
            raise ValueError(
                "M_BH is required when any band uses lag_mode=\"physical\" "
                "(the default for add_lightcurve)."
            )
        if has_free and self.driver_data is None:
            warnings.warn(
                "Band(s) with lag_mode=\"free\" are registered but no driver "
                "light curve was added via add_driver_lightcurve(). A global "
                "shift of the driver, compensated by an equal shift of every "
                "free-lag band's tau, leaves the likelihood unchanged -- the "
                "absolute lag origin (and hence each such band's tau) is not "
                "identifiable without a driver light curve to anchor it."
            )
        if self.fit_temperature_slope and has_physical:
            import inspect

            if "viscous_slope" not in inspect.signature(_model.response_function).parameters:
                raise ValueError(
                    "fit_temperature_slope=True needs a response function that accepts "
                    "viscous_slope (e.g. forward_model.thin_disk_response); the active "
                    "pycream2.model.response_function does not."
                )
        unknown = set(self.fixed_params) - self._valid_fixed_param_names()
        if unknown:
            raise ValueError(
                f"fixed_params has key(s) that aren't a real site given the "
                f"currently registered bands/driver: {sorted(unknown)}. "
                f"Valid names right now: {sorted(self._valid_fixed_param_names())}."
            )

    # ------------------------------------------------------------------
    @classmethod
    def resume(cls, title: str, run_id: str = "latest", output_dir: Optional[str] = None) -> "EchoFit":
        """Load a previous (possibly interrupted) run's data/config/progress
        so ``.fit()`` continues it instead of starting over.

        Parameters
        ----------
        title : str
            The run title it was originally created with.
        run_id : str
            ``"latest"`` (default) picks the most recent run under that
            title, or pass an exact ``"run_<timestamp>"`` string.
        output_dir : str, optional
            Same output-root override as ``EchoFit(..., output_dir=...)``.
        """
        output_root = run_manager.resolve_output_root(output_dir)
        run_dir = run_manager.find_run_dir(output_root, title, run_id)
        manifest = run_manager.load_json(run_dir / "manifest.json")

        ef = cls(
            M_BH=manifest["M_BH"], title=title, output_dir=output_dir,
            fixed_params=manifest.get("fixed_params"), drw_prior=manifest.get("drw_prior", False),
            marginalise_linear=manifest.get("marginalise_linear", False),
            fit_temperature_slope=manifest.get("fit_temperature_slope", False),
            redshift=manifest.get("redshift"), flux_unit=manifest.get("flux_unit", "mJy"),
            flux_scale=manifest.get("flux_scale", 1.0), ebv_galactic=manifest.get("ebv_galactic", 0.0),
            sed_analysis=manifest.get("sed_analysis", False), sed_options=manifest.get("sed_options"),
        )
        ef.run_dir = run_dir
        ef._fit_config = manifest["fit_config"]

        bands = run_manager.load_bands_npz(run_dir / "data.npz")
        for name, d in bands.items():
            ef.add_lightcurve(
                name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"],
                lag_mode=d["lag_mode"], fit_error_model=d.get("fit_error_model", False),
                diffuse_continuum=d.get("diffuse_continuum", False), background_order=d.get("background_order", 0),
            )
        driver_path = run_dir / "driver.npz"
        if driver_path.exists():
            driver = run_manager.load_driver_npz(driver_path)
            ef.add_driver_lightcurve(
                t=driver["t"], y=driver["y"], yerr=driver["yerr"],
                fit_error_model=driver.get("fit_error_model", False),
                background_order=driver.get("background_order", 0),
            )

        grid = run_manager.load_samples_npz(run_dir / "grid.npz")
        ef.freqs = jnp.asarray(grid["freqs"])
        ef.tau_grid = jnp.asarray(grid["tau_grid"])

        checkpoint_dir = run_dir / "checkpoint"
        state_path = checkpoint_dir / "state.pkl"
        if state_path.exists():
            samples = run_manager.load_samples_npz(checkpoint_dir / "samples.npz")
            extra_fields = run_manager.load_samples_npz(checkpoint_dir / "extra_fields.npz")
            n_done = int(next(iter(samples.values())).shape[0])
            ef._resume_state = dict(
                last_state=run_manager.load_state(state_path),
                samples=samples, extra_fields=extra_fields, n_done=n_done,
            )
            print(f"Found checkpoint: {n_done}/{ef._fit_config['num_samples']} samples already collected.")
        else:
            print("No checkpoint found (likely interrupted during warmup) -- restarting this run from scratch.")

        return ef

    # ------------------------------------------------------------------
    def fit(
        self,
        rng_seed=_UNSET,
        num_warmup=_UNSET,
        num_samples=_UNSET,
        num_chains: int = 1,
        max_tree_depth=_UNSET,
        dense_mass=_UNSET,
        chain_method: str = "parallel",
        checkpoint_every=_UNSET,
        report_every: Optional[int] = None,
        progress_bar: bool = True,
        generate_report: bool = True,
        init_from_optimum: bool = False,
    ):
        """Run NUTS and store the posterior samples on ``self.samples``.

        If this instance has no ``title``, this is a single non-resumable
        in-memory run (the original behaviour) -- ``num_chains``/
        ``chain_method`` apply normally.

        If ``title`` was given (directly, or via ``.resume()``), this
        instead runs single-chain NUTS in checkpointed chunks of
        ``checkpoint_every`` samples (``num_chains``/``chain_method`` are
        ignored; a warning is raised if ``num_chains != 1``), saving
        progress after every chunk so an interrupted fit can be resumed
        with ``EchoFit.resume(title)``. When complete, writes the final
        posterior (as ``chains.nc``, an ArviZ ``InferenceData``) and (if
        ``generate_report``) the same plots + report.html as
        ``scripts/smoke_test.py`` to the run directory.

        dense_mass : bool, optional
            Full covariance-based NUTS mass matrix (the default, ``True``)
            rather than a diagonal one (``False``) -- see
            ``inference.run_mcmc``'s docstring and ``CLAUDE.md`` decisions
            #17/#21. This model's parameters are correlated enough that a
            diagonal mass matrix makes NUTS spend nearly every sample pinned
            at ``max_tree_depth``'s ceiling: measured ~17x fewer effective
            samples per wall-clock second than dense on a 5-band test set,
            with the same posterior. A dense matrix needs a reasonably long
            ``num_warmup`` to adapt (the default 1000 is fine; a few dozen is
            not): check ``ef.extra_fields["diverging"]`` and increase
            ``num_warmup`` if it's above a few percent. A resumed run keeps
            whatever it was started with.

        init_from_optimum : bool
            Start NUTS at ``.optimise()``'s peak (call ``.optimise()``
            first) rather than the data-anchored default guess. Single-chain
            only, for the same reason as ``_init_strategy``'s: identical
            starts defeat multi-chain R-hat checks.

        report_every : int, optional
            Only used on the checkpointed path. If given, ``report.html``
            (and the PNGs it references) are refreshed in ``run_dir`` after
            every checkpoint that adds up to at least this many new
            samples since the last refresh, so a long-running fit's report
            can be watched as it progresses rather than only seen once at
            the end. Off by default, since re-rendering the full plot set
            (corner plots, posterior-predictive fits, ...) on every
            checkpoint would add real overhead to short ``checkpoint_every``
            values; the final report (governed by ``generate_report``) is
            always written regardless of this setting.

        After ``.resume()``, any arguments left unset here reuse the
        original run's settings (so ``ef.fit()`` with no arguments "just
        continues"); pass a value explicitly to override it.
        """
        if self.freqs is None or self.tau_grid is None:
            self.build_grid()
        self._validate_before_fit()

        if self.title is None:
            rng_seed = 0 if rng_seed is _UNSET else rng_seed
            num_warmup = 1000 if num_warmup is _UNSET else num_warmup
            num_samples = 1000 if num_samples is _UNSET else num_samples
            max_tree_depth = None if max_tree_depth is _UNSET else max_tree_depth
            dense_mass = True if dense_mass is _UNSET else dense_mass
            rng_key = jax.random.PRNGKey(rng_seed)
            self.mcmc = run_mcmc(
                reverberation_model, self._model_kwargs(), rng_key,
                num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                max_tree_depth=max_tree_depth, chain_method=chain_method, progress_bar=progress_bar,
                init_strategy=self._fit_init_strategy(num_chains, init_from_optimum), dense_mass=dense_mass,
            )
            self._samples_by_chain = self._add_linear_draws(
                self.mcmc.get_samples(group_by_chain=True), rng_seed
            )
            self.samples = {k: v.reshape((-1,) + v.shape[2:]) for k, v in self._samples_by_chain.items()}
            self.extra_fields = self.mcmc.get_extra_fields()
            self._extra_fields_by_chain = self.mcmc.get_extra_fields(group_by_chain=True)
            self._maybe_disc_sed()
            return self

        # -- title given: checkpointed/resumable single-chain path --
        if num_chains != 1:
            warnings.warn(
                f"EchoFit(title=...) checkpointing only supports num_chains=1; "
                f"ignoring num_chains={num_chains}."
            )

        def _pick(value, key, default):
            if value is not _UNSET:
                return value
            if self._fit_config is not None and key in self._fit_config:
                return self._fit_config[key]
            return default

        rng_seed = _pick(rng_seed, "rng_seed", 0)
        num_warmup = _pick(num_warmup, "num_warmup", 1000)
        num_samples = _pick(num_samples, "num_samples", 1000)
        max_tree_depth = _pick(max_tree_depth, "max_tree_depth", None)
        # A run resumed from before dense_mass was recorded keeps its original
        # (diagonal) mass matrix rather than switching mid-run to the new default.
        dense_mass = _pick(dense_mass, "dense_mass", False if self._fit_config is not None else True)
        checkpoint_every = _pick(checkpoint_every, "checkpoint_every", 100)

        if self.run_dir is None:
            self.run_dir = run_manager.new_run_dir(self._output_root, self.title)
        checkpoint_dir = self.run_dir / "checkpoint"
        manifest_path = self.run_dir / "manifest.json"

        if self._resume_state is not None:
            init_last_state = self._resume_state["last_state"]
            prev_samples = self._resume_state["samples"]
            prev_extra = self._resume_state["extra_fields"]
            n_already_done = self._resume_state["n_done"]
            if n_already_done >= num_samples:
                print(f"'{self.title}' at {self.run_dir} is already complete "
                      f"({n_already_done}/{num_samples} samples) -- nothing to resume.")
            else:
                print(
                    f"Resuming '{self.title}' from {self.run_dir} "
                    f"({n_already_done}/{num_samples} samples already collected)."
                )
        else:
            init_last_state = None
            prev_samples, prev_extra, n_already_done = {}, {}, 0
            self._fit_config = dict(
                rng_seed=rng_seed, num_warmup=num_warmup, num_samples=num_samples,
                max_tree_depth=max_tree_depth, dense_mass=dense_mass, checkpoint_every=checkpoint_every,
            )
            if not manifest_path.exists():
                run_manager.save_bands_npz(self.run_dir / "data.npz", self.bands)
                if self.driver_data is not None:
                    run_manager.save_driver_npz(self.run_dir / "driver.npz", self.driver_data)
                run_manager.save_samples_npz(
                    self.run_dir / "grid.npz",
                    dict(freqs=np.asarray(self.freqs), tau_grid=np.asarray(self.tau_grid)),
                )
                run_manager.save_json(manifest_path, dict(
                    title=self.title, M_BH=self.M_BH,
                    bands={n: d["wavelength"] for n, d in self.bands.items()},
                    fit_config=self._fit_config,
                    fixed_params=self.fixed_params,
                    drw_prior=self.drw_prior,
                    marginalise_linear=self.marginalise_linear,
                    fit_temperature_slope=self.fit_temperature_slope,
                    redshift=self.redshift, flux_unit=self.flux_unit, flux_scale=self.flux_scale,
                    ebv_galactic=self.ebv_galactic, sed_analysis=self.sed_analysis,
                    sed_options=self.sed_options,
                ))

        chunk_samples_so_far, chunk_extra_so_far = [], []
        t0 = time.time()
        last_report_at = [n_already_done]

        def _on_chunk_done(mcmc, last_state, n_done_total):
            chunk_samples_so_far.append(mcmc.get_samples())
            chunk_extra_so_far.append(mcmc.get_extra_fields())
            merged_samples = _merge_dicts([prev_samples] + chunk_samples_so_far)
            merged_extra = _merge_dicts([prev_extra] + chunk_extra_so_far)
            run_manager.save_samples_npz(checkpoint_dir / "samples.npz", merged_samples)
            run_manager.save_samples_npz(checkpoint_dir / "extra_fields.npz", merged_extra)
            run_manager.save_state(checkpoint_dir / "state.pkl", last_state)
            print(f"  checkpoint: {n_done_total}/{num_samples} samples saved.")

            if report_every is not None and n_done_total - last_report_at[0] >= report_every:
                last_report_at[0] = n_done_total
                self._samples_by_chain = self._add_linear_draws(
                    {k: v[None, ...] for k, v in merged_samples.items()}, rng_seed
                )
                self.samples = {k: v[0] for k, v in self._samples_by_chain.items()}
                self.extra_fields = merged_extra
                self._extra_fields_by_chain = {k: v[None, ...] for k, v in merged_extra.items()}
                reporting.generate_report(
                    self, self.run_dir, fit_seconds=time.time() - t0, title=self.title
                )
                print(f"  report refreshed at {n_done_total}/{num_samples} samples.")

        rng_key = jax.random.PRNGKey(rng_seed)
        samples_this_call, samples_by_chain_this_call, extra_this_call, _ = run_mcmc_chunked(
            reverberation_model, self._model_kwargs(), rng_key,
            num_warmup=num_warmup, num_samples=num_samples, checkpoint_every=checkpoint_every,
            max_tree_depth=max_tree_depth, progress_bar=progress_bar,
            init_last_state=init_last_state, n_already_done=n_already_done,
            on_chunk_done=_on_chunk_done, init_strategy=self._fit_init_strategy(1, init_from_optimum),
            dense_mass=dense_mass,
        )
        fit_seconds = time.time() - t0

        merged = _merge_dicts([prev_samples, samples_this_call])
        self._samples_by_chain = self._add_linear_draws({k: v[None, ...] for k, v in merged.items()}, rng_seed)
        self.samples = {k: v[0] for k, v in self._samples_by_chain.items()}
        self.extra_fields = _merge_dicts([prev_extra, extra_this_call])
        self._extra_fields_by_chain = {k: v[None, ...] for k, v in self.extra_fields.items()}
        self.mcmc = None  # no single mcmc object spans all chunks in this path

        self._save_chains(self.run_dir)
        self._maybe_disc_sed()
        if generate_report:
            reporting.generate_report(
                self, self.run_dir, fit_seconds=fit_seconds, title=self.title
            )
            print(f"Report written to {self.run_dir / 'report.html'}")

        return self

    def optimise(
        self, num_samples: int = 1000, num_restarts: Optional[int] = None, rng_seed: int = 0,
        restart_scale: float = 0.5, method: str = "nested_laplace", **nested_kwargs,
    ):
        """Directly solve for the posterior, without MCMC.

        ``method="nested_laplace"`` (the default since October 2026) runs
        :meth:`nested_laplace`: a grid over ``log_mdot`` and inclination with
        the other parameters integrated at every point, corrected by
        importance sampling. It follows a curved ridge or a second mode,
        which a single Gaussian cannot; see ``docs/nested_laplace.md``.
        ``nested_kwargs`` are passed to it, as are ``num_restarts`` and
        ``restart_scale`` (random restarts of its mode search; default 4).

        ``method="laplace"`` is the original single-Gaussian solve (the
        default before), described below, with its multi-start check
        (``optimise_restarts``). It is faster (by 1.4-1.8x on the synthetic
        cases measured) and exact for a Gaussian posterior, but understated
        the uncertainty up to six-fold on weakly constraining data. It runs one
        optimisation by default; ``num_restarts=4`` adds the multi-start
        reproducibility check.
        """
        if num_restarts is None:
            num_restarts = 4 if method == "nested_laplace" else 1
        if method == "nested_laplace":
            return self.nested_laplace(num_samples=num_samples, rng_seed=rng_seed, num_restarts=num_restarts,
                                       restart_scale=restart_scale, **nested_kwargs)
        if method != "laplace":
            raise ValueError(f"optimise(method=...) must be 'nested_laplace' or 'laplace', not {method!r}")
        if nested_kwargs:
            raise TypeError(f"optimise(method='laplace') got unexpected arguments {sorted(nested_kwargs)}")
        return self._optimise_laplace(num_samples=num_samples, num_restarts=num_restarts, rng_seed=rng_seed,
                                      restart_scale=restart_scale)

    def _optimise_laplace(
        self, num_samples: int = 1000, num_restarts: int = 1, rng_seed: int = 0,
        restart_scale: float = 0.5,
    ):
        """``optimise(method="laplace")``. Directly solve for the posterior, without MCMC: maximise the
        linear-marginalised posterior over the ~10 nonlinear parameters with
        L-BFGS, then approximate the posterior around that peak as a
        Gaussian from its Hessian (the Laplace approximation), and draw the
        linear parameters exactly for each Laplace sample.

        On a 5-band, 500-point synthetic set, the whole first call took
        ~27s (~5s of it L-BFGS; most of the rest one-off JIT compilation of
        the Hessian and the posterior draws) against ~56s for a 500+500
        dense-mass NUTS run, and its ``log_mdot`` posterior (0.194 +/- 0.012)
        matched NUTS's (0.193 +/- 0.011). The gap widens for longer runs,
        since NUTS's cost grows with the number of samples and this doesn't;
        see ``docs/performance_improvements.md``. It is weakest for broad,
        bounded, non-Gaussian parameters: inclination's posterior piles up
        against its prior bound, which a Gaussian in unconstrained space
        can't reproduce. Always uses the marginalised
        model, whatever ``marginalise_linear`` is set to: its maximum is the
        peak of the *marginal* posterior of the nonlinear parameters, which
        is what the Laplace approximation needs (a joint maximum over the
        Fourier coefficients too would be a different, biased estimator).

        The Laplace approximation is only as good as the posterior is
        Gaussian in NumPyro's unconstrained space: fine for a single,
        well-constrained peak, not for a multimodal or strongly skewed
        posterior (e.g. ``lag_mode="free"`` bands, CLAUDE.md's
        rough-edges note, or ``log_mdot`` and inclination on weakly
        constraining data): use :meth:`nested_laplace`, which grids those
        parameters and follows a ridge or a second mode, or ``.fit()``.
        The peak is found in unconstrained coordinates, where a bounded
        uniform prior's Jacobian favours the middle of its range, so a
        weakly constrained parameter's mode is pulled towards it; the
        Hessian's curvature is checked over one posterior standard deviation
        (``_posterior_scale_curvature``), since tiny ripples in the thin-disc
        potential can dominate it at a point.

        Fills ``self.samples``/``self._samples_by_chain`` (one "chain" of
        ``num_samples`` Laplace draws) exactly like ``.fit()``, so every
        ``plot_*`` method and ``reporting.generate_report`` work on the
        result; ``self.extra_fields`` stays empty (no NUTS diagnostics
        exist). Also sets ``self.optimum`` (the constrained peak values),
        ``self.laplace_covariance`` (unconstrained space),
        ``self.log_evidence`` (the Laplace estimate of the log evidence,
        ``-U* + d/2 ln 2 pi + 1/2 ln det(cov)``, for comparing models fitted
        to the same data, e.g. with and without an extra component),
        ``self.optimise_result`` (scipy's ``OptimizeResult``) and
        ``self.optimise_timings`` (seconds spent in L-BFGS, the Hessian and
        the posterior draws, all including one-off JIT compilation, plus
        ``newton_offset_in_sd``, the convergence diagnostic, and
        ``restarts_agreeing``).

        Multi-start reproducibility, the direct-solve counterpart of running
        several MCMC chains from different starting points: every restart is
        polished to its own optimum (not just the best one, so a restart that
        L-BFGS merely stopped short on isn't mistaken for a different
        answer), then compared with the best. ``self.optimise_restarts`` has
        one dict per restart: ``start_values`` (its constrained scalar
        parameters at the start), ``potential`` and ``delta_potential`` (above the
        best), ``max_offset_in_sd`` (the largest distance from the best
        optimum over every parameter, in Laplace posterior standard
        deviations), ``start_offset_in_sd`` (how far its starting point was
        from the data-anchored one, on the same scale), ``agrees``
        (``max_offset_in_sd`` below ``RESTART_AGREEMENT_SD``) and ``values``
        (its constrained scalar parameters). It warns if any restart ends at
        a different optimum.

        Parameters
        ----------
        num_samples : int
            Laplace posterior draws to generate.
        num_restarts : int
            Optimisations, from the data-anchored initial point
            (``_init_strategy``) plus ``num_restarts - 1`` random
            perturbations of it; the best is kept. Guards against a local
            optimum, and measures reproducibility (above). Default 1 (since
            October 2026, for speed: each restart costs a full optimisation);
            pass 4 or more for the reproducibility check.
        rng_seed : int
            Seeds the restart perturbations and the Laplace/linear draws.
        restart_scale : float
            Standard deviation of the restart perturbations, in NumPyro's
            unconstrained space. Raise it (e.g. 2.0) for a stricter
            reproducibility test from more widely spread starting points.
        """
        from jax.flatten_util import ravel_pytree
        from numpyro.infer.util import constrain_fn
        from scipy.optimize import minimize

        if self.freqs is None or self.tau_grid is None:
            self.build_grid()
        self._validate_before_fit()

        kwargs = dict(self._model_kwargs(), marginalise_linear=True)
        info = _initialize_marginal_model(rng_seed, kwargs, self._init_strategy(1))
        z0, unravel = ravel_pytree(info.param_info.z)
        potential = lambda z: info.potential_fn(unravel(z))
        value_and_grad = jax.jit(jax.value_and_grad(potential))

        def objective(z):
            v, g = value_and_grad(jnp.asarray(z, dtype=z0.dtype))
            v, g = float(v), np.asarray(g, dtype=np.float64)
            if not (np.isfinite(v) and np.all(np.isfinite(g))):
                # A trial step far outside the posterior (e.g. L-BFGS's first,
                # gradient-scaled step from a start high up a steep slope) can
                # overflow to NaN. SciPy's line search cannot backtrack from a
                # NaN and aborts the whole restart; a huge finite value makes it
                # backtrack instead. Found when a restart ran to sigma_drw = inf,
                # log_mdot = 33 on its first step.
                return _NONFINITE_POTENTIAL, np.zeros_like(g)
            return v, g

        timings = {}
        t0 = time.perf_counter()
        rng = np.random.default_rng(rng_seed)
        starts = [np.asarray(z0)] + [np.asarray(z0) + restart_scale * rng.normal(size=z0.shape) for _ in range(num_restarts - 1)]
        # Tolerances matched to a float32 objective: the default ftol (~2e-9
        # relative) is below float32's own resolution (~1e-7), so L-BFGS's line
        # search otherwise "fails" right at the optimum it has already found.
        options = dict(ftol=1e-7, gtol=1e-3, maxiter=1000)
        results = [minimize(objective, x, jac=True, method="L-BFGS-B", options=options) for x in starts]
        timings["lbfgs_seconds"] = time.perf_counter() - t0
        timings["lbfgs_evaluations_total"] = int(sum(r.nfev for r in results))
        # Newton polishing with the exact Hessian. L-BFGS alone can stop well
        # short of the peak along the prior-dominated driver-amplitude/band-gain
        # ridge (CLAUDE.md decision #13): on NGC 5548 it stopped 6.5 units of
        # potential above it, with the implied Newton step 3.5 posterior standard
        # deviations long. Each step is damped until the potential falls and
        # only ever accepted if it does (see polish), so polishing can't make
        # things worse; it stops once the implied step is under 0.01 standard
        # deviations. Convergence is judged on that posterior scale rather than
        # by L-BFGS's own flag, whose line search on a float32 objective often
        # ends "abnormally" right at the optimum. Every restart is polished, not
        # just the lowest, so the restarts can be compared at their true optima
        # and the best is chosen after polishing.
        t0 = time.perf_counter()
        # The curvature check needs ~2 potential values per parameter: one at a
        # time through a plain compiled potential, not jax.vmap's batched one,
        # whose separate compilation cost ~4 s for a 5-parameter problem.
        potential_jit = jax.jit(potential)
        corrections = {}
        # Finite-difference Hessians (_fd_hessian): the first pass's steps from
        # L-BFGS's own inverse-Hessian estimate, later ones along the previous
        # Hessian's eigenvectors.
        scale = {"sd": _lbfgs_sd(min((r for r in results if np.isfinite(r.fun)), key=lambda r: r.fun, default=results[0])),
                 "basis": None}

        def batch_potential(points):
            return np.array([float(potential_jit(jnp.asarray(x, dtype=z0.dtype))) for x in np.atleast_2d(points)])

        def curvature(z, posterior_scale=False):
            # posterior_scale: check each eigenvalue over one posterior sd
            # (_posterior_scale_curvature); ~2 potential evaluations per
            # parameter, so only at each polished optimum, not every Newton step.
            _, vals, vecs = _fd_hessian(lambda x: objective(x)[1], z, sd=scale["sd"], basis=scale["basis"])
            scale["basis"] = (vals, vecs)
            if posterior_scale:
                f0 = float(batch_potential(np.asarray(z)[None])[0])
                vals, corrections["n"] = _posterior_scale_curvature(batch_potential, np.asarray(z), f0, vals, vecs)
            return vals, vecs

        def laplace_cov(vals, vecs):
            return (vecs / np.clip(vals, 1e-8, None)) @ vecs.T

        def polish(result):
            # Damped, saddle-free Newton: step = (|H| + damping I)^-1 grad, in
            # H's eigenbasis, with the damping raised until the step lowers the
            # potential and relaxed after each success (Levenberg-Marquardt
            # style). A plain Newton step with a backtracking line search stalled
            # on the curved inclination/log_mdot ridge of the NGC 5548 thin-disc
            # fit: tiny or negative eigenvalues there make the undamped step
            # enormous, every backtracked fraction of it still went uphill, and
            # restarts stopped 4-110 sd short of the optimum along the ridge.
            z, f = np.asarray(result.x, dtype=np.float64), float(result.fun)
            if not np.isfinite(f):
                return None
            vals, vecs = curvature(z)
            n_newton, offset_in_sd, damping = 0, np.inf, 0.0
            for _ in range(20):
                grad = objective(z)[1]
                cov = laplace_cov(vals, vecs)
                offset_in_sd = float(np.max(np.abs(cov @ grad) / np.sqrt(np.diag(cov))))
                if offset_in_sd < 0.01:
                    break
                grad_eig = vecs.T @ grad
                floor = 1e-6 * float(np.max(np.abs(vals)))
                for _ in range(15):
                    step = vecs @ (grad_eig / (np.abs(vals) + damping))
                    f_new = objective(z - step)[0]
                    if f_new < f:
                        break
                    damping = max(10.0 * damping, floor)
                else:
                    break
                z, f = z - step, f_new
                damping /= 10.0
                vals, vecs = curvature(z)
                n_newton += 1
            # Judge convergence on the posterior-scale curvature: tiny ripples in
            # the potential can make the pointwise Hessian claim a huge remaining
            # step where the peak has in fact been reached.
            vals, vecs = curvature(z, posterior_scale=True)
            cov = laplace_cov(vals, vecs)
            offset_in_sd = float(np.max(np.abs(cov @ objective(z)[1]) / np.sqrt(np.diag(cov))))
            return dict(z=z, f=f, eigvals=vals, cov=cov, n_newton=n_newton, offset_in_sd=offset_in_sd,
                        curvature_corrections=corrections["n"])

        polished = [polish(r) for r in results]
        best_index = min((i for i, p in enumerate(polished) if p is not None), key=lambda i: polished[i]["f"])
        best, peak = results[best_index], polished[best_index]
        eigvals, cov = peak["eigvals"], peak["cov"]
        timings["curvature_corrections"] = peak["curvature_corrections"]
        n_newton, offset_in_sd = peak["n_newton"], peak["offset_in_sd"]
        best.x, best.fun = peak["z"], peak["f"]
        timings["hessian_and_newton_seconds"] = time.perf_counter() - t0
        timings["newton_iterations"] = n_newton
        timings["newton_offset_in_sd"] = offset_in_sd
        if offset_in_sd > 0.25:
            warnings.warn(
                f"optimise(): the optimum may not have converged: one more Newton step would "
                f"move it by {offset_in_sd:.2f} posterior standard deviations."
            )
        if eigvals.min() <= 0:
            warnings.warn(
                "optimise(): the Hessian at the optimum is not positive definite "
                f"(smallest eigenvalue {eigvals.min():.3g}), so the peak is not a clean "
                "Gaussian mode; clipping those directions. Treat the Laplace posterior "
                "with suspicion and cross-check with .fit()."
            )
        z_hat = jnp.asarray(best.x, dtype=z0.dtype)

        def scalar_values(z):
            values = constrain_fn(
                reverberation_model, (), kwargs, unravel(jnp.asarray(z, dtype=z0.dtype)), return_deterministic=True,
            )
            return {k: float(v) for k, v in values.items() if np.size(v) == 1 and not k.startswith("y_pred_")}

        sd = np.sqrt(np.diag(cov))
        self.optimise_restarts = []
        for i, (start, result, p) in enumerate(zip(starts, results, polished)):
            entry = dict(
                index=i, start_offset_in_sd=float(np.max(np.abs(start - starts[0]) / sd)),
                start_values=scalar_values(start),
                lbfgs_evaluations=int(result.nfev), potential=np.nan, delta_potential=np.nan,
                max_offset_in_sd=np.nan, newton_iterations=0, agrees=False, values={},
            )
            if p is not None:
                offset = float(np.max(np.abs(p["z"] - peak["z"]) / sd))
                entry.update(
                    potential=p["f"], delta_potential=p["f"] - peak["f"], max_offset_in_sd=offset,
                    newton_iterations=p["n_newton"], agrees=offset < RESTART_AGREEMENT_SD,
                    values=scalar_values(p["z"]),
                )
            self.optimise_restarts.append(entry)
        n_agree = sum(r["agrees"] for r in self.optimise_restarts)
        timings["restarts_agreeing"] = n_agree
        if n_agree < num_restarts:
            worst = [r for r in self.optimise_restarts if not r["agrees"]]
            warnings.warn(
                f"optimise(): only {n_agree} of {num_restarts} restarts reached the same optimum; "
                + ", ".join(
                    f"restart {r['index']} ended {r['max_offset_in_sd']:.2g} sd away, "
                    f"{r['delta_potential']:.3g} above it in potential" for r in worst
                )
                + ". The best is kept; a multimodal posterior needs .nested_laplace() or .fit() with several chains."
            )

        t0 = time.perf_counter()
        z_draws = rng.multivariate_normal(best.x, cov, size=num_samples).astype(np.asarray(z0).dtype)
        constrained = jax.jit(jax.vmap(
            lambda z: constrain_fn(reverberation_model, (), kwargs, unravel(z), return_deterministic=True)
        ))(jnp.asarray(z_draws))
        nonlinear = {k: np.asarray(v)[None, ...] for k, v in constrained.items()}

        self._set_lazy_linear_draws(nonlinear, rng_seed)
        self.extra_fields, self._extra_fields_by_chain = {}, {}
        self.mcmc = None

        self.optimum = {
            k: np.asarray(v) for k, v in
            constrain_fn(reverberation_model, (), kwargs, unravel(z_hat), return_deterministic=True).items()
        }
        self.laplace_covariance = cov
        self._laplace_peak = dict(self.optimum)
        # Laplace estimate of the log evidence of the marginal posterior (the
        # linear parameters are integrated out exactly):
        # ln Z ~ -U* + d/2 ln(2 pi) + 1/2 ln det(cov), for comparing models,
        # e.g. with and without a diffuse-continuum component or background.
        _, logdet = np.linalg.slogdet(np.asarray(cov, dtype=np.float64))
        self.log_evidence = -float(best.fun) + 0.5 * cov.shape[0] * np.log(2.0 * np.pi) + 0.5 * logdet
        self.optimise_result = best
        timings["draws_seconds"] = time.perf_counter() - t0
        timings["method"] = "laplace"
        self.optimise_timings = timings
        self._maybe_disc_sed()
        return self

    # ------------------------------------------------------------------
    def nested_laplace(
        self, num_samples: int = 1000, rng_seed: int = 0, grid_params=None, n_pass=None,
        n_fine=None, drop: float = 8.0, num_restarts: int = 4, restart_scale: float = 0.5,
    ):
        """Posterior by a nested Laplace approximation (after INLA, Rue,
        Martino & Chopin 2009), corrected by Pareto-smoothed importance
        sampling: see :mod:`pycream2.nested_laplace`.

        ``optimise()`` fits one Gaussian at one peak. Where the data constrain
        ``log_mdot`` and inclination only weakly, their posterior is a long,
        curved ridge or has two modes, and that Gaussian understates the
        uncertainty several-fold or sits on one mode. This method grids those
        parameters (``grid_params``, default ``log_mdot``, ``cos_inclination``
        and, when fitted, ``temperature_slope``), integrates the remaining
        nonlinear parameters by a Laplace approximation at every grid point and
        the linear ones exactly, and so follows a ridge or a second mode. Its
        draws are importance-weighted against the exact posterior and
        resampled; ``k_hat`` (Pareto shape) below ~0.7 means they are reliable.

        This is what ``.optimise()`` runs by default. Fills
        ``self.samples``/``self._samples_by_chain`` (with the exact linear
        draws), sets ``self.optimum``/``self.laplace_covariance`` at the best
        mode (as ``optimise(method="laplace")`` does at its peak),
        ``self.log_evidence`` (the
        importance-sampling estimate), ``self.nested_laplace_result`` (the
        grids, their log posterior, ``k_hat``, the effective sample size of
        the weights, both evidence estimates) and
        ``self.nested_laplace_timings``. Warns if ``k_hat`` > 0.7 or if the
        posterior has not fallen off at an edge of the grid.

        Parameters
        ----------
        num_samples : int
            Proposal draws, importance-resampled to the same number.
        rng_seed : int
            Seeds the draws.
        grid_params : sequence of str, optional
            Scalar sample sites to grid; others are integrated by Laplace.
        n_pass : sequence of int, optional
            Cells per gridded parameter in each search pass (default (12, 9)
            for two).
        n_fine : sequence of int, optional
            Cells per gridded parameter in the final grid (default (20, 15)
            for two).
        drop : float
            Log-density drop from the peak that bounds the region gridded
            (8 leaves out ~0.03 per cent of a Gaussian's mass).
        num_restarts, restart_scale : int, float
            Random restarts of the mode search, perturbing the data-anchored
            start as ``optimise(method="laplace")`` does (same defaults).
        """
        from numpyro.infer.util import constrain_fn

        from . import nested_laplace as _nl

        if self.freqs is None or self.tau_grid is None:
            self.build_grid()
        self._validate_before_fit()
        kwargs = dict(self._model_kwargs(), marginalise_linear=True)
        info = _initialize_marginal_model(rng_seed, kwargs, self._init_strategy(1))
        result = _nl.run(reverberation_model, kwargs, info, num_samples=num_samples, rng_seed=rng_seed,
                         grid_params=grid_params or _nl.DEFAULT_GRID_PARAMS, n_pass=n_pass, n_fine=n_fine,
                         drop=drop, num_restarts=num_restarts, restart_scale=restart_scale)
        t0 = time.perf_counter()
        unravel = result["unravel"]
        constrained = jax.jit(jax.vmap(
            lambda z: constrain_fn(reverberation_model, (), kwargs, unravel(z), return_deterministic=True)
        ))(jnp.asarray(result["z"], dtype=result["dtype"]))
        nonlinear = {k: np.asarray(v)[None, ...] for k, v in constrained.items()}
        self._set_lazy_linear_draws(nonlinear, rng_seed)
        self.extra_fields, self._extra_fields_by_chain = {}, {}
        self.mcmc = None
        result["timings"]["linear_draws_seconds"] = time.perf_counter() - t0
        self.log_evidence = result["log_evidence_is"]
        self.nested_laplace_result = {k: v for k, v in result.items() if k not in ("unravel", "dtype")}
        self.nested_laplace_timings = result["timings"]
        # The same peak attributes as optimise(method="laplace"), at the best mode
        # (highest Laplace evidence), so fit(init_from_optimum=True), the SED
        # analysis and plot_landscape work after either solve.
        from scipy.optimize import OptimizeResult

        best = max(result["modes"], key=lambda o: o["log_evidence"])
        self.optimum = {
            k: np.asarray(v) for k, v in constrain_fn(
                reverberation_model, (), kwargs, unravel(jnp.asarray(best["z"], dtype=result["dtype"])),
                return_deterministic=True,
            ).items()
        }
        self.laplace_covariance = np.asarray(best["cov"])
        self.optimise_result = OptimizeResult(x=np.asarray(best["z"]), fun=float(best["U"]))
        self.optimise_restarts = None
        self.optimise_timings = dict(result["timings"], method="nested_laplace")
        if result["k_hat"] > 0.7:
            warnings.warn(
                f"nested_laplace(): Pareto k_hat = {result['k_hat']:.2f} > 0.7, so the importance-weighted "
                "draws are unreliable; cross-check with .fit()."
            )
        if result["edge_drop"] < drop - 2.0:
            warnings.warn(
                f"nested_laplace(): the posterior is only {result['edge_drop']:.1f} below its peak at an edge "
                "of the grid; it may extend beyond it."
            )
        self._maybe_disc_sed()
        return self

    # ------------------------------------------------------------------
    def disc_sed_analysis(self, **overrides) -> dict:
        """The disc's variable and mean SED, luminosity distance and ``H_0``
        from this fit (Cackett, Horne & Winkler 2007's test): see
        :mod:`pycream2.disc_sed`. Uses the constructor's ``redshift``,
        ``flux_unit``, ``flux_scale``, ``ebv_galactic`` and ``sed_options``;
        ``overrides`` replace any of them for this call. Stores the result
        in ``self.disc_sed`` (``"summary"`` and per-draw ``"draws"``) and
        returns the summary."""
        options = dict(redshift=self.redshift, flux_unit=self.flux_unit, flux_scale=self.flux_scale,
                       ebv_galactic=self.ebv_galactic, **self.sed_options)
        options.update(overrides)
        if options["redshift"] is None:
            raise ValueError("disc_sed_analysis needs the redshift: EchoFit(..., redshift=...) or redshift=...")
        self.disc_sed = disc_sed.disc_sed_analysis(self, **options)
        return self.disc_sed["summary"]

    def _maybe_disc_sed(self):
        if not self.sed_analysis:
            return
        try:
            self.disc_sed_analysis()
        except Exception as exc:  # never lose a finished fit to the post-processing
            warnings.warn(f"disc SED analysis failed: {exc}")

    def plot_disc_sed(self, **kwargs):
        """Flux-flux diagram, variable and mean disc SED, and ``H_0``: see
        :func:`plotting.plot_disc_sed`. Runs :meth:`disc_sed_analysis` first
        if it has not been run."""
        if self.disc_sed is None:
            self.disc_sed_analysis()
        return plotting.plot_disc_sed(self.disc_sed, self.bands, **kwargs)

    def _optimum_init_values(self) -> dict:
        """Initial values for every sample site of the model ``.fit()`` will
        run, at ``.optimise()``'s peak: the nonlinear parameters directly,
        and (when the fit samples them) the linear ones at their conditional
        posterior mean given that peak, via ``draw_linear=True`` with
        ``linear_eps`` pinned to zero."""
        from numpyro import handlers

        values = {k: v for k, v in self.optimum.items() if np.ndim(v) == 0 and not k.startswith("y_pred_")}
        if self.marginalise_linear:
            return values
        kwargs = dict(self._model_kwargs(), marginalise_linear=True, draw_linear=True)
        curves = dict(self.bands, **({"driver": self.driver_data} if self.driver_data is not None else {}))
        n_linear = 2 * len(self.freqs) + sum(
            (f"C_{n}" not in self.fixed_params) + d.get("background_order", 0) for n, d in curves.items()
        )
        substituted = dict(values, linear_eps=jnp.zeros(n_linear))
        trace = handlers.trace(handlers.substitute(handlers.seed(reverberation_model, 0), substituted)).get_trace(**kwargs)
        for name in ["S_raw", "C_raw"] + [f"{p}_{n}" for n in curves for p in ("C", "bg")]:
            if name in trace and name not in self.fixed_params:
                values[name] = np.asarray(trace[name]["value"])
        return values

    def _linear_draw_sites(self) -> list:
        """Site names :meth:`_add_linear_draws` adds."""
        names = list(self.bands) + (["driver"] if self.driver_data is not None else [])
        sites = ["S_raw", "C_raw", "S", "C"] + [f"y_pred_{n}" for n in names]
        sites += [f"C_{n}" for n in names if f"C_{n}" not in self.fixed_params]
        curves = dict(self.bands, **({"driver": self.driver_data} if self.driver_data is not None else {}))
        sites += [f"bg_{n}" for n, d in curves.items() if d.get("background_order", 0)]
        return sites

    def _set_lazy_linear_draws(self, nonlinear_by_chain: dict, rng_seed: int):
        """Set ``samples``/``_samples_by_chain`` after a direct solve, with the
        exact linear draws (driver coefficients, offsets, backgrounds, model
        light curves) made only when first needed: they cost ~17 ms per sample
        on a 2-band set (a QR solve each), 17-30 s for 1000, more than the
        rest of ``optimise(method="laplace")``. Reading a nonlinear parameter
        (``ef.samples["log_mdot"]``) never triggers them; reading a linear one,
        iterating the dict (every plot and report does) or saving it does,
        once, for both dicts. They use the model as it is at that moment, so
        don't change the bands or the response function in between."""
        resolver = _LinearDraws(self, nonlinear_by_chain, rng_seed, self._linear_draw_sites())
        self._samples_by_chain = _LazySamples(nonlinear_by_chain, resolver, by_chain=True)
        self.samples = _LazySamples({k: v[0] for k, v in nonlinear_by_chain.items()}, resolver, by_chain=False)

    def _add_linear_draws(self, samples_by_chain: dict, rng_seed: int) -> dict:
        """With ``marginalise_linear``, NUTS never samples the driver's
        Fourier coefficients or the offsets. Draw them here, once per
        posterior sample, from their exact conditional Gaussian posterior
        given that sample's nonlinear parameters (``reverberation_model(...,
        draw_linear=True)`` via ``numpyro.infer.Predictive``), and add them
        under the same site names the default model uses (``S_raw``,
        ``C_raw``, ``S``, ``C``, ``C_{band}``, ``C_driver``, ``y_pred_*``),
        so every plot/report/saved chain works unchanged. A no-op
        otherwise. Uses its own fixed key (derived from ``rng_seed``), so
        re-running the same fit reproduces the same draws.
        """
        if not self.marginalise_linear:
            return samples_by_chain
        from numpyro.infer import Predictive

        n_chains, n_draws = next(iter(samples_by_chain.values())).shape[:2]
        flat = {k: jnp.reshape(jnp.asarray(v), (-1,) + v.shape[2:]) for k, v in samples_by_chain.items()}
        return_sites = self._linear_draw_sites()
        # Compiled once and run sample by sample (Predictive's default, a
        # lax.map): eagerly, every operation of every draw was dispatched and
        # many compiled on their own, ~3-8x slower; vectorising the draws
        # instead was no faster on CPU and needs memory for all of them at once.
        model_kwargs = self._model_kwargs()
        draw = jax.jit(lambda samples, key: Predictive(
            reverberation_model, posterior_samples=samples, return_sites=return_sites)(key, **model_kwargs, draw_linear=True))
        draws = draw(flat, jax.random.fold_in(jax.random.PRNGKey(rng_seed), 1))
        out = dict(samples_by_chain)
        for k, v in draws.items():
            out[k] = np.asarray(v).reshape((n_chains, n_draws) + v.shape[1:])
        return out

    def _save_chains(self, run_dir):
        """Save the final posterior. ``chains.npz`` (plain numpy, no extra
        dependencies) is always written; ``chains.nc`` (an ArviZ
        InferenceData, more standard for further MCMC analysis/diagnostics)
        is best-effort -- skipped with a warning if a netCDF backend isn't
        available, rather than failing an otherwise-successful fit."""
        posterior = {
            k: v for k, v in self._samples_by_chain.items() if not k.startswith("y_pred_")
        }
        # drop the leading (num_chains=1) axis: (1, n_samples, ...) -> (n_samples, ...)
        run_manager.save_samples_npz(run_dir / "chains.npz", {k: v[0] for k, v in posterior.items()})
        try:
            import arviz as az

            az.from_dict(posterior=posterior).to_netcdf(str(run_dir / "chains.nc"))
        except Exception as e:
            warnings.warn(f"Could not write chains.nc (ArviZ/netCDF backend issue): {e}")

    # ------------------------------------------------------------------
    def plot_raw_lightcurves(self, **kwargs):
        return plotting.plot_raw_lightcurves(self.bands, driver=self.driver_data, **kwargs)

    def plot_power_spectrum(self, **kwargs):
        """Posterior driver power spectrum vs. the fitted prior shape.

        Sanity check that the driver's Fourier coefficients (S, C) are
        actually behaving like the assumed prior under the posterior, not
        just the prior itself: P(w) = (S**2 + C**2) / (2*dw) should track
        the fitted curve -- a Lorentzian (from posterior sigma_drw/tau_drw
        draws), flattening into a w**-2 slope above 1/tau_drw, if
        ``drw_prior=True``; a pure w**-2 power law everywhere otherwise
        (the default -- see model.py's "random-walk prior" note).
        """
        if self.samples is None:
            raise RuntimeError("Call .fit() before plotting the power spectrum.")
        tau_drw = np.asarray(self.samples["tau_drw"]) if "tau_drw" in self.samples else None
        return plotting.plot_power_spectrum(
            np.asarray(self.freqs),
            np.asarray(self.samples["S"]),
            np.asarray(self.samples["C"]),
            np.asarray(self.samples["sigma_drw"]),
            tau_drw,
            **kwargs,
        )

    def plot_mcmc_diagnostics(self, param_names=None, **kwargs):
        if self._samples_by_chain is None:
            raise RuntimeError("Call .fit() before plotting diagnostics.")
        samples_by_chain = self._samples_by_chain
        if param_names is None:
            scalar_like = [
                k for k in samples_by_chain
                if not k.startswith("y_pred_") and k not in ("S", "C")
            ]
            param_names = scalar_like
        return plotting.plot_mcmc_diagnostics(samples_by_chain, param_names=param_names, **kwargs)

    def plot_corner(self, param_names=("log_mdot", "inclination"), true_values=None, **kwargs):
        """Corner plot (pairwise joint posteriors + marginals, coloured per
        chain) -- see :func:`plotting.plot_corner`. Defaults to
        ``log_mdot``/``inclination`` (present only if at least one band
        used ``lag_mode="physical"``); pass e.g.
        ``param_names=("tau_line_a", "tau_line_b")`` for a free-lag fit.
        """
        if self._samples_by_chain is None:
            raise RuntimeError("Call .fit() before plotting the corner plot.")
        return plotting.plot_corner(
            self._samples_by_chain, param_names=param_names, true_values=true_values, **kwargs
        )

    def plot_corner_bands(self, true_values=None, **kwargs):
        """Corner plot of every band's offset/stretch parameters
        (``S_{band}``, ``C_{band}`` -- the linear scale/offset absorbing
        each band's own flux calibration, per ``model.reverberation_model``'s
        per-band loop). Band names come from ``self.bands``, so this always
        covers every band in the fit, however many there are.
        """
        if self._samples_by_chain is None:
            raise RuntimeError("Call .fit() before plotting the corner plot.")
        param_names = [f"{p}_{name}" for name in self.bands for p in ("S", "C")]
        return plotting.plot_corner(
            self._samples_by_chain, param_names=param_names, true_values=true_values, **kwargs
        )

    def plot_corner_free_lag(self, true_values=None, **kwargs):
        """Corner plot of every ``lag_mode="free"`` band's independently
        inferred lag (``tau_{band}``) -- the top-hat centroid parameters
        (see ``forward_model.tophat_response_free``). Raises if no band in
        this fit used ``lag_mode="free"``.
        """
        if self._samples_by_chain is None:
            raise RuntimeError("Call .fit() before plotting the corner plot.")
        free_lag_bands = [name for name, d in self.bands.items() if d["lag_mode"] == "free"]
        if not free_lag_bands:
            raise ValueError("plot_corner_free_lag: no lag_mode=\"free\" bands in this fit.")
        param_names = [f"tau_{name}" for name in free_lag_bands]
        return plotting.plot_corner(
            self._samples_by_chain, param_names=param_names, true_values=true_values, **kwargs
        )

    def plot_fourier_correlation(self, **kwargs):
        """Posterior correlation matrix of the driver's Fourier
        coefficients -- see :func:`plotting.plot_fourier_correlation` for
        why this is a heatmap rather than a corner plot (``S``/``C`` are
        each one vector-valued site with ``n_freq`` components, not
        individually-named scalar sites, and ``n_freq`` is often in the
        tens).
        """
        if self.samples is None:
            raise RuntimeError("Call .fit() before plotting the Fourier correlation.")
        return plotting.plot_fourier_correlation(
            self.samples["S"], self.samples["C"], np.asarray(self.freqs), **kwargs
        )

    def plot_bof(self, checkpoint_every=None, **kwargs):
        """Badness-of-Fit trace (2 x NUTS potential energy, one line per
        chain) -- see :func:`plotting.plot_bof`. Requires ``extra_fields``
        to include ``potential_energy``, which every fit since this feature
        was added requests by default (``inference.run_mcmc``/
        ``run_mcmc_chunked``); raises if it's missing, e.g. after resuming
        a checkpoint saved before this feature existed.
        """
        if not self._extra_fields_by_chain or "potential_energy" not in self._extra_fields_by_chain:
            raise RuntimeError(
                "plot_bof: no 'potential_energy' in extra_fields -- call .fit() first, "
                "or (if resuming) this checkpoint predates BOF tracking."
            )
        return plotting.plot_bof(
            self._extra_fields_by_chain["potential_energy"], checkpoint_every=checkpoint_every, **kwargs
        )

    def plot_landscape(self, truth=None, **kwargs):
        """Badness-of-Fit landscape of ``log_mdot`` against inclination from
        :meth:`nested_laplace` (the scout grid over the prior range, and the
        final grid with 68/95/99.7 per cent contours) -- see
        :func:`plotting.plot_landscape`. Marks ``truth`` if given and the
        single-Gaussian solve's peak if ``optimise(method="laplace")`` has
        run on this object."""
        if not self.nested_laplace_result:
            raise RuntimeError("plot_landscape: call .optimise() or .nested_laplace() first.")
        return plotting.plot_landscape(self.nested_laplace_result, truth=truth, optimum=self._laplace_peak, **kwargs)

    def plot_optimise_restarts(self, param_names=None, **kwargs):
        """Multi-start reproducibility of :meth:`optimise` -- see
        :func:`plotting.plot_optimise_restarts`."""
        if not self.optimise_restarts:
            raise RuntimeError("plot_optimise_restarts: call .optimise() first.")
        return plotting.plot_optimise_restarts(
            self.optimise_restarts, self.samples, param_names=param_names,
            agreement_sd=RESTART_AGREEMENT_SD, **kwargs,
        )

    def plot_lightcurve_fits(
        self, n_fine: int = 200, n_pred_samples: int = 200, extrapolate_days: float = 30.0, **kwargs
    ):
        """Draw posterior-predictive light curves and response functions.

        Subsamples up to ``n_pred_samples`` posterior draws for speed, and
        evaluates them (vectorised with ``jax.vmap``) on a dense time grid
        per band plus the shared lag grid. This all happens *after* ``.fit()``
        -- extending ``extrapolate_days`` does not slow down NUTS, only the
        (cheap, matrix-multiply) posterior-predictive evaluation here.

        Parameters
        ----------
        extrapolate_days : float
            Extend the plotted time range this far (days) before the first
            and after the last observation, so the credible band's growth
            outside the data is visible (for a DRW-like driver this should
            look roughly like a t^(1/2) widening before saturating).
        """
        if self.samples is None:
            raise RuntimeError("Call .fit() before plotting fits.")

        all_t = np.concatenate([d["t"] for d in self.bands.values()])
        t_fine = jnp.linspace(
            all_t.min() - extrapolate_days, all_t.max() + extrapolate_days, n_fine
        )

        n_total = self.samples["S"].shape[0]
        idx = np.random.default_rng(0).choice(
            n_total, size=min(n_pred_samples, n_total), replace=False
        )

        S = jnp.asarray(self.samples["S"])[idx]
        C = jnp.asarray(self.samples["C"])[idx]
        has_physical = any(d["lag_mode"] == "physical" for d in self.bands.values())
        if has_physical:
            log_mdot = jnp.asarray(self.samples["log_mdot"])[idx]
            inclination = jnp.asarray(self.samples["inclination"])[idx]
            # A fixed slope (the default) is simply not passed, as in the model.
            slope = (jnp.asarray(self.samples["temperature_slope"])[idx]
                     if "temperature_slope" in self.samples else jnp.full(len(idx), jnp.nan))

        def physical_draw(log_mdot_s, incl_s, slope_s, wavelength):
            # Read via the model module's attribute, not a direct import of our
            # own, so that swapping model.response_function (see CLAUDE.md's
            # "swappable by contract" design decision) is reflected here too --
            # a direct `from .forward_model import response_function` would
            # bind an independent copy that a swap on model.py wouldn't reach,
            # leaving the fit and this plot inconsistent with each other.
            slope_kwargs = {"viscous_slope": slope_s} if "temperature_slope" in self.samples else {}
            psi = _model.response_function(
                self.tau_grid, log_mdot=log_mdot_s, wavelength=wavelength,
                inclination=incl_s, M_BH=self.M_BH, **slope_kwargs,
            )
            return psi

        def free_draw(tau_s):
            return tophat_response_free(self.tau_grid, tau_mean=tau_s)

        def echo_draw(S_s, C_s, psi, S_band_s, C_band_s):
            A, B = transfer_coeffs(self.tau_grid, psi, self.freqs)
            echo = compute_echo(S_s, C_s, self.freqs, A, B, t_fine)
            return S_band_s * echo + C_band_s

        def sample(site):
            return jnp.asarray(self.samples[site])[idx]

        y_pred_samples, psi_samples, baseline_samples = {}, {}, {}
        for name, d in self.bands.items():
            S_band, C_band = sample(f"S_{name}"), sample(f"C_{name}")
            if d["lag_mode"] == "physical":
                psi = jax.vmap(physical_draw, in_axes=(0, 0, 0, None))(log_mdot, inclination, slope, d["wavelength"])
            else:
                psi = jax.vmap(free_draw)(sample(f"tau_{name}"))
            if d.get("diffuse_continuum", False):
                # The same mixture as the model (model.reverberation_model).
                psi = jax.vmap(lambda p, f, m, w: mix_diffuse_continuum(p, self.tau_grid, f, m, w))(
                    psi, sample(f"dce_fraction_{name}"), sample(f"dce_delay_{name}"), sample(f"dce_width_{name}"))
            y_pred = jax.vmap(echo_draw)(S, C, psi, S_band, C_band)
            basis_fine = self._background_basis(d, t=t_fine)
            if basis_fine is not None:
                background = sample(f"bg_{name}") @ basis_fine.T
                y_pred = y_pred + background
                # The non-reverberating part, offset plus background, drawn in
                # place of the flat offset line.
                baseline_samples[name] = np.asarray(C_band[:, None] + background)
            y_pred_samples[name] = np.asarray(y_pred)
            psi_samples[name] = np.asarray(psi)

        driver_samples = jax.vmap(
            lambda S_s, C_s: driver_at(S_s, C_s, self.freqs, t_fine)
        )(S, C)

        driver_points = None
        driver_sigma_eff = None
        if self.driver_data is not None:
            S_driver = float(np.mean(self.samples["S_driver"][idx]))
            C_driver = float(np.mean(self.samples["C_driver"][idx]))
            driver_offset = C_driver
            basis_driver = self._background_basis(self.driver_data)
            if basis_driver is not None:
                driver_offset = C_driver + np.asarray(basis_driver) @ np.mean(self.samples["bg_driver"][idx], axis=0)
            driver_points = (
                self.driver_data["t"],
                (self.driver_data["y"] - driver_offset) / S_driver,
                self.driver_data["yerr"] / abs(S_driver),
            )
            if self.driver_data.get("fit_error_model", False):
                sigma_scale_driver = float(np.mean(self.samples["sigma_scale_driver"][idx]))
                sigma_jitter_driver = float(np.mean(self.samples["sigma_jitter_driver"][idx]))
                sigma_eff_driver = np.sqrt(
                    (sigma_scale_driver * self.driver_data["yerr"]) ** 2 + sigma_jitter_driver ** 2
                )
                driver_sigma_eff = sigma_eff_driver / abs(S_driver)

        # Effective error bars (sigma_scale/sigma_jitter, decision #18) are only
        # meaningful for a band actually fitted with fit_error_model=True.
        sigma_eff_by_band = {}
        for name, d in self.bands.items():
            if not d.get("fit_error_model", False):
                continue
            sigma_scale = float(np.mean(self.samples[f"sigma_scale_{name}"][idx]))
            sigma_jitter = float(np.mean(self.samples[f"sigma_jitter_{name}"][idx]))
            sigma_eff_by_band[name] = np.sqrt((sigma_scale * d["yerr"]) ** 2 + sigma_jitter ** 2)

        c_band_samples = {name: np.asarray(self.samples[f"C_{name}"]) for name in self.bands}

        return plotting.plot_lightcurve_fits(
            self.bands, np.asarray(t_fine), y_pred_samples,
            np.asarray(self.tau_grid), psi_samples,
            driver_samples=np.asarray(driver_samples), driver_points=driver_points,
            driver_sigma_eff=driver_sigma_eff, sigma_eff_by_band=sigma_eff_by_band or None,
            c_band_samples=c_band_samples, baseline_samples=baseline_samples or None,
            **kwargs,
        )
