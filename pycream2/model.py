"""
model.py
========

The NumPyro probabilistic model tying together:

* a stochastic driver, represented as a truncated Fourier series whose
  sine/cosine amplitudes are given Gaussian priors matching a target power
  spectrum -- by default a pure random-walk (RW) power law (``drw_prior=False``,
  ``sigma_drw`` the only hyperparameter, inferred), or, when ``drw_prior=True``,
  a damped random walk (DRW)'s Lorentzian power spectrum (``sigma_drw`` and
  ``tau_drw`` both inferred) -- see the "random-walk prior" note below;
* an optional driver light curve, a direct (zero-lag) observation of the
  driver itself -- see the module docstring note on identifiability below;
* a per-band causal response function -- either the physical, thin-disk-
  scaling one (``forward_model.response_function``, tied to a single shared
  ``log_mdot``/``inclination``/``M_BH`` across all such bands) or a free-lag
  one (``forward_model.tophat_response_free``, an independently inferred
  lag per band, e.g. for emission-line reverberation mapping) -- that
  convolves the driver into each band's echo;
* a Gaussian observation likelihood for irregularly sampled multi-band light
  curves.

Inferred, always: sigma_drw, {S_k, C_k} driver Fourier coefficients,
{S_band, C_band} per band; tau_drw too if ``drw_prior=True``. Inferred if
any band uses the physical response: log_mdot, inclination (shared across
those bands). Inferred per band using the free-lag response: tau_{band}.
Inferred if a driver light curve is given: S_driver, C_driver.

M_BH is a fixed input, never a latent variable.

Identifiability note: a global shift of the driver by any Δ, compensated by
shifting every band's response lag by -Δ, leaves the likelihood exactly
unchanged (see CLAUDE.md) -- the absolute lag origin is not identifiable
from the echoes alone. The physical response ties every such band's lag to
one shared ``log_mdot`` through a fixed, monotonic wavelength scaling
(``lag_scaling``), which breaks that degeneracy across 2+ bands at
different wavelengths -- a multiplicative rescaling of the shared parameter
can't mimic an additive common shift. The free-lag response has no such
tie: each band's lag is independent, so the degeneracy is exact, and a
fit using it for any band needs a driver light curve (or some other
external anchor) to be identifiable.

Driver amplitude note: a global rescale of the driver by any lambda != 0
(``S, C -> lambda*S, lambda*C``), compensated by rescaling every band's
gain by ``1/lambda`` (``S_band -> S_band/lambda`` for every band), also
leaves every predicted light curve, and hence the likelihood, exactly
unchanged -- unlike the shift degeneracy above, this holds regardless of
lag_mode or how many bands there are, since it's a property of the model's
linear driver-amplitude/per-band-gain separability, not the disk physics.
It's broken only by the priors, not the data: with `sigma_drw`'s prior
scale a fixed constant unrelated to the actual light curves' units, this
direction is only weakly regularised, which shows up as the inferred
driver visibly growing/shrinking with little constraint early in a chain
(see the GIF in README.md). ``EchoFit`` anchors `sigma_drw`'s prior scale
to the registered data's own amplitude instead (see
``EchoFit._sigma_drw_prior_scale``) -- the same fix, by the same
mechanism, as the author's PhD-era CREAM Fortran code's optional Gaussian
prior on its power-spectrum normalisation `P0` (`cream_f90.f90`'s `bof4`,
gated by `sigp0square`/`siglogp0`), which plays the same role as
`sigma_drw` here.

Random-walk prior note: `drw_prior` (`EchoFit(drw_prior=...)`) chooses
between a pure random-walk (RW) driver prior, `rw_prior_scale` (default,
`P(w) = sigma_drw**2 / w**2`, no damping timescale, no `tau_drw` site at
all), and the original damped random walk (DRW) prior, `drw_prior_scale`
(`P(w) = sigma_drw**2 * tau_drw / (1 + (w*tau_drw)**2)`, with `tau_drw`
inferred alongside `sigma_drw`). RW is the default: for the short,
irregularly-sampled campaigns this package targets, `tau_drw` is often
weakly identified anyway (see "Known rough edges" in CLAUDE.md), and a
plain power law is one fewer hyperparameter to identify for the same
qualitative "smooth stochastic variability" driver.

This is deliberately *not* implemented as "fix `tau_drw` to a large
constant inside `drw_prior_scale`" (the seemingly obvious way to remove
the turnover) -- that limit sends the power spectrum to *zero* everywhere
at fixed `sigma_drw` (`sigma_drw**2 * tau_drw / (w*tau_drw)**2 =
sigma_drw**2 / (tau_drw * w**2) -> 0` as `tau_drw -> infinity`), not to a
finite power law. Getting a genuine, non-trivial `1/w**2` law in that
limit needs `sigma_drw**2 / tau_drw` held fixed as `tau_drw` grows, i.e.
`sigma_drw`'s own prior would need rescaling to compensate for whatever
constant `tau_drw` was fixed to -- confirmed directly, not assumed.
Writing `rw_prior_scale`'s `1/w**2` form directly sidesteps this
entirely: exact, no large-constant tuning, no `tau_drw` site or rescaling
needed, and `sigma_drw` keeps the *same* site name and the *same*
data-anchored prior (`EchoFit._sigma_drw_prior_scale`) in both cases,
even though its physical meaning differs (a saturating asymptotic
variability scale for the DRW; a non-saturating power-law amplitude for
the RW, since a pure random walk's variance grows without bound).

Inclination note: `inclination` is sampled as `cos_inclination ~
Uniform(cos(INCLINATION_MAX_DEG), 1)`, not `inclination ~
Uniform(0, INCLINATION_MAX_DEG)` directly, with `inclination` itself a
`numpyro.deterministic` transform (`arccos`) of it. This is the standard
"isotropic orientation" prior: solid angle `dOmega = sin(i) di dphi`
integrates to a flat density in `cos(i)`, not in `i` itself -- a uniform
prior in `i` directly over-weights edge-on orientations relative to a
population with no preferred axis. `inclination` (in degrees) remains the
name every other function reads (plotting, `response_function`, the disk
visualisation) -- only the sampling parameterisation changed, not the
site's public meaning.

Fixed-parameter note: `fixed_params` (`EchoFit(fixed_params={...})`) lets
any of this model's scalar sites (`sigma_drw`, `tau_drw` if `drw_prior=True`
(it isn't a site at all otherwise), `log_mdot`,
`inclination`, `S_driver`, `C_driver`, `S_{band}`, `C_{band}`, free-lag
bands' `tau_{band}`, and any band/driver's `sigma_scale_{name}`/
`sigma_jitter_{name}` if its error model is turned on, see below) be held
at a known value instead of inferred, e.g. `fixed_params={"inclination":
0.0}` to assume a face-on disk while fitting everything else. This
generalises decision #3's "`M_BH` is always fixed" to any parameter a
caller already knows or wants to hold fixed for a particular fit, via the
same mechanism throughout (`_param`: substitute a `numpyro.deterministic`
constant for the `numpyro.sample` call) rather than a bespoke flag per
parameter. A fixed `inclination` bypasses the `cos_inclination`
reparameterisation above entirely -- the fixed value is used directly, in
degrees, matching how every other caller of `inclination` already expects
it.

Error-model note: each band (and the driver light curve, if registered)
can optionally fit its own error rescaling, via `EchoFit.add_lightcurve(...,
fit_error_model=True)` (default `False`, off, exactly today's behaviour --
this is opt-in per light curve, not a global switch). When on, the
band's/driver's reported `yerr` is treated as only approximately correct
and combined with two extra nuisance parameters into an effective sigma:

    sigma_eff = sqrt((sigma_scale * yerr)**2 + sigma_jitter**2)

`sigma_scale_{name}` (`LogNormal(0, 0.5)`, median 1: "no rescaling" is the
prior's own central value) multiplicatively rescales the whole error
array, and `sigma_jitter_{name}` (`HalfNormal(mean(yerr))`, i.e. anchored
to that light curve's own typical quoted error, the same data-anchoring
philosophy as `sigma_drw`'s prior, decision #13) adds a constant "floor"
variance term. This is a direct, checked adaptation of the author's
PhD-era CREAM Fortran code's own `sigexpand`/`varexpand` nuisance
parameters (`cream_f90.f90`, `ernew2 = (er(it)*fnow)**2 + varnow`,
confirmed by reading the source -- see decision #18 and
`docs/mcmc_implementation.md`), for the same reason it exists there: real
quoted photometric/measurement errors are often mis-calibrated (too small
or too large), and fitting the rescaling instead of trusting `yerr`
verbatim avoids an overconfident (or underconfident) posterior on
everything else. Off by default per light curve so synthetic-data fits
(where `yerr` genuinely is correct by construction) and any existing
analysis keep exactly today's likelihood unless explicitly opted in.

Extra-components note: two optional, per-light-curve components, both off
by default (``EchoFit.add_lightcurve(..., diffuse_continuum=True,
background_order=K)``; ``docs/extra_components.md`` has the physics, priors
and references).

* **Diffuse continuum** (bands only): a second reprocessor with a log-normal
  delay distribution (``forward_model.lognormal_response``), mixed into the
  band's response, ``psi -> (1 - f) psi + f psi_LN(median, width)``, after
  Cackett, Zoghbi & Ulrich (2022). Three parameters per band:
  ``dce_fraction_{name}`` (``f``, the diffuse component's share of the band's
  integrated response; ``Uniform(0, 1)``), ``dce_delay_{name}`` (its median
  delay, days; ``LogUniform(DCE_DELAY_MIN, tau_max)``) and ``dce_width_{name}``
  (its rms width in dex; ``Uniform(*DCE_WIDTH_PRIOR_DEX)``). Because both
  parts are area-normalised, the band's gain still multiplies the whole
  response, so nothing else changes, including the linear marginalisation.
* **Slow background** (bands and the driver light curve): ``K`` Legendre
  polynomials in time, ``P_1 .. P_K`` over the whole campaign
  (``forward_model.legendre_background_basis``; the basis arrives
  precomputed as each light curve's ``"background_basis"``), added to the
  constant offset with coefficients ``bg_{name}`` (``bg_driver``) ~
  ``Normal(0, BACKGROUND_PRIOR_WIDTH * std(y))``. They are linear, so with
  ``marginalise_linear`` they are integrated out exactly like the offsets.

**Linear-parameter marginalisation note** (``marginalise_linear=True``).
With every nonlinear parameter held fixed (``sigma_drw``/``tau_drw``,
``log_mdot``/``inclination``/``tau_{band}``, each ``S_{band}``/``S_driver``,
the error-model parameters), every predicted light curve is *linear* in the
driver's Fourier coefficients and in each band's/the driver's constant
offset ``C_{band}``/``C_driver``, and all of those have Gaussian priors. With
a Gaussian likelihood they can therefore be integrated out exactly: the
data are jointly Gaussian, ``y ~ N(0, D + M M^T)``, with ``D`` the diagonal
of effective error variances and ``M`` the (prior-whitened) design matrix
mapping the linear parameters onto every observation (every band plus the
driver, stacked). NUTS then samples only the ~10 nonlinear parameters
instead of ~2*n_freq more, removing the strongly correlated,
high-dimensional directions that otherwise push most samples to
``max_tree_depth``. Nothing is lost: given each posterior sample of the
nonlinear parameters, the linear parameters' conditional posterior is an
exact Gaussian, drawn from after the fit (``draw_linear=True``, see
``EchoFit._add_linear_draws``), so ``S``/``C``/``C_{band}`` etc. come back
under their usual names. It is the same joint posterior as the default
model, factorised differently.

Evaluated via the Woodbury/matrix-determinant identities in the
prior-whitened basis (``P = I + M^T D^-1 M``, ``(p, p)``, ``p = 2*n_freq +
n_offsets``): ``log|D + M M^T| = log|D| + log|P|``, and the quadratic form
as ``min_theta [(y - M theta)^T D^-1 (y - M theta) + theta^T theta]``,
evaluated at its minimiser ``theta_hat = P^-1 M^T D^-1 y``. That form is
used on purpose rather than the textbook ``y^T D^-1 y - b^T P^-1 b``: the
latter subtracts two large, nearly equal numbers and loses the answer to
float32 cancellation, whereas an error in ``theta_hat`` only perturbs a
minimum at second order.
"""

from __future__ import annotations

from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .forward_model import (
    response_function, tophat_response_free, transfer_coeffs, compute_echo, driver_at, mix_diffuse_continuum,
)

INCLINATION_MAX_DEG = 80.0
# Uniform prior bounds on the free temperature slope alpha (T ~ r**-alpha,
# fit_temperature_slope=True). The upper bound was 1.5 until NGC 5548's
# posterior piled up against it.
TEMPERATURE_SLOPE_PRIOR = (0.5, 2.5)
# Standard deviation of the Normal(0, sd) prior on log_mdot. It was 1 until
# September 2026, when NGC 5548's thin-disc fits sat 3 and 6 prior sd into its
# tail (log_mdot 3.1 with alpha = 3/4, 5.9 with alpha free) and the prior was
# pulling the free-slope fit by ~0.8. At 5 it is broad on every scale seen so
# far but still keeps the optimiser off absurd values along the flat
# log_mdot/alpha direction.
LOG_MDOT_PRIOR_SD = 5.0
# Diffuse-continuum component (add_lightcurve(..., diffuse_continuum=True)):
# its median delay ~ LogUniform(DCE_DELAY_MIN, tau_max) days (scale-free,
# broad-line-region delays span decades), its width ~ Uniform(*DCE_WIDTH_PRIOR_DEX)
# in dex (0.05 dex is a 12% spread, about the narrowest a lag grid resolves;
# 1 dex a factor-of-ten spread), its share of the response ~ Uniform(0, 1).
DCE_DELAY_MIN = 0.1
DCE_WIDTH_PRIOR_DEX = (0.05, 1.0)
# Slow background (add_lightcurve(..., background_order=K)): each Legendre
# coefficient ~ Normal(0, BACKGROUND_PRIOR_WIDTH * std(y)), i.e. a trend up to
# about the light curve's own variability. |P_k| <= 1 over the campaign, so a
# coefficient is roughly the trend's amplitude.
BACKGROUND_PRIOR_WIDTH = 1.0


def drw_prior_scale(freqs: jnp.ndarray, sigma_drw, tau_drw) -> jnp.ndarray:
    """Standard deviation of each Fourier coefficient under a DRW prior.

    The DRW has a Lorentzian power spectrum in angular frequency w:

        P(w) = sigma_drw**2 * tau_drw / (1 + (w * tau_drw)**2)

    Discretising onto the fixed frequency grid ``freqs`` with local spacing
    ``dw_k``, the implied standard deviation of each independent
    sine/cosine amplitude is ``sqrt(P(w_k) * dw_k)``. This turns the
    "arbitrary sinusoids" driver into a proper (approximate) DRW Gaussian
    process, with ``sigma_drw`` (long-term variability amplitude) and
    ``tau_drw`` (damping timescale, days) as the two inferred hyperparameters.
    """
    power = sigma_drw ** 2 * tau_drw / (1.0 + (freqs * tau_drw) ** 2)
    # local frequency spacing (grid is expected to be sorted, positive)
    dw = jnp.gradient(freqs)
    dw = jnp.clip(dw, 1e-8, None)
    return jnp.sqrt(power * dw)


def rw_prior_scale(freqs: jnp.ndarray, sigma_drw) -> jnp.ndarray:
    """Standard deviation of each Fourier coefficient under a pure
    random-walk (RW) prior -- the "no damping timescale" limit of the DRW
    above, with a genuine power-law power spectrum at every frequency on
    the grid:

        P(w) = sigma_drw**2 / w**2

    Written directly rather than by taking ``tau_drw -> infinity`` in
    ``drw_prior_scale``: that limit sends the power spectrum to *zero*
    everywhere at fixed ``sigma_drw`` (``sigma_drw**2 * tau_drw / (w*tau_drw)**2
    = sigma_drw**2 / (tau_drw * w**2) -> 0`` as ``tau_drw -> infinity``),
    not to a finite power law -- getting a non-trivial limit needs
    ``sigma_drw**2 / tau_drw`` held fixed as ``tau_drw`` grows, which is
    exactly what writing the ``1/w**2`` form directly does, with no
    ``tau_drw`` site or rescaling required. ``sigma_drw`` keeps the same
    site name and the same data-anchored prior (``EchoFit._sigma_drw_prior_scale``)
    as the DRW case, but its physical meaning changes: it's now this
    power law's amplitude, not a saturating asymptotic variability scale
    (a pure random walk's variance grows without bound, it never
    saturates the way a DRW's does).
    """
    dw = jnp.gradient(freqs)
    dw = jnp.clip(dw, 1e-8, None)
    power = sigma_drw ** 2 / freqs ** 2
    return jnp.sqrt(power * dw)


def reverberation_model(
    freqs: jnp.ndarray,
    tau_grid: jnp.ndarray,
    M_BH: Optional[float],
    bands: Dict[str, dict],
    driver: Optional[dict] = None,
    sigma_drw_prior_scale: float = 2.0,
    fixed_params: Optional[Dict[str, float]] = None,
    drw_prior: bool = False,
    transfer_mats: Optional[tuple] = None,
    marginalise_linear: bool = False,
    draw_linear: bool = False,
    fit_temperature_slope: bool = False,
):
    """NumPyro model for multi-band reverberation-mapped light curves.

    Parameters
    ----------
    freqs : (n_freq,) array
        Fixed grid of driver angular frequencies (rad/day).
    tau_grid : (n_tau,) array
        Fixed grid of lags (days) used to evaluate/normalise psi and its
        Fourier transform.
    M_BH : float, optional
        Fixed black hole mass (solar masses). Not inferred. Only needed
        (may be ``None`` otherwise) if at least one band uses
        ``lag_mode="physical"``.
    bands : dict
        Mapping ``band_name -> {"t", "y", "yerr", "wavelength", "lag_mode",
        "fit_error_model"}`` for each observed light curve. ``lag_mode`` is
        ``"physical"`` (mean lag tied to the shared ``log_mdot`` via
        ``lag_scaling``) or ``"free"`` (an independently inferred
        ``tau_{band_name}``) -- see the module docstring's identifiability
        note for when ``"free"`` needs a ``driver`` to be identifiable.
        ``fit_error_model`` (optional, default ``False``) turns on that
        band's ``sigma_scale_{band_name}``/``sigma_jitter_{band_name}`` --
        see the module docstring's "error-model" note.
    driver : dict, optional
        ``{"t", "y", "yerr", "fit_error_model"}`` for a light curve that
        directly (zero-lag) observes the driver itself, e.g. an
        X-ray/lamppost continuum, or a directly-monitored AGN continuum
        anchoring an emission-line fit. ``fit_error_model`` works the same
        way as for a band, turning on ``sigma_scale_driver``/
        ``sigma_jitter_driver``.
    sigma_drw_prior_scale : float
        Scale of ``sigma_drw``'s ``HalfNormal`` prior. ``EchoFit`` sets this
        from the registered light curves' own data (see
        ``EchoFit._sigma_drw_prior_scale``) rather than leaving it a fixed
        constant -- see the module docstring's "driver amplitude" note for
        why. Defaults to the old fixed value only for direct/standalone
        calls to this function.
    fixed_params : dict, optional
        ``{site_name: value}`` for any scalar site this model would
        otherwise sample -- see the module docstring's "fixed-parameter"
        note. Unrecognised keys are silently unused (``EchoFit`` validates
        them against the actual registered bands/driver before fitting).
    drw_prior : bool
        ``False`` (default): the driver's Fourier coefficients follow a
        pure random-walk (RW) prior, ``rw_prior_scale`` -- no ``tau_drw``
        site at all. ``True``: the original damped random walk (DRW)
        prior, ``drw_prior_scale``, with ``tau_drw`` inferred alongside
        ``sigma_drw``. See the module docstring's "random-walk prior" note.
    transfer_mats : tuple, optional
        ``(Wc, Ws)`` from ``forward_model.transfer_matrices(tau_grid,
        freqs)``, and likewise an optional ``"basis"`` entry
        (``forward_model.fourier_basis(freqs, t)``) in each band/driver
        dict: fixed trig matrices precomputed once instead of rebuilt on
        every gradient evaluation (~5x faster per NUTS step, identical
        result). ``EchoFit`` always supplies both; omitted, everything is
        computed on the fly as before.
    marginalise_linear : bool
        Integrate the driver's Fourier coefficients and every
        ``C_{band}``/``C_driver`` offset out analytically instead of
        sampling them -- see the module docstring's "linear-parameter
        marginalisation" note. Only the nonlinear parameters remain NUTS
        sample sites; the likelihood becomes a single ``numpyro.factor``.
    fit_temperature_slope : bool
        Infer the disk temperature slope as ``temperature_slope``
        (``Uniform(0.5, 1.5)``) and pass it to ``response_function`` as
        ``viscous_slope`` (see ``EchoFit(fit_temperature_slope=...)``).
    draw_linear : bool
        Only with ``marginalise_linear``, and only for post-processing (via
        ``numpyro.infer.Predictive`` with the nonlinear posterior samples
        substituted in, never under NUTS): also draw the linear parameters
        from their exact conditional Gaussian posterior and expose them as
        ``S_raw``/``C_raw``/``S``/``C``/``C_{band}``/``C_driver``/
        ``y_pred_*`` deterministic sites, under the same names the default
        (sampled) model uses.
    """
    fixed_params = fixed_params or {}

    def _param(name, dist_obj):
        """Sample ``name``, unless ``fixed_params`` pins it to a constant."""
        if name in fixed_params:
            return numpyro.deterministic(name, jnp.asarray(fixed_params[name], dtype=jnp.float32))
        return numpyro.sample(name, dist_obj)

    # -- shared driving-source hyperparameters ---------------------------
    sigma_drw = _param("sigma_drw", dist.HalfNormal(sigma_drw_prior_scale))
    if drw_prior:
        tau_drw = _param("tau_drw", dist.LogNormal(loc=jnp.log(20.0), scale=1.0))
        prior_scale = drw_prior_scale(freqs, sigma_drw, tau_drw)
    else:
        prior_scale = rw_prior_scale(freqs, sigma_drw)
    n_freq = freqs.shape[0]

    # Non-centred parameterisation: S, C's scale is itself a sampled
    # hyperparameter (via prior_scale(sigma_drw[, tau_drw])), which produces
    # a Neal's-funnel geometry if sampled directly ("centred") -- NUTS then
    # can't find one step size that works both where prior_scale is small
    # and where it's large, and every trajectory runs to max tree depth.
    # Sampling unit-scale S_raw/C_raw and pushing the hyperparameter
    # dependence into a deterministic transform removes that coupling.
    if not marginalise_linear:
        with numpyro.plate("freq", n_freq):
            S_raw = numpyro.sample("S_raw", dist.Normal(0.0, 1.0))
            C_raw = numpyro.sample("C_raw", dist.Normal(0.0, 1.0))
        S = numpyro.deterministic("S", S_raw * prior_scale)
        C = numpyro.deterministic("C", C_raw * prior_scale)

    # Marginalised mode only: every light curve's contribution to the joint
    # linear-Gaussian system, stacked and solved once after the band loop.
    blocks = []

    # -- driver light curve: a direct, zero-lag anchor on X(t) itself ----
    if driver is not None:
        S_driver = _param("S_driver", dist.LogNormal(_gain_prior_loc(driver["y"], sigma_drw_prior_scale), 1.0))
        offset_is_linear = marginalise_linear and "C_driver" not in fixed_params
        offset_loc, offset_sd = _offset_prior(driver["y"])
        if not offset_is_linear:
            C_driver = _param("C_driver", dist.Normal(offset_loc, offset_sd))
        bg_basis_driver, bg_sd_driver = _background(driver)
        if not marginalise_linear:
            y_pred_driver = S_driver * driver_at(S, C, freqs, driver["t"], basis=driver.get("basis")) + C_driver
            if bg_basis_driver is not None:
                bg_driver = numpyro.sample(
                    "bg_driver", dist.Normal(0.0, bg_sd_driver).expand([bg_basis_driver.shape[1]]).to_event(1))
                y_pred_driver = y_pred_driver + bg_basis_driver @ bg_driver
            numpyro.deterministic("y_pred_driver", y_pred_driver)
        if driver.get("fit_error_model", False):
            sigma_scale_driver = _param("sigma_scale_driver", dist.LogNormal(0.0, 0.5))
            sigma_jitter_driver = _param("sigma_jitter_driver", dist.HalfNormal(jnp.mean(driver["yerr"])))
            sigma_eff_driver = jnp.sqrt((sigma_scale_driver * driver["yerr"]) ** 2 + sigma_jitter_driver ** 2)
        else:
            sigma_eff_driver = driver["yerr"]
        if marginalise_linear:
            sin_wt, cos_wt = driver.get("basis") or _basis(freqs, driver["t"])
            blocks.append(dict(
                name="driver", y=driver["y"], sigma=sigma_eff_driver, gain=S_driver,
                S_cols=sin_wt, C_cols=cos_wt, known_offset=None if offset_is_linear else C_driver,
                offset_loc=offset_loc, offset_sd=offset_sd, bg_cols=bg_basis_driver, bg_sd=bg_sd_driver,
            ))
        else:
            numpyro.sample("obs_driver", dist.Normal(y_pred_driver, sigma_eff_driver), obs=driver["y"])

    # -- shared physical reprocessing parameters (physical-mode bands only) --
    if any(d["lag_mode"] == "physical" for d in bands.values()):
        log_mdot = _param("log_mdot", dist.Normal(0.0, LOG_MDOT_PRIOR_SD))
        if "inclination" in fixed_params:
            inclination = numpyro.deterministic(
                "inclination", jnp.asarray(fixed_params["inclination"], dtype=jnp.float32)
            )
        else:
            # Uniform in cos(inclination), not inclination itself -- see the
            # module docstring's "inclination note". inclination stays the
            # public (degrees) site every other function reads.
            cos_incl_min = jnp.cos(jnp.deg2rad(INCLINATION_MAX_DEG))
            cos_inclination = numpyro.sample("cos_inclination", dist.Uniform(cos_incl_min, 1.0))
            inclination = numpyro.deterministic("inclination", jnp.rad2deg(jnp.arccos(cos_inclination)))
        # Free temperature slope alpha (T ~ r**-alpha), Starkey et al. (2017)'s
        # Model 2; only passed to the response when requested, so responses
        # without a viscous_slope argument (the skew-normal) are unaffected.
        slope_kwargs = {}
        if fit_temperature_slope:
            slope_kwargs["viscous_slope"] = _param("temperature_slope", dist.Uniform(*TEMPERATURE_SLOPE_PRIOR))

    # tau_grid[-1], not float(...): under NUTS's internal while_loop tracing
    # tau_grid can be an abstract tracer, and dist.Uniform accepts a JAX
    # scalar directly -- no need to (and, when traced, can't) concretise it.
    tau_max = tau_grid[-1]

    # -- per-band amplitude / offset + likelihood ------------------------
    for band_name, d in bands.items():
        S_band = _param(f"S_{band_name}", dist.LogNormal(_gain_prior_loc(d["y"], sigma_drw_prior_scale), 1.0))
        offset_is_linear = marginalise_linear and f"C_{band_name}" not in fixed_params
        offset_loc, offset_sd = _offset_prior(d["y"])
        if not offset_is_linear:
            C_band = _param(f"C_{band_name}", dist.Normal(offset_loc, offset_sd))

        if d["lag_mode"] == "physical":
            psi = response_function(
                tau_grid,
                log_mdot=log_mdot,
                wavelength=d["wavelength"],
                inclination=inclination,
                M_BH=M_BH,
                **slope_kwargs,
            )
        else:
            tau_band = _param(f"tau_{band_name}", dist.Uniform(0.0, tau_max))
            psi = tophat_response_free(tau_grid, tau_mean=tau_band)
        if d.get("diffuse_continuum", False):
            psi = mix_diffuse_continuum(
                psi, tau_grid,
                fraction=_param(f"dce_fraction_{band_name}", dist.Uniform(0.0, 1.0)),
                median=_param(f"dce_delay_{band_name}", dist.LogUniform(DCE_DELAY_MIN, tau_max)),
                width_dex=_param(f"dce_width_{band_name}", dist.Uniform(*DCE_WIDTH_PRIOR_DEX)),
            )

        bg_basis, bg_sd = _background(d)
        A, B = transfer_coeffs(tau_grid, psi, freqs, matrices=transfer_mats)
        if not marginalise_linear:
            echo = compute_echo(S, C, freqs, A, B, d["t"], basis=d.get("basis"))
            y_pred = S_band * echo + C_band
            if bg_basis is not None:
                bg = numpyro.sample(
                    f"bg_{band_name}", dist.Normal(0.0, bg_sd).expand([bg_basis.shape[1]]).to_event(1))
                y_pred = y_pred + bg_basis @ bg
            numpyro.deterministic(f"y_pred_{band_name}", y_pred)

        if d.get("fit_error_model", False):
            sigma_scale = _param(f"sigma_scale_{band_name}", dist.LogNormal(0.0, 0.5))
            sigma_jitter = _param(f"sigma_jitter_{band_name}", dist.HalfNormal(jnp.mean(d["yerr"])))
            sigma_eff = jnp.sqrt((sigma_scale * d["yerr"]) ** 2 + sigma_jitter ** 2)
        else:
            sigma_eff = d["yerr"]

        if marginalise_linear:
            # echo = sin_wt @ (S*A + C*B) + cos_wt @ (C*A - S*B), so these are
            # the columns multiplying S and C (see forward_model.compute_echo).
            sin_wt, cos_wt = d.get("basis") or _basis(freqs, d["t"])
            blocks.append(dict(
                name=band_name, y=d["y"], sigma=sigma_eff, gain=S_band,
                S_cols=sin_wt * A - cos_wt * B, C_cols=sin_wt * B + cos_wt * A,
                known_offset=None if offset_is_linear else C_band,
                offset_loc=offset_loc, offset_sd=offset_sd, bg_cols=bg_basis, bg_sd=bg_sd,
            ))
        else:
            numpyro.sample(
                f"obs_{band_name}",
                dist.Normal(y_pred, sigma_eff),
                obs=d["y"],
            )

    if marginalise_linear:
        _linear_marginal(blocks, prior_scale, draw_linear)


# Offset and gain priors are anchored to each light curve's own data, so they
# mean the same thing in any flux units: C ~ Normal(mean(y), 10 std(y)), and
# S ~ LogNormal(log(std(y) / sigma_drw_prior_scale), 1), i.e. centred on the
# gain that maps a driver of the anchored amplitude onto the band's own
# variability (the idea behind EchoFit._init_strategy and decision #13).
# Before September 2026 they were Normal(0, 5) and LogNormal(0, 1) in absolute
# units: on NGC 5548 the 1158 A offset sat ~9 sigma from its prior mean and the
# z-band gain ~4 sigma below its prior median, a real pull on the fit. The
# width is 10 std(y), not 5: an offset is strongly correlated with the
# driver's slowest Fourier terms, and at 5 std(y) that direction was tight
# enough to make NUTS diverge (tests/test_rw_prior.py: 1-24% divergent over
# four seeds, against 0.4-3% at 10 std(y) and 0.2-4% with the old prior).
_OFFSET_PRIOR_WIDTH = 10.0


def _offset_prior(y):
    """(loc, sd) of a light curve's offset prior: its own mean, and
    ``_OFFSET_PRIOR_WIDTH`` times its own standard deviation."""
    return jnp.mean(y), _OFFSET_PRIOR_WIDTH * jnp.std(y)


def _gain_prior_loc(y, sigma_drw_prior_scale):
    """Log-median of a light curve's gain prior: its own standard deviation
    relative to the driver-amplitude prior scale."""
    return jnp.log(jnp.std(y) / sigma_drw_prior_scale)


def _background(d):
    """(basis, prior sd) of a light curve's slow background, or (None, None)
    if it has none: the precomputed Legendre basis ``d["background_basis"]``
    (``(n, K)``, K >= 1) and ``BACKGROUND_PRIOR_WIDTH * std(y)``."""
    basis = d.get("background_basis")
    if basis is None or basis.shape[1] == 0:
        return None, None
    return basis, BACKGROUND_PRIOR_WIDTH * jnp.std(d["y"])


def _basis(freqs, t):
    wt = freqs[None, :] * t[:, None]
    return jnp.sin(wt), jnp.cos(wt)


def _linear_marginal(blocks, prior_scale, draw_linear):
    """Add the exact marginal likelihood of every light curve in ``blocks``
    (linear parameters integrated out) as a ``numpyro.factor`` and, if
    ``draw_linear``, draw the linear parameters from their conditional
    posterior -- see the module docstring's "linear-parameter
    marginalisation" note for the maths.

    Linear parameters, in the prior-whitened basis (unit-Normal prior):
    ``S_raw`` (n_freq), ``C_raw`` (n_freq), then one offset per block whose
    ``known_offset`` is ``None`` (``offset = offset_loc + offset_sd * theta``,
    so the prior-mean offset is subtracted from that block's data first),
    then each block's slow-background coefficients, if it has any
    (``bg_cols``, ``(n, K)``; ``coefficient = bg_sd * theta``).
    """
    n_freq = prior_scale.shape[0]
    offset_names = [b["name"] for b in blocks if b["known_offset"] is None]
    n_off = len(offset_names)
    bg_blocks = [b for b in blocks if b.get("bg_cols") is not None]
    bg_start, n_bg = {}, 0
    for b in bg_blocks:
        bg_start[b["name"]] = n_bg
        n_bg += b["bg_cols"].shape[1]

    rows, ys, sigmas = [], [], []
    for b in blocks:
        n = b["y"].shape[0]
        onehot = jnp.zeros((n_off,))
        if b["known_offset"] is None:
            onehot = onehot.at[offset_names.index(b["name"])].set(b["offset_sd"])
            ys.append(b["y"] - b["offset_loc"])
        else:
            ys.append(b["y"] - b["known_offset"])
        bg_part = jnp.zeros((n, n_bg))
        if b.get("bg_cols") is not None:
            k0 = bg_start[b["name"]]
            bg_part = bg_part.at[:, k0:k0 + b["bg_cols"].shape[1]].set(b["bg_cols"] * b["bg_sd"])
        rows.append(jnp.concatenate([
            b["gain"] * b["S_cols"] * prior_scale,
            b["gain"] * b["C_cols"] * prior_scale,
            jnp.ones((n, 1)) * onehot[None, :],
            bg_part,
        ], axis=1))
        sigmas.append(jnp.broadcast_to(b["sigma"], (n,)))
    y = jnp.concatenate(ys)
    sigma = jnp.concatenate(sigmas)
    Mw = jnp.concatenate(rows, axis=0) / sigma[:, None]   # D^-1/2 M, (N, p)
    yw = y / sigma                                        # D^-1/2 y
    p = Mw.shape[1]

    # QR of the stacked least-squares system [D^-1/2 M; I], not a Cholesky
    # of P = I + M^T D^-1 M: R^T R = P, but R's condition number is only the
    # square root of P's. P reached ~3e7 on a 5-band, 500-point test set, and
    # a float32 Cholesky there gave gradients ~9% off, enough to shrink
    # NUTS's adapted step size from ~0.4 to a tiny fraction of that and
    # multiply trajectory lengths ~30x. See CLAUDE.md decision #21.
    Q, R = jnp.linalg.qr(jnp.concatenate([Mw, jnp.eye(p)], axis=0))
    theta_hat = jax.scipy.linalg.solve_triangular(R, Q[: Mw.shape[0]].T @ yw, lower=False)
    resid = yw - Mw @ theta_hat
    quad = resid @ resid + theta_hat @ theta_hat
    log_det = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(R)))) + 2.0 * jnp.sum(jnp.log(sigma))
    numpyro.factor("linear_marginal_loglik", -0.5 * (quad + log_det + y.shape[0] * jnp.log(2.0 * jnp.pi)))

    if not draw_linear:
        return
    # Conditional posterior is N(theta_hat, P^-1); with P = R^T R,
    # theta_hat + R^-1 eps (eps ~ N(0, I)) has exactly that covariance.
    eps = numpyro.sample("linear_eps", dist.Normal(0.0, 1.0).expand([p]).to_event(1))
    theta = theta_hat + jax.scipy.linalg.solve_triangular(R, eps, lower=False)
    S_raw = numpyro.deterministic("S_raw", theta[:n_freq])
    C_raw = numpyro.deterministic("C_raw", theta[n_freq:2 * n_freq])
    numpyro.deterministic("S", S_raw * prior_scale)
    numpyro.deterministic("C", C_raw * prior_scale)
    blocks_by_name = {b["name"]: b for b in blocks}
    for i, name in enumerate(offset_names):
        b = blocks_by_name[name]
        numpyro.deterministic(f"C_{name}", b["offset_loc"] + theta[2 * n_freq + i] * b["offset_sd"])
    for b in bg_blocks:
        k0 = 2 * n_freq + n_off + bg_start[b["name"]]
        numpyro.deterministic(f"bg_{b['name']}", theta[k0:k0 + b["bg_cols"].shape[1]] * b["bg_sd"])
    for b, row in zip(blocks, rows):
        known = b["offset_loc"] if b["known_offset"] is None else b["known_offset"]
        numpyro.deterministic(f"y_pred_{b['name']}", row @ theta + known)
