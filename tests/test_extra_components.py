"""
Tests for the two optional extra components (both off by default):

- a diffuse-continuum reprocessor per band, a log-normal delay distribution
  mixed into the band's response (``add_lightcurve(..., diffuse_continuum=True)``);
- a slowly varying Legendre background per light curve
  (``add_lightcurve(..., background_order=K)``, ``add_driver_lightcurve(...,
  background_order=K)``).

See docs/extra_components.md.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import legendre
from numpyro import handlers
from scipy.stats import multivariate_normal

import pycream2.model as model_module
from pycream2 import EchoFit
from pycream2.forward_model import legendre_background_basis, lognormal_response, mix_diffuse_continuum
from pycream2.model import reverberation_model
from pycream2.synthetic import generate_synthetic_dataset

_np_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def test_lognormal_response_is_causal_normalised_and_has_the_right_median():
    tau = jnp.linspace(-2.0, 80.0, 8001)
    tau_np = np.asarray(tau)
    psi = np.asarray(lognormal_response(tau, 6.0, 0.2))
    assert np.all(psi[tau_np <= 0.0] == 0.0)
    assert _np_trapz(psi, tau_np) == pytest.approx(1.0, abs=1e-4)
    cdf = np.cumsum(psi) * (tau_np[1] - tau_np[0])
    assert tau_np[np.searchsorted(cdf, 0.5)] == pytest.approx(6.0, rel=0.01)
    # The log10 tau spread is the requested width in dex.
    pos = tau_np > 0
    w = psi[pos] / psi[pos].sum()
    lg = np.log10(tau_np[pos])
    assert np.sqrt(np.sum(w * (lg - np.sum(w * lg)) ** 2)) == pytest.approx(0.2, rel=0.02)
    # Differentiable in both parameters (CLAUDE.md decision #7's gradient trap).
    g = jax.grad(lambda m, s: jnp.sum(lognormal_response(tau, m, s) * tau), argnums=(0, 1))(6.0, 0.2)
    assert all(np.isfinite(float(x)) and float(x) != 0.0 for x in g)


def test_mix_diffuse_continuum_keeps_the_area_and_weights_the_parts():
    tau = jnp.linspace(0.0, 60.0, 4001)
    disc = lognormal_response(tau, 1.0, 0.3)
    mixed = np.asarray(mix_diffuse_continuum(disc, tau, 0.25, 10.0, 0.2))
    assert _np_trapz(mixed, np.asarray(tau)) == pytest.approx(1.0, abs=1e-4)
    mean = lambda p: _np_trapz(np.asarray(p) * np.asarray(tau), np.asarray(tau))  # noqa: E731
    expected = 0.75 * mean(disc) + 0.25 * mean(lognormal_response(tau, 10.0, 0.2))
    assert mean(mixed) == pytest.approx(expected, rel=1e-4)


def test_legendre_background_basis_matches_numpy():
    t = np.linspace(10.0, 110.0, 37)
    basis = np.asarray(legendre_background_basis(t, (10.0, 110.0), 4))
    x = 2.0 * (t - 10.0) / 100.0 - 1.0
    for k in range(1, 5):
        np.testing.assert_allclose(basis[:, k - 1], legendre.legval(x, [0] * k + [1]), atol=1e-6)
    assert np.asarray(legendre_background_basis(t, (10.0, 110.0), 0)).shape == (37, 0)


def _small_echofit(diffuse_continuum=False, background_order=0, driver_background=0, with_driver=True, **kwargs):
    rng = np.random.default_rng(0)
    ef = EchoFit(M_BH=1.0e8, **kwargs)
    for name, wav in {"g": 4770.0, "i": 7625.0}.items():
        t = np.sort(rng.uniform(0.0, 60.0, 15))
        ef.add_lightcurve(
            name, wavelength=wav, t=t, y=rng.normal(1.0, 0.5, 15), yerr=np.full(15, 0.1),
            diffuse_continuum=diffuse_continuum and name == "i", background_order=background_order if name == "g" else 0,
        )
    if with_driver:
        ef.add_driver_lightcurve(t=np.sort(rng.uniform(0.0, 60.0, 12)), y=rng.normal(size=12), yerr=np.full(12, 0.2),
                                 background_order=driver_background)
    ef.build_grid(n_freq=10, n_tau=80)
    return ef


def _sites(tr):
    return {k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed", False)}


def test_extra_components_are_off_by_default():
    ef = _small_echofit()
    tr = handlers.trace(handlers.seed(reverberation_model, 0)).get_trace(**ef._model_kwargs())
    assert not {k for k in tr if k.startswith(("dce_", "bg_"))}


@pytest.mark.parametrize("marginalise", [False, True])
def test_extra_component_sites_appear_when_switched_on(marginalise):
    ef = _small_echofit(diffuse_continuum=True, background_order=2, driver_background=1, marginalise_linear=marginalise)
    tr = handlers.trace(handlers.seed(reverberation_model, 0)).get_trace(**ef._model_kwargs())
    sampled = _sites(tr)
    assert {"dce_fraction_i", "dce_delay_i", "dce_width_i"} <= sampled
    assert not {"dce_fraction_g", "bg_i"} & set(tr)
    if marginalise:
        # Linear, so integrated out like the offsets.
        assert not {"bg_g", "bg_driver"} & sampled
    else:
        assert tr["bg_g"]["value"].shape == (2,) and tr["bg_driver"]["value"].shape == (1,)


_NONLINEAR = dict(
    sigma_drw=jnp.asarray(0.7), log_mdot=jnp.asarray(0.2), cos_inclination=jnp.asarray(0.6),
    S_g=jnp.asarray(0.8), S_i=jnp.asarray(1.3), S_driver=jnp.asarray(1.1),
    dce_fraction_i=jnp.asarray(0.3), dce_delay_i=jnp.asarray(5.0), dce_width_i=jnp.asarray(0.3),
)


def test_marginal_likelihood_with_extra_components_matches_brute_force_integration():
    """The exact marginal (background coefficients integrated out with the
    Fourier terms and offsets) against dense float64 integration, with the
    design matrix from jax.jacobian of the *sampled* model's predictions."""
    ef = _small_echofit(diffuse_continuum=True, background_order=2, driver_background=1)
    kwargs = ef._model_kwargs()
    kwargs.pop("marginalise_linear", None)
    tr = handlers.trace(handlers.substitute(handlers.seed(reverberation_model, 0), _NONLINEAR)).get_trace(
        **kwargs, marginalise_linear=True)
    marginal = float(tr["linear_marginal_loglik"]["fn"].log_prob(tr["linear_marginal_loglik"]["value"]))

    names = ["g", "i", "driver"]
    n_freq = kwargs["freqs"].shape[0]
    ys = {n: np.asarray(kwargs["bands"][n]["y"] if n != "driver" else kwargs["driver"]["y"]) for n in names}

    def predictions(S_raw, C_raw, offsets, bg_g, bg_driver):
        values = dict(_NONLINEAR, S_raw=S_raw, C_raw=C_raw, bg_g=bg_g, bg_driver=bg_driver,
                      **{f"C_{n}": o for n, o in zip(names, offsets)})
        t = handlers.trace(handlers.substitute(handlers.seed(reverberation_model, 0), values)).get_trace(**kwargs)
        preds = jnp.concatenate([t[f"y_pred_{n}"]["value"] for n in names])
        sigmas = jnp.concatenate([jnp.broadcast_to(t[f"obs_{n}"]["fn"].scale, t[f"obs_{n}"]["value"].shape)
                                  for n in names])
        return preds, sigmas

    prior_mean = (jnp.zeros(n_freq), jnp.zeros(n_freq), jnp.asarray([ys[n].mean() for n in names]),
                  jnp.zeros(2), jnp.zeros(1))
    mean, sigma = predictions(*prior_mean)
    J = jax.jacobian(lambda *a: predictions(*a)[0], argnums=(0, 1, 2, 3, 4))(*prior_mean)
    J = np.concatenate([np.asarray(j, dtype=np.float64) for j in J], axis=1)
    w_bg = model_module.BACKGROUND_PRIOR_WIDTH
    prior_var = np.concatenate([
        np.ones(2 * n_freq),
        [(model_module._OFFSET_PRIOR_WIDTH * ys[n].std()) ** 2 for n in names],
        [(w_bg * ys["g"].std()) ** 2] * 2, [(w_bg * ys["driver"].std()) ** 2],
    ])
    cov = np.diag(np.asarray(sigma, dtype=np.float64) ** 2) + (J * prior_var) @ J.T
    y = np.concatenate([ys[n] for n in names])
    expected = multivariate_normal(mean=np.asarray(mean, dtype=np.float64), cov=cov).logpdf(y)
    assert marginal == pytest.approx(expected, rel=1e-4)


def test_fixed_params_accepts_the_diffuse_continuum_sites_only_when_switched_on():
    ef = _small_echofit(diffuse_continuum=True)
    assert {"dce_fraction_i", "dce_delay_i", "dce_width_i"} <= ef._valid_fixed_param_names()
    assert "dce_fraction_g" not in ef._valid_fixed_param_names()
    assert "dce_fraction_i" not in _small_echofit()._valid_fixed_param_names()


def test_background_order_must_not_be_negative():
    with pytest.raises(ValueError, match="background_order"):
        EchoFit(M_BH=1e8).add_lightcurve("g", 4770.0, [0.0, 1.0], [1.0, 2.0], [0.1, 0.1], background_order=-1)


def test_extra_components_persist_across_resume(tmp_path):
    ef = _small_echofit(diffuse_continuum=True, background_order=2, driver_background=1,
                        title="extras_resume", output_dir=str(tmp_path))
    ef.fit(num_warmup=5, num_samples=4, checkpoint_every=2, progress_bar=False, generate_report=False)
    resumed = EchoFit.resume("extras_resume", output_dir=str(tmp_path))
    assert resumed.bands["i"]["diffuse_continuum"] is True
    assert resumed.bands["g"]["background_order"] == 2
    assert resumed.driver_data["background_order"] == 1
    resumed.fit(num_samples=6, progress_bar=False, generate_report=False)
    assert resumed.samples["bg_g"].shape == (6, 2)


@pytest.mark.slow
def test_optimise_recovers_injected_diffuse_continuum_and_backgrounds():
    """End to end: synthetic light curves with a diffuse-continuum component in
    two bands and slow trends in two others, fitted with optimise(). Also the
    case that exposed optimise()'s NaN restart: restart 1's first L-BFGS step
    overflowed (sigma_drw = inf) until non-finite trial points were mapped to a
    large finite potential."""
    truth_dce = {"r": (0.3, 8.0, 0.25), "i": (0.4, 8.0, 0.25)}
    truth_bg = {"u": [0.4, -0.2], "z": [-0.3, 0.15]}
    data = generate_synthetic_dataset(M_BH=1e8, log_mdot_true=0.3, n_obs_per_band=120, t_span=250,
                                      noise_level=0.03, seed=3, diffuse_continuum=truth_dce, background=truth_bg)
    ef = EchoFit(M_BH=1e8)
    for n, d in data["bands"].items():
        ef.add_lightcurve(n, d["wavelength"], d["t"], d["y"], d["yerr"],
                          diffuse_continuum=n in truth_dce, background_order=2 if n in truth_bg else 0)
    ef.build_grid(tau_max=60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ef.optimise(num_samples=300, num_restarts=2)
    assert ef.optimise_timings["restarts_agreeing"] == 2
    # The Laplace evidence prefers the components that are really there.
    with_components = ef.log_evidence
    plain = EchoFit(M_BH=1e8)
    for n, d in data["bands"].items():
        plain.add_lightcurve(n, d["wavelength"], d["t"], d["y"], d["yerr"])
    plain.build_grid(tau_max=60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain.optimise(num_samples=50, num_restarts=1)
    assert with_components > plain.log_evidence + 10
    s = ef.samples
    assert abs(np.mean(s["log_mdot"]) - 0.3) < 0.1
    for n, (frac, delay, width) in truth_dce.items():
        assert abs(np.mean(s[f"dce_fraction_{n}"]) - frac) < 0.1
        assert abs(np.mean(s[f"dce_delay_{n}"]) - delay) < 2.0
        assert abs(np.mean(s[f"dce_width_{n}"]) - width) < 0.1
    for n, coeffs in truth_bg.items():
        np.testing.assert_allclose(np.mean(s[f"bg_{n}"], axis=0), coeffs, atol=0.05)
    # The plot draws each background band's offset + background curve (not a
    # flat offset line), labelled, in its light-curve panel.
    fig, _ = ef.plot_lightcurve_fits()
    labels = [line.get_label() for ax in fig.axes for line in ax.get_lines()]
    assert labels.count("offset + background") == len(truth_bg)
    import matplotlib.pyplot as plt

    plt.close(fig)
