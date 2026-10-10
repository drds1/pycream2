"""``reverberation_model(marginalise_linear=True)`` integrates the driver's
Fourier coefficients and every band's/the driver's offset out analytically.
It must give exactly the same marginal likelihood as brute-force
integration of the default (sampled) model, and the post-fit conditional
draws must reproduce the default model's posterior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from scipy.stats import multivariate_normal

from pycream2.echofit import EchoFit, _posterior_scale_curvature
import pycream2.model as model_module
from pycream2.model import reverberation_model
from pycream2.synthetic import generate_synthetic_dataset


def _small_echofit(with_driver=True, fit_error_model=False, fixed_params=None, **kwargs):
    rng = np.random.default_rng(0)
    ef = EchoFit(M_BH=1.0e8, fixed_params=fixed_params, **kwargs)
    for name, wav in {"g": 4770.0, "i": 7625.0}.items():
        t = np.sort(rng.uniform(0.0, 60.0, 15))
        ef.add_lightcurve(
            name, wavelength=wav, t=t, y=rng.normal(1.0, 0.5, 15), yerr=np.full(15, 0.1),
            fit_error_model=fit_error_model, background_order=0,  # backgrounds: test_extra_components.py
        )
    if with_driver:
        ef.add_driver_lightcurve(t=np.sort(rng.uniform(0.0, 60.0, 12)), y=rng.normal(size=12), yerr=np.full(12, 0.2),
                                 background_order=0)
    ef.build_grid(n_freq=10, n_tau=80)
    return ef


_NONLINEAR = dict(
    sigma_drw=jnp.asarray(0.7), log_mdot=jnp.asarray(0.2), cos_inclination=jnp.asarray(0.6),
    S_g=jnp.asarray(0.8), S_i=jnp.asarray(1.3), S_driver=jnp.asarray(1.1),
    sigma_scale_g=jnp.asarray(1.2), sigma_jitter_g=jnp.asarray(0.05),
    sigma_scale_i=jnp.asarray(0.9), sigma_jitter_i=jnp.asarray(0.02),
)


def _marginal_factor(kwargs):
    tr = handlers.trace(handlers.substitute(handlers.seed(reverberation_model, 0), _NONLINEAR)).get_trace(
        **kwargs, marginalise_linear=True
    )
    return float(tr["linear_marginal_loglik"]["fn"].log_prob(tr["linear_marginal_loglik"]["value"]))


def _brute_force_marginal(kwargs, fixed_params):
    """Dense float64 Gaussian marginal, with the design matrix taken from
    ``jax.jacobian`` of the *default* model's predictions w.r.t. its linear
    sample sites -- an independent route to the same quantity."""
    names = [n for n in list(kwargs["bands"]) + (["driver"] if kwargs["driver"] else []) if f"C_{n}" not in fixed_params]
    n_freq = kwargs["freqs"].shape[0]

    def predictions(S_raw, C_raw, offsets):
        values = dict(_NONLINEAR, S_raw=S_raw, C_raw=C_raw, **{f"C_{n}": o for n, o in zip(names, offsets)})
        tr = handlers.trace(handlers.substitute(handlers.seed(reverberation_model, 0), values)).get_trace(**kwargs)
        preds = [tr[f"y_pred_{n}"]["value"] for n in kwargs["bands"]]
        sigmas = [jnp.broadcast_to(tr[f"obs_{n}"]["fn"].scale, tr[f"obs_{n}"]["value"].shape) for n in kwargs["bands"]]
        if kwargs["driver"]:
            preds.append(tr["y_pred_driver"]["value"])
            sigmas.append(jnp.broadcast_to(tr["obs_driver"]["fn"].scale, tr["obs_driver"]["value"].shape))
        return jnp.concatenate(preds), jnp.concatenate(sigmas)

    # Offset priors are data-anchored: Normal(mean(y), (width * std(y))**2) per light curve.
    ys = {n: np.asarray(kwargs["bands"][n]["y"] if n in kwargs["bands"] else kwargs["driver"]["y"]) for n in names}
    locs = jnp.asarray([ys[n].mean() for n in names])
    prior_mean = (jnp.zeros(n_freq), jnp.zeros(n_freq), locs)
    mean, sigma = predictions(*prior_mean)
    J = jax.jacobian(lambda *a: predictions(*a)[0], argnums=(0, 1, 2))(*prior_mean)
    J = np.concatenate([np.asarray(j, dtype=np.float64) for j in J], axis=1)
    prior_var = np.concatenate([np.ones(2 * n_freq), [(model_module._OFFSET_PRIOR_WIDTH * ys[n].std()) ** 2 for n in names]])
    cov = np.diag(np.asarray(sigma, dtype=np.float64) ** 2) + (J * prior_var) @ J.T
    y = np.concatenate([np.asarray(d["y"]) for d in kwargs["bands"].values()]
                       + ([np.asarray(kwargs["driver"]["y"])] if kwargs["driver"] else []))
    return multivariate_normal(mean=np.asarray(mean, dtype=np.float64), cov=cov).logpdf(y)


@pytest.mark.parametrize("with_driver, fit_error_model, fixed_params", [
    (True, False, None),
    (False, True, None),
    (True, False, {"C_g": 0.9, "C_driver": -0.1}),
])
def test_marginal_likelihood_matches_brute_force_integration(with_driver, fit_error_model, fixed_params):
    ef = _small_echofit(with_driver, fit_error_model, fixed_params)
    kwargs = ef._model_kwargs()
    kwargs.pop("marginalise_linear", None)
    expected = _brute_force_marginal(kwargs, fixed_params or {})
    assert _marginal_factor(kwargs) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("fit_error_model", [False, True])
def test_data_space_marginal_matches_parameter_space_and_brute_force(fit_error_model, monkeypatch):
    """Fewer data points (42) than linear parameters (~63): the marginal
    likelihood is factorised in data space; it must agree with the
    parameter-space factorisation and the dense float64 integral, in value
    and gradient."""
    ef = _small_echofit(True, fit_error_model)
    ef.build_grid(n_freq=30, n_tau=80)
    kwargs = ef._model_kwargs()
    kwargs.pop("marginalise_linear", None)
    n_obs = sum(len(d["t"]) for d in ef.bands.values()) + len(ef.driver_data["t"])
    assert n_obs < 2 * len(ef.freqs)
    expected = _brute_force_marginal(kwargs, {})
    data_space = _marginal_factor(kwargs)
    monkeypatch.setattr(model_module, "_FORCE_PARAMETER_SPACE", True)
    jax.clear_caches()
    parameter_space = _marginal_factor(kwargs)
    assert data_space == pytest.approx(expected, rel=1e-4)
    assert data_space == pytest.approx(parameter_space, rel=1e-5)

    def gradient(force):
        monkeypatch.setattr(model_module, "_FORCE_PARAMETER_SPACE", force)
        jax.clear_caches()

        def loglik(log_mdot, sigma_drw):
            values = dict(_NONLINEAR, log_mdot=log_mdot, sigma_drw=sigma_drw)
            tr = handlers.trace(handlers.substitute(handlers.seed(reverberation_model, 0), values)).get_trace(
                **kwargs, marginalise_linear=True)
            return tr["linear_marginal_loglik"]["fn"].log_prob(tr["linear_marginal_loglik"]["value"])

        return np.asarray(jax.grad(loglik, argnums=(0, 1))(jnp.asarray(0.2), jnp.asarray(0.7)))

    np.testing.assert_allclose(gradient(False), gradient(True), rtol=1e-3)


def test_marginalised_model_has_no_linear_sample_sites():
    ef = _small_echofit(marginalise_linear=True)
    tr = handlers.trace(handlers.seed(reverberation_model, 0)).get_trace(**ef._model_kwargs())
    sampled = {k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed", False)}
    assert not sampled & {"S_raw", "C_raw", "C_g", "C_i", "C_driver"}
    assert {"S_g", "S_i", "S_driver", "log_mdot", "sigma_drw"} <= sampled


@pytest.mark.slow
def test_marginalised_fit_matches_sampled_fit():
    """Same data, both ways: the nonlinear posterior and the posterior-mean
    driver/offsets should agree, and the conditional draws must give the
    plots everything they read (S, C, C_band)."""
    data = generate_synthetic_dataset(
        bands={"g": 4770.0, "r": 6200.0, "i": 7625.0}, M_BH=1.0e8, n_obs_per_band=60,
        n_freq=30, n_tau=200, noise_level=0.03, seed=1,
    )
    fits = {}
    for marginalise in (False, True):
        ef = EchoFit(M_BH=1.0e8, marginalise_linear=marginalise)
        for name, d in data["bands"].items():
            ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
        ef.build_grid(n_freq=30, n_tau=200, period_max=2 * np.pi / float(data["freqs"].min()))
        ef.fit(num_warmup=500, num_samples=500, progress_bar=False, rng_seed=0)
        fits[marginalise] = ef

    sampled, marg = fits[False].samples, fits[True].samples
    for key in ("S", "C", "C_g", "C_r", "C_i", "y_pred_g"):
        assert key in marg
    assert marg["S"].shape == sampled["S"].shape
    for key in ("log_mdot", "S_g", "C_g"):
        a, b = np.asarray(sampled[key]), np.asarray(marg[key])
        assert abs(a.mean() - b.mean()) < 0.5 * max(a.std(), b.std()) + 1e-3, key
    t = np.linspace(0.0, 100.0, 50)
    for ef in fits.values():
        ef._driver_mean = np.mean(np.sin(np.outer(t, ef.freqs)) @ np.asarray(ef.samples["S"]).T
                                  + np.cos(np.outer(t, ef.freqs)) @ np.asarray(ef.samples["C"]).T, axis=1)
    d0, d1 = fits[False]._driver_mean, fits[True]._driver_mean
    assert np.corrcoef(d0, d1)[0, 1] > 0.95
    plt_fig = fits[True].plot_lightcurve_fits()
    assert plt_fig is not None


def test_marginalise_linear_persists_across_resume_and_fills_linear_sites(tmp_path):
    """marginalise_linear changes which sites exist, so (like fixed_params/
    fit_error_model/drw_prior before it) it must survive the checkpointed
    save/resume round trip, and the post-fit conditional draws must be
    present on both the original and the resumed fit."""
    ef = _small_echofit(title="marginal_resume_test", output_dir=str(tmp_path), marginalise_linear=True)
    ef.fit(rng_seed=0, num_warmup=5, num_samples=10, checkpoint_every=5, progress_bar=False, generate_report=False)
    for key in ("S", "C", "S_raw", "C_g", "C_driver", "y_pred_g"):
        assert ef.samples[key].shape[0] == 10, key
    assert ef._samples_by_chain["S"].shape[:2] == (1, 10)

    ef2 = EchoFit.resume("marginal_resume_test", output_dir=str(tmp_path))
    assert ef2.marginalise_linear is True
    ef2.fit(num_samples=15, progress_bar=False, generate_report=False)
    assert ef2.samples["S"].shape[0] == 15


def test_in_memory_marginalised_fit_fills_linear_sites_per_chain():
    ef = _small_echofit(marginalise_linear=True)
    ef.fit(rng_seed=0, num_warmup=5, num_samples=6, num_chains=2, chain_method="sequential", progress_bar=False)
    assert ef._samples_by_chain["S"].shape[:2] == (2, 6)
    assert ef.samples["S"].shape[0] == 12
    assert ef.samples["C_g"].shape == (12,)


def test_laplace_curvature_ignores_a_tiny_ripple_in_the_potential():
    # U = x**2/2 + y**2 + 1e-4 sin(200 x): at x0 = pi/400 the ripple's
    # curvature (-4) dominates the Hessian's x entry (1 - 4 = -3), yet over the
    # posterior's own width (one sd, ~1) U is the unit-curvature quadratic. The
    # stiff, clean y direction (curvature 2) must be left alone.
    def u(p):
        p = np.atleast_2d(p)
        return 0.5 * p[:, 0] ** 2 + p[:, 1] ** 2 + 1e-4 * np.sin(200 * p[:, 0])

    z = np.array([np.pi / 400, 0.0])
    vals, vecs = np.array([-3.0, 2.0]), np.eye(2)
    corrected, n = _posterior_scale_curvature(u, z, float(u(z)[0]), vals, vecs)
    assert n == 1
    np.testing.assert_allclose(corrected, [1.0, 2.0], atol=5e-3)
    # A spuriously *huge* pointwise curvature (a sharp feature) must not shrink
    # the probe inside the feature and confirm itself.
    corrected, n = _posterior_scale_curvature(u, z, float(u(z)[0]), np.array([1e6, 2.0]), vecs)
    assert n == 1
    np.testing.assert_allclose(corrected, [1.0, 2.0], atol=5e-3)


def test_laplace_curvature_keeps_a_genuinely_stiff_direction():
    # Curvature 1e4 (sd 0.01, well below the 0.05 minimum starting step): the
    # probe starts wide, converges to the true width and leaves it unchanged.
    def u(p):
        p = np.atleast_2d(p)
        return 0.5e4 * p[:, 0] ** 2 + 0.5 * p[:, 1] ** 2

    z = np.zeros(2)
    corrected, n = _posterior_scale_curvature(u, z, 0.0, np.array([1e4, 1.0]), np.eye(2))
    assert n == 0
    np.testing.assert_allclose(corrected, [1e4, 1.0])


def test_laplace_curvature_keeps_a_merely_anharmonic_direction():
    # U = x**2/2 + x**4/5: curvature 1 at the peak, about 1.4 over one sd (the
    # central difference at h ~ 1 picks up the quartic). That is anharmonicity,
    # not an artefact, and the pointwise value is kept.
    def u(p):
        p = np.atleast_2d(p)
        return 0.5 * p[:, 0] ** 2 + 0.2 * p[:, 0] ** 4 + 0.5 * p[:, 1] ** 2

    corrected, n = _posterior_scale_curvature(u, np.zeros(2), 0.0, np.array([1.0, 1.0]), np.eye(2))
    assert n == 0
    np.testing.assert_allclose(corrected, [1.0, 1.0])


def test_optimise_restarts_are_polished_and_compared():
    """optimise()'s multi-start bookkeeping: every restart is polished to its
    own optimum and measured against the best. The "data" here are pure noise,
    whose posterior is genuinely multimodal (two of these restarts share a
    second mode 15 units above the best), so agreement is tested on real
    synthetic light curves in test_optimise_laplace_matches_nuts instead."""
    ef = _small_echofit()
    with pytest.warns(UserWarning, match="restarts reached the same optimum"):
        ef.optimise(num_samples=50, num_restarts=3, restart_scale=1.0, method="laplace")
    restarts = ef.optimise_restarts
    assert len(restarts) == 3
    # The kept restart (highest Laplace evidence, not necessarily the lowest
    # potential) is the reference everything else is measured from.
    kept = [r for r in restarts if r["delta_potential"] == 0.0 and r["max_offset_in_sd"] == 0.0]
    assert len(kept) >= 1
    assert restarts[0]["start_offset_in_sd"] == 0.0
    assert all(r["start_offset_in_sd"] > 0.0 for r in restarts[1:])
    assert ef.optimise_timings["restarts_agreeing"] == sum(r["agrees"] for r in restarts) < 3
    assert all(r["agrees"] for r in restarts if r["delta_potential"] == 0.0)
    assert {"log_mdot", "inclination"} <= set(restarts[0]["values"])
    assert ef.plot_optimise_restarts() is not None


def test_optimise_fills_samples_and_can_seed_nuts(tmp_path):
    """EchoFit.optimise(): the direct (no-MCMC) solve fills everything the
    plots read, and fit(init_from_optimum=True) can start NUTS at its peak,
    for both the sampled and the marginalised model."""
    ef = _small_echofit()
    ef.optimise(num_samples=50, num_restarts=2, method="laplace")
    assert ef.optimise_timings["newton_offset_in_sd"] < 0.1
    for key in ("log_mdot", "inclination", "S", "C", "C_g", "C_driver", "y_pred_g"):
        assert ef.samples[key].shape[0] == 50, key
    assert np.all(np.linalg.eigvalsh(ef.laplace_covariance) > 0)
    assert ef.extra_fields == {}
    assert ef.plot_lightcurve_fits() is not None
    assert ef.optimise_timings["newton_offset_in_sd"] < 0.01

    from pycream2 import reporting
    html = reporting.generate_report(ef, tmp_path / "report").read_text()
    assert "direct solve" in html and "divergent transitions" not in html
    assert "Direct-solve reproducibility" in html

    ef.fit(num_warmup=5, num_samples=5, progress_bar=False, init_from_optimum=True)
    assert "S_raw" in ef.samples

    with pytest.raises(ValueError, match="single-chain"):
        ef.fit(num_warmup=2, num_samples=2, num_chains=2, progress_bar=False, init_from_optimum=True)
    with pytest.raises(ValueError, match="optimise"):
        _small_echofit().fit(num_warmup=2, num_samples=2, progress_bar=False, init_from_optimum=True)


@pytest.mark.slow
def test_optimise_laplace_matches_nuts():
    """The direct solve's Laplace posterior agrees with a full NUTS run on
    the well-constrained physical parameters."""
    data = generate_synthetic_dataset(
        bands={"g": 4770.0, "r": 6200.0, "i": 7625.0}, M_BH=1.0e8, n_obs_per_band=60,
        n_freq=30, n_tau=200, noise_level=0.03, seed=1,
    )
    fits = []
    for direct in (True, False):
        ef = EchoFit(M_BH=1.0e8)
        for name, d in data["bands"].items():
            ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
        ef.build_grid(n_freq=30, n_tau=200)
        if direct:
            ef.optimise(num_restarts=4, restart_scale=1.0, method="laplace")
            # Real light curves: every restart, from widely spread starts,
            # must reach the same optimum.
            assert ef.optimise_timings["restarts_agreeing"] == 4
        else:
            ef.fit(num_warmup=500, num_samples=500, progress_bar=False)
        fits.append(np.asarray(ef.samples["log_mdot"]))
    laplace, nuts = fits
    assert abs(laplace.mean() - nuts.mean()) < 0.5 * nuts.std()
    assert 0.6 < laplace.std() / nuts.std() < 1.6
