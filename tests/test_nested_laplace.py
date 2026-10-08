"""The nested Laplace approximation (pycream2/nested_laplace.py,
EchoFit.nested_laplace): exact on a toy posterior that is bimodal in the
gridded parameter (where a single Laplace approximation cannot be), and
end-to-end on a small EchoFit problem."""

import warnings

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model

from pycream2 import nested_laplace
from pycream2.echofit import EchoFit
from pycream2.synthetic import generate_synthetic_dataset

Y_B = np.array([1.0, 1.6, 0.4, 1.3, 0.9])


def _toy(y_b):
    # a**2 is measured as 4, so a is near +2 or -2; the prior on a makes the two
    # modes unequal (~73/27). b's data depend on a; c is a bounded inner parameter.
    a = numpyro.sample("a", dist.Normal(1.0, 2.0))
    b = numpyro.sample("b", dist.Normal(0.0, 2.0))
    c = numpyro.sample("c", dist.HalfNormal(2.0))
    numpyro.sample("obs_a", dist.Normal(a ** 2, 0.6), obs=4.0)
    numpyro.sample("obs_b", dist.Normal(b + 0.15 * a, c), obs=y_b)


def _exact_marginal_of_a(a):
    """p(a | y) on the grid ``a`` by brute-force integration over b and c."""
    from scipy.stats import halfnorm, norm

    b = np.linspace(-4.0, 6.0, 250)[:, None]
    c = np.linspace(1e-3, 4.0, 250)[None, :]
    log = np.empty((len(a),) + np.broadcast_shapes(b.shape, c.shape))
    for k, ak in enumerate(a):  # one a at a time keeps memory small
        lk = (norm.logpdf(ak, 1, 2) + norm.logpdf(b, 0, 2) + halfnorm.logpdf(c, scale=2)
              + norm.logpdf(4.0, ak ** 2, 0.6))
        for y in Y_B:
            lk = lk + norm.logpdf(y, b + 0.15 * ak, c)
        log[k] = lk
    dens = np.exp(log - log.max()).sum(axis=(1, 2))
    return dens / np.trapezoid(dens, a) if hasattr(np, "trapezoid") else dens / np.trapz(dens, a)


def test_nested_laplace_recovers_a_bimodal_marginal_exactly():
    kwargs = dict(y_b=Y_B)
    info = initialize_model(jax.random.PRNGKey(0), _toy, model_kwargs=kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = nested_laplace.run(_toy, kwargs, info, num_samples=4000, rng_seed=1, grid_params=("a",))
    a_index = int(result["unravel"](np.arange(result["z"].shape[1], dtype=np.float32))["a"])
    a_draws = result["z"][:, a_index]  # a's support is the real line: unconstrained = constrained

    grid = np.linspace(-4.0, 4.0, 401)
    exact = _exact_marginal_of_a(grid)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    mean = trap(grid * exact, grid)
    sd = np.sqrt(trap((grid - mean) ** 2 * exact, grid))
    p_positive = trap(np.where(grid > 0, exact, 0.0), grid)
    assert 0.65 < p_positive < 0.85  # the modes are genuinely unequal
    # Within three Monte Carlo standard errors at the importance weights'
    # effective sample size (over six seeds the method averaged a mean of
    # 0.998 +/- 0.015 and a positive-mode weight of 0.752 +/- 0.004).
    ess = result["ess"]
    assert abs(np.mean(a_draws > 0) - p_positive) < 3 * np.sqrt(p_positive * (1 - p_positive) / ess) + 0.005
    assert abs(np.mean(a_draws) - mean) < 3 * sd / np.sqrt(ess) + 0.01
    assert abs(np.std(a_draws) - sd) < 3 * sd / np.sqrt(2 * ess) + 0.03
    assert result["k_hat"] < 0.7
    # A single Gaussian at either mode would have sd ~0.15; the bimodal one is ~1.9.
    assert np.std(a_draws) > 1.0


def test_echofit_nested_laplace_end_to_end():
    data = generate_synthetic_dataset(bands={"g": 4770.0, "i": 7625.0}, M_BH=1.0e8, n_obs_per_band=30,
                                      n_freq=10, n_tau=60, noise_level=0.03, seed=3)
    ef = EchoFit(M_BH=1.0e8)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=10, n_tau=60, period_max=2 * np.pi / float(data["freqs"].min()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ef.nested_laplace(num_samples=200, n_pass=(6, 5), n_fine=(10, 8))
    r = ef.nested_laplace_result
    assert set(r["names"]) == {"log_mdot", "cos_inclination"}
    for key in ("log_mdot", "inclination", "sigma_drw", "S", "C", "y_pred_g"):
        assert key in ef.samples and len(ef.samples[key]) == 200
    assert np.isfinite(ef.log_evidence) and np.isfinite(r["k_hat"])
    assert abs(np.mean(ef.samples["log_mdot"]) - data["truth"]["log_mdot"]) < 4 * np.std(ef.samples["log_mdot"]) + 0.1
    assert r["timings"]["final_points"] == 80


def _narrow_and_broad():
    # A narrow mode (sd 0.01) that falls between the scout grid's points, and a
    # broad one (sd 1) that a coarse grid sees easily; b is an unrelated inner
    # parameter. The regression: a search that trusted the coarse grid zoomed
    # onto the broad mode and lost the narrow one.
    import jax.numpy as jnp

    a = numpyro.sample("a", dist.Normal(0.0, 5.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    numpyro.factor("modes", jnp.logaddexp(jnp.log(0.6) + dist.Normal(0.37, 0.01).log_prob(a),
                                          jnp.log(0.4) + dist.Normal(5.0, 1.0).log_prob(a)))
    numpyro.sample("obs_b", dist.Normal(b, 0.5), obs=np.array([0.3, -0.2]))


def test_nested_laplace_keeps_a_narrow_mode_beside_a_broad_one():
    from numpyro.infer import init_to_value
    from scipy.stats import norm

    info = initialize_model(jax.random.PRNGKey(0), _narrow_and_broad, model_kwargs={},
                            init_strategy=init_to_value(values={"a": 0.36, "b": 0.0}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = nested_laplace.run(_narrow_and_broad, {}, info, num_samples=4000, rng_seed=2, grid_params=("a",))
    a_index = int(result["unravel"](np.arange(result["z"].shape[1], dtype=np.float32))["a"])
    a_draws = result["z"][:, a_index]
    # Exact mass of the narrow mode: prior times each component, integrated.
    narrow = 0.6 * norm.pdf(0.37, 0, 5)
    grid = np.linspace(-2.0, 12.0, 20001)
    broad = 0.4 * np.trapezoid(norm.pdf(grid, 5, 1) * norm.pdf(grid, 0, 5), grid) if hasattr(np, "trapezoid") else \
        0.4 * np.trapz(norm.pdf(grid, 5, 1) * norm.pdf(grid, 0, 5), grid)
    p_narrow = narrow / (narrow + broad)
    assert abs(np.mean(np.abs(a_draws - 0.37) < 0.1) - p_narrow) < 0.04
    assert result["k_hat"] < 0.7


def test_plot_landscape_and_report_section(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from pycream2 import reporting

    data = generate_synthetic_dataset(bands={"g": 4770.0, "i": 7625.0}, M_BH=1.0e8, n_obs_per_band=30,
                                      n_freq=10, n_tau=60, noise_level=0.03, seed=3)
    ef = EchoFit(M_BH=1.0e8)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=10, n_tau=60, period_max=2 * np.pi / float(data["freqs"].min()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ef.nested_laplace(num_samples=200, n_pass=(6, 5), n_fine=(10, 8))
        fig, axes = ef.plot_landscape(truth=data["truth"])
        assert len(axes) == 2
        html = reporting.generate_report(ef, tmp_path)
    assert "Badness-of-Fit landscape" in open(html).read()
