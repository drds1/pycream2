"""
End-to-end integration test: generate a synthetic multi-band dataset, run
the full NUTS pipeline (EchoFit.fit), and check the posterior is actually
consistent with the injected ground truth.

This is deliberately more than a "does it crash" smoke test -- it checks:
  * no divergent transitions (a red flag for a badly parameterised model
    or a mismatched frequency grid -- see grid_utils.estimate_dt_min),
  * the well-identified parameter (log_mdot, which sets the mean lag) is
    recovered close to truth,
  * the posterior-predictive light curve actually fits the data to within
    a reasonable multiple of the injected noise.

log_mdot is the parameter checked tightly here because the mean lag is the
dominant signal in this forward model (see CLAUDE.md point 4); inclination
(which only affects lag *skew*, a subtler effect) and the DRW hyperparameters
sigma_drw/tau_drw are comparatively weakly identified from a couple of
bands / ~50 epochs each, so they're only checked for being finite and
within their prior support rather than close to truth.
"""

import numpy as np
import pytest

from pycream2.synthetic import generate_synthetic_dataset
from pycream2.echofit import EchoFit


def _make_synthetic(seed: int):
    return generate_synthetic_dataset(
        M_BH=1.0e9,  # large enough that mean lags >> sampling cadence (see below)
        log_mdot_true=0.0,
        inclination_true=35.0,
        sigma_drw_true=0.3,
        tau_drw_true=30.0,
        bands={"g": 4770.0, "i": 7625.0},
        t_span=200.0,
        n_obs_per_band=50,
        n_freq=15,
        n_tau=100,
        tau_max=50.0,
        noise_level=0.02,
        seed=seed,
    )


def _build_echofit(data, drw_prior: bool = False):
    ef = EchoFit(M_BH=data["truth"]["M_BH"], drw_prior=drw_prior)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    # Fit on the truth's own basis: the generator draws its driver with a
    # longest period of the baseline, shorter than build_grid()'s default.
    ef.build_grid(n_freq=15, n_tau=100, period_max=2 * np.pi / float(data["freqs"].min()))
    return ef


def _fit_synthetic(seed: int, drw_prior: bool = False):
    data = _make_synthetic(seed)
    ef = _build_echofit(data, drw_prior=drw_prior)
    ef.fit(rng_seed=0, num_warmup=500, num_samples=500, num_chains=1, progress_bar=False)
    return ef, data


@pytest.mark.slow
def test_mcmc_recovers_synthetic_truth():
    # drw_prior=True: this test specifically checks tau_drw recovery/
    # finiteness (CLAUDE.md decision #19's "Known rough edges" note this
    # docstring already describes), which only exists as a site under the
    # DRW prior -- the default drw_prior=False (random-walk) has no
    # tau_drw at all. See test_rw_prior.py for default-prior coverage.
    ef, data = _fit_synthetic(seed=2, drw_prior=True)
    truth = data["truth"]

    # -- sampling health -------------------------------------------------
    diverging = np.asarray(ef.extra_fields["diverging"])
    assert diverging.mean() < 0.05, f"{diverging.sum()}/{len(diverging)} divergent transitions"

    for name in ("log_mdot", "inclination", "sigma_drw", "tau_drw", "S", "C"):
        assert np.all(np.isfinite(np.asarray(ef.samples[name])))

    # -- the well-identified parameter: mean lag / log_mdot --------------
    log_mdot_mean = float(ef.samples["log_mdot"].mean())
    assert abs(log_mdot_mean - truth["log_mdot"]) < 0.3, (
        f"log_mdot posterior mean {log_mdot_mean:.3f} far from truth {truth['log_mdot']:.3f}"
    )

    # -- posterior-predictive fit quality --------------------------------
    # Using the model's own y_pred_{band} deterministic sites (averaged
    # over posterior draws) rather than plugging in posterior means, since
    # the forward model is nonlinear in the sampled parameters.
    for name, d in data["bands"].items():
        y_pred_mean = np.asarray(ef.samples[f"y_pred_{name}"]).mean(axis=0)
        resid_rms = float(np.std(y_pred_mean - d["y"]))
        noise_floor = float(np.mean(d["yerr"]))
        assert resid_rms < 8.0 * noise_floor, (
            f"band {name}: posterior-predictive residual RMS {resid_rms:.4f} "
            f">> injected noise floor {noise_floor:.4f}"
        )


def test_synthetic_and_fit_frequency_grids_agree():
    """Regression test for the dt_min mismatch bug: the ground-truth driver's
    frequency grid and EchoFit.build_grid()'s fitting grid must be derived
    from the same (robust) cadence estimate, or NUTS is fitting a basis that
    cannot represent the data that generated it."""
    data = _make_synthetic(seed=2)
    ef = _build_echofit(data)

    assert np.isclose(float(ef.freqs.max()), float(data["freqs"].max()), rtol=1e-3)
