"""EchoFit(fit_temperature_slope=True): the disk temperature slope alpha
(T ~ r**-alpha) as a fitted parameter, Starkey et al. (2017)'s Model 2."""

import numpy as np
import pytest

import pycream2.model as model_mod
from pycream2 import EchoFit
from pycream2.forward_model import thin_disk_response
from pycream2.synthetic import generate_synthetic_dataset


@pytest.fixture
def thin_disk():
    original = model_mod.response_function
    model_mod.response_function = thin_disk_response
    try:
        yield
    finally:
        model_mod.response_function = original


def _echofit(**kwargs):
    data = generate_synthetic_dataset(
        M_BH=1.0e8, bands={"g": 4770.0, "i": 7625.0}, n_obs_per_band=20,
        n_freq=8, n_tau=80, tau_max=30.0, noise_level=0.05, seed=0,
    )
    ef = EchoFit(M_BH=1.0e8, fit_temperature_slope=True, **kwargs)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=8, n_tau=80)
    return ef


def test_temperature_slope_is_a_fitted_site(thin_disk):
    ef = _echofit()
    ef.optimise(num_samples=20, num_restarts=1, method="laplace")
    assert "temperature_slope" in ef.samples
    lo, hi = model_mod.TEMPERATURE_SLOPE_PRIOR
    assert np.all((ef.samples["temperature_slope"] > lo) & (ef.samples["temperature_slope"] < hi))
    assert ef.plot_lightcurve_fits() is not None


def test_temperature_slope_can_be_fixed(thin_disk):
    ef = _echofit(fixed_params={"temperature_slope": 1.0})
    ef.optimise(num_samples=10, num_restarts=1, method="laplace")
    assert np.allclose(ef.samples["temperature_slope"], 1.0)


def test_temperature_slope_needs_a_response_that_accepts_it():
    """The default skew-normal response has no viscous_slope argument."""
    ef = _echofit()
    with pytest.raises(ValueError, match="viscous_slope"):
        ef.optimise(num_samples=10, num_restarts=1, method="laplace")


def test_temperature_slope_persists_across_resume(thin_disk, tmp_path):
    ef = _echofit(title="slope_resume", output_dir=str(tmp_path))
    ef.fit(num_warmup=5, num_samples=4, checkpoint_every=2, progress_bar=False, generate_report=False)
    resumed = EchoFit.resume("slope_resume", output_dir=str(tmp_path))
    assert resumed.fit_temperature_slope is True
    resumed.fit(num_samples=6, progress_bar=False, generate_report=False)
    assert resumed.samples["temperature_slope"].shape[0] == 6
