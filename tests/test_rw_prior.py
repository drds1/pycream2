"""
Tests for the driver's prior family (CLAUDE.md decision #19):
``EchoFit(drw_prior=False)`` (the default) uses a pure random-walk (RW)
power-law prior on the driver's Fourier coefficients, no ``tau_drw`` site
at all; ``EchoFit(drw_prior=True)`` restores the original damped random
walk (DRW) prior with ``tau_drw`` inferred alongside ``sigma_drw``.
"""

import numpy as np
import pytest

from pycream2.synthetic import generate_synthetic_dataset
from pycream2.echofit import EchoFit


def _dataset():
    return generate_synthetic_dataset(
        M_BH=1.0e8, bands={"g": 4770.0, "i": 7625.0}, n_obs_per_band=25,
        n_freq=8, n_tau=40, tau_max=30.0, noise_level=0.05, seed=0,
    )


def _build(data, **kwargs):
    ef = EchoFit(M_BH=1.0e8, **kwargs)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=8, n_tau=40)
    return ef


def test_rw_prior_is_default_no_tau_drw_site():
    ef = _build(_dataset())  # drw_prior not passed -> False
    assert ef.drw_prior is False
    ef.fit(num_warmup=5, num_samples=5, progress_bar=False)
    assert "tau_drw" not in ef.samples
    assert "sigma_drw" in ef.samples
    assert np.all(np.isfinite(ef.samples["sigma_drw"]))


def test_drw_prior_true_restores_tau_drw_site():
    ef = _build(_dataset(), drw_prior=True)
    ef.fit(num_warmup=5, num_samples=5, progress_bar=False)
    assert "tau_drw" in ef.samples
    assert np.all(np.isfinite(ef.samples["tau_drw"]))


def test_fixing_tau_drw_without_drw_prior_raises():
    ef = _build(_dataset(), fixed_params={"tau_drw": 20.0})  # drw_prior=False default
    with pytest.raises(ValueError, match="tau_drw"):
        ef.fit(num_warmup=5, num_samples=5, progress_bar=False)


def test_fixing_tau_drw_works_with_drw_prior_true():
    ef = _build(_dataset(), drw_prior=True, fixed_params={"tau_drw": 20.0})
    ef.fit(num_warmup=5, num_samples=5, progress_bar=False)
    assert np.all(ef.samples["tau_drw"] == 20.0)


def test_plot_power_spectrum_does_not_crash_in_rw_mode():
    ef = _build(_dataset())
    ef.fit(num_warmup=5, num_samples=5, progress_bar=False)
    fig, ax = ef.plot_power_spectrum()
    assert fig is not None


def test_plot_power_spectrum_does_not_crash_in_drw_mode():
    ef = _build(_dataset(), drw_prior=True)
    ef.fit(num_warmup=5, num_samples=5, progress_bar=False)
    fig, ax = ef.plot_power_spectrum()
    assert fig is not None


def test_drw_prior_persists_across_resume(tmp_path):
    """drw_prior must survive the checkpointed save/resume round trip --
    the same class of bug decision #15/#18 hit with fixed_params/
    fit_error_model: a constructor-level setting that changes which sites
    exist has to be threaded through the manifest and EchoFit.resume()."""
    data = _dataset()
    ef = EchoFit(M_BH=1.0e8, title="drw_prior_resume_test", output_dir=str(tmp_path), drw_prior=True)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=8, n_tau=40)
    ef.fit(rng_seed=0, num_warmup=5, num_samples=10, checkpoint_every=5, progress_bar=False)
    assert "tau_drw" in ef.samples

    ef2 = EchoFit.resume("drw_prior_resume_test", output_dir=str(tmp_path))
    assert ef2.drw_prior is True
    ef2.fit(progress_bar=False)
    assert "tau_drw" in ef2.samples


@pytest.mark.slow
def test_rw_prior_recovers_log_mdot_on_synthetic_data():
    """Same recovery-quality bar as test_recovery.py's DRW check (decision
    #19 flips the default, so the default path needs the same end-to-end
    validation), just under the new default random-walk prior."""
    data = generate_synthetic_dataset(
        M_BH=1.0e9, log_mdot_true=0.0, inclination_true=35.0,
        sigma_drw_true=0.3, tau_drw_true=30.0,
        bands={"g": 4770.0, "i": 7625.0}, t_span=200.0, n_obs_per_band=50,
        n_freq=15, n_tau=100, tau_max=50.0, noise_level=0.02, seed=2,
    )
    ef = EchoFit(M_BH=data["truth"]["M_BH"])  # drw_prior=False default
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=15, n_tau=100, period_max=2 * np.pi / float(data["freqs"].min()))
    # 1000 warmup steps, not 500: with log_mdot's prior widened from N(0, 1) to
    # N(0, 5**2) (model.LOG_MDOT_PRIOR_SD) 500 left 6% of transitions divergent.
    ef.fit(rng_seed=0, num_warmup=1000, num_samples=500, num_chains=1, progress_bar=False)

    diverging = np.asarray(ef.extra_fields["diverging"])
    assert diverging.mean() < 0.05, f"{diverging.sum()}/{len(diverging)} divergent transitions"
    assert "tau_drw" not in ef.samples

    log_mdot_mean = float(ef.samples["log_mdot"].mean())
    assert abs(log_mdot_mean - data["truth"]["log_mdot"]) < 0.3, (
        f"log_mdot posterior mean {log_mdot_mean:.3f} far from truth {data['truth']['log_mdot']:.3f}"
    )
