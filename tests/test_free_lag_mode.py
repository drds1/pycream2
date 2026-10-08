"""
Tests for lag_mode="free" bands and EchoFit.add_driver_lightcurve(): the
validation guardrails (M_BH required for physical bands, a warning when a
free-lag band has no driver anchor), and a real MCMC recovery check that a
driver-anchored free-lag fit actually recovers the right lag ordering --
which tests/test_shift_degeneracy.py already proved is impossible without
one, exactly.
"""

import warnings

import numpy as np
import pytest

from pycream2.echofit import EchoFit
from pycream2.synthetic import generate_synthetic_dataset, generate_free_lag_dataset


def test_physical_band_without_M_BH_raises():
    data = generate_synthetic_dataset(
        M_BH=1.0e9, bands={"g": 4770.0}, n_obs_per_band=15, n_freq=6, n_tau=40, seed=0,
    )
    ef = EchoFit(M_BH=None)  # no M_BH given
    d = data["bands"]["g"]
    ef.add_lightcurve("g", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=6, n_tau=40)
    with pytest.raises(ValueError, match="M_BH is required"):
        ef.fit(num_warmup=5, num_samples=5, progress_bar=False)


def test_free_band_without_driver_warns():
    data = generate_free_lag_dataset(
        lines={"line_a": 8.0}, n_obs_per_line=15, n_freq=6, n_tau=40,
        include_driver=False, seed=0,
    )
    ef = EchoFit(M_BH=None)
    d = data["bands"]["line_a"]
    ef.add_lightcurve("line_a", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"], lag_mode="free")
    ef.build_grid(n_freq=6, n_tau=40)
    with pytest.warns(UserWarning, match="not identifiable"):
        ef.fit(num_warmup=5, num_samples=5, progress_bar=False)


def test_free_band_with_driver_does_not_warn():
    data = generate_free_lag_dataset(
        lines={"line_a": 8.0}, n_obs_per_line=15, n_obs_driver=20, n_freq=6, n_tau=40,
        include_driver=True, seed=0,
    )
    ef = EchoFit(M_BH=None)
    d = data["bands"]["line_a"]
    ef.add_lightcurve("line_a", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"], lag_mode="free")
    ef.add_driver_lightcurve(t=data["driver"]["t"], y=data["driver"]["y"], yerr=data["driver"]["yerr"])
    ef.build_grid(n_freq=6, n_tau=40)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        ef.fit(num_warmup=5, num_samples=5, progress_bar=False)


@pytest.mark.slow
def test_free_lag_recovery_with_driver_anchor():
    """The real payoff: with a driver light curve registered, a multi-line
    free-lag fit should recover the true lags -- exactly what
    test_shift_degeneracy.py shows is mathematically impossible without the
    driver anchoring the absolute scale.

    A *single* NUTS chain is not good enough evidence of that on its own:
    a broad Uniform(0, tau_max) prior on each tau_{band}, combined with a
    stochastic driver that has its own autocorrelation structure, gives a
    genuinely multimodal likelihood -- a lone chain can converge cleanly
    (0 divergences, tight posterior) while sitting in the *wrong* mode, which
    looks deceptively healthy. Confirmed directly: at a sparser data budget
    (35 obs/line, noise_level=0.05), 2 of 3 single-chain seeds recovered the
    true lags almost exactly, but one converged confidently to values 3-4x
    too large. Picking whichever seed happens to match a truth you already
    know would be exactly the mistake this test exists to catch, not
    validate -- on real data there's no known answer to seed-shop against.

    The honest fix is what you'd actually have to do without knowing the
    truth: run several independently-initialised chains and check they
    *agree* (Gelman-Rubin R-hat) before trusting any of them. That's the
    real assertion here; recovering the true values is checked only once
    convergence is established.
    """
    from numpyro.diagnostics import summary

    data = generate_free_lag_dataset(
        lines={"line_a": 8.0, "line_b": 18.0, "line_c": 30.0},
        sigma_drw_true=0.3, tau_drw_true=25.0, t_span=200.0,
        n_obs_per_line=60, n_obs_driver=100, include_driver=True,
        n_freq=15, n_tau=150, tau_max=60.0, noise_level=0.02, seed=1,
    )
    ef = EchoFit(M_BH=None)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"], lag_mode="free")
    ef.add_driver_lightcurve(t=data["driver"]["t"], y=data["driver"]["y"], yerr=data["driver"]["yerr"])
    # tau_max must match the synthetic generator's (60.0) -- build_grid()'s
    # own default (half the observed baseline) would otherwise give a much
    # wider, weaker Uniform(0, tau_max) prior on each tau_{band} than intended.
    ef.build_grid(n_freq=15, n_tau=150, tau_max=60.0, period_max=2 * np.pi / float(data["freqs"].min()))

    # 1000 warmup steps, not 400: with the Filon transfer coefficients
    # (forward_model._filon_weights, CLAUDE.md decision #22) one of the 4
    # chains stayed in a local mode ~660 units of potential worse, every lag
    # shifted ~+0.1 d, after 400; the trapezoid rule's high-frequency aliasing
    # had happened to smooth that trap away. With 1000 all 4 agree.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ef.fit(
            rng_seed=0, num_warmup=1000, num_samples=400,
            num_chains=4, chain_method="vectorized", progress_bar=False,
        )

    diverging = np.asarray(ef.extra_fields["diverging"])
    assert diverging.mean() < 0.1

    diag = summary(ef._samples_by_chain, prob=0.9)
    for name in data["bands"]:
        r_hat = diag[f"tau_{name}"]["r_hat"]
        assert r_hat < 1.05, (
            f"tau_{name} r_hat={r_hat:.3f}: the 4 independently-initialised chains "
            f"don't agree (didn't converge to the same mode), so nothing below this "
            f"point can be trusted from this fit -- see this test's docstring."
        )

    for name, truth in data["truth"]["bands"].items():
        recovered = float(np.median(ef.samples[f"tau_{name}"]))
        assert abs(recovered - truth["tau"]) < 2.0, (
            f"tau_{name} recovered={recovered:.2f} true={truth['tau']:.2f}: chains agree "
            f"(R-hat is fine) but converged together on the wrong value."
        )
