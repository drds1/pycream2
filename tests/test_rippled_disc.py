"""
Rimmed and rippled discs (pycream2.rippled_disc; Starkey, Huang, Horne & Lin
2023, MNRAS 519, 2754): the geometry, the shadows, the response function's
limits, and a synthetic-data recovery. See docs/rippled_disc.md.
"""

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pycream2.model as model_module
from pycream2 import EchoFit
from pycream2 import disc_sed
from pycream2.forward_model import disk_t1_kelvin, thin_disk_response
from pycream2.grid_utils import graded_tau_grid
from pycream2.responses import get_response
from pycream2.rippled_disc import (
    NGC5548_RIM,
    disc_height,
    disc_surface,
    mean_delay,
    response_elements,
    rippled_disc_fnu,
    rippled_disc_response,
)
from pycream2.synthetic import generate_synthetic_dataset

M_BH = 7e7
LOG_MDOT = 0.0587  # a viscous disc at 1500 K at 5 light-days, as in the paper's NGC 5548 example
TAU = jnp.asarray(graded_tau_grid(30.0, 600))


def _median(tau, psi):
    tau, psi = np.asarray(tau), np.asarray(psi)
    cdf = np.cumsum(psi * np.gradient(tau))
    return float(np.interp(0.5, cdf / cdf[-1], tau))


@pytest.mark.parametrize("wavelength", [2000.0, 9000.0])
@pytest.mark.parametrize("inclination", [0.0, 45.0])
def test_flat_disc_matches_thin_disk_response(wavelength, inclination):
    """A flat disc without irradiation heating is thin_disk_response's viscous
    disc: the 2-D quadrature here and the exact 1-D azimuthal reduction there
    must give the same delays."""
    psi = rippled_disc_response(TAU, 0.0, wavelength, inclination, M_BH, lamppost_efficiency=0.0)
    ref = thin_disk_response(TAU, 0.0, wavelength, inclination, M_BH, include_irradiation=False)
    assert mean_delay(TAU, psi) == pytest.approx(mean_delay(TAU, ref), rel=0.01)
    assert _median(TAU, psi) == pytest.approx(_median(TAU, ref), rel=0.01)


def test_response_is_causal_and_normalised():
    psi = np.asarray(rippled_disc_response(TAU, LOG_MDOT, 5000.0, 45.0, M_BH, **NGC5548_RIM))
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    assert np.all(np.isfinite(psi)) and np.all(psi >= 0.0)
    assert trapz(psi, np.asarray(TAU)) == pytest.approx(1.0, abs=1e-3)


def test_height_profile_follows_the_papers_equations():
    r = np.linspace(1.0, 5.0, 9)
    np.testing.assert_allclose(disc_height(r, height=0.15, r_ref=5.0, beta=100.0), 0.15 * (r / 5.0) ** 100, rtol=1e-5, atol=1e-12)
    rippled = disc_height(r, height=0.02, r_ref=1.0, beta=1.0, ripple_amplitude=0.3, ripple_wavelength=1.0,
                          ripple_slope=0.5)
    expected = 0.02 * r * (1 + 0.3 * r ** -0.5 * np.cos(2 * np.pi * r))
    np.testing.assert_allclose(rippled, expected, rtol=1e-5)


def test_convex_disc_is_shadowed_beyond_r_beta():
    """Starkey et al. eq. 16-17: a convex power law (beta < 1) is lit only
    inside r_beta = r_1 (H_LP / (H_1 (1 - beta)))**(1 / beta)."""
    h1, beta = 0.05, 0.5
    s = disc_surface(0.0, M_BH, height=h1, r_ref=1.0, beta=beta, r_out=20.0)
    r, lit = np.asarray(s["r"]), np.asarray(s["lit"])
    r_beta = (s["h_lp"] / (h1 * (1 - beta))) ** (1 / beta)
    assert np.all(lit[r < 0.95 * r_beta] == 1.0)
    assert np.all(lit[r > 1.05 * r_beta] == 0.0)


def test_ripple_troughs_lie_in_the_crests_shadow():
    """On a rippled disc only the slopes facing the lamppost are lit, and some
    of those are hidden behind the crest inside them; shadowed gas is only
    viscously heated, so it is much cooler."""
    s = disc_surface(LOG_MDOT, M_BH, height=0.02, r_ref=1.0, beta=1.2, ripple_amplitude=0.3,
                     ripple_wavelength=1.0, r_out=10.0, lamppost_efficiency=0.2)
    r, lit, dH, T = (np.asarray(s[k]) for k in ("r", "lit", "dH", "T"))
    band = (r > 3.0) & (r < 8.0)
    assert np.all(lit[band & (dH < 0)] == 0.0)  # slopes facing away
    rising = band & (dH > 0)
    assert 0.0 < lit[rising].mean() < 1.0  # some rising slopes are in a crest's shadow
    assert np.median(T[band & (lit == 1)]) > 2.0 * np.median(T[band & (lit == 0)])


def test_papers_irradiation_form_is_available():
    """exact_irradiation=False is Starkey et al. eq. 3 as written:
    T**4 = T_1**4 r**-3 [(1 - sqrt(r_in/r)) + (4/3) eps (r/r_g) f]."""
    eps = 0.2
    s = disc_surface(LOG_MDOT, M_BH, exact_irradiation=False, **dict(NGC5548_RIM, lamppost_efficiency=eps))
    r, f, lit, T = (np.asarray(s[k], dtype=float) for k in ("r", "f", "lit", "T"))
    r_g = s["r_in"] / 6.0
    t1 = float(disk_t1_kelvin(LOG_MDOT, M_BH))
    expected = t1 * (r ** -3 * ((1 - np.sqrt(s["r_in"] / r)) + lit * (4 / 3) * eps * (r / r_g) * f)) ** 0.25
    np.testing.assert_allclose(T, expected, rtol=1e-4)
    # The exact form is cooler on the steep rim face (by up to sqrt(1 + H'**2) in T**4).
    exact = np.asarray(disc_surface(LOG_MDOT, M_BH, **dict(NGC5548_RIM, lamppost_efficiency=eps))["T"])
    face = r > 4.9
    assert exact[face].max() < T[face].max()


@pytest.mark.parametrize("inclination", [30.0, 60.0])
def test_vertical_wall_mean_delay(inclination):
    """A near-vertical rim's own mean delay: each part of the face weighted by
    its projected area gives r_out (1 + (pi/4) sin i), plus (H_LP - <H>) cos i
    from the face's height. Only the steep part of the face (H above a tenth of
    H_out, where H' > 40) is a wall; it is uniformly lit, so <H> = 0.55 H_out."""
    h_out, r_out = 0.5, 5.0
    e = response_elements(LOG_MDOT, 9000.0, inclination, M_BH, n_phi=720, height=h_out, r_ref=r_out,
                          beta=4000.0, r_out=r_out, lamppost_efficiency=0.2)
    w, d = np.asarray(e["weight"]), np.asarray(e["delay"])
    face = np.asarray(e["surface"]["H"]) > 0.1 * h_out
    sini, cosi = np.sin(np.deg2rad(inclination)), np.cos(np.deg2rad(inclination))
    expected = r_out * (1 + np.pi / 4 * sini) + (e["surface"]["h_lp"] - 0.55 * h_out) * cosi
    assert (w[face] * d[face]).sum() / w[face].sum() == pytest.approx(expected, rel=0.01)


def test_rim_lengthens_and_flattens_the_red_delays():
    """The rim's irradiated face adds a late response at r_out: longer optical
    delays than a flat disc's, and a lag spectrum that flattens at long
    wavelengths as the delays approach the rim's."""
    flat = dict(r_out=NGC5548_RIM["r_out"], lamppost_efficiency=NGC5548_RIM["lamppost_efficiency"])
    lags = {name: [mean_delay(TAU, rippled_disc_response(TAU, LOG_MDOT, lam, 45.0, M_BH, **geom))
                   for lam in (4770.0, 9134.0)]
            for name, geom in (("flat", flat), ("rim", NGC5548_RIM))}
    assert lags["rim"][0] > 2.5 * lags["flat"][0]
    assert lags["rim"][1] > 1.5 * lags["flat"][1]
    assert lags["rim"][1] / lags["rim"][0] < lags["flat"][1] / lags["flat"][0]


def test_gradients_are_finite():
    f = jax.jit(lambda lm, inc: jnp.sum(TAU * rippled_disc_response(TAU, lm, 9000.0, inc, M_BH, **NGC5548_RIM)))
    g = jax.grad(f, argnums=(0, 1))(LOG_MDOT, 45.0)
    assert all(np.isfinite(float(x)) and float(x) != 0.0 for x in g)


def test_flat_disc_spectrum_matches_disc_sed():
    """A flat disc's spectrum (no irradiation heating) is disc_sed's viscous one."""
    lam = np.array([2000.0, 5000.0, 9000.0])
    t1 = float(disk_t1_kelvin(1.0, 1e8))
    ref = disc_sed.disc_fnu_at_1cm(lam, 0.0, t1, 0.75, 30.0, 1e8, include_irradiation=False) / (100 * disc_sed.MPC_CM) ** 2
    ours = rippled_disc_fnu(lam, 1.0, 30.0, 1e8, dl_mpc=100.0, lamppost_efficiency=0.0, r_out=1000.0, n_r=4000)
    np.testing.assert_allclose(ours, ref, rtol=0.01)


def test_registered_response():
    assert get_response("rippled_disc") is rippled_disc_response


@pytest.mark.slow
def test_rimmed_disc_recovery_and_flat_disc_bias(monkeypatch):
    """Light curves from the NGC 5548 rim: the rimmed model (geometry fixed at the
    truth) recovers log_mdot and the inclination, while a flat thin disc fitted to
    the same data is far worse and much hotter, the signature of the disc-size
    problem."""
    bands = {"w2": 1928.0, "g": 4770.0, "i": 7625.0, "z": 9134.0}
    rim = functools.partial(rippled_disc_response, **NGC5548_RIM)
    data = generate_synthetic_dataset(bands=bands, M_BH=M_BH, log_mdot_true=LOG_MDOT, inclination_true=45.0,
                                      t_span=250.0, n_obs_per_band=120, noise_level=0.02, tau_max=30.0, seed=3,
                                      response=rim)

    def fit(response):
        monkeypatch.setattr(model_module, "response_function", response)
        ef = EchoFit(M_BH=M_BH)
        for name, d in data["bands"].items():
            ef.add_lightcurve(name, d["wavelength"], d["t"], d["y"], d["yerr"], background_order=0)
        # The truth's own driver basis (the synthetic generator draws on it), not
        # build_grid()'s default; the synthetic truth has no background.
        ef.build_grid(tau_max=30.0, period_max=2 * np.pi / float(data["freqs"].min()), n_freq=len(data["freqs"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ef.optimise(num_samples=200, num_restarts=1, method="laplace")
        return ef

    ef_rim, ef_flat = fit(rim), fit(thin_disk_response)
    lm = np.asarray(ef_rim.samples["log_mdot"])
    # Within 0.1 dex (this four-band realisation lands 0.08 dex low, ~4 Laplace sd;
    # the six-band set in docs/rippled_disc.md lands within 1 sd): small next to
    # the flat disc's 1.5 dex bias below.
    assert abs(np.median(lm) - LOG_MDOT) < 0.1
    assert abs(np.median(ef_rim.samples["inclination"]) - 45.0) < 5.0
    assert np.median(ef_flat.samples["log_mdot"]) > LOG_MDOT + 0.5
    assert ef_rim.log_evidence > ef_flat.log_evidence + 100.0
