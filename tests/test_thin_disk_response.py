"""
Tests for forward_model.thin_disk_response (the accretion-disk response
ported from the PhD-era CREAM Fortran, see CLAUDE.md) and the small
pycream2.responses registry that makes it (and any custom response) a
one-line, discoverable swap-in for "physical"-mode bands.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pycream2.forward_model import thin_disk_response, disk_temperature_profile, lag_scaling, _schwarzschild_radius_light_days, _WIEN_B_ANGSTROM_KELVIN
from pycream2.responses import available_responses, get_response, register_response

_np_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def test_thin_disk_response_is_causal_and_normalised():
    tau_grid = jnp.linspace(-5.0, 80.0, 300)
    psi = thin_disk_response(
        tau_grid, log_mdot=0.0, wavelength=5000.0, inclination=30.0, M_BH=1e8
    )
    psi_np = np.asarray(psi)
    tau_np = np.asarray(tau_grid)
    assert not np.isnan(psi_np).any()
    assert np.all(psi_np[tau_np < 0] == 0.0)
    area = _np_trapz(psi_np, tau_np)
    assert np.isclose(area, 1.0, atol=1e-2)


def test_thin_disk_response_median_lag_scales_with_wavelength():
    """Longer wavelengths respond from further out, so the median delay grows
    with wavelength. (This used to test the peak lag, which only moved with
    wavelength because the old Gaussian smoothing in tau pushed the peak
    outwards; unsmoothed, a lamppost-irradiated thin disc peaks at its inner
    edge at every wavelength, where the zero-torque cooling and the h_x/d**3
    dilution make the innermost ring respond most strongly.)"""
    tau_grid = jnp.linspace(-1.0, 20.0, 4000)
    tau_np = np.asarray(tau_grid)

    def median_lag(wavelength):
        psi = np.asarray(thin_disk_response(
            tau_grid, log_mdot=0.0, wavelength=wavelength, inclination=20.0, M_BH=1e8
        ))
        cdf = np.cumsum(psi) / np.sum(psi)
        return float(tau_np[np.searchsorted(cdf, 0.5)])

    assert median_lag(3000.0) < median_lag(5000.0) < median_lag(9000.0)


def test_thin_disk_response_has_nonzero_gradients():
    """The same failure mode as tophat_response_free's hard edges: a
    zero-gradient response would look fine (finite, normalised) but leave
    NUTS unable to move log_mdot/inclination at all."""
    tau_grid = jnp.linspace(0.0, 80.0, 200)

    def mean_lag(log_mdot, inclination):
        psi = thin_disk_response(
            tau_grid, log_mdot, wavelength=5000.0, inclination=inclination, M_BH=1e8
        )
        return jnp.sum(psi * tau_grid)

    grad_mdot = jax.grad(mean_lag, argnums=0)(0.0, 30.0)
    grad_incl = jax.grad(mean_lag, argnums=1)(0.0, 30.0)
    assert float(grad_mdot) != 0.0
    assert float(grad_incl) != 0.0


def test_thin_disk_response_inclination_increases_skew_not_just_shifts_mean():
    """Mirrors CLAUDE.md decision #4 (checked already for response_function):
    inclination should reshape the response, not simply translate it -- a
    face-on disk's iso-delay surface is flatter than an inclined one's."""
    tau_grid = jnp.linspace(0.0, 100.0, 500)
    tau_np = np.asarray(tau_grid)

    psi_face_on = np.asarray(
        thin_disk_response(tau_grid, log_mdot=0.0, wavelength=5000.0, inclination=1.0, M_BH=1e8)
    )
    psi_inclined = np.asarray(
        thin_disk_response(tau_grid, log_mdot=0.0, wavelength=5000.0, inclination=70.0, M_BH=1e8)
    )

    def skewness(psi):
        mean = _np_trapz(psi * tau_np, tau_np)
        var = _np_trapz(psi * (tau_np - mean) ** 2, tau_np)
        third = _np_trapz(psi * (tau_np - mean) ** 3, tau_np)
        return third / var ** 1.5

    assert skewness(psi_inclined) > skewness(psi_face_on)


def test_thin_disk_response_smoothing_days_zero_gives_exact_mean_lag_independence():
    """With smoothing off (smoothing_days=0.0), the analytic azimuthal
    integral is exact, so decision #4's mean-lag-independent-of-inclination
    claim should hold tightly, not just approximately -- this is the
    "escape hatch" for anyone who wants that exactness back at the cost of
    the visible kink the default smoothing exists to remove (see the next
    test). Since the lamppost-dilution fix the response is compact, with a
    sharp contribution from just outside the ISCO at high inclination, so
    the grid must be fine: at 800 points the quadrature alone drifted 2%,
    at 8000 points 0.2%.

    Since the lamppost height entered the delay (tau = d + h_x cos i +
    r sin i cos phi, September 2026), the mean delay is <d> + h_x cos i
    exactly (the r sin i cos phi term averages to zero), so it is
    <tau> - h_x cos i that is independent of inclination; the h_x cos i term
    itself is only ~0.03 d here."""
    tau_grid = jnp.linspace(-1.0, 8.0, 8000)
    tau_np = np.asarray(tau_grid)

    from pycream2.forward_model import _schwarzschild_radius_light_days
    hx = 3.0 * _schwarzschild_radius_light_days(1e8)

    lags = []
    for inclination in [0.0, 40.0, 80.0]:
        psi = np.asarray(thin_disk_response(
            tau_grid, log_mdot=0.0, wavelength=4000.0, inclination=inclination,
            M_BH=1e8, smoothing_days=0.0,
        ))
        lags.append(_np_trapz(psi * tau_np, tau_np) - hx * np.cos(np.deg2rad(inclination)))

    drift = abs(lags[-1] - lags[0]) / lags[0]
    assert drift < 0.01, f"expected near-exact mean-lag independence with smoothing off, got {drift:.4f} drift"


def test_thin_disk_response_default_smoothing_mean_lag_drift_is_bounded():
    """The default smoothing (a 5 per cent Gaussian in ln tau since September
    2026; before that a Gaussian in tau, smoothing_frac 0.4 then 0.1) is
    causal, so it no longer breaks decision #4's inclination-independence of
    the mean delay the way the Gaussian in tau did (~7-10% face-on to 80
    degrees): <tau> - h_x cos i (see the exact test above) drifts by only
    ~1% on this grid."""
    from pycream2.forward_model import _schwarzschild_radius_light_days
    hx = 3.0 * _schwarzschild_radius_light_days(1e8)
    tau_grid = jnp.linspace(-1.0, 8.0, 4000)
    tau_np = np.asarray(tau_grid)

    lags = []
    for inclination in [0.0, 40.0, 80.0]:
        psi = np.asarray(thin_disk_response(
            tau_grid, log_mdot=0.0, wavelength=4000.0, inclination=inclination, M_BH=1e8,
        ))
        lags.append(_np_trapz(psi * tau_np, tau_np) - hx * np.cos(np.deg2rad(inclination)))

    drift = abs(lags[-1] - lags[0]) / lags[0]
    assert drift < 0.03, (
        f"expected the causal default smoothing to keep the mean delay (less h_x cos i) "
        f"inclination-independent to a few per cent, got {drift:.4f} -- see docs/cream_response_comparison.md"
    )


def test_registry_looks_up_built_ins():
    assert "thin_disk" in available_responses()
    assert "skew_normal" in available_responses()
    assert get_response("thin_disk") is thin_disk_response


def test_registry_raises_with_helpful_message_for_unknown_name():
    with pytest.raises(KeyError, match="not_a_real_response"):
        get_response("not_a_real_response")


def test_registry_supports_custom_registration():
    def my_response(tau_grid, log_mdot, wavelength, inclination, M_BH, **kwargs):
        return thin_disk_response(tau_grid, log_mdot, wavelength, inclination, M_BH)

    register_response("my_response", my_response)
    try:
        assert get_response("my_response") is my_response
        assert "my_response" in available_responses()
    finally:
        from pycream2.responses import _REGISTRY

        del _REGISTRY["my_response"]


def test_thin_disk_response_is_usable_via_the_existing_swap_mechanism():
    """thin_disk_response matches the same (tau_grid, log_mdot, wavelength,
    inclination, M_BH) contract as response_function, so it plugs into the
    existing pycream2.model.response_function swap point (CLAUDE.md decision
    #5) with no further wiring -- exactly the point of the registry."""
    import pycream2.model as model_mod
    from pycream2.synthetic import generate_synthetic_dataset
    from pycream2.echofit import EchoFit

    original = model_mod.response_function
    try:
        model_mod.response_function = get_response("thin_disk")

        data = generate_synthetic_dataset(
            M_BH=1.0e8, bands={"g": 4770.0}, n_obs_per_band=15,
            n_freq=6, n_tau=60, tau_max=40.0, noise_level=0.08, seed=0,
        )
        ef = EchoFit(M_BH=1.0e8)
        for name, d in data["bands"].items():
            ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
        ef.build_grid(n_freq=6, n_tau=60)
        ef.fit(rng_seed=0, num_warmup=10, num_samples=10, progress_bar=False)
    finally:
        model_mod.response_function = original


def test_disk_temperature_profile_matches_wien_law_at_the_reference_radius():
    """The power law T_1 r**(-3/4) reaches the Wien temperature b/lambda
    exactly at lag_scaling's tau_ref(lambda), at every wavelength (so T_1 is
    wavelength-independent); the full profile adds the ISCO factor
    (1 - sqrt(r_in/r))**(1/4) on top."""
    M_BH, log_mdot = 1e8, 0.0
    r_in = 3.0 * _schwarzschild_radius_light_days(M_BH)
    for wavelength in (1500.0, 5000.0, 9000.0):
        tau_ref = lag_scaling(log_mdot, wavelength, M_BH)
        T = float(disk_temperature_profile(tau_ref, log_mdot, wavelength, M_BH))
        inner = (1.0 - np.sqrt(r_in / tau_ref)) ** 0.25
        assert T == pytest.approx(_WIEN_B_ANGSTROM_KELVIN / wavelength * inner, rel=1e-5)


def test_disk_temperature_profile_does_not_depend_on_wavelength_for_any_slope():
    """The disk's temperature is a property of the disk: the profile must not
    change with the observing wavelength, whatever the slope (before
    September 2026 it did, for any slope other than 0.75)."""
    r = jnp.array([0.3, 1.0, 5.0])
    for slope in (0.75, 1.0):
        a = np.asarray(disk_temperature_profile(r, 0.5, 1500.0, 1e8, viscous_slope=slope))
        b = np.asarray(disk_temperature_profile(r, 0.5, 9000.0, 1e8, viscous_slope=slope))
        assert np.allclose(a, b, rtol=1e-6)


def test_thin_disk_mean_lag_obeys_the_standard_lag_temperature_relation():
    """Regression test for the missing lamppost dilution (h_x/x**3) in the
    response weight: with it, the mean delay of the unsmoothed response
    satisfies <tau> = (X k lambda T_1 / hc)**(4/3) light-days with X ~ 3.2
    (responsivity-weighted, including the ISCO term) at every wavelength and
    inclination. Without it X came out ~8, a ~3x temperature error. The
    relation is for the delay r (1 + sin i cos phi); the exact lamppost delay
    (September 2026) adds h_x cos i to the mean, which is subtracted here."""
    from pycream2.forward_model import _schwarzschild_radius_light_days, disk_t1_kelvin
    hx = 3.0 * _schwarzschild_radius_light_days(10 ** 7.5)
    tau = jnp.linspace(0.0, 40.0, 8000)
    tau_np = np.asarray(tau)
    t1 = float(disk_t1_kelvin(1.0, 10 ** 7.5))
    for wavelength in (1367.0, 9157.0):
        for inclination in (0.0, 45.0):
            psi = np.asarray(thin_disk_response(tau, 1.0, wavelength, inclination, 10 ** 7.5, smoothing_days=0.0))
            mean = _np_trapz(psi * tau_np, tau_np) / _np_trapz(psi, tau_np) - hx * np.cos(np.deg2rad(inclination))
            x_eff = 1.4387773538e8 / (wavelength * t1) * mean ** 0.75
            assert 3.0 < x_eff < 3.4, (wavelength, inclination, x_eff)


def test_disk_temperature_profile_vanishes_at_the_isco():
    """No viscous dissipation right at the inner boundary -- the standard
    Shakura-Sunyaev condition, T(r_in) = 0 (down to the function's own
    numerical floor). Checked relative to the temperature well clear of
    the ISCO, rather than an absolute Kelvin value, since the floor itself
    scales with the Wien-law reference temperature (M_BH/wavelength
    dependent)."""
    M_BH, wavelength, log_mdot = 1e8, 5000.0, 0.0
    r_in = 3.0 * _schwarzschild_radius_light_days(M_BH)
    T_at_isco = float(disk_temperature_profile(r_in, log_mdot, wavelength, M_BH))
    T_away = float(disk_temperature_profile(10.0 * r_in, log_mdot, wavelength, M_BH))
    assert T_at_isco < 1e-2 * T_away


def test_disk_temperature_profile_declines_far_from_the_isco():
    """Away from the inner-boundary region (where the profile genuinely
    peaks just outside the ISCO before declining -- not monotonic from
    r_in itself), temperature should fall off smoothly with radius."""
    M_BH, wavelength, log_mdot = 1e8, 5000.0, 0.0
    r_in = 3.0 * _schwarzschild_radius_light_days(M_BH)
    r = np.geomspace(50.0 * r_in, 50.0, 20)  # well clear of the near-ISCO peak
    T = np.asarray(disk_temperature_profile(r, log_mdot, wavelength, M_BH))
    assert np.all(np.diff(T) < 0.0)


def test_thin_disk_response_starts_at_zero_and_smoothing_is_causal():
    """The response is zero at zero lag and at every negative lag with the
    default (log-normal) smoothing: before September 2026 the default was a
    Gaussian in tau, which spread the near-vertical onset of the response
    across tau = 0 and left psi(0) at its peak value for inclined discs (see
    docs/cream_response_comparison.md)."""
    tau_grid = jnp.linspace(-2.0, 20.0, 4001)
    tau_np = np.asarray(tau_grid)
    for inclination in (0.0, 45.0, 80.0):
        psi = np.asarray(thin_disk_response(tau_grid, 3.0, 5000.0, inclination, 10 ** 7.5))
        assert np.all(psi[tau_np <= 0.0] == 0.0)
        assert abs(_np_trapz(psi, tau_np) - 1.0) < 1e-3
        assert psi[tau_np > 0.0][0] < 1e-3 * psi.max()


@pytest.mark.parametrize("include_irradiation", [False, True])
def test_thin_disk_response_no_light_before_the_lamppost_delay(include_irradiation):
    """With the lamppost height in the delay, nothing can respond before the
    shortest lamppost-disc-observer path: face-on that is sqrt(r**2 + h_x**2)
    + h_x at the response's inner edge. Viscous-only, that edge is the
    temperature peak just outside the ISCO (r_pk); with irradiation, which
    keeps the inner disk warm, the sigmoid mask still starts at r_pk but its
    tail reaches in towards the ISCO itself, so the bound is the ISCO's delay."""
    from pycream2.forward_model import _schwarzschild_radius_light_days
    M = 10 ** 7.5
    rs = _schwarzschild_radius_light_days(M)
    hx = 3.0 * rs
    r_edge = 3.0 * rs * ((3.5 / 3.0) ** 2 if not include_irradiation else 1.0)
    earliest = np.sqrt(r_edge ** 2 + hx ** 2) + hx
    tau_grid = jnp.linspace(0.0, 2.0, 20001)
    tau_np = np.asarray(tau_grid)
    psi = np.asarray(thin_disk_response(tau_grid, 3.0, 5000.0, 0.0, M, smoothing_log=0.0,
                                        include_irradiation=include_irradiation))
    # The inner-edge sigmoid is 0.05 r_in wide, so allow a sliver below the edge.
    assert np.all(psi[tau_np < 0.9 * earliest] < 1e-6 * psi.max())
    assert psi[tau_np > 1.5 * earliest].max() > 0.0


def test_log_smoothing_keeps_the_mean_delay():
    """A 5 per cent Gaussian in ln tau shifts the mean delay by
    exp(sigma**2 / 2) - 1 ~ 0.1 per cent (plus quadrature error)."""
    tau_grid = jnp.linspace(0.0, 60.0, 12001)
    tau_np = np.asarray(tau_grid)
    exact = np.asarray(thin_disk_response(tau_grid, 3.0, 5000.0, 30.0, 10 ** 7.5, smoothing_log=0.0))
    smooth = np.asarray(thin_disk_response(tau_grid, 3.0, 5000.0, 30.0, 10 ** 7.5))
    mean = lambda psi: _np_trapz(psi * tau_np, tau_np) / _np_trapz(psi, tau_np)  # noqa: E731
    assert abs(mean(smooth) / mean(exact) - 1.0) < 0.01


@pytest.mark.parametrize("wavelength", [2055.0, 5425.0])
def test_default_response_has_no_inner_edge_spike(wavelength):
    """The viscous-only response peaks at the disk's inner edge in every band
    (the zero-torque factor makes the inner disk cool, and the T**-3 weighting
    makes cool gas respond strongly), a sharp spike at a fixed ~0.25-day delay
    for Fairall 9's parameters that CREAM's responses don't have. The default,
    irradiated response peaks well outside it, later at longer wavelengths."""
    from pycream2.forward_model import _schwarzschild_radius_light_days
    M = 2.55e8
    rs = _schwarzschild_radius_light_days(M)
    hx, r_pk = 3.0 * rs, 3.0 * rs * (3.5 / 3.0) ** 2
    edge = np.sqrt(r_pk ** 2 + hx ** 2) + hx
    tau_grid = jnp.linspace(0.0, 10.0, 4001)
    tau_np = np.asarray(tau_grid)
    peak = tau_np[np.argmax(np.asarray(thin_disk_response(tau_grid, 2.44, wavelength, 0.0, M)))]
    peak_viscous = tau_np[np.argmax(np.asarray(
        thin_disk_response(tau_grid, 2.44, wavelength, 0.0, M, include_irradiation=False)))]
    assert peak_viscous < 1.3 * edge
    assert peak > 2.0 * edge
