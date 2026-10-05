"""The disc SED, luminosity distance and H_0 analysis (pycream2.disc_sed):
known-answer tests on constructed light curves (no fitting), and its
integration with EchoFit (sed_analysis=True, the report, resume)."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from pycream2 import EchoFit
from pycream2 import disc_sed
from pycream2.forward_model import disk_t1_kelvin, driver_at
from pycream2.synthetic import generate_synthetic_dataset

M_BH = 1e8
Z = 0.03
LAMS = {"uv": 2000.0, "u": 3500.0, "g": 4800.0, "i": 7500.0}  # rest frame
LOG_MDOT, INCL, D_TRUE_MPC = 1.0, 30.0, 140.0


def test_ccm_extinction_at_v_is_rv_times_ebv():
    # CCM (1989): a(x) = 1, b(x) = 0 at x = 1.82 / micron, so A = R_V E(B-V).
    assert disc_sed.ccm_a_lambda(1e4 / 1.82, 0.1) == pytest.approx(0.31, rel=1e-6)
    a = disc_sed.ccm_a_lambda([1500.0, 2200.0, 5000.0, 9000.0], 0.1)
    assert a[1] > a[0] > a[2] > a[3] > 0  # the 2175 A bump, then falling with wavelength


def test_flux_unit_conversion():
    assert np.all(disc_sed.to_mjy([5000.0, 6000.0], "mJy", 2.0) == 2.0)
    # f_nu = f_lambda lambda^2 / c: 1e-15 erg/s/cm2/A at 5000 A is 8.34e-3 mJy.
    assert disc_sed.to_mjy([5000.0], "f_lambda", 1e-15)[0] == pytest.approx(1e-15 * 5000.0 ** 2 / 2.99792458e18 * 1e26)
    with pytest.raises(ValueError):
        disc_sed.to_mjy([5000.0], "magnitudes")


def test_h0_round_trip_and_low_redshift_limit():
    dl = 300.0
    h0 = disc_sed.h0_from_dl(dl, 0.05)
    # Flat Lambda-CDM, Omega_m = 0.3: D_L ~ (cz/H0)(1 + (1 - q0) z / 2), q0 = -0.55.
    assert h0 == pytest.approx(2.99792458e5 * 0.05 / dl * (1 + 0.775 * 0.05), rel=2e-3)


def test_standard_disc_spectra_have_the_textbook_slopes():
    lam = np.array([2500.0, 3500.0, 5000.0])
    t1 = 3e5  # hot enough for the nu^(1/3) regime to reach 2500 A
    flux = disc_sed.disc_fnu_at_1cm(lam, 0.0, t1, 0.75, 0.0, M_BH)
    shape = disc_sed.response_sed_shape(lam, t1, 0.75, M_BH)
    nu = 1.0 / lam
    # A T ~ r^-3/4 disc: f_nu ~ nu^(1/3), and so is its lamppost response.
    assert np.polyfit(np.log(nu), np.log(flux), 1)[0] == pytest.approx(1 / 3, abs=0.04)
    assert np.polyfit(np.log(nu), np.log(shape), 1)[0] == pytest.approx(1 / 3, abs=0.04)


def _constructed_echofit(ebv_galactic=0.0, ebv_intrinsic=0.0, flux_unit="mJy", flux_scale=1.0, dce_fraction=None,
                         variable_sed="mean"):
    """An EchoFit whose 'posterior' (a few identical draws) describes light
    curves made from a model disc at D_TRUE_MPC, a host galaxy (none in the
    bluest band) and dust, all known. ``variable_sed="mean"``: the variable
    flux is proportional to the mean disc flux (the flux-flux assumption);
    ``"response"``: it follows the disc's lamppost response spectrum."""
    t = np.linspace(0.0, 100.0, 60)
    ef = EchoFit(M_BH=M_BH, redshift=Z, flux_unit=flux_unit, flux_scale=flux_scale, ebv_galactic=ebv_galactic)
    for n, lam in LAMS.items():
        ef.add_lightcurve(n, lam, t, np.ones_like(t), np.full_like(t, 0.01))
    ef.build_grid(tau_max=20.0)
    n_draws, n_freq = 4, len(ef.freqs)
    rng = np.random.default_rng(0)
    S_f, C_f = rng.normal(size=n_freq) * 0.3, rng.normal(size=n_freq) * 0.3
    t_fine = jnp.linspace(0.0, 100.0, 1500)
    x = np.asarray(driver_at(jnp.asarray(S_f), jnp.asarray(C_f), ef.freqs, t_fine))
    x0 = x.min() - 2.0  # the disc's zero point, below the faint state
    names = list(LAMS)
    lam_rest = np.array([LAMS[n] for n in names])
    lam_obs = lam_rest * (1 + Z)
    t1 = float(disk_t1_kelvin(LOG_MDOT, M_BH))
    disc_true = disc_sed.disc_fnu_at_1cm(lam_rest, Z, t1, 0.75, INCL, M_BH) / (D_TRUE_MPC * disc_sed.MPC_CM) ** 2
    disc_true = disc_true * 10 ** (-0.4 * disc_sed.ccm_a_lambda(lam_rest, ebv_intrinsic))
    host = np.array([0.0, 1.0, 2.0, 3.0])  # mJy, none in the bluest band
    dimming = 10 ** (-0.4 * disc_sed.ccm_a_lambda(lam_obs, ebv_galactic)) / disc_sed.to_mjy(lam_obs, flux_unit, flux_scale)
    share = np.ones(len(names)) if dce_fraction is None else 1.0 - np.asarray(dce_fraction)
    samples = {"log_mdot": np.full(n_draws, LOG_MDOT), "inclination": np.full(n_draws, INCL),
               "S": np.tile(S_f, (n_draws, 1)), "C": np.tile(C_f, (n_draws, 1))}
    shape = disc_sed.response_sed_shape(lam_rest, t1, 0.75, M_BH)
    for k, n in enumerate(names):
        s_band = disc_true[k] / share[k] / (x.mean() - x0) * dimming[k]  # light-curve units per unit X
        if variable_sed == "response":
            s_band = disc_true[0] / (x.mean() - x0) * shape[k] / shape[0] * dimming[k]
        samples[f"S_{n}"] = np.full(n_draws, s_band)
        # Mean flux = host + disc whatever the variable SED's shape.
        samples[f"C_{n}"] = np.full(n_draws, (host[k] + disc_true[k] / share[k]) * dimming[k] - s_band * x.mean())
        if dce_fraction is not None:
            samples[f"dce_fraction_{n}"] = np.full(n_draws, dce_fraction[k])
    ef.samples = samples
    return ef, host


@pytest.mark.parametrize("kwargs", [
    {},
    {"ebv_galactic": 0.08},
    {"flux_unit": "f_lambda", "flux_scale": 1e-15},
    {"dce_fraction": [0.0, 0.3, 0.2, 0.1]},
])
def test_recovers_the_distance_host_and_h0_of_a_constructed_disc(kwargs):
    ef, host = _constructed_echofit(**kwargs)
    s = ef.disc_sed_analysis()
    assert s["dl_mpc"][1] == pytest.approx(D_TRUE_MPC, rel=1e-3)
    assert s["dl_mpc_flux_flux"][1] == pytest.approx(D_TRUE_MPC, rel=1e-3)  # exact when variable ~ mean
    assert s["h0"][1] == pytest.approx(float(disc_sed.h0_from_dl(D_TRUE_MPC, Z)), rel=1e-3)
    for k, b in enumerate(s["bands"].values()):
        assert b["dl_mpc"][1] == pytest.approx(D_TRUE_MPC, rel=1e-3)  # every band agrees: the SED's shape is right
        assert b["constant_mjy"][1] == pytest.approx(host[k], abs=1e-6)
    assert s["zero_point_below_faint_state"] == 1.0


def test_flux_flux_estimator_is_biased_when_the_variable_sed_is_the_lamppost_response():
    ef, host = _constructed_echofit(variable_sed="response")
    s = ef.disc_sed_analysis()
    assert s["dl_mpc"][1] == pytest.approx(D_TRUE_MPC, rel=1e-3)  # host band alone: exact
    for k, b in enumerate(s["bands"].values()):
        assert b["constant_mjy"][1] == pytest.approx(host[k], abs=1e-6)
    # The response is bluer than the disc, so flux-flux leaves disc light in the
    # redder bands' "host" and puts the disc too far away.
    assert s["dl_mpc_flux_flux"][1] > 1.02 * D_TRUE_MPC


def test_fits_intrinsic_reddening_jointly_with_the_distance():
    ef, _ = _constructed_echofit(ebv_galactic=0.02, ebv_intrinsic=0.1)
    plain = ef.disc_sed_analysis(distance_method="flux_flux")
    assert plain["dl_mpc"][1] > 1.1 * D_TRUE_MPC  # ignoring the dust, the disc looks further away
    with pytest.raises(ValueError, match="flux_flux"):
        ef.disc_sed_analysis(fit_intrinsic_ebv=True)
    s = ef.disc_sed_analysis(fit_intrinsic_ebv=True, distance_method="flux_flux")
    assert s["ebv_intrinsic"][1] == pytest.approx(0.1, abs=1e-4)
    assert s["dl_mpc"][1] == pytest.approx(D_TRUE_MPC, rel=1e-3)


def test_warns_when_the_fluxes_are_not_absolute():
    ef, _ = _constructed_echofit()
    # Mean-subtracting the host band moves the zero point (where its model flux
    # vanishes) above the mean state, so every band's disc flux turns negative.
    ef.samples["C_uv"] = ef.samples["C_uv"] - 50.0
    with pytest.warns(UserWarning, match="not positive"):
        ef.disc_sed_analysis()


def test_needs_a_redshift_and_a_disc_model():
    with pytest.raises(ValueError, match="redshift"):
        EchoFit(M_BH=M_BH, sed_analysis=True)
    with pytest.raises(ValueError, match="flux_unit"):
        EchoFit(M_BH=M_BH, flux_unit="Jy")
    ef, _ = _constructed_echofit()
    ef.redshift = None
    with pytest.raises(ValueError, match="redshift"):
        ef.disc_sed_analysis()
    del ef.samples["log_mdot"]
    with pytest.raises(ValueError, match="disc model"):
        ef.disc_sed_analysis(redshift=Z)


def _small_fit(**kwargs):
    data = generate_synthetic_dataset(M_BH=M_BH, log_mdot_true=0.3, n_obs_per_band=40, t_span=120,
                                      noise_level=0.03, seed=1)
    ef = EchoFit(M_BH=M_BH, redshift=Z, sed_analysis=True, **kwargs)
    for n, d in data["bands"].items():
        # Absolute fluxes: the synthetic light curves are mean-subtracted.
        ef.add_lightcurve(n, d["wavelength"], d["t"], np.asarray(d["y"]) + 5.0, d["yerr"])
    ef.build_grid(tau_max=40)
    return ef


def test_runs_after_optimise_and_appears_in_the_report(tmp_path):
    from pycream2 import reporting

    ef = _small_fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ef.optimise(num_samples=60, num_restarts=1)
    assert ef.disc_sed is not None
    assert np.isfinite(ef.disc_sed["summary"]["dl_mpc"][1])
    html = reporting.generate_report(ef, tmp_path).read_text()
    assert "luminosity distance" in html and (tmp_path / "disc_sed.png").exists()


def test_a_failure_only_warns_and_keeps_the_fit():
    ef = _small_fit(sed_options={"host_band": "no_such_band"})
    with pytest.warns(UserWarning, match="disc SED analysis failed"):
        ef.optimise(num_samples=20, num_restarts=1)
    assert ef.samples is not None and ef.disc_sed is None


def test_sed_settings_persist_across_resume(tmp_path):
    options = {"fit_intrinsic_ebv": True, "distance_method": "flux_flux"}
    ef = _small_fit(flux_unit="f_lambda", flux_scale=1e-15, ebv_galactic=0.03,
                    sed_options=options, title="sed_resume", output_dir=str(tmp_path))
    ef.fit(num_warmup=5, num_samples=4, checkpoint_every=2, progress_bar=False, generate_report=False)
    resumed = EchoFit.resume("sed_resume", output_dir=str(tmp_path))
    assert (resumed.redshift, resumed.flux_unit, resumed.flux_scale, resumed.ebv_galactic) == (Z, "f_lambda", 1e-15, 0.03)
    assert resumed.sed_analysis is True and resumed.sed_options == options
