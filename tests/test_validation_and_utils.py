"""
Targeted tests for real coverage gaps found via `pytest --cov=pycream2`
(see README's "Tests and coverage"): EchoFit's own input-validation/
guard-clause paths, run_manager.py's filesystem/serialisation helpers,
synthetic.py's default-argument and observing-gap paths, grid_utils.py's
degenerate cases, and a couple of inference.py/echofit.py branches no
other test happened to exercise (max_tree_depth passthrough, a
checkpointed run with a driver light curve, the chains.nc best-effort
fallback). Not an attempt at 100% coverage -- see that section for which
gaps were left alone deliberately (defensive branches that can't happen
given this codebase's own guarantees).
"""

import warnings

import numpy as np
import pytest

from pycream2 import grid_utils, run_manager
from pycream2.echofit import EchoFit
from pycream2.synthetic import generate_synthetic_dataset, generate_free_lag_dataset


# -- EchoFit validation / guard clauses ------------------------------------

def test_add_lightcurve_rejects_unknown_lag_mode():
    ef = EchoFit(M_BH=1e8)
    with pytest.raises(ValueError, match="lag_mode must be"):
        ef.add_lightcurve("g", wavelength=4770.0, t=[0.0], y=[0.0], yerr=[0.1], lag_mode="nonsense")


def test_build_grid_period_max_sets_the_lowest_driver_frequency():
    t = np.linspace(0.0, 100.0, 101)
    ef = EchoFit(M_BH=1e8)
    ef.add_lightcurve("g", wavelength=4770.0, t=t, y=np.sin(t / 10.0), yerr=np.full_like(t, 0.1))
    ef.build_grid(n_freq=10, n_tau=30)
    default_max = float(ef.freqs.max())
    assert np.isclose(float(ef.freqs.min()), 2.0 * np.pi / 100.0)
    ef.build_grid(n_freq=10, n_tau=30, period_max=400.0)
    assert np.isclose(float(ef.freqs.min()), 2.0 * np.pi / 400.0)
    assert np.isclose(float(ef.freqs.max()), default_max)  # the upper bound is unchanged


def test_build_grid_without_any_lightcurve_raises():
    ef = EchoFit(M_BH=1e8)
    with pytest.raises(ValueError, match="Add at least one light curve"):
        ef.build_grid()


@pytest.mark.parametrize("method_name", [
    "plot_power_spectrum", "plot_mcmc_diagnostics", "plot_corner",
    "plot_corner_bands", "plot_corner_free_lag", "plot_fourier_correlation",
    "plot_lightcurve_fits",
])
def test_plot_methods_raise_before_fit(method_name):
    ef = EchoFit(M_BH=1e8)
    ef.add_lightcurve("g", wavelength=4770.0, t=[0.0, 1.0], y=[0.0, 0.1], yerr=[0.1, 0.1])
    with pytest.raises(RuntimeError, match="Call .fit\\(\\)"):
        getattr(ef, method_name)()


def test_plot_bof_raises_before_fit():
    """plot_bof has its own message (no potential_energy yet), not the
    generic "Call .fit()" one the other plot_* guards share."""
    ef = EchoFit(M_BH=1e8)
    ef.add_lightcurve("g", wavelength=4770.0, t=[0.0, 1.0], y=[0.0, 0.1], yerr=[0.1, 0.1])
    with pytest.raises(RuntimeError, match="potential_energy"):
        ef.plot_bof()


def test_fit_auto_builds_grid_if_not_called_explicitly():
    data = generate_synthetic_dataset(
        M_BH=1e8, bands={"g": 4770.0}, n_obs_per_band=15, n_freq=6, n_tau=30, seed=0,
    )
    ef = EchoFit(M_BH=1e8)
    d = data["bands"]["g"]
    ef.add_lightcurve("g", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    assert ef.freqs is None
    ef.fit(num_warmup=5, num_samples=5, progress_bar=False)  # no build_grid() call
    assert ef.freqs is not None and ef.tau_grid is not None


def test_max_tree_depth_is_passed_through(monkeypatch):
    """Regression check that inference._build_kernel actually receives and
    applies max_tree_depth, not just accepts it silently."""
    import pycream2.inference as inference_mod

    seen = {}
    real_nuts = inference_mod.NUTS

    def spy_nuts(model, **kwargs):
        seen.update(kwargs)
        return real_nuts(model, **kwargs)

    monkeypatch.setattr(inference_mod, "NUTS", spy_nuts)

    data = generate_synthetic_dataset(
        M_BH=1e8, bands={"g": 4770.0}, n_obs_per_band=15, n_freq=6, n_tau=30, seed=0,
    )
    ef = EchoFit(M_BH=1e8)
    d = data["bands"]["g"]
    ef.add_lightcurve("g", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=6, n_tau=30)
    ef.fit(num_warmup=5, num_samples=5, max_tree_depth=4, progress_bar=False)
    assert seen.get("max_tree_depth") == 4


def test_dense_mass_is_passed_through_to_nuts(monkeypatch):
    """Regression check that inference._build_kernel actually forwards
    dense_mass to NUTS -- see CLAUDE.md decision #17: with the default
    diagonal mass matrix, NUTS was found to spend nearly every sample
    pinned at max_tree_depth's ceiling on this model (confirmed directly:
    mean ~858 leapfrog steps/sample); dense_mass=True cut that ~7.5x with
    no loss of recovery accuracy, so it needs to actually reach NUTS, not
    just be accepted and silently dropped."""
    import pycream2.inference as inference_mod

    seen = {}
    real_nuts = inference_mod.NUTS

    def spy_nuts(model, **kwargs):
        seen.update(kwargs)
        return real_nuts(model, **kwargs)

    monkeypatch.setattr(inference_mod, "NUTS", spy_nuts)

    data = generate_synthetic_dataset(
        M_BH=1e8, bands={"g": 4770.0}, n_obs_per_band=15, n_freq=6, n_tau=30, seed=0,
    )
    ef = EchoFit(M_BH=1e8)
    d = data["bands"]["g"]
    ef.add_lightcurve("g", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=6, n_tau=30)
    ef.fit(num_warmup=5, num_samples=5, dense_mass=True, progress_bar=False)
    assert seen.get("dense_mass") is True


def test_checkpointed_run_with_driver_persists_and_resumes(tmp_path):
    """A registered driver light curve is saved/reloaded across a
    checkpointed run + resume, and a non-default num_chains warns rather
    than silently being ignored."""
    data = generate_free_lag_dataset(
        lines={"line_a": 8.0}, n_obs_per_line=15, n_obs_driver=20,
        include_driver=True, n_freq=6, n_tau=30, tau_max=25.0, noise_level=0.05, seed=0,
    )
    ef = EchoFit(M_BH=None, title="driver_checkpoint_test", output_dir=str(tmp_path))
    d = data["bands"]["line_a"]
    ef.add_lightcurve("line_a", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"], lag_mode="free")
    ef.add_driver_lightcurve(t=data["driver"]["t"], y=data["driver"]["y"], yerr=data["driver"]["yerr"])
    ef.build_grid(n_freq=6, n_tau=30, tau_max=25.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ef.fit(rng_seed=0, num_warmup=5, num_samples=10, checkpoint_every=10, num_chains=2, progress_bar=False)
    assert any("num_chains=1" in str(w.message) for w in caught)
    assert (ef.run_dir / "driver.npz").exists()

    ef2 = EchoFit.resume("driver_checkpoint_test", output_dir=str(tmp_path))
    assert ef2.driver_data is not None
    np.testing.assert_allclose(ef2.driver_data["t"], data["driver"]["t"])


def test_dense_mass_persists_across_resume(tmp_path):
    """dense_mass is part of the checkpointed path's persisted fit_config
    (CLAUDE.md decision #17), same as max_tree_depth/checkpoint_every --
    a resumed run must keep using it, not silently fall back to the
    diagonal-mass-matrix default partway through a run."""
    data = generate_synthetic_dataset(
        M_BH=1e8, bands={"g": 4770.0}, n_obs_per_band=15, n_freq=6, n_tau=30, seed=0,
    )
    ef = EchoFit(M_BH=1e8, title="dense_mass_resume_test", output_dir=str(tmp_path))
    d = data["bands"]["g"]
    ef.add_lightcurve("g", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=6, n_tau=30)
    ef.fit(rng_seed=0, num_warmup=5, num_samples=10, checkpoint_every=5, dense_mass=True, progress_bar=False)
    assert ef._fit_config["dense_mass"] is True

    ef2 = EchoFit.resume("dense_mass_resume_test", output_dir=str(tmp_path))
    assert ef2._fit_config["dense_mass"] is True
    ef2.fit(progress_bar=False)
    assert len(ef2.samples["log_mdot"]) == 10


def test_dense_mass_is_the_default(monkeypatch):
    """CLAUDE.md decision #21 flipped the default to a dense mass matrix
    (~17x more effective samples per second than diagonal on this model)."""
    import pycream2.inference as inference_mod

    seen = {}
    real_nuts = inference_mod.NUTS

    def spy_nuts(model, **kwargs):
        seen.update(kwargs)
        return real_nuts(model, **kwargs)

    monkeypatch.setattr(inference_mod, "NUTS", spy_nuts)
    data = generate_synthetic_dataset(M_BH=1e8, bands={"g": 4770.0}, n_obs_per_band=15, n_freq=6, n_tau=30, seed=0)
    ef = EchoFit(M_BH=1e8)
    d = data["bands"]["g"]
    ef.add_lightcurve("g", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=6, n_tau=30)
    ef.fit(num_warmup=5, num_samples=5, progress_bar=False)
    assert seen.get("dense_mass") is True


def test_resumed_run_without_recorded_dense_mass_stays_diagonal(tmp_path):
    """A checkpoint written before dense_mass was persisted must not switch
    to the new dense default partway through its run."""
    import json

    data = generate_synthetic_dataset(M_BH=1e8, bands={"g": 4770.0}, n_obs_per_band=15, n_freq=6, n_tau=30, seed=0)
    ef = EchoFit(M_BH=1e8, title="old_manifest_test", output_dir=str(tmp_path))
    d = data["bands"]["g"]
    ef.add_lightcurve("g", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=6, n_tau=30)
    ef.fit(rng_seed=0, num_warmup=5, num_samples=5, checkpoint_every=5, dense_mass=False,
           progress_bar=False, generate_report=False)
    manifest_path = ef.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["fit_config"]["dense_mass"]
    manifest_path.write_text(json.dumps(manifest))

    ef2 = EchoFit.resume("old_manifest_test", output_dir=str(tmp_path))
    ef2.fit(num_samples=10, progress_bar=False, generate_report=False)
    assert ef2._fit_config.get("dense_mass", False) is False


def test_save_chains_falls_back_gracefully_when_netcdf_write_fails(tmp_path, monkeypatch):
    """The ArviZ/netCDF write in _save_chains is documented as best-effort
    (CLAUDE.md) -- confirm the fallback actually works, not just that the
    code has a try/except around it."""
    import arviz as az

    data = generate_synthetic_dataset(
        M_BH=1e8, bands={"g": 4770.0}, n_obs_per_band=15, n_freq=6, n_tau=30, seed=0,
    )
    ef = EchoFit(M_BH=1e8, title="netcdf_fallback_test", output_dir=str(tmp_path))
    d = data["bands"]["g"]
    ef.add_lightcurve("g", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=6, n_tau=30)
    ef.fit(num_warmup=5, num_samples=5, checkpoint_every=5, progress_bar=False)
    # .fit() with title= already called _save_chains once (successfully) --
    # remove that chains.nc so the monkeypatched call below is a clean
    # check of the failure path, not just "the old file is still there".
    (ef.run_dir / "chains.nc").unlink(missing_ok=True)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated netCDF backend failure")

    monkeypatch.setattr(az, "from_dict", boom)
    with pytest.warns(UserWarning, match="Could not write chains.nc"):
        ef._save_chains(ef.run_dir)
    assert (ef.run_dir / "chains.npz").exists()
    assert not (ef.run_dir / "chains.nc").exists()


# -- run_manager.py ---------------------------------------------------------

def test_resolve_output_root_priority(tmp_path, monkeypatch):
    monkeypatch.delenv(run_manager.ENV_VAR, raising=False)
    assert run_manager.resolve_output_root(str(tmp_path)) == tmp_path  # explicit arg wins

    monkeypatch.setenv(run_manager.ENV_VAR, str(tmp_path / "from_env"))
    assert run_manager.resolve_output_root(None) == tmp_path / "from_env"

    monkeypatch.delenv(run_manager.ENV_VAR, raising=False)
    assert run_manager.resolve_output_root(None) == run_manager.Path("outputs")


def test_find_run_dir_raises_for_missing_title_and_run_id(tmp_path):
    with pytest.raises(FileNotFoundError, match="No runs found"):
        run_manager.find_run_dir(tmp_path, "nonexistent_title")

    title_dir = tmp_path / "some_title"
    title_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No run_\\* directories"):
        run_manager.find_run_dir(tmp_path, "some_title")

    (title_dir / "run_20260101_000000").mkdir()
    with pytest.raises(FileNotFoundError, match="not found"):
        run_manager.find_run_dir(tmp_path, "some_title", run_id="run_99990101_000000")


def test_json_default_converts_numpy_types_and_rejects_others():
    assert run_manager._json_default(np.int64(3)) == 3
    assert run_manager._json_default(np.float32(1.5)) == pytest.approx(1.5)
    assert run_manager._json_default(np.array([1, 2, 3])) == [1, 2, 3]
    with pytest.raises(TypeError, match="Not JSON serialisable"):
        run_manager._json_default(object())


def test_driver_npz_round_trip(tmp_path):
    driver = dict(t=np.array([0.0, 1.0, 2.0]), y=np.array([0.1, 0.2, 0.3]), yerr=np.array([0.01, 0.01, 0.01]))
    path = tmp_path / "driver.npz"
    run_manager.save_driver_npz(path, driver)
    loaded = run_manager.load_driver_npz(path)
    for key in driver:
        np.testing.assert_allclose(loaded[key], driver[key])


# -- synthetic.py -------------------------------------------------------

def test_generate_synthetic_dataset_default_bands():
    data = generate_synthetic_dataset(n_obs_per_band=10, n_freq=6, n_tau=30, seed=0)
    assert set(data["bands"]) == {"u", "g", "r", "i", "z"}


def test_generate_synthetic_dataset_respects_observing_gaps():
    gaps = [(50.0, 14.0), (150.0, 21.0)]
    data = generate_synthetic_dataset(
        bands={"g": 4770.0}, t_span=200.0, n_obs_per_band=60, n_freq=6, n_tau=30,
        gaps=gaps, seed=0,
    )
    t = data["bands"]["g"]["t"]
    for start, duration in gaps:
        assert not np.any((t > start) & (t < start + duration))


def test_generate_free_lag_dataset_default_lines():
    data = generate_free_lag_dataset(n_obs_per_line=10, n_obs_driver=15, n_freq=6, n_tau=30, seed=0)
    assert set(data["bands"]) == {"line_a", "line_b", "line_c"}


# -- grid_utils.py --------------------------------------------------------

def test_estimate_dt_min_degenerate_single_point_per_band_uses_t_span():
    dt_min = grid_utils.estimate_dt_min([np.array([1.0]), np.array([2.0])], t_span=100.0)
    assert dt_min == pytest.approx(100.0)


def test_estimate_dt_min_without_t_span_skips_floor():
    dt_min = grid_utils.estimate_dt_min([np.array([0.0, 1.0, 5.0])])
    assert dt_min == pytest.approx(float(np.percentile([1.0, 4.0], 5.0)))
