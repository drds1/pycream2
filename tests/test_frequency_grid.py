"""
build_grid()'s default driver frequency grid (harmonics of the longest period,
then log spacing, up to f_max cycles per day) and the default linear background,
both October 2026 (CLAUDE.md decision #29). No fits: grid construction only.
"""

import numpy as np
import pytest

from pycream2 import run_manager
from pycream2.echofit import DEFAULT_BACKGROUND_ORDER, F_MAX_CYCLES_PER_DAY, EchoFit
from pycream2.grid_utils import hybrid_frequency_grid


def _echofit(t_end=100.0):
    t = np.arange(0.0, t_end, 1.0)
    ef = EchoFit(M_BH=1e8)
    ef.add_lightcurve("g", wavelength=4770.0, t=t, y=np.sin(t / 10.0), yerr=np.full_like(t, 0.1))
    return ef


def test_hybrid_grid_is_harmonic_then_logarithmic():
    period_max, f_max, log_step = 300.0, 2.0, 0.03
    w = hybrid_frequency_grid(period_max, f_max, log_step)
    dw = 2.0 * np.pi / period_max
    steps = np.diff(w)
    assert np.isclose(w[0], dw)
    assert np.all(steps > 0)
    assert w[-1] <= 2.0 * np.pi * f_max * (1 + 1e-9)
    assert w[-1] * (1 + log_step) > 2.0 * np.pi * f_max  # one more step would overshoot
    linear = w[:-1] * log_step < dw
    assert linear.any() and (~linear).any()
    # Harmonics of the longest period below the crossover, a constant ratio above it.
    assert np.allclose(steps[linear], dw)
    assert np.allclose(steps[~linear] / w[:-1][~linear], log_step)
    assert np.allclose(w[: linear.sum() + 1] / dw, np.arange(1, linear.sum() + 2))


def test_hybrid_grid_needs_far_fewer_modes_than_linear_spacing():
    # An NGC 5548-length campaign: linear spacing to 2 cycles/day needs ~2 period_max modes.
    period_max = 2.0 * (289.0 + 144.5)
    n = len(hybrid_frequency_grid(period_max, 2.0, 0.03))
    assert n < 200 < 2.0 * period_max


def test_hybrid_grid_rejects_non_positive_settings():
    with pytest.raises(ValueError):
        hybrid_frequency_grid(100.0, 0.0, 0.03)


def test_build_grid_default_reaches_f_max_whatever_the_cadence():
    for cadence_end in (100.0, 400.0):
        ef = _echofit(cadence_end)
        ef.build_grid(n_tau=30)
        w = np.asarray(ef.freqs)
        f_top = w.max() / (2.0 * np.pi)
        assert F_MAX_CYCLES_PER_DAY / (1 + 0.03) < f_top <= F_MAX_CYCLES_PER_DAY
        assert np.isclose(w.min(), 2.0 * np.pi / (2.0 * (cadence_end - 1.0 + 0.5 * (cadence_end - 1.0))))


def test_build_grid_f_max_and_dt_min_set_the_upper_frequency():
    ef = _echofit()
    ef.build_grid(n_tau=30, f_max=0.5)
    assert np.isclose(float(ef.freqs.max()) / (2 * np.pi), 0.5, rtol=0.03)
    ef.build_grid(n_tau=30, dt_min=0.25)  # w_max = pi / dt_min, i.e. 2 cycles/day
    assert np.isclose(float(ef.freqs.max()) / (2 * np.pi), 2.0, rtol=0.03)


def test_build_grid_n_freq_keeps_the_old_log_grid():
    ef = _echofit()
    ef.build_grid(n_tau=30, n_freq=12, dt_min=1.0)
    w = np.asarray(ef.freqs)
    assert len(w) == 12
    assert np.isclose(w.max(), np.pi / 1.0)
    assert np.allclose(np.diff(np.log(w)), np.log(w[1] / w[0]))


def test_light_curves_get_a_linear_background_by_default():
    ef = _echofit()
    ef.add_driver_lightcurve(t=[0.0, 1.0, 2.0], y=[1.0, 2.0, 1.0], yerr=[0.1, 0.1, 0.1])
    assert DEFAULT_BACKGROUND_ORDER == 1
    assert ef.bands["g"]["background_order"] == 1
    assert ef.driver_data["background_order"] == 1


def test_checkpoints_saved_before_the_background_default_resume_without_one(tmp_path):
    # A checkpoint written before background_order existed has no such key; it
    # must reload as 0 (its model), not as today's default.
    t = np.arange(5.0)
    np.savez(tmp_path / "bands.npz", names=np.array(["g"]), g__t=t, g__y=t, g__yerr=t + 1, g__wavelength=np.asarray(4770.0))
    np.savez(tmp_path / "driver.npz", t=t, y=t, yerr=t + 1)
    assert run_manager.load_bands_npz(tmp_path / "bands.npz")["g"]["background_order"] == 0
    assert run_manager.load_driver_npz(tmp_path / "driver.npz")["background_order"] == 0
