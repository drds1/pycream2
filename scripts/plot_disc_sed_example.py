"""
Worked example of the disc SED, distance and H_0 analysis (docs/disc_sed.md):
synthetic light curves given absolute fluxes from a model disc at the distance
H_0 = 70 km/s/Mpc gives at z = 0.03, behind a host galaxy and Galactic dust,
fitted with optimise() and EchoFit(sed_analysis=True). Writes
docs/images/disc_sed_example.png and prints the recovered values against the
truth.

Usage (from the repository root, about a minute):
    MPLBACKEND=Agg poetry run python scripts/plot_disc_sed_example.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

from pycream2 import EchoFit
from pycream2.synthetic import generate_synthetic_dataset, with_disc_fluxes

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "disc_sed_example.png"
REDSHIFT, EBV = 0.03, 0.05
HOST = {"g": 1.0, "r": 2.0, "i": 3.0, "z": 3.5}  # mJy; none in u, the bluest band


def main():
    data = generate_synthetic_dataset(M_BH=1e8, log_mdot_true=0.3, inclination_true=30.0, n_obs_per_band=150,
                                      t_span=250, noise_level=0.03, seed=2)
    data = with_disc_fluxes(data, redshift=REDSHIFT, h0=70.0, host_mjy=HOST, ebv_galactic=EBV)
    ef = EchoFit(M_BH=1e8, redshift=REDSHIFT, ebv_galactic=EBV, sed_analysis=True)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, d["wavelength"], d["t"], d["y"], d["yerr"])
    ef.build_grid(tau_max=60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ef.optimise(num_samples=500, num_restarts=2)
    s, truth = ef.disc_sed["summary"], data["truth"]["disc_sed"]
    print(f"D_L {s['dl_mpc'][1]:.1f} (+{s['dl_mpc'][2] - s['dl_mpc'][1]:.1f}/-{s['dl_mpc'][1] - s['dl_mpc'][0]:.1f}) "
          f"Mpc, truth {truth['dl_mpc']:.1f}")
    print(f"H_0 {s['h0'][1]:.1f} (+{s['h0'][2] - s['h0'][1]:.1f}/-{s['h0'][1] - s['h0'][0]:.1f}), truth {truth['h0']:.1f}")
    ff = s["h0_flux_flux"]
    print(f"H_0 from the flux-flux estimator instead: {ff[1]:.1f} (+{ff[2] - ff[1]:.1f}/-{ff[1] - ff[0]:.1f})")
    for name, b in s["bands"].items():
        print(f"  {name}: host {b['constant_mjy'][1]:.2f} (truth {truth['bands'][name]['host_mjy']:.2f}), "
              f"disc {b['disc_mean_mjy'][1]:.2f} (truth {truth['bands'][name]['disc_mean_mjy']:.2f}) mJy")
    fig, _ = ef.plot_disc_sed()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
