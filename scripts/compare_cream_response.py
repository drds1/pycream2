"""
Compare pycream2's thin-disc response function with the original CREAM
Fortran's (``tfbx`` in ``cream_f90.f90``) for the same disc, and write the
figures and numbers behind ``docs/cream_response_comparison.md``.

CREAM's ``tfbx`` and the helpers it calls are extracted verbatim from the
Fortran source, compiled with gfortran next to a small driver, and run on
the same lag grid as ``forward_model.thin_disk_response``. Both codes are
given the same disc: black-hole mass, T_1 at one light-day, temperature slope
3/4, inner radius 3 R_S, wavelength and inclination, with CREAM's own
smoothing convention (a Gaussian as wide as the lag-grid spacing) and, for
the temperature, the viscous term only (CREAM's irradiation temperature
T1i = 0; its response weight still carries the lamppost dilution).

Usage (needs gfortran and a checkout of pycecream):
    MPLBACKEND=Agg poetry run python scripts/compare_cream_response.py \\
        --cream-source ../pycecream/pycecream/cream_f90.f90
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from pycream2.forward_model import disk_t1_kelvin, thin_disk_response

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "images"
M_BH = 10 ** 7.51  # NGC 5548 (Pancoast et al. 2014)
T1 = 19400.0  # K at one light-day: NGC 5548's thin-disc fit
WAVELENGTH = 5000.0  # Angstrom
INCLINATIONS = [0.0, 30.0, 60.0, 80.0]
TAU_MAX = 10.0  # days
GRIDS = {"coarse": 201, "fine": 2001}  # lag-grid points (0.05 d and 0.005 d spacing)
ROUTINES = ["subroutine tfbx", "function f_rs2ld", "subroutine tr4visc", "subroutine tr4irad",
            "function ranu", "function ran3", "subroutine myedrat2mdot"]
END = re.compile(r"^\s*end(\s+(subroutine|function)(\s+\w+)?)?\s*(!.*)?$", re.I)

DRIVER = """
program driver
  implicit none
  integer :: ntau, i
  real :: taumax, wav, t1v, t1i, sv, si, embh, deginc, rinsch
  real, allocatable :: tau(:), psi(:)
  read(*,*) ntau, taumax, wav, t1v, t1i, sv, si, embh, deginc, rinsch
  allocate(tau(ntau), psi(ntau))
  do i = 1, ntau
    tau(i) = taumax * real(i - 1) / real(ntau - 1)
  end do
  call tfbx(tau, ntau, wav, t1v, t1i, sv, si, embh, deginc, rinsch, psi, 0.0)
  do i = 1, ntau
    write(*,'(2es16.8)') tau(i), psi(i)
  end do
end program driver
"""


def build_cream(source: Path, work: Path) -> Path:
    """Extract tfbx and its helpers verbatim, compile with a driver."""
    lines = source.read_text().split("\n")
    parts = []
    for name in ROUTINES:
        start = next(k for k, line in enumerate(lines)
                     if re.match(rf"^\s*{name.split()[0]}\s+{name.split()[1]}\b", line, re.I))
        end = start + 1
        while not END.match(lines[end]):
            end += 1
        parts.append("\n".join(lines[start:end + 1]))
    (work / "tfbx.f90").write_text("\n\n".join(parts) + "\n")
    (work / "driver.f90").write_text(DRIVER)
    exe = work / "tfbx_driver"
    subprocess.run(["gfortran", "-O2", "-ffree-line-length-none", "-o", str(exe),
                    str(work / "tfbx.f90"), str(work / "driver.f90")], check=True)
    return exe


def cream_psi(exe: Path, ntau: int, inclination: float) -> tuple[np.ndarray, np.ndarray]:
    args = f"{ntau} {TAU_MAX} {WAVELENGTH} {T1} 0 0.75 0.75 {M_BH} {inclination} 1\n"
    out = subprocess.run([str(exe)], input=args, capture_output=True, text=True, check=True).stdout
    data = np.array([[float(x) for x in line.split()] for line in out.strip().split("\n")])
    return data[:, 0], data[:, 1]


def pycream_psi(tau: np.ndarray, inclination: float, log_mdot: float, **kwargs) -> np.ndarray:
    # Viscous-only, as CREAM is run here (irradiation temperature 0): the
    # comparison of docs/cream_response_comparison.md predates irradiation
    # becoming pycream2's default (October 2026).
    kwargs.setdefault("include_irradiation", False)
    return np.asarray(thin_disk_response(jnp.asarray(tau), log_mdot, WAVELENGTH, inclination, M_BH, **kwargs))


def stats(tau, psi):
    """Mean and median delay, and the lag of the peak, of a response on a grid."""
    w = np.clip(psi, 0, None)
    cdf = np.cumsum(w) / np.sum(w)
    return dict(mean=float(np.sum(tau * w) / np.sum(w)), median=float(tau[np.searchsorted(cdf, 0.5)]),
                peak=float(tau[np.argmax(w)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cream-source", type=Path, required=True, help="path to CREAM's cream_f90.f90")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_mdot = brentq(lambda x: float(disk_t1_kelvin(x, M_BH)) - T1, -10, 15)
    results = {"log_mdot": log_mdot, "T1": T1, "M_BH": M_BH, "wavelength": WAVELENGTH, "grids": {}}
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_cream(args.cream_source, Path(tmp))
        fig, axes = plt.subplots(2, len(INCLINATIONS), figsize=(11, 5), sharex=True)
        for row, (grid, ntau) in enumerate(GRIDS.items()):
            results["grids"][grid] = {}
            for col, inc in enumerate(INCLINATIONS):
                tau, c = cream_psi(exe, ntau, inc)
                dtau = tau[1] - tau[0]
                # CREAM's smoothing and CREAM's delay (no lamppost height in it),
                # for the like-for-like comparison; then pycream2's defaults, the
                # old one (a Gaussian in tau, 0.1 of the Wien radius, and h_x = 0 in
                # the delay, before September 2026) and the current one (causal
                # log-normal smoothing and the exact lamppost delay).
                p_matched = pycream_psi(tau, inc, log_mdot, smoothing_days=dtau, delay_lamppost_height=False)
                p_old = pycream_psi(tau, inc, log_mdot, smoothing_frac=0.1, delay_lamppost_height=False)
                p_default = pycream_psi(tau, inc, log_mdot)
                # CREAM normalises to the peak and forces psi[0] = 0; show pycream2 the
                # same way for the shape comparison.
                cn = c / c.max()
                pm = p_matched / p_matched.max()
                po = p_old / p_old.max()
                pd = p_default / p_default.max()
                results["grids"][grid][f"{inc:.0f}"] = dict(
                    dtau=float(dtau), cream=stats(tau, c), pycream2_matched=stats(tau, p_matched),
                    pycream2_old_default=stats(tau, p_old), pycream2_default=stats(tau, p_default),
                    pycream2_default_psi0_over_peak=float(pd[0]), pycream2_old_psi0_over_peak=float(po[0]),
                    cream_psi1_over_peak=float(cn[1]), pycream2_psi0_over_peak=float(pm[0]),
                    pycream2_psi1_over_peak=float(pm[1]),
                    max_abs_diff_after_first_bin=float(np.max(np.abs(cn[1:] - pm[1:]))),
                )
                ax = axes[row, col]
                ax.plot(tau, cn, color="k", lw=1.4, label="CREAM tfbx")
                ax.plot(tau, pm, color="tab:red", lw=1.0, ls="--", label="pycream2, CREAM's smoothing and delay")
                ax.plot(tau, po, color="tab:blue", lw=0.8, ls=":", label="pycream2, old default")
                ax.plot(tau, pd, color="tab:green", lw=1.0, ls="-.", label="pycream2, new default")
                ax.set_xlim(0, 6)
                ax.grid(alpha=0.6, lw=0.4)
                if row == 0:
                    ax.set_title(f"i = {inc:.0f} deg")
                if col == 0:
                    ax.set_ylabel(f"psi / peak\n({grid} grid, {dtau:.3f} d)")
                if row == 1:
                    ax.set_xlabel("tau (days)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(OUT_DIR / "cream_response_comparison.png", dpi=150)
        plt.close(fig)
    (OUT_DIR / "cream_response_comparison.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
