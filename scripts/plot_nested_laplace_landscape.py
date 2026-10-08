"""
Badness-of-Fit landscapes of log_mdot against inclination (EchoFit.plot_landscape)
for synthetic cases from scripts/synthetic_recovery_grid.py, with the truth and
EchoFit.optimise()'s peak marked: the figures in docs/nested_laplace.md.

Usage (from the repository root):
    MPLBACKEND=Agg poetry run python scripts/plot_nested_laplace_landscape.py \
        --out docs/images/nested_laplace gi:100:1:1 gi:100:1:2
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_recovery_grid as grid  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="+", help="BAND_SET:SNR:CADENCE:SEED")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    truth = dict(log_mdot=grid.TRUE_LOG_MDOT, inclination=grid.TRUE_INCLINATION)
    for case in args.cases:
        band_set, snr, cadence, seed = case.split(":")
        snr, cadence, seed = float(snr), float(cadence), int(seed)
        bands = grid.simulate(band_set, snr, cadence, seed)
        ef = grid.make_echofit(bands, frequency_grid="auto")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ef.optimise(rng_seed=seed)
            optimum = ef.optimum
            ef.nested_laplace(rng_seed=seed)
        ef.optimum = optimum
        r, smp = ef.nested_laplace_result, ef.samples
        print(f"{case}: log_mdot {smp['log_mdot'].mean():.2f}+/-{smp['log_mdot'].std():.2f} "
              f"inclination {smp['inclination'].mean():.1f}+/-{smp['inclination'].std():.1f} "
              f"k_hat {r['k_hat']:.2f} ess {r['ess']:.0f} grid seconds {r['timings']['total_seconds']:.0f}", flush=True)
        fig, _ = ef.plot_landscape(truth=truth)
        fig.suptitle(f"{band_set}, SNR {snr:g}, {cadence:g}-day cadence, seed {seed}")
        path = args.out / f"landscape_{grid.config_label(band_set, snr, cadence, seed)}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(path, flush=True)


if __name__ == "__main__":
    main()
