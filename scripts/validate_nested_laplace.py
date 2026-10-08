"""
Validate EchoFit.nested_laplace() against an independent reference: nested
sampling (nautilus; Lange 2023, MNRAS 525, 3181) of the same linear-
marginalised posterior, on synthetic data from scripts/synthetic_recovery_grid.py.
EchoFit.optimise() is run alongside for comparison.

NUTS is not used as the reference: on the weakly constraining cases this is
for, its chains barely move between modes (a single chain jumped once in 300
draws, split R-hat 2.1), so it cannot weigh them. Nested sampling can.

nautilus samples exp(-U(z)), NumPyro's potential in unconstrained coordinates,
inside a uniform box: log_mdot over a fixed wide range, the logit of
cos(inclination) over +/-8 (inclination from ~0.03 degrees to the 80-degree
bound), and each remaining parameter over the range of nested_laplace()'s
proposal draws, widened by its own span plus 2 (in unconstrained units). The
box only has to contain the posterior; its volume does not bias the samples.

Writes <out>/<case>.json (summaries and timings) and <out>/<case>.png.

Usage (from the repository root, after `poetry install --with validation`):
    MPLBACKEND=Agg poetry run python scripts/validate_nested_laplace.py gi 100 1 1 --out experiments/nested_laplace
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_recovery_grid as grid  # noqa: E402

LOG_MDOT_BOX = (-4.0, 14.0)
CHUNK = 256


def _summary(log_mdot, inclination, seconds, **extra):
    q = lambda x: [float(v) for v in np.percentile(x, [2.5, 16, 50, 84, 97.5])]
    return dict(log_mdot_mean=float(np.mean(log_mdot)), log_mdot_sd=float(np.std(log_mdot)), log_mdot_q=q(log_mdot),
                inclination_mean=float(np.mean(inclination)), inclination_sd=float(np.std(inclination)),
                inclination_q=q(inclination), seconds=seconds, **extra)


def reference(ef, nl_result, seed, n_live):
    """nautilus on the marginalised posterior; returns equal-weight draws of
    (log_mdot, inclination) and the run's statistics."""
    from nautilus import Prior, Sampler
    from numpyro.infer.util import initialize_model

    from pycream2.model import reverberation_model

    kwargs = dict(ef._model_kwargs(), marginalise_linear=True)
    info = initialize_model(jax.random.PRNGKey(0), reverberation_model, model_kwargs=kwargs,
                            init_strategy=ef._init_strategy(1))
    from jax.flatten_util import ravel_pytree

    z0, unravel = ravel_pytree(info.param_info.z)
    names = {k: int(np.asarray(v)) for k, v in unravel(jnp.arange(z0.size, dtype=z0.dtype)).items()}
    batch = jax.jit(jax.vmap(lambda z: -info.potential_fn(unravel(z))))
    proposal = np.asarray(nl_result["z_proposal"])
    prior = Prior()
    order = sorted(names, key=names.get)
    for name in order:
        i = names[name]
        if name == "log_mdot":
            lo, hi = LOG_MDOT_BOX
        elif name == "cos_inclination":
            lo, hi = -8.0, 8.0
        else:
            span = np.ptp(proposal[:, i])
            lo, hi = proposal[:, i].min() - span - 2.0, proposal[:, i].max() + span + 2.0
        prior.add_parameter(name, dist=(float(lo), float(hi)))

    def log_likelihood(x):
        x = np.atleast_2d(x)
        out = np.empty(len(x))
        for s in range(0, len(x), CHUNK):
            part = x[s:s + CHUNK]
            pad = np.zeros((CHUNK, x.shape[1]))
            pad[:len(part)] = part
            v = np.asarray(batch(jnp.asarray(pad, dtype=z0.dtype)), dtype=np.float64)[:len(part)]
            out[s:s + len(part)] = np.where(np.isfinite(v), v, -1e30)
        return out

    sampler = Sampler(prior, log_likelihood, n_live=n_live, vectorized=True, pass_dict=False, seed=seed)
    t0 = time.perf_counter()
    sampler.run(verbose=False, discard_exploration=True, n_eff=4000)
    seconds = time.perf_counter() - t0
    points, log_w, _ = sampler.posterior()
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    rng = np.random.default_rng(seed)
    draws = points[rng.choice(len(points), size=4000, p=w)]
    from numpyro.distributions.transforms import biject_to
    import numpyro.distributions as dist

    from pycream2.model import INCLINATION_MAX_DEG

    lm = draws[:, order.index("log_mdot")]
    cos_t = biject_to(dist.Uniform(np.cos(np.deg2rad(INCLINATION_MAX_DEG)), 1.0).support)
    inc = np.rad2deg(np.arccos(np.asarray(cos_t(jnp.asarray(draws[:, order.index("cos_inclination")])))))
    return lm, inc, dict(seconds=seconds, n_like=int(getattr(sampler, "n_like", -1)), ess=float(1.0 / np.sum(w ** 2)),
                         log_evidence=float(sampler.log_z))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("band_set")
    parser.add_argument("snr", type=float)
    parser.add_argument("cadence", type=float)
    parser.add_argument("seed", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-live", type=int, default=2000)
    parser.add_argument("--nuts", type=Path, help="optional NUTS samples (.npz with log_mdot, inclination) to overlay")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    label = grid.config_label(args.band_set, args.snr, args.cadence, args.seed)
    bands = grid.simulate(args.band_set, args.snr, args.cadence, args.seed)
    results = {}

    ef = grid.make_echofit(bands, frequency_grid="auto")
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ef.optimise(rng_seed=args.seed)
    results["optimise"] = _summary(ef.samples["log_mdot"], ef.samples["inclination"], time.perf_counter() - t0,
                                   restarts_agreeing=ef.optimise_timings["restarts_agreeing"],
                                   log_evidence=float(ef.log_evidence))
    draws = {"optimise": (np.asarray(ef.samples["log_mdot"]), np.asarray(ef.samples["inclination"]))}

    ef = grid.make_echofit(bands, frequency_grid="auto")
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ef.nested_laplace(rng_seed=args.seed)
    r = ef.nested_laplace_result
    results["nested_laplace"] = _summary(ef.samples["log_mdot"], ef.samples["inclination"], time.perf_counter() - t0,
                                         k_hat=r["k_hat"], ess=r["ess"], log_evidence=r["log_evidence_is"],
                                         log_evidence_grid=r["log_evidence_grid"], timings=r["timings"])
    draws["nested_laplace"] = (np.asarray(ef.samples["log_mdot"]), np.asarray(ef.samples["inclination"]))

    lm, inc, stats = reference(ef, r, args.seed, args.n_live)
    results["nautilus"] = _summary(lm, inc, stats.pop("seconds"), **stats)
    draws["nautilus"] = (lm, inc)
    if args.nuts and args.nuts.exists():
        d = np.load(args.nuts)
        draws["NUTS (one chain, 400-day grid)"] = (d["log_mdot"], d["inclination"])
        results["nuts_one_chain"] = _summary(d["log_mdot"], d["inclination"], None)

    results["truth"] = dict(log_mdot=grid.TRUE_LOG_MDOT, inclination=grid.TRUE_INCLINATION)
    (args.out / f"{label}.json").write_text(json.dumps(results, indent=1, default=float))

    colours = {"nautilus": "0.15", "nested_laplace": "#2a78d6", "optimise": "#eb6834",
               "NUTS (one chain, 400-day grid)": "#1baf7a"}
    styles = {"nautilus": dict(lw=2.5), "nested_laplace": dict(lw=2), "optimise": dict(lw=2, ls="--"),
              "NUTS (one chain, 400-day grid)": dict(lw=1.5, ls=":")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, k, truth, xlabel in ((axes[0], 0, grid.TRUE_LOG_MDOT, "log_mdot"),
                                 (axes[1], 1, grid.TRUE_INCLINATION, "inclination (degrees)")):
        allv = np.concatenate([d[k] for d in draws.values()])
        bins = np.linspace(np.percentile(allv, 0.2), np.percentile(allv, 99.8), 60)
        for name, d in draws.items():
            ax.hist(d[k], bins=bins, density=True, histtype="step", color=colours[name], label=name, **styles[name])
        ax.axvline(truth, color="0.5", lw=1, zorder=0)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.6)
    axes[0].set_ylabel("posterior density")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{args.band_set}, SNR {args.snr:g}, {args.cadence:g}-day cadence, seed {args.seed} (grey line: truth)")
    fig.tight_layout()
    fig.savefig(args.out / f"{label}.png", dpi=150)
    for name in ("optimise", "nested_laplace", "nautilus"):
        s = results[name]
        print(f"{label} {name}: log_mdot {s['log_mdot_mean']:.2f}+/-{s['log_mdot_sd']:.2f} "
              f"i {s['inclination_mean']:.1f}+/-{s['inclination_sd']:.1f}  {s['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
