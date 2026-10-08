"""
plot_performance_analysis.py
============================

Regenerates every chart and number in ``docs/performance_improvements.md``:
the before/after analysis of (1) precomputing the fixed Fourier/lag trig
matrices, (2) the dense NUTS mass matrix as the default, (3) integrating the
linear parameters out analytically (``marginalise_linear``) and the direct
``EchoFit.optimise()`` solve built on it, and (4) the true-blackbody disk
colours.

Everything is measured under ``jax.jit`` (see CLAUDE.md decision #9 for why
eager timings mislead), one run at a time, on one fixed synthetic dataset,
and written to ``docs/images/perf/`` (PNGs plus ``results.json``, which the
doc's tables quote). Run it on an otherwise idle machine: anything else
competing for the CPU skews the wall-clock numbers.

Usage
-----
    python scripts/plot_performance_analysis.py          # ~15 minutes
    python scripts/plot_performance_analysis.py --quick  # shorter NUTS runs, for a smoke check
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpyro import handlers
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model

from pycream2.echofit import EchoFit
from pycream2.forward_model import (
    disk_temperature_profile, driver_at, lag_scaling, transfer_coeffs, transfer_matrices,
    _schwarzschild_radius_light_days,
)
from pycream2.grid_utils import graded_tau_grid
from pycream2.model import reverberation_model
from pycream2.plotting import blackbody_to_colour
from pycream2.synthetic import generate_synthetic_dataset

BANDS = {"u": 3500.0, "g": 4770.0, "r": 6200.0, "i": 7625.0, "z": 9000.0}
M_BH = 1.0e8
KEY_PARAMS = ("log_mdot", "cos_inclination", "sigma_drw", "S_g")

# One fixed colour per entity, identical in every chart.
COLOURS = {
    "rebuilt": "#8a8984",          # model variant: trig matrices rebuilt every step (before)
    "sampled": "#2a78d6",          # model variant: precomputed, everything sampled
    "marginalised": "#1baf7a",     # model variant: precomputed + linear params marginalised
    "sampled_diag": "#eb6834",     # fit: sampled model, diagonal mass (old default)
    "sampled_dense": "#2a78d6",    # fit: sampled model, dense mass (new default)
    "marginalised_dense": "#1baf7a",
    "laplace": "#4a3aa7",          # fit: EchoFit.optimise() direct solve
}
LABELS = {
    "rebuilt": "Trig matrices rebuilt every step (before)",
    "sampled": "Precomputed trig matrices (after)",
    "marginalised": "Precomputed + linear params marginalised (QR)",
    "sampled_diag": "NUTS, diagonal mass (old default)",
    "sampled_dense": "NUTS, dense mass (new default)",
    "marginalised_dense": "NUTS, marginalised, dense mass",
    "laplace": "optimise(): L-BFGS + Laplace, no MCMC",
}
FITS = ("sampled_diag", "sampled_dense", "marginalised_dense")


def _style(ax):
    ax.grid(alpha=0.6, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _bench(f, *args, n=200):
    jax.block_until_ready(f(*args))
    t0 = time.perf_counter()
    for _ in range(n):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n * 1e3  # ms


def _make_echofit(data, marginalise=False):
    ef = EchoFit(M_BH=M_BH, marginalise_linear=marginalise)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid()
    return ef


def _strip_precomputation(kwargs):
    out = dict(kwargs, transfer_mats=None)
    out["bands"] = {n: {k: v for k, v in d.items() if k != "basis"} for n, d in kwargs["bands"].items()}
    return out


# ----------------------------------------------------------------------
def gradient_costs(data):
    """Cost of one potential-energy + gradient evaluation (one NUTS leapfrog
    step) for each model variant."""
    out = {}
    for variant in ("rebuilt", "sampled", "marginalised"):
        ef = _make_echofit(data, marginalise=(variant == "marginalised"))
        kwargs = ef._model_kwargs()
        if variant == "rebuilt":
            kwargs = _strip_precomputation(kwargs)
        info = initialize_model(jax.random.PRNGKey(0), reverberation_model, model_kwargs=kwargs)
        vg = jax.jit(jax.value_and_grad(info.potential_fn))
        n_dims = int(sum(np.size(v) for v in info.param_info.z.values()))
        out[variant] = dict(ms_per_eval=_bench(vg, info.param_info.z), n_sampled_dims=n_dims)
    return out


def transfer_scaling():
    """transfer_coeffs cost vs n_tau, rebuilt vs precomputed matrices."""
    freqs = jnp.asarray(np.geomspace(0.01, 5.0, 60), dtype=jnp.float32)
    rows = []
    for n_tau in (100, 200, 400, 800, 1600):
        tau = jnp.asarray(graded_tau_grid(100.0, n_tau, power=3.0), dtype=jnp.float32)
        psi = jnp.exp(-tau / 2.0)
        mats = transfer_matrices(tau, freqs)
        slow = jax.jit(lambda p, tau=tau: transfer_coeffs(tau, p, freqs))
        fast = jax.jit(lambda p, tau=tau, mats=mats: transfer_coeffs(tau, p, freqs, matrices=mats))
        rows.append(dict(n_tau=n_tau, rebuilt_ms=_bench(slow, psi), precomputed_ms=_bench(fast, psi)))
    return rows


def likelihood_accuracy(data):
    """The marginal log-likelihood's float32 accuracy, three ways, against a
    float64 reference, on this dataset's own (whitened) linear system at one
    fixed set of nonlinear parameters: Cholesky of P = I + M^T D^-1 M with
    the textbook quadratic form, Cholesky with the residual quadratic form,
    and QR of [D^-1/2 M; I] with the residual form (as implemented)."""
    ef = _make_echofit(data)
    kwargs = ef._model_kwargs()
    nonlinear = dict(sigma_drw=jnp.asarray(0.5), log_mdot=jnp.asarray(0.0), cos_inclination=jnp.asarray(0.7))
    nonlinear.update({f"S_{n}": jnp.asarray(1.0) for n in BANDS})
    names = list(BANDS)
    n_freq = kwargs["freqs"].shape[0]

    def predictions(S_raw, C_raw, offsets):
        values = dict(nonlinear, S_raw=S_raw, C_raw=C_raw, **{f"C_{n}": o for n, o in zip(names, offsets)})
        t = handlers.trace(handlers.substitute(handlers.seed(reverberation_model, 0), values)).get_trace(**kwargs)
        return jnp.concatenate([t[f"y_pred_{n}"]["value"] for n in names])

    zeros = (jnp.zeros(n_freq), jnp.zeros(n_freq), jnp.zeros(len(names)))
    J = jax.jacobian(predictions, argnums=(0, 1, 2))(*zeros)
    M = np.concatenate([np.asarray(j, dtype=np.float64) for j in J], axis=1)
    M[:, 2 * n_freq:] *= 5.0
    y = np.concatenate([np.asarray(kwargs["bands"][n]["y"], dtype=np.float64) for n in names])
    sigma = np.concatenate([np.asarray(kwargs["bands"][n]["yerr"], dtype=np.float64) for n in names])
    Mw64, yw64 = M / sigma[:, None], y / sigma
    P64 = np.eye(M.shape[1]) + Mw64.T @ Mw64
    theta64 = np.linalg.solve(P64, Mw64.T @ yw64)
    r64 = yw64 - Mw64 @ theta64
    ref = -0.5 * (r64 @ r64 + theta64 @ theta64 + np.linalg.slogdet(P64)[1])

    def chol(Mw, yw, textbook):
        p = Mw.shape[1]
        L = jnp.linalg.cholesky(jnp.eye(p) + Mw.T @ Mw)
        b = Mw.T @ yw
        th = jax.scipy.linalg.cho_solve((L, True), b)
        r = yw - Mw @ th
        quad = (yw @ yw - b @ th) if textbook else (r @ r + th @ th)
        return -0.5 * (quad + 2 * jnp.sum(jnp.log(jnp.diag(L))))

    def qr(Mw, yw):
        p = Mw.shape[1]
        Q, R = jnp.linalg.qr(jnp.concatenate([Mw, jnp.eye(p)], 0))
        th = jax.scipy.linalg.solve_triangular(R, Q[: Mw.shape[0]].T @ yw, lower=False)
        r = yw - Mw @ th
        return -0.5 * (r @ r + th @ th + 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(R)))))

    Mw, yw = jnp.asarray(Mw64, jnp.float32), jnp.asarray(yw64, jnp.float32)
    variants = {
        "cholesky_textbook": lambda a, b: chol(a, b, True),
        "cholesky_residual": lambda a, b: chol(a, b, False),
        "qr_residual": qr,
    }
    grads = {k: jax.grad(f)(Mw, yw) for k, f in variants.items()}
    out = dict(reference_float64=float(ref), condition_number_P=float(np.linalg.cond(P64)),
               n_obs=int(len(y)), n_linear=int(M.shape[1]))
    for k, f in variants.items():
        vg = jax.jit(jax.value_and_grad(f))
        out[k] = dict(
            error=float(vg(Mw, yw)[0]) - float(ref),
            grad_rel_diff_vs_qr=float(jnp.linalg.norm(grads[k] - grads["qr_residual"]) / jnp.linalg.norm(grads["qr_residual"])),
            ms_value_and_grad=_bench(vg, Mw, yw),
        )
    return out


def nuts_runs(data, n):
    """NUTS fits on the same data/seed, warmup and sampling timed apart."""
    out, samples = {}, {}
    for fit in FITS:
        ef = _make_echofit(data, marginalise=(fit == "marginalised_dense"))
        kwargs = ef._model_kwargs()
        mcmc = MCMC(
            NUTS(reverberation_model, init_strategy=ef._init_strategy(1), dense_mass=(fit != "sampled_diag")),
            num_warmup=n, num_samples=n, progress_bar=False,
        )
        t0 = time.perf_counter()
        mcmc.warmup(jax.random.PRNGKey(0), extra_fields=("num_steps",), collect_warmup=True, **kwargs)
        warmup_s = time.perf_counter() - t0
        warmup_steps = int(np.sum(mcmc.get_extra_fields()["num_steps"]))
        t0 = time.perf_counter()
        mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=("num_steps", "diverging"), **kwargs)
        sampling_s = time.perf_counter() - t0
        by_chain = ef._add_linear_draws(mcmc.get_samples(group_by_chain=True), 0)
        extra = mcmc.get_extra_fields()
        ess = {p: float(effective_sample_size(np.asarray(by_chain[p]))) for p in KEY_PARAMS}
        out[fit] = dict(
            warmup_seconds=warmup_s, sampling_seconds=sampling_s, warmup_steps=warmup_steps,
            num_steps=np.asarray(extra["num_steps"]).tolist(), divergences=int(np.sum(extra["diverging"])),
            step_size=float(mcmc.last_state.adapt_state.step_size),
            ess=ess, min_ess=min(ess.values()),
            min_ess_per_sampling_second=min(ess.values()) / sampling_s,
            min_ess_per_total_second=min(ess.values()) / (warmup_s + sampling_s),
        )
        samples[fit] = (ef, {k: np.asarray(v[0]) for k, v in by_chain.items()})
    return out, samples


def laplace_run(data, n):
    """EchoFit.optimise(), one call, as a user would make it (first call in
    the process, so including JIT compilation), with its own stage timings."""
    ef = _make_echofit(data)
    t0 = time.perf_counter()
    ef.optimise(num_samples=n, method="laplace")
    total = time.perf_counter() - t0
    return dict(
        seconds_total=total, **ef.optimise_timings,
        evaluations_best_start=int(ef.optimise_result.nfev), num_samples=n,
    ), (ef, {k: np.asarray(v) for k, v in ef.samples.items()})


# ----------------------------------------------------------------------
def chart_gradient_costs(costs, outdir):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    variants = list(costs)
    vals = [costs[v]["ms_per_eval"] for v in variants]
    ax.barh(range(len(variants)), vals, color=[COLOURS[v] for v in variants], height=0.6)
    for i, (v, x) in enumerate(zip(variants, vals)):
        ax.text(x, i, f"  {x:.2f} ms  ({costs[v]['n_sampled_dims']} sampled dims)", va="center", fontsize=9, color="0.2")
    ax.set_yticks(range(len(variants)), [LABELS[v] for v in variants], fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(right=max(vals) * 8)
    ax.set_xlabel("ms per potential + gradient evaluation (log scale)")
    ax.set_title("Cost of one NUTS leapfrog step", loc="left", fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(outdir / "gradient_cost.png", dpi=150)
    plt.close(fig)


def chart_transfer_scaling(rows, outdir):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    n = [r["n_tau"] for r in rows]
    for key, variant in (("rebuilt_ms", "rebuilt"), ("precomputed_ms", "sampled")):
        v = [r[key] for r in rows]
        ax.plot(n, v, "-o", color=COLOURS[variant], lw=2, ms=6, label=LABELS[variant])
        ax.annotate(f"{v[-1]:.3f} ms", (n[-1], v[-1]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8, color="0.2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(n, [str(x) for x in n])
    ax.set_xlim(right=n[-1] * 1.8)
    ax.set_xlabel("lag grid points n_tau (n_freq = 60)")
    ax.set_ylabel("ms per transfer_coeffs call")
    ax.set_title("Transfer-coefficient cost vs lag-grid size", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(outdir / "transfer_scaling.png", dpi=150)
    plt.close(fig)


def chart_steps(nuts, outdir):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    edges = np.arange(0, 12) - 0.5
    for fit in FITS:
        depth = np.log2(np.asarray(nuts[fit]["num_steps"]) + 1)
        counts, _ = np.histogram(depth, bins=edges)
        ax.plot(np.arange(0, 11), counts, "-o", color=COLOURS[fit], lw=2, ms=6,
                label=f"{LABELS[fit]}: median {int(np.median(nuts[fit]['num_steps']))} steps")
    ax.set_xticks(range(0, 11), [f"{2 ** d - 1}" for d in range(0, 11)], fontsize=8)
    ax.set_xlabel("leapfrog steps per sample (1023 = max_tree_depth ceiling)")
    ax.set_ylabel("samples")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
    ax.set_title("Trajectory length per NUTS sample", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(outdir / "steps_per_sample.png", dpi=150)
    plt.close(fig)


def chart_time_to_posterior(nuts, laplace, outdir):
    """Wall time to a usable posterior, split warmup/sampling, one bar each."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    rows = list(FITS) + ["laplace"]
    for i, r in enumerate(rows):
        if r == "laplace":
            solve = laplace["lbfgs_seconds"]
            ax.barh(i, solve, color=COLOURS[r], height=0.6)
            ax.barh(i, laplace["seconds_total"] - solve, left=solve, color=COLOURS[r], alpha=0.35, height=0.6)
            total = laplace["seconds_total"]
            note = f"{solve:.0f} s L-BFGS + {total - solve:.0f} s Hessian and {laplace['num_samples']} exact draws"
        else:
            w, s = nuts[r]["warmup_seconds"], nuts[r]["sampling_seconds"]
            ax.barh(i, w, color=COLOURS[r], alpha=0.35, height=0.6)
            ax.barh(i, s, left=w, color=COLOURS[r], height=0.6)
            total, note = w + s, f"{w:.0f} s warmup + {s:.0f} s sampling; min ESS {nuts[r]['min_ess']:.0f}"
        ax.text(total, i, f"  {note}", va="center", fontsize=8, color="0.2")
    ax.set_yticks(range(len(rows)), [LABELS[r] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(right=max(n["warmup_seconds"] + n["sampling_seconds"] for n in nuts.values()) * 2.1)
    ax.set_xlabel("wall-clock seconds, including JIT compilation\n(pale: NUTS warmup, or optimise()'s Hessian + draws)")
    ax.set_title("Time to a posterior, same data", loc="left", fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(outdir / "time_to_posterior.png", dpi=150)
    plt.close(fig)


def chart_ess_per_second(nuts, outdir):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(KEY_PARAMS))
    w = 0.26
    for i, fit in enumerate(FITS):
        v = [nuts[fit]["ess"][p] / nuts[fit]["sampling_seconds"] for p in KEY_PARAMS]
        ax.bar(x + (i - 1) * w, v, width=w - 0.02, color=COLOURS[fit], label=LABELS[fit])
    ax.set_yscale("log")
    ax.set_xticks(x, KEY_PARAMS)
    ax.set_ylabel("ESS per sampling second (log)")
    ax.set_ylim(top=ax.get_ylim()[1] * 6)
    ax.set_title("Sampling efficiency once warmed up", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(outdir / "ess_per_second.png", dpi=150)
    plt.close(fig)


def chart_posteriors(fits, truth_log_mdot, outdir):
    params = [("log_mdot", "log_mdot"), ("inclination", "inclination (deg)"), ("sigma_drw", "sigma_drw"), ("C_g", "C_g")]
    order = list(FITS) + ["laplace"]
    fig, axes = plt.subplots(1, len(params), figsize=(12, 3.0))
    for ax, (p, label) in zip(axes, params):
        lo = min(np.percentile(fits[f][1][p], 0.5) for f in order)
        hi = max(np.percentile(fits[f][1][p], 99.5) for f in order)
        bins = np.linspace(lo, hi, 30)
        for f in order:
            ax.hist(fits[f][1][p], bins=bins, density=True, histtype="step", lw=2, color=COLOURS[f], label=LABELS[f])
        if p == "log_mdot":
            ax.axvline(truth_log_mdot, color="0.2", lw=1, ls="--")
        ax.set_xlabel(label, fontsize=9)
        ax.set_yticks([])
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Posterior agreement across all four methods (dashed: true log_mdot)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "posterior_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_driver(fits, outdir):
    ef = fits["sampled_dense"][0]
    t_all = np.concatenate([d["t"] for d in ef.bands.values()])
    t = jnp.linspace(t_all.min(), t_all.max(), 400)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for f in ("sampled_dense", "laplace"):
        ef_f, s = fits[f]
        X = np.asarray(jax.vmap(lambda S, C: driver_at(S, C, ef_f.freqs, t))(jnp.asarray(s["S"]), jnp.asarray(s["C"])))
        lo, med, hi = np.percentile(X, [2.5, 50, 97.5], axis=0)
        tt = np.asarray(t)
        if f == "sampled_dense":  # filled envelope, solid median
            ax.fill_between(tt, lo, hi, color=COLOURS[f], alpha=0.25, lw=0)
            ax.plot(tt, med, color=COLOURS[f], lw=2, label=f"{LABELS[f]}: median (solid), 95% envelope (filled)")
        else:  # dashed on top, so both stay visible where they coincide
            ax.plot(tt, med, color=COLOURS[f], lw=1.5, ls=(0, (4, 3)), label=f"{LABELS[f]}: median (dashed), 95% bounds (dotted)")
            ax.plot(tt, lo, color=COLOURS[f], lw=1, ls=":")
            ax.plot(tt, hi, color=COLOURS[f], lw=1, ls=":")
    ax.set_xlabel("time (days)")
    ax.set_ylabel("driver X(t)")
    ax.set_title("Inferred driving light curve: full MCMC vs direct solve", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(outdir / "driver_agreement.png", dpi=150)
    plt.close(fig)


def chart_blackbody(outdir):
    T = np.geomspace(1000.0, 1.0e5, 600)
    rgb = blackbody_to_colour(T)
    r_in = 3.0 * float(_schwarzschild_radius_light_days(M_BH))
    tau_ref = float(lag_scaling(0.0, 5000.0, M_BH))
    r = np.geomspace(r_in * 1.02, 6 * tau_ref, 400)
    T_r = np.asarray(disk_temperature_profile(r, 0.0, 5000.0, M_BH))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 4.8), gridspec_kw={"height_ratios": [1, 2.2]})
    ax0.imshow(rgb[None, :, :], aspect="auto", extent=(np.log10(T[0]), np.log10(T[-1]), 0, 1))
    ax0.set_yticks([])
    ticks = [1e3, 2e3, 3e3, 5e3, 6.5e3, 1e4, 2e4, 5e4, 1e5]
    ax0.set_xticks(np.log10(ticks), [f"{t:,.0f}" for t in ticks], fontsize=8)
    ax0.set_xlabel("blackbody temperature (K)", fontsize=9)
    ax0.set_title("Blackbody colour vs temperature (Planck x CIE 1931 -> sRGB)", loc="left", fontsize=11)

    ax1.set_facecolor("black")
    ax1.scatter(r / tau_ref, T_r, c=blackbody_to_colour(T_r), s=14, lw=0)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.axvline(1.0, color="0.7", lw=1, ls="--")
    ax1.text(1.08, T_r.max() * 0.5, "Wien radius at 5000 A", color="0.85", fontsize=8)
    ax1.set_xlabel("radius / Wien radius at 5000 A", fontsize=9)
    ax1.set_ylabel("T (K)", fontsize=9)
    ax1.set_title("Disk temperature profile (log_mdot = 0), each point in its true colour", loc="left", fontsize=11)
    ax1.grid(alpha=0.3, lw=0.6, color="0.6")
    fig.tight_layout()
    fig.savefig(outdir / "blackbody_colours.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path("docs/images/perf"))
    parser.add_argument("--quick", action="store_true", help="Short NUTS runs (smoke check only).")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_dataset(bands=BANDS, M_BH=M_BH, n_obs_per_band=100, n_freq=60, n_tau=400, noise_level=0.05, seed=0)
    n = 100 if args.quick else 500
    results = dict(settings=dict(n_bands=len(BANDS), n_obs_per_band=100, n_freq=60, n_tau=400,
                                 num_warmup=n, num_samples=n, truth_log_mdot=float(data["truth"]["log_mdot"])))
    print("1/6 gradient cost per leapfrog step ...", flush=True)
    results["gradient_costs"] = gradient_costs(data)
    print("2/6 transfer_coeffs scaling ...", flush=True)
    results["transfer_scaling"] = transfer_scaling()
    print("3/6 marginal likelihood accuracy ...", flush=True)
    results["likelihood_accuracy"] = likelihood_accuracy(data)
    print("4/6 NUTS fits ...", flush=True)
    results["nuts"], fits = nuts_runs(data, n)
    print("5/6 optimise() direct solve ...", flush=True)
    results["laplace"], fits["laplace"] = laplace_run(data, 1000)
    results["posterior_summary"] = {
        f: {p: [float(np.mean(fits[f][1][p])), float(np.std(fits[f][1][p]))] for p in ("log_mdot", "inclination", "sigma_drw", "S_g", "C_g")}
        for f in fits
    }
    print("6/6 charts ...", flush=True)
    chart_gradient_costs(results["gradient_costs"], args.outdir)
    chart_transfer_scaling(results["transfer_scaling"], args.outdir)
    chart_steps(results["nuts"], args.outdir)
    chart_time_to_posterior(results["nuts"], results["laplace"], args.outdir)
    chart_ess_per_second(results["nuts"], args.outdir)
    chart_posteriors(fits, results["settings"]["truth_log_mdot"], args.outdir)
    chart_driver(fits, args.outdir)
    chart_blackbody(args.outdir)
    (args.outdir / "results.json").write_text(json.dumps(results, indent=2))
    printable = {k: v for k, v in results.items() if k != "nuts"}
    printable["nuts"] = {f: {k: v for k, v in r.items() if k != "num_steps"} for f, r in results["nuts"].items()}
    print(json.dumps(printable, indent=1))


if __name__ == "__main__":
    main()
