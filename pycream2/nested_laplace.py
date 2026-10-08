"""
nested_laplace.py
=================

A nested Laplace approximation (after Rue, Martino & Chopin 2009, INLA) for
the few nonlinear parameters whose posterior is not Gaussian, with a Pareto-
smoothed importance-sampling correction (Vehtari et al. 2024; Berild et al.
2022's IS-INLA). Used by :meth:`pycream2.EchoFit.nested_laplace`.

Why: ``EchoFit.optimise()`` fits one Gaussian at one peak. On weakly
constraining data (e.g. g and i at SNR 100) the posterior of ``log_mdot`` and
inclination is a long, curved ridge or two separate modes, and a single
Gaussian understates the uncertainty several-fold, or sits on one mode
(``scripts/synthetic_recovery_grid.py``). The difficulty is confined to those
few "outer" parameters; the rest (``sigma_drw``, the band gains, any error-model
terms) are close to Gaussian given them, and the driver's Fourier coefficients
and the offsets are integrated out exactly (``model._linear_marginal``).

How:

1. The outer parameters (default ``log_mdot``, ``cos_inclination`` and, when
   fitted, ``temperature_slope``) are gridded in their *constrained*
   coordinates. At each grid point the inner parameters are optimised (damped
   Newton, warm-started from a solved neighbour), and the Laplace approximation
   of the integral over them gives the log marginal posterior of the outer ones:
   ``log p(theta | y) = -U* - log|d theta / d z| + d/2 log 2 pi - 1/2 log det H``,
   with ``U`` NumPyro's potential in unconstrained coordinates ``z``.
2. Mode finding: a scout grid over the prior range (bounded parameters over
   their whole support, unbounded ones such as ``log_mdot`` over +/-3.3 prior
   sd) locates candidate regions, and a full optimisation from each of its
   local maxima, and from the data-anchored start, climbs to every mode's peak,
   however narrow. A coarse grid's values alone are not enough: at SNR 1000 a
   0.01-dex mode fell between scout points and a zoom that trusted them settled
   on a broader, wrong mode. Modes are weighed by their Laplace evidence.
   Then a box around the kept modes is refined pass by pass (small grids solved
   without exact Hessians, ``log det H`` held at the start's): extended where
   the posterior reaches an edge that isn't a hard bound, shrunk while it fills
   too little of the box.
3. The final grid solves a sparse sub-lattice (every fourth point) with exact
   Hessians; every other point takes its nearest sub-lattice point's Hessian,
   for its Newton steps and a first-order shift of the inner optimum, and the
   ``log det H`` term is interpolated over the sub-lattice. So a grid point
   costs a few gradient evaluations.
4. Draws: a grid cell by its posterior mass, the outer parameters uniformly
   within it, the inner ones from the conditional Gaussian. Each is weighted by
   the exact potential over its proposal density; Pareto smoothing gives the
   ``k_hat`` diagnostic (reliable below ~0.7) and the draws are resampled by
   the smoothed weights, which corrects what the grid and the inner Gaussians
   miss. The cell is chosen from the grid's log density interpolated onto
   ``upsample``-times finer sub-cells, which follows a narrow ridge within a
   grid cell. A fraction ``mode_fraction`` of the draws comes from a Gaussian at
   each mode instead (defensive importance sampling, Hesterberg 1995), with
   every draw weighted against the whole mixture, so a mode narrower than the
   grid's cells is still sampled, with correct weight. Both the inner
   conditionals and the mode components are multivariate Student-t (``t_dof``
   degrees of freedom) rather than Gaussian, the standard defence against a
   proposal with lighter tails than the target (the skewed posterior of a noise
   scale, say), which is what drives ``k_hat`` up.

Costs are in ``docs/nested_laplace.md``.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp

DEFAULT_GRID_PARAMS = ("log_mdot", "cos_inclination", "temperature_slope")
# Modes are kept (and scout maxima tried) more generously than the final grid
# is sized: a mode within MODE_DROP of the best in Laplace log evidence is kept.
# The final box only needs the region within ``drop`` (default 8) of the peak,
# which leaves out ~0.03 per cent of a Gaussian's mass; a looser cut let a
# broad plateau at ~e^-10 relative density (responses running far beyond the
# lag grid at log_mdot ~10-15) stretch the grid to ~1 dex per row.
MODE_DROP = 12.0


def _transforms(model, kwargs, names):
    """NumPyro's constrained<->unconstrained transform for each named site."""
    from numpyro import handlers
    from numpyro.distributions.transforms import biject_to

    trace = handlers.trace(handlers.seed(model, 0)).get_trace(**kwargs)
    return {n: biject_to(trace[n]["fn"].support) for n in names}


def _prior_sds(model, kwargs, names):
    """Each named site's prior standard deviation (NaN where it has none)."""
    from numpyro import handlers

    trace = handlers.trace(handlers.seed(model, 0)).get_trace(**kwargs)
    out = []
    for n in names:
        try:
            out.append(float(np.sqrt(trace[n]["fn"].variance)))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def _support_bounds(transform) -> tuple:
    lo = float(transform(jnp.asarray(-np.inf)))
    hi = float(transform(jnp.asarray(np.inf)))
    return lo, hi


class _Problem:
    """The potential split into outer (gridded) and inner (Laplace) parts."""

    def __init__(self, model, kwargs, info, grid_params):
        from jax.flatten_util import ravel_pytree

        self.model, self.kwargs = model, kwargs
        self.z0, self.unravel = ravel_pytree(info.param_info.z)
        self.dtype = self.z0.dtype
        index = self.unravel(jnp.arange(self.z0.size, dtype=self.dtype))
        sizes = {k: int(np.size(v)) for k, v in index.items()}
        self.names = [n for n in grid_params if n in index and sizes[n] == 1]
        if not self.names:
            raise ValueError(f"none of {grid_params} is a scalar sample site of this model")
        self.outer = np.array([int(np.asarray(index[n])) for n in self.names])
        self.inner = np.array([i for i in range(self.z0.size) if i not in set(self.outer)])
        self.transforms = _transforms(model, kwargs, self.names)
        potential = lambda z: info.potential_fn(self.unravel(z))
        self.potential = jax.jit(potential)
        self.batch_potential = jax.jit(jax.vmap(potential))

        def assemble(zr, zo):
            return jnp.zeros_like(self.z0).at[self.inner].set(zr).at[self.outer].set(zo)

        self.assemble = assemble
        inner_pot = lambda zr, zo: potential(assemble(zr, zo))
        self.inner_vg = jax.jit(jax.value_and_grad(inner_pot))
        self.full_hessian = jax.jit(jax.hessian(potential))
        # The inner rows of the Hessian only (forward derivatives along the
        # inner directions of the gradient): gives H_rr and H_ro, which is all
        # a grid point needs, for d_inner rather than d_total directions.
        grad = jax.grad(potential)
        self.inner_hessian = jax.jit(
            lambda zr, zo: jax.jacfwd(lambda r: grad(assemble(r, zo)))(zr)
        )
        self.n_grad = 0
        self.n_hess = 0

    # constrained outer values theta <-> unconstrained zo, and log |d theta / d zo|
    def to_unconstrained(self, theta):
        return np.array([float(self.transforms[n].inv(jnp.asarray(t))) for n, t in zip(self.names, theta)])

    def log_jacobian(self, zo, theta):
        return sum(float(self.transforms[n].log_abs_det_jacobian(jnp.asarray(z), jnp.asarray(t)))
                   for n, z, t in zip(self.names, zo, theta))

    def vg(self, zr, zo):
        self.n_grad += 1
        f, g = self.inner_vg(jnp.asarray(zr, dtype=self.dtype), jnp.asarray(zo, dtype=self.dtype))
        return float(f), np.asarray(g, dtype=np.float64)

    def hessian(self, zr, zo):
        """Full Hessian at (zr, zo); returns the inner block H_rr and the cross block H_ro."""
        self.n_hess += 1
        cols = np.asarray(self.inner_hessian(jnp.asarray(zr, dtype=self.dtype), jnp.asarray(zo, dtype=self.dtype)),
                          dtype=np.float64)  # (d_total, d_inner): d grad / d zr
        h_rr = cols[self.inner]
        return 0.5 * (h_rr + h_rr.T), cols[self.outer].T


def _inner_solve(prob: _Problem, zr, zo, h_rr, tol=1e-2, max_iter=30):
    """Minimise the potential over the inner parameters at fixed outer zo by
    damped Newton steps with the (possibly stale) Hessian h_rr, from zr.
    Converged when the Newton step is below ``tol`` inner posterior sds.
    Returns (zr, U, converged)."""
    vals, vecs = np.linalg.eigh(h_rr)
    vals = np.abs(vals)
    vals = np.clip(vals, 1e-6 * max(vals.max(), 1e-12), None)
    f, g = prob.vg(zr, zo)
    if not np.isfinite(f):
        return zr, np.inf, False
    for _ in range(max_iter):
        ge = vecs.T @ g
        if np.max(np.abs(ge) / np.sqrt(vals)) < tol:
            return zr, f, True
        damping, accepted = 0.0, False
        for _ in range(12):
            step = vecs @ (ge / (vals + damping))
            f_new, g_new = prob.vg(zr - step, zo)
            if np.isfinite(f_new) and f_new < f:
                accepted = True
                break
            damping = max(10.0 * damping, 1e-3 * vals.max())
        if not accepted:
            return zr, f, False
        zr, f, g = zr - step, f_new, g_new
    return zr, f, False


def _solve_point(prob, theta, zr_start, h_rr, exact_hessian, rough=False):
    """Inner optimum and log marginal density at constrained outer values theta.
    ``rough`` (search passes, which only need the density to ~1 unit): a looser
    tolerance, capped iterations and no exact-Hessian retry."""
    zo = prob.to_unconstrained(theta)
    if not np.all(np.isfinite(zo)):
        return None
    # Search passes need the density to ~1 unit; elsewhere 0.05 inner sd changes
    # it by ~0.001, far below anything that matters, at half the Newton steps
    # of a tighter tolerance. Exact-Hessian points (the Laplace volume term and
    # the neighbours' references) are converged harder.
    tol, max_iter = (0.1, 8) if rough else ((1e-2, 30) if exact_hessian else (0.05, 30))
    zr, f, ok = _inner_solve(prob, np.asarray(zr_start, dtype=np.float64), zo, h_rr, tol=tol, max_iter=max_iter)
    if not ok and np.isfinite(f) and not rough:
        # A stale Hessian can stall; retry once with the exact one here.
        h_rr = prob.hessian(zr, zo)[0]
        zr, f, ok = _inner_solve(prob, zr, zo, h_rr)
    if not np.isfinite(f):
        return None
    out = dict(theta=np.asarray(theta, dtype=np.float64), zo=zo, zr=zr, U=f, converged=ok,
               log_jac=prob.log_jacobian(zo, theta), h_used=h_rr)
    if exact_hessian:
        h_rr, h_ro = prob.hessian(zr, zo)
        h_reg, indefinite = _saddle_free(h_rr)
        out.update(h_rr=h_reg, h_ro=h_ro, logdet=float(np.linalg.slogdet(h_reg)[1]), indefinite=indefinite)
    return out


def _polish(prob, objective, z, max_iter=20, tol=0.01):
    """Damped, saddle-free Newton steps on the full potential from ``z``, each
    accepted only if the potential falls, until the step is below ``tol``
    posterior sds: the same polish as optimise(method="laplace") (CLAUDE.md
    decision #21). L-BFGS alone can stop well short of the peak along
    prior-dominated directions; without this, the demo's diffuse-continuum
    fit found a "mode" ~3400 in log evidence below the real one."""
    z = np.asarray(z, dtype=np.float64)
    f, g = objective(z)
    damping = 0.0
    for _ in range(max_iter):
        h = np.asarray(prob.full_hessian(jnp.asarray(z, dtype=prob.dtype)), dtype=np.float64)
        vals, vecs = np.linalg.eigh(0.5 * (h + h.T))
        vals = np.clip(np.abs(vals), 1e-8 * max(np.abs(vals).max(), 1e-12), None)
        ge = vecs.T @ g
        if np.max(np.abs(vecs @ (ge / vals)) / np.sqrt(np.diag((vecs / vals) @ vecs.T))) < tol:
            break
        for _ in range(15):
            step = vecs @ (ge / (vals + damping))
            f_new, g_new = objective(z - step)
            if f_new < f:
                break
            damping = max(10.0 * damping, 1e-6 * vals.max())
        else:
            break
        z, f, g = z - step, f_new, g_new
        damping /= 10.0
    return z, f


def _saddle_free(h):
    """``h`` with its eigenvalues replaced by their absolute values (floored at
    1e-8 of the largest), and whether any was not positive. An inner Hessian
    need not be positive definite at every grid point (a component pressed
    against a bound, say: first seen on the demo's diffuse-continuum fit), and
    the covariance, the first-order shift and log det H all need it to be; the
    same convention as optimise(method="laplace")'s Newton polish. Points where
    it applies are counted in the timings, and k_hat shows any harm."""
    vals, vecs = np.linalg.eigh(0.5 * (h + h.T))
    indefinite = bool(vals.min() <= 0)
    vals = np.clip(np.abs(vals), 1e-8 * max(np.abs(vals).max(), 1e-12), None)
    return (vecs * vals) @ vecs.T, indefinite


def _log_marginal(point, logdet, d_inner):
    return -point["U"] - point["log_jac"] + 0.5 * d_inner * np.log(2 * np.pi) - 0.5 * logdet


def _solve_points(prob, thetas, idx, start, h_ref, exact=None, rough=False):
    """Solve the points ``thetas`` (constrained, (N, d)) with grid indices
    ``idx``. Points flagged in ``exact`` are solved first, nearest the start
    first, with exact Hessians; the rest then take their nearest exact point's
    Hessian and the first-order shift of its inner optimum (``dzr/dzo = -H_rr^-1
    H_ro``). Without ``exact``, every point is warm-started from its nearest
    solved neighbour with ``h_ref``."""
    n = len(thetas)
    exact = np.zeros(n, bool) if exact is None else np.asarray(exact, bool)
    first = int(np.argmin(np.sum((thetas - start["theta"]) ** 2 / np.maximum(np.var(thetas, axis=0), 1e-12), axis=1)))
    dist0 = np.sum((idx - idx[first]) ** 2, axis=1)
    solved = {}

    def nearest(k, pool):
        pool = list(pool)
        return pool[int(np.argmin(np.sum((idx[pool] - idx[k]) ** 2, axis=1)))]

    for k in [int(k) for k in np.argsort(dist0, kind="stable") if exact[k]]:
        base = solved[nearest(k, solved)] if solved else start
        p = _solve_point(prob, thetas[k], base["zr"], base.get("h_rr", h_ref), exact_hessian=True)
        solved[k] = p if p is not None else _failed(prob, thetas[k], base)
    refs = [k for k in solved if "h_rr" in solved[k] and np.isfinite(solved[k]["U"])]
    for k in [int(k) for k in np.argsort(dist0, kind="stable") if not exact[k]]:
        if refs:
            r = solved[nearest(k, refs)]
            zr0 = r["zr"] - np.linalg.solve(r["h_rr"], r["h_ro"] @ (prob.to_unconstrained(thetas[k]) - r["zo"]))
            if not np.all(np.isfinite(zr0)):
                zr0 = r["zr"]
            h = r["h_rr"]
        else:
            base = solved[nearest(k, solved)] if solved else start
            zr0, h = base["zr"], base.get("h_rr", base.get("h_used", h_ref))
        p = _solve_point(prob, thetas[k], zr0, h, exact_hessian=False, rough=rough)
        solved[k] = p if p is not None else _failed(prob, thetas[k], start)
    return [solved[k] for k in range(n)]


def _failed(prob, theta, base):
    return dict(theta=np.asarray(theta, float), zr=base["zr"], U=np.inf, log_jac=0.0, converged=False,
                zo=prob.to_unconstrained(theta), logdet=np.nan)


def _cell_axis(lo, hi, n):
    """n cell centres spanning [lo, hi]."""
    edges = np.linspace(lo, hi, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _product(axes):
    shape = tuple(len(a) for a in axes)
    idx = np.array(list(np.ndindex(*shape)))
    return np.stack([axes[d][idx[:, d]] for d in range(len(axes))], axis=1), idx, shape


def _region(axes, logp, drop):
    """Per dimension, the cell-edge span of the cells within ``drop`` of the
    peak, and whether it reaches the first/last cell."""
    finite = np.where(np.isfinite(logp), logp, -np.inf)
    keep = finite > finite.max() - drop
    out = []
    for d in range(len(axes)):
        other = tuple(i for i in range(len(axes)) if i != d)
        used = np.where(keep.any(axis=other) if other else keep)[0]
        w = axes[d][1] - axes[d][0] if len(axes[d]) > 1 else 0.0
        out.append((axes[d][used[0]] - 0.5 * w, axes[d][used[-1]] + 0.5 * w, used[0] == 0,
                    used[-1] == len(axes[d]) - 1, w))
    return out


def _interp(sub_axes, values, thetas):
    """Multilinear interpolation of ``values`` (on the product grid ``sub_axes``)
    at ``thetas``; NaNs are first filled with the nearest finite value."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt

    grid = np.array(values, dtype=np.float64)
    good = np.isfinite(grid)
    if not good.any():
        return np.zeros(len(thetas))
    if not good.all():
        _, nearest = distance_transform_edt(~good, return_indices=True)
        grid = grid[tuple(nearest)]
    axes = [a if len(a) > 1 else np.array([a[0] - 1.0, a[0] + 1.0]) for a in sub_axes]
    grid = np.broadcast_to(grid, tuple(len(a) for a in axes))
    f = RegularGridInterpolator(axes, grid, bounds_error=False, fill_value=None)
    return f(np.clip(thetas, [a[0] for a in axes], [a[-1] for a in axes]))


def run(model, kwargs, info, num_samples: int = 1000, rng_seed: int = 0,
        grid_params: Sequence[str] = DEFAULT_GRID_PARAMS, n_pass: Optional[Sequence[int]] = None,
        n_fine: Optional[Sequence[int]] = None, drop: float = 8.0, max_passes: int = 8,
        stride: int = 4, prior_span_sd: float = 3.3, upsample: int = 4, max_seeds: int = 4,
        mode_fraction: float = 0.2, t_dof: float = 6.0, num_restarts: int = 4, restart_scale: float = 0.5) -> dict:
    """The nested Laplace approximation of ``model``'s marginalised posterior;
    see the module docstring. ``info`` is NumPyro's ``initialize_model`` result
    for ``model(**kwargs)``. Returns a dict with the draws (unconstrained ``z``,
    PSIS-resampled), the final grid and its log posterior, ``k_hat``, the
    evidence estimates and timings."""
    import arviz as az
    from scipy.optimize import minimize

    timings = {}
    t_all = time.perf_counter()
    prob = _Problem(model, kwargs, info, grid_params)
    d_out, d_in = len(prob.names), len(prob.inner)
    n_pass = tuple(n_pass or {1: (24,), 2: (12, 9), 3: (8, 6, 5)}.get(d_out, (5,) * d_out))
    n_fine = tuple(n_fine or {1: (80,), 2: (20, 15), 3: (10, 8, 6)}.get(d_out, (6,) * d_out))

    # 1. A starting point: the joint optimum from the data-anchored start.
    t0 = time.perf_counter()
    vg_full = jax.jit(jax.value_and_grad(lambda z: info.potential_fn(prob.unravel(z))))

    def objective(z):
        v, g = vg_full(jnp.asarray(z, dtype=prob.dtype))
        v, g = float(v), np.asarray(g, dtype=np.float64)
        return (v, g) if np.isfinite(v) and np.all(np.isfinite(g)) else (1e10, np.zeros_like(g))

    res = minimize(objective, np.asarray(prob.z0, dtype=np.float64), jac=True, method="L-BFGS-B",
                   options=dict(ftol=1e-7, gtol=1e-3, maxiter=1000))
    z_hat, _ = _polish(prob, objective, res.x)
    theta_hat = np.array([float(prob.transforms[n](jnp.asarray(z_hat[i]))) for n, i in zip(prob.names, prob.outer)])
    h_full = np.asarray(prob.full_hessian(jnp.asarray(z_hat, dtype=prob.dtype)), dtype=np.float64)
    h_full = 0.5 * (h_full + h_full.T)
    h_rr0 = _saddle_free(h_full[np.ix_(prob.inner, prob.inner)])[0]
    logdet0 = np.linalg.slogdet(h_rr0)[1]
    start = dict(theta=theta_hat, zr=z_hat[prob.inner], h_rr=h_rr0)
    timings["start_seconds"] = time.perf_counter() - t0

    # 2. Find every mode. A scout grid over the prior range (bounded parameters
    #    over their support, unbounded ones over +/-prior_span_sd prior sd)
    #    locates candidate regions; a full optimisation from each of its local
    #    maxima, and from the data-anchored start, then climbs to every mode's
    #    true peak however narrow, which a coarse grid's values cannot be trusted
    #    to show. Modes are weighed by their Laplace evidence.
    from scipy.ndimage import maximum_filter

    t0 = time.perf_counter()
    bounds = [_support_bounds(prob.transforms[n]) for n in prob.names]
    unbounded = [not (np.isfinite(lo) and np.isfinite(hi)) for lo, hi in bounds]
    prior_sd = _prior_sds(model, kwargs, prob.names)
    scout_box = []
    for d, (lo, hi) in enumerate(bounds):
        if unbounded[d]:
            half = prior_span_sd * (prior_sd[d] if np.isfinite(prior_sd[d]) else 10.0)
            scout_box.append((theta_hat[d] - half, theta_hat[d] + half))
        else:
            scout_box.append((lo, hi))
    axes = [_cell_axis(lo, hi, n) for (lo, hi), n in zip(scout_box, n_pass)]
    thetas, idx, shape = _product(axes)
    scout = _solve_points(prob, thetas, idx, start, h_rr0, rough=True)
    scout_logp = np.array([_log_marginal(p, logdet0, d_in) for p in scout]).reshape(shape)
    finite = np.where(np.isfinite(scout_logp), scout_logp, -np.inf)
    peaks = np.where((maximum_filter(finite, size=3, mode="nearest") == finite).ravel()
                     & (finite.ravel() > finite.max() - 3 * MODE_DROP))[0]
    peaks = peaks[np.argsort(-finite.ravel()[peaks])][:max_seeds]
    # Seeds: the polished data-anchored optimum, the scout grid's local maxima,
    # and random perturbations of the data-anchored start, as
    # optimise(method="laplace")'s restarts (same scale), so the mode search is
    # never weaker than that solve's. The perturbations matter where the *inner*
    # parameters have optima of their own: on the demo's diffuse-continuum fit
    # the data-anchored start and the scout seeds all led to a basin ~3400 in
    # log evidence below the one a perturbed restart found.
    rng_seeds = np.random.default_rng(rng_seed)
    seeds = [z_hat] + [np.asarray(prob.assemble(jnp.asarray(scout[k]["zr"], dtype=prob.dtype),
                                                jnp.asarray(scout[k]["zo"], dtype=prob.dtype)), dtype=np.float64)
                       for k in peaks]
    seeds += [np.asarray(prob.z0, dtype=np.float64) + restart_scale * rng_seeds.normal(size=prob.z0.size)
              for _ in range(num_restarts - 1)]
    modes = []
    for zs in seeds:
        m = minimize(objective, zs, jac=True, method="L-BFGS-B", options=dict(ftol=1e-7, gtol=1e-3, maxiter=1000))
        if not np.isfinite(m.fun) or m.fun >= 1e9:
            continue
        m.x, m.fun = _polish(prob, objective, m.x)
        h = np.asarray(prob.full_hessian(jnp.asarray(m.x, dtype=prob.dtype)), dtype=np.float64)
        h = 0.5 * (h + h.T)
        vals, vecs = np.linalg.eigh(h)
        if vals.min() <= 0:
            vals = np.clip(np.abs(vals), 1e-8, None)
        cov = (vecs / vals) @ vecs.T
        zo = m.x[prob.outer]
        sd_z = np.sqrt(np.diag(cov)[prob.outer])
        theta = np.array([float(prob.transforms[n](jnp.asarray(z))) for n, z in zip(prob.names, zo)])
        # Outer posterior sd in constrained units, via d theta / d z at the mode.
        dtheta = np.exp([float(prob.transforms[n].log_abs_det_jacobian(jnp.asarray(z), jnp.asarray(t)))
                         for n, z, t in zip(prob.names, zo, theta)])
        mode = dict(theta=theta, sd=sd_z * dtheta, z=m.x, U=float(m.fun), cov=cov, logdet_h=float(np.sum(np.log(vals))),
                    log_evidence=-float(m.fun) - 0.5 * float(np.sum(np.log(vals))))
        if not any(np.all(np.abs(mode["theta"] - o["theta"]) < 0.5 * np.maximum(o["sd"], 1e-6)) for o in modes):
            modes.append(mode)
    if not modes:
        raise RuntimeError("nested_laplace: no mode found")
    top = max(o["log_evidence"] for o in modes)
    modes = [o for o in modes if o["log_evidence"] > top - MODE_DROP]
    timings["scout_points"] = len(scout)
    timings["modes"] = len(modes)

    # 3. Zoom and extend: grid a box around every kept mode (+/-6 sd), then, pass
    #    by pass, extend it wherever the region within ``drop`` of the peak
    #    reaches an edge that isn't a hard bound (a ridge longer than the modes'
    #    local widths), and shrink it while that region fills too little of it.
    box = []
    for d, (lo, hi) in enumerate(bounds):
        a = min(o["theta"][d] - 6 * o["sd"][d] for o in modes)
        b = max(o["theta"][d] + 6 * o["sd"][d] for o in modes)
        box.append((max(a, lo), min(b, hi)))
    best_mode = max(modes, key=lambda o: o["log_evidence"])
    best = dict(theta=best_mode["theta"], zr=best_mode["z"][prob.inner], h_rr=h_rr0)
    passes = [dict(axes=axes, logp=scout_logp)]
    for _ in range(max_passes):
        axes = [_cell_axis(lo, hi, n) for (lo, hi), n in zip(box, n_pass)]
        thetas, idx, shape = _product(axes)
        points = _solve_points(prob, thetas, idx, best, h_rr0, rough=True)
        logp = np.array([_log_marginal(p, logdet0, d_in) for p in points]).reshape(shape)
        passes.append(dict(axes=axes, logp=logp))
        flat = np.where(np.isfinite(logp.ravel()), logp.ravel(), -np.inf)
        best = points[int(np.argmax(flat))]
        region = _region(axes, logp, drop)
        new_box, done = [], True
        for d, (lo, hi, at_lo, at_hi, w) in enumerate(region):
            blo, bhi = bounds[d]
            width = box[d][1] - box[d][0]
            # Mass at an edge that isn't a hard bound: extend by half the box.
            nlo = box[d][0] - 0.5 * width if at_lo and box[d][0] > blo else lo - w
            nhi = box[d][1] + 0.5 * width if at_hi and box[d][1] < bhi else hi + w
            if (at_lo and box[d][0] > blo) or (at_hi and box[d][1] < bhi):
                done = False
            # Within a cell of a hard bound: run to the bound, so the final grid
            # doesn't cut off mass piled against it.
            nlo = blo if nlo < blo + w else nlo
            nhi = bhi if nhi > bhi - w else nhi
            if (nhi - nlo) < 0.6 * width:
                done = False
            new_box.append((nlo, nhi))
        box = new_box
        if done:
            break
    # The modes' own neighbourhoods always stay inside the final box.
    box = [(min(lo, min(o["theta"][d] - 3 * o["sd"][d] for o in modes)),
            max(hi, max(o["theta"][d] + 3 * o["sd"][d] for o in modes))) for d, (lo, hi) in enumerate(box)]
    box = [(max(lo, b[0]), min(hi, b[1])) for (lo, hi), b in zip(box, bounds)]
    timings["search_seconds"] = time.perf_counter() - t0
    timings["search_passes"] = len(passes)

    # 4. Final grid: exact Hessians on a sub-lattice, the rest from it.
    t0 = time.perf_counter()
    fine_axes = [_cell_axis(lo, hi, n) for (lo, hi), n in zip(box, n_fine)]
    thetas, idx, fine_shape = _product(fine_axes)
    sub = [np.array(sorted(set(range(0, n, stride)) | {n - 1})) for n in n_fine]
    exact = np.all([np.isin(idx[:, d], sub[d]) for d in range(d_out)], axis=0)
    fine_points = _solve_points(prob, thetas, idx, best, h_rr0, exact=exact)
    sub_shape = tuple(len(s_) for s_ in sub)
    sub_logdet = np.array([p.get("logdet", np.nan) for p, e in zip(fine_points, exact) if e]).reshape(sub_shape)
    logdet = _interp([fine_axes[d][sub[d]] for d in range(d_out)], sub_logdet, thetas)
    logdet = np.where(exact, [p.get("logdet", np.nan) for p in fine_points], logdet)
    fine_logp = np.array([_log_marginal(p, ld, d_in) for p, ld in zip(fine_points, logdet)]).reshape(fine_shape)
    timings["final_seconds"] = time.perf_counter() - t0
    timings["final_points"] = int(np.prod(fine_shape))
    timings["indefinite_hessians"] = int(sum(bool(p.get("indefinite")) for p in fine_points))

    finite = np.where(np.isfinite(fine_logp), fine_logp, -np.inf)
    edge_drop = np.inf
    for d, (lo, hi) in enumerate(bounds):
        w = fine_axes[d][1] - fine_axes[d][0] if len(fine_axes[d]) > 1 else 0.0
        for end, bound, edge in ((0, lo, fine_axes[d][0] - 0.5 * w), (-1, hi, fine_axes[d][-1] + 0.5 * w)):
            if np.isclose(edge, bound):
                continue
            edge_drop = min(edge_drop, finite.max() - np.take(finite, end, axis=d).max())

    # 5. Draws, by defensive importance sampling (Hesterberg 1995): most from the
    #    grid (a cell by mass, the outer values uniform within it, the inner ones
    #    from the conditional Student-t of the nearest exact point, first-order
    #    shifted), a fraction ``mode_fraction`` from a Student-t at each mode, and
    #    every draw weighted against the whole mixture. A mode narrower than the
    #    grid's cells still gets draws of its own, with correct weights.
    t0 = time.perf_counter()
    rng = np.random.default_rng(rng_seed)
    flat_logp = finite.ravel()
    widths = np.array([a[1] - a[0] if len(a) > 1 else 1.0 for a in fine_axes])
    cell_volume = float(np.prod(widths))
    # Grid part: the grid's log density interpolated (multilinearly, in log
    # space) onto sub-cells ``upsample`` times finer, so a narrow or curved ridge
    # is followed within a cell instead of being smeared across it.
    sub_axes = [_cell_axis(a[0] - 0.5 * w, a[-1] + 0.5 * w, len(a) * upsample) for a, w in zip(fine_axes, widths)]
    sub_thetas, sub_idx, sub_shape = _product(sub_axes)
    sub_logp = _interp(fine_axes, np.where(np.isfinite(fine_logp), fine_logp, finite[np.isfinite(finite)].min() - 50.0),
                       sub_thetas)
    sub_logp = np.where(np.isfinite(sub_logp), sub_logp, -np.inf)
    prob_sub = np.exp(sub_logp - sub_logp.max())
    prob_sub /= prob_sub.sum()
    sub_widths = widths / upsample
    sub_volume = float(np.prod(sub_widths))
    sub_lo = np.array([ax[0] - 0.5 * w for ax, w in zip(sub_axes, sub_widths)])
    ref_index = [k for k in np.where(exact)[0] if "h_rr" in fine_points[k] and np.isfinite(fine_points[k]["U"])]
    if not ref_index:
        raise RuntimeError("nested_laplace: no grid point with a usable Hessian")
    ref_of_cell, cache = {}, {}

    def cell_gaussian(c):
        if c not in ref_of_cell:
            ref_of_cell[c] = ref_index[int(np.argmin(np.sum((idx[ref_index] - idx[c]) ** 2, axis=1)))]
        r_i = ref_of_cell[c]
        if r_i not in cache:
            r = fine_points[r_i]
            cov = np.linalg.inv(r["h_rr"])
            cache[r_i] = (np.linalg.cholesky(0.5 * (cov + cov.T)), np.linalg.slogdet(r["h_rr"])[1],
                          np.linalg.solve(r["h_rr"], r["h_ro"]), r["h_rr"])
        return cache[r_i]

    def t_logpdf(delta2, d, logdet_h):
        """Multivariate Student-t (``t_dof`` degrees of freedom) log density with
        precision matrix H at squared Mahalanobis distance delta2."""
        from scipy.special import gammaln

        return (gammaln(0.5 * (t_dof + d)) - gammaln(0.5 * t_dof) - 0.5 * d * np.log(t_dof * np.pi)
                + 0.5 * logdet_h - 0.5 * (t_dof + d) * np.log1p(delta2 / t_dof))

    def t_draw(d):
        return rng.normal(size=d) * np.sqrt(t_dof / rng.chisquare(t_dof))

    def grid_log_q(theta, zo, zr):
        """Log density of the grid part at (theta, zr), per unit theta and zr."""
        k = np.floor((theta - sub_lo) / sub_widths).astype(int)
        if np.any(k < 0) or np.any(k >= np.array(sub_shape)):
            return -np.inf
        s_flat = int(np.ravel_multi_index(tuple(k), sub_shape))
        if prob_sub[s_flat] <= 0:
            return -np.inf
        c = int(np.ravel_multi_index(tuple(np.minimum(k // upsample, np.array(fine_shape) - 1)), fine_shape))
        _, logdet_r, shift, h = cell_gaussian(c)
        resid = zr - (fine_points[c]["zr"] - shift @ (zo - fine_points[c]["zo"]))
        return np.log(prob_sub[s_flat]) - np.log(sub_volume) + t_logpdf(float(resid @ h @ resid), d_in, logdet_r)

    n_modes = len(modes)
    n_mode_draws = int(round(mode_fraction * num_samples / n_modes)) if n_modes else 0
    n_grid = num_samples - n_modes * n_mode_draws
    frac = np.array([n_grid] + [n_mode_draws] * n_modes, dtype=float) / num_samples
    chol_modes = [np.linalg.cholesky(0.5 * (o["cov"] + o["cov"].T)) for o in modes]

    def modes_log_q_z(z):
        """Log of each mode's (Student-t) density at z (unconstrained, full)."""
        out = []
        for o, L in zip(modes, chol_modes):
            e = np.linalg.solve(L, z - o["z"])
            out.append(t_logpdf(float(e @ e), len(z), o["logdet_h"]))
        return np.array(out)

    picks = rng.choice(len(sub_logp), size=n_grid, p=prob_sub)
    thetas_g = sub_thetas[picks] + sub_widths * rng.uniform(-0.5, 0.5, size=(n_grid, d_out))
    thetas_g = np.clip(thetas_g, [b[0] for b in bounds], [b[1] for b in bounds])
    zo_g = np.stack([np.asarray(prob.transforms[n].inv(jnp.asarray(thetas_g[:, d])), dtype=np.float64)
                     for d, n in enumerate(prob.names)], axis=1)
    z_draws = np.empty((num_samples, prob.z0.size))
    for j in range(n_grid):
        k = np.minimum(sub_idx[picks[j]] // upsample, np.array(fine_shape) - 1)
        c = int(np.ravel_multi_index(tuple(k), fine_shape))
        chol, _, shift, _ = cell_gaussian(c)
        z_draws[j, prob.inner] = fine_points[c]["zr"] - shift @ (zo_g[j] - fine_points[c]["zo"]) + chol @ t_draw(d_in)
        z_draws[j, prob.outer] = zo_g[j]
    j = n_grid
    for o, L in zip(modes, chol_modes):
        for _ in range(n_mode_draws):
            z_draws[j] = o["z"] + L @ t_draw(prob.z0.size)
            j += 1
    zo_all = z_draws[:, prob.outer]
    thetas_all = np.stack([np.asarray(prob.transforms[n](jnp.asarray(zo_all[:, d])), dtype=np.float64)
                           for d, n in enumerate(prob.names)], axis=1)
    thetas_all[:n_grid] = thetas_g  # exact values for the grid part (no float32 round trip)
    log_jac = sum(np.asarray(prob.transforms[n].log_abs_det_jacobian(jnp.asarray(zo_all[:, d]),
                                                                    jnp.asarray(thetas_all[:, d])), dtype=np.float64)
                  for d, n in enumerate(prob.names))
    # Mixture density per unit (theta, zr): a Gaussian in z converts by 1/|d theta/d z|.
    log_q = np.empty(num_samples)
    for j in range(num_samples):
        parts = [np.log(frac[0]) + grid_log_q(thetas_all[j], zo_all[j], z_draws[j, prob.inner])
                 if frac[0] > 0 else -np.inf]
        if n_modes and n_mode_draws:
            parts += list(np.log(frac[1:]) + modes_log_q_z(z_draws[j]) - log_jac[j])
        top = max(parts)
        log_q[j] = top + np.log(np.sum(np.exp(np.array(parts) - top))) if np.isfinite(top) else -np.inf
    u = np.concatenate([np.asarray(prob.batch_potential(jnp.asarray(z_draws[i:i + 250], dtype=prob.dtype)))
                        for i in range(0, num_samples, 250)]).astype(np.float64)
    log_target = -u - log_jac
    log_w = np.where(np.isfinite(log_target), log_target - log_q, -np.inf)
    good = np.isfinite(log_w)
    if not good.any():
        raise RuntimeError("nested_laplace: every draw has a non-finite potential")
    # Draws with a non-finite potential get (effectively) zero weight.
    smoothed, k_hat = az.psislw(np.where(good, log_w, log_w[good].min() - 1000.0))
    weights = np.exp(smoothed - np.max(smoothed))
    weights /= weights.sum()
    resample = rng.choice(num_samples, size=num_samples, p=weights)
    timings["draws_seconds"] = time.perf_counter() - t0
    timings["total_seconds"] = time.perf_counter() - t_all
    timings["gradient_evaluations"] = prob.n_grad
    timings["hessian_evaluations"] = prob.n_hess

    top = log_w[good].max()
    log_z_is = float(np.log(np.sum(np.exp(log_w[good] - top))) + top - np.log(num_samples))
    log_z_grid = float(np.log(np.sum(np.exp(flat_logp - flat_logp.max()))) + flat_logp.max() + np.log(cell_volume))
    return dict(
        z=z_draws[resample], z_proposal=z_draws, log_weights=log_w, k_hat=float(k_hat),
        ess=float(1.0 / np.sum(weights ** 2)), names=prob.names, unravel=prob.unravel, dtype=prob.dtype,
        fine_axes=fine_axes, fine_logp=fine_logp, passes=passes, edge_drop=float(edge_drop),
        log_evidence_is=log_z_is, log_evidence_grid=log_z_grid, z_hat=z_hat, timings=timings,
        modes=[{k: o[k] for k in ("theta", "sd", "z", "U", "cov", "log_evidence")} for o in modes],
    )
