# Nested Laplace: a fast posterior that survives ridges and multiple modes

`EchoFit.nested_laplace()` is a direct solve, like `optimise()`, that keeps
its speed without its main failure: on weakly constraining data, the
posterior of `log_mdot` and inclination is a long curved ridge or has two
separate modes, and `optimise()`'s single Gaussian then understates the
uncertainty several-fold or reports one mode as if it were the whole answer.

```python
ef.build_grid()
ef.nested_laplace()            # ~2-4 min on two bands of 100 points (laptop CPU)
r = ef.nested_laplace_result
print(r["k_hat"])              # Pareto k: below 0.7 means the draws are reliable
ef.plot_corner()               # every plot and the report work as after .fit()
```

## The problem it solves

The recovery tests on synthetic, CREAM-paper-style light curves
(`scripts/synthetic_recovery_grid.py`; random-walk driver, M Mdot =
10^8 Msun^2/yr, inclination 30 degrees) showed, on g and i at SNR 100 and a
1-day cadence:

| Seed | `optimise()` | NUTS (one chain) | What NUTS showed |
|---|---|---|---|
| 1 | log_mdot 1.97 ± 0.20, all 4 restarts agree | 2.94 ± 1.26 | two separate modes; the chain jumped between them once (split R-hat 2.1) |
| 2 | 4.21 ± 0.50, restarts disagree | 3.26 ± 1.26 | one long ridge from log_mdot ~1.5 to ~6 |

The true log_mdot is 1.95. On seed 1 `optimise()` happened to land on the
true mode but gave no sign that a second one exists; on seed 2 it reported one
end of a ridge with an error bar a third of the true width. NUTS was not a
reliable reference either: a chain that crosses between modes once cannot
weigh them.

## How it works

The difficulty is confined to a few parameters, `log_mdot` and the
inclination (and `temperature_slope` when it is fitted), which are also the
science parameters. Given them, everything else is well behaved: the driver's
Fourier coefficients and the band offsets are integrated out exactly
(`marginalise_linear`, [performance_improvements.md](performance_improvements.md)),
and what remains (`sigma_drw`, the band gains, any error-model terms) is close
to Gaussian. That is the setting of the integrated nested Laplace
approximation (INLA; Rue, Martino & Chopin 2009): grid the few awkward
parameters $\theta$, and at each grid point integrate the rest, $z_r$, by a
Laplace approximation,

```math
\log p(\theta \mid y) \approx -U(\theta, \hat z_r(\theta)) - \log\left|\frac{\partial\theta}{\partial z_\theta}\right| + \frac{d_r}{2}\log 2\pi - \frac12 \log\det H_{rr}(\theta),
```

with $U$ NumPyro's potential in unconstrained coordinates, $\hat z_r(\theta)$
the inner optimum and $H_{rr}$ its Hessian. The grid follows a ridge or a
second mode wherever it goes.

The steps (`pycream2/nested_laplace.py`):

1. **Find every mode.** A scout grid covers the prior range (cos i over its
   whole support, `log_mdot` over ±3.3 prior standard deviations). A full
   optimisation from each of its local maxima, and from the usual
   data-anchored start, climbs to every mode's peak, and each mode is weighed
   by its Laplace evidence. The scout grid's own values are not trusted to
   find modes: at SNR 1000 a 0.01-dex mode fell between scout points, and a
   version that zoomed in on the best scout value settled confidently on a
   broader, wrong mode (log_mdot 4.91 ± 0.02 for a true 1.95).
2. **Size the box.** A box around the kept modes is refined pass by pass
   with small, cheap grids: extended wherever the posterior reaches an edge
   that is not a hard prior bound (a ridge), and shrunk while it fills too
   little of the box (a narrow peak). Next to a hard bound it runs to the
   bound, so mass piled against, say, face-on is not cut off.
3. **The final grid** (20 × 15 cells for two parameters) solves a sub-lattice
   of every fourth point with exact Hessians; every other point borrows its
   nearest sub-lattice point's Hessian, for its Newton steps and a
   first-order shift of the inner optimum ($\partial \hat z_r/\partial z_\theta =
   -H_{rr}^{-1} H_{r\theta}$), and $\log\det H_{rr}$ is interpolated. Each
   grid point then costs a few gradient evaluations.
4. **Draws, importance-weighted.** Most draws come from the grid: a sub-cell
   chosen by the grid's log density interpolated onto a 4×-finer mesh, the
   gridded parameters uniform within it, the others from a multivariate
   Student-t (6 degrees of freedom) around the conditional optimum. A fifth
   come from a Student-t at each mode (defensive importance sampling,
   Hesterberg 1995), so a mode narrower than a grid cell is still sampled.
   Every draw is weighted by the exact posterior over the whole mixture's
   density, the weights are Pareto-smoothed (PSIS; Vehtari et al. 2024) and
   the draws resampled by them, as in IS-INLA (Berild et al. 2022). This
   corrects what the grid and the Laplace step miss, and gives the
   diagnostic $\hat k$: below 0.7 the weighted draws are reliable. The linear
   parameters are then drawn exactly, as after `optimise()`.

Two outputs carry the checks: `nested_laplace_result["k_hat"]` (with a
warning above 0.7) and `["edge_drop"]`, how far the posterior has fallen at
the grid's edges (with a warning if it may extend beyond them).

## Validation

### Exact toy posteriors (`tests/test_nested_laplace.py`)

| Toy | Exact | `nested_laplace()` |
|---|---|---|
| Two modes (a² = 4 measured, prior centred off zero), plus one unbounded and one positive inner parameter: mean, sd, weight of the a > 0 mode | 1.02, 1.70, 0.757 | 0.95, 1.74, 0.740 (k̂ 0.48) |
| A 0.01-wide mode beside a broad one, the narrow one between scout points: weight of the narrow mode | 0.712 | 0.719 |

A single Laplace approximation gives an sd of ~0.15 for the first toy.

### Synthetic disc fits, against nested sampling

`scripts/validate_nested_laplace.py` fits the same simulated light curves
with `optimise()`, `nested_laplace()` and, as the reference, nested sampling
of the identical marginalised posterior with nautilus (Lange 2023), an
independent algorithm built for multimodal posteriors. g and i, SNR 100,
1-day cadence; truth log_mdot = 1.95, inclination 30 degrees:

| Seed | Method | log_mdot | Inclination (°) | Wall time |
|---|---|---|---|---|
| 1 | `optimise()` | 1.99 ± 0.20 | 35.0 ± 11.0 | 113 s |
| 1 | **`nested_laplace()`** | **2.63 ± 1.24** | **40.4 ± 18.0** | **190 s** |
| 1 | nautilus (reference) | 2.59 ± 1.24 | 38.6 ± 17.9 | 2610 s |
| 2 | `optimise()` | 4.34 ± 0.47 | 57.3 ± 7.7 | 118 s |
| 2 | **`nested_laplace()`** | **3.76 ± 1.30** | **56.4 ± 15.7** | **167 s** |
| 2 | nautilus (reference) | 3.79 ± 1.34 | 57.1 ± 15.4 | 1871 s |

(The figures below come from the validation script's own run, at an earlier
setting of the final grid: its `nested_laplace()` curves differ from the
table by Monte Carlo noise.)

![Seed 1: two modes](images/nested_laplace/gi_snr100_dt1_s1.png)

![Seed 2: a long ridge](images/nested_laplace/gi_snr100_dt1_s2.png)

On seed 1 the posterior has two modes, in log_mdot (near 2 and 4 to 5) and
in inclination (near 25 and 63 degrees); `nested_laplace()` reproduces both
and their weights, `optimise()` sees only the first, and the single NUTS
chain finds both but weighs them wrongly. On seed 2 it is one long ridge
running into the 80-degree bound, which `nested_laplace()` follows end to end.
The NUTS chain shown was run on the 400-day frequency grid of the recovery
study, not `build_grid()`'s default, so it is indicative only.

Note what these cases are: g and i at SNR 100 genuinely cannot pin down the
disc. The *answer* is a broad posterior; the point is that `nested_laplace()`
reports that honestly and `optimise()` does not.

At SNR 1000 on ugriz (truth as above), where the posterior is a single
0.01-dex peak, `nested_laplace()` gave log_mdot 1.95 ± 0.01 and inclination
30.4 ± 1.1 degrees (k̂ below 0.2), as `optimise()` did.

## The Badness-of-Fit landscape

The grid `nested_laplace()` builds is itself the clearest picture of what
the data say about the disc. `ef.plot_landscape(truth=...)` shows it as
$\Delta\mathrm{BOF} = -2\,\Delta\log p$ (Badness of Fit, as in Starkey,
Horne & Villforth 2016, eq. 12, with everything other than `log_mdot` and the
inclination optimised and integrated out at each point): on the left the
scout grid over the whole prior range, on the right the final grid with
contours enclosing 68, 95 and 99.7 per cent. It is also in `report.html`
after a `nested_laplace()` fit.

![Seed 1: two basins](images/nested_laplace/landscape_gi_snr100_dt1_s1.png)

![Seed 2: one long valley](images/nested_laplace/landscape_gi_snr100_dt1_s2.png)

The cross marks `optimise()`'s peak: the bottom of one basin, with nothing
to say about the other basin (seed 1) or how far the valley runs (seed 2).
In both, the alternative to the truth is a more inclined disc with a higher
accretion rate (log_mdot 4 to 6 at 55 to 80 degrees), which g and i at SNR
100 cannot rule out.

## Cost

Wall times on a laptop (2 CPU cores, macOS, JAX on CPU), one fit at a time,
including JIT compilation:

| Case | Data points | `optimise()` | `nested_laplace()` | nautilus (reference) | NUTS, one chain, 300 + 300 |
|---|---|---|---|---|---|
| g, i; SNR 100; 1-day; seed 1 (two modes) | 200 | 113 s | 190 s | 2610 s | 662 s (unconverged) |
| g, i; SNR 100; 1-day; seed 2 (ridge) | 200 | 118 s | 167 s | 1871 s | |
| g, r, i; SNR 300; 2-day | 150 | 151 s | 251 s | | |
| u, g, r, i, z; SNR 1000; 1-day | 500 | 246 s | 444 s | | |

Where `nested_laplace()`'s time goes on the 200-point seed 1 fit: 20 s
finding the joint optimum (mostly compilation), 58 s for the scout grid and
mode search, 44 s for the final grid, 11 s for the weighted draws, and 55 s for
the exact draws of the linear parameters. The last step is shared with
`optimise()` (where it is 60 s of 113 s, and 141 s of 246 s on 500 points),
and is the obvious next thing to speed up. A gradient of the marginalised
posterior costs ~37 ms on 2 bands of 100 points and ~90 ms on 5, so the
direct solves scale with the data roughly as NUTS does, at a small fraction of
its number of evaluations.

`optimise()` is the faster of the two, by 1.4 to 1.8 times on these cases. It agrees with
`nested_laplace()` where the posterior is one clean peak (the SNR 1000 case:
1.947 ± 0.014 against 1.948 ± 0.014), and differs where a bounded parameter is
skewed (inclination near face-on on the g, r, i case: 20.5 ± 7.9 against
17.0 ± 7.7 degrees) and badly where the posterior is a ridge or has two modes.

## When to use which

- **`nested_laplace()`**: the default direct solve whenever the answer
  matters and the data may not pin down `log_mdot` and the inclination: few
  bands, modest SNR, a short or gappy campaign. Check `k_hat`.
- **`optimise()`**: a quick first look, or data that clearly constrain both
  (it agrees with the others there, and is somewhat faster).
- **`.fit()` (NUTS)**: `lag_mode="free"` bands, which have a lag parameter
  per band (too many to grid), and anything where `k_hat` stays above 0.7.

## Limitations

- Up to about three gridded parameters; the cost grows as the product of the
  grid sizes.
- A mode narrower than the scout grid's spacing is only found if an
  optimisation reaches it, from the data-anchored start or a scout maximum.
  `k_hat` cannot flag a mode that no draw came near.
- The inner parameters are assumed close to Gaussian (Student-t in the
  proposal) given the gridded ones; the importance weights correct moderate
  departures and `k_hat` flags large ones.

## References

- Rue, Martino & Chopin (2009), JRSS B 71, 319: INLA.
- Berild, Martino, Gómez-Rubio & Rue (2022), J. Comput. Graph. Stat. 31(4): importance sampling with INLA.
- Vehtari, Simpson, Gelman, Yao & Gabry (2024), JMLR 25(72): Pareto smoothed importance sampling.
- Hesterberg (1995), Technometrics 37(2): defensive mixture distributions for importance sampling.
- Yao, Vehtari & Gelman (2022), JMLR 23(79): multimodal posteriors and the limits of mode-based approximations.
- Lange (2023), MNRAS 525, 3181: nautilus, the nested sampler used as the reference.
