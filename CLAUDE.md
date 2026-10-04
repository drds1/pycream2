# CLAUDE.md

Context for Claude (or any future contributor) picking this repo back up.

## What this is

`pycream2`: a JAX + NumPyro package that fits multi-band AGN reverberation
light curves as a delayed, smoothed echo of an unobserved driving light
curve. Built from a single detailed spec; see `README.md` for the physics
and package layout.

**Formerly `echofit`, renamed to `pycream2` (Continuum Reprocessing AGN
MCMC) in September 2026** as the successor to the author's CREAM Fortran
code and its `pycecream` wrapper, and published to PyPI under that name
(`.github/workflows/publish.yml`, trusted publishing on each GitHub
release). The `EchoFit` class and its module `pycream2/echofit.py` kept
their old names deliberately, to keep the user-facing API unchanged; don't
treat that as a missed rename. Published work should cite Starkey, Horne &
Villforth (2016): keep `CITATION.cff` and README's "Citing pycream2"
section in step with each other. The version lives in `pyproject.toml`
and `CITATION.cff` only (`tests/test_version_consistency.py` checks they
match; `pycream2.__version__` reads the installed metadata);
`docs/releasing.md` is the release procedure.

## Key design decisions (don't relitigate without reason)

1. **Driver = Fourier series, not a literal DRW GP kernel.** `X(t) = Σ_k
   [S_k sin(w_k t) + C_k cos(w_k t)]` on a *fixed* frequency grid built once
   in `EchoFit.build_grid()`. `S_k, C_k` are deterministic transforms
   (`S_k = S_raw_k * prior_scale_k`, non-centred) of unit-Normal
   `S_raw, C_raw` NumPyro sample sites, with `prior_scale` set by the DRW's
   Lorentzian power spectrum (`model.drw_prior_scale`), parameterised by
   inferred `sigma_drw`, `tau_drw`. The Fourier-series driver was requested
   explicitly in the spec and also happens to make the whole model
   analytically convolvable (see next point) instead of needing a GP solve.
   The non-centred form is deliberate: sampling `S, C` directly
   ("centred") creates Neal's-funnel geometry against `sigma_drw`/`tau_drw`
   that pins NUTS near its max-tree-depth ceiling. Don't revert to centred
   without re-checking `tests/test_recovery.py`'s step-count/divergence
   behaviour.

2. **Convolution is closed-form, not numerical double-integration.**
   Because the driver is a sum of sinusoids, `∫ ψ(τ) X(t-τ) dτ` reduces to a
   sum over frequencies weighted by the response function's own Fourier
   transform (`forward_model.transfer_coeffs` → `A_k, B_k`), then
   `forward_model.compute_echo` is a single `(n_obs, n_freq)` matrix
   contraction. **Do not** replace this with a brute-force `(n_obs, n_tau)`
   trapezoidal double loop: that's both slower and was explicitly
   prohibited by the "no unnecessary loops over time" requirement.

3. **`M_BH` is always fixed, never a `numpyro.sample` site.** It's a plain
   Python float passed through `EchoFit(M_BH=...)` into
   `forward_model.lag_scaling` / `response_function` / `model.reverberation_model`.
   If someone asks to infer it later, that's a deliberate scope change:
   flag it, don't just add a prior silently.

4. **Inclination affects skew only, not mean lag.** Mean lag comes from
   `lag_scaling(log_mdot, wavelength, M_BH)` alone. `response_function`'s
   skew-normal `alpha` parameter is a separate function of inclination. Keep
   these decoupled if you touch `forward_model.response_function`. This is
   exact for `response_function` by construction. `thin_disk_response`
   satisfies it exactly too, analytically -- *without* its default
   smoothing (`smoothing_days=0.0`); with the default smoothing on (see
   decision #8), it holds only approximately (~10% mean-lag drift face-on
   to 80 degrees), a deliberate, documented trade-off, not an oversight.

5. **Response function is swappable by contract, not inheritance.** Any
   replacement must accept `(tau_grid, log_mdot, wavelength, inclination,
   M_BH, ...)` and return a causal (`τ<0 → 0`), area-normalised-on-`tau_grid`
   array. `transfer_coeffs`/`compute_echo`/plotting never assume the skew-
   normal specifically. To actually swap it at runtime (e.g. for a quick
   experiment, without editing `forward_model.py`), reassign
   `pycream2.model.response_function`, e.g. `import pycream2.model as model;
   model.response_function = my_fn`. This is the *only* place that needs
   patching: `echofit.py`'s plotting code reads it via `model.response_function`
   (attribute access on the module, evaluated at call time) rather than its
   own `from .forward_model import response_function`, specifically so a
   swap there is honoured by both fitting and plotting consistently. Don't
   reintroduce a direct `response_function` import in `echofit.py` --that
   was a real bug once (the fit used the swapped function but the plot
   silently kept re-deriving ψ from the original, producing a plot
   inconsistent with what NUTS actually fit).

6. **On-disk run management (`title=`) is opt-in and single-chain only.**
   `EchoFit(M_BH=..., title=...)` switches `.fit()` from the original
   purely-in-memory path to a checkpointed one (`inference.run_mcmc_chunked`)
   that saves progress every `checkpoint_every` samples and can be resumed
   via `EchoFit.resume(title)`. This only covers the *sampling* phase (not
   warmup) and forces `num_chains=1` -- both deliberate scope limits, not
   oversights; see the "Fitting your own light curves" section of
   `README.md`. Without `title`, behaviour is byte-for-byte the original
   `EchoFit` -- don't let the checkpointed path's bookkeeping leak into it.
   `run_manager.py` owns the on-disk layout/serialisation,
   `reporting.py` owns the shared plots+HTML (used by both this path and
   `scripts/smoke_test.py` -- don't duplicate report-building logic back
   into either call site).

7. **Free-lag bands (`lag_mode="free"`) need a driver light curve to be
   identifiable -- this is exact, not a rule of thumb.** A global shift of
   the driver by any Δ, compensated by shifting every band's response lag
   by -Δ, leaves the predicted light curves *exactly* unchanged (proved
   directly, no MCMC, in `tests/test_shift_degeneracy.py`). The physical
   response (`lag_mode="physical"`, the default) escapes this because every
   such band's lag is tied to one shared `log_mdot` through a fixed
   `λ^(4/3)` scaling law -- a multiplicative rescaling of one shared
   parameter can't mimic an additive common shift once you have 2+ bands at
   different wavelengths. A `lag_mode="free"` band's `tau_{band}` has no
   such tie, so the degeneracy is exact; `add_driver_lightcurve()` (a
   direct, zero-lag `y = S_driver*X(t) + C_driver` observation of the
   driver, via `forward_model.driver_at`) is what anchors it.
   `EchoFit.fit()` warns (doesn't raise) if a free-lag band has no driver
   registered.

   **Gradient trap, already hit once:** `forward_model.tophat_response_free`
   must stay a *smoothed* box (sigmoid edges), never a literal hard
   `jnp.where(|tau - tau_mean| <= half_width, 1, 0)`. A hard top-hat has
   **exactly zero gradient** w.r.t. `tau_mean` almost everywhere (confirmed
   directly with `jax.grad` -- autodiff doesn't backprop through a
   comparison's operands), so NUTS gets no signal at all to move
   `tau_{band}` and the chain just sits near its init. This is invisible
   from `diverging`/acceptance-rate diagnostics alone (0 divergences, looks
   "fine") -- the tell was a posterior with essentially zero width. If you
   add another free-form response family, check its gradient w.r.t.
   whatever parameter NUTS samples before trusting a fit that used it.

8. **`thin_disk_response` and `pycream2/responses.py` are a second
   `lag_mode="physical"` response family, not a new mechanism.** They plug
   into the *existing* swap point from decision #5
   (`pycream2.model.response_function = ...`) -- `pycream2/responses.py` is
   only a small registry (`register_response`/`get_response`) for
   discoverability, it does not change how a response actually gets wired
   into a fit. `thin_disk_response` follows Starkey, Horne & Villforth
   (2016, MNRAS 456, 1960; arXiv:1511.06162, the CREAM paper -- confirmed
   directly against the paper's own equations, not just the Fortran): real
   Shakura-Sunyaev viscous + lamppost-irradiation temperature profile
   (their eq. 2), the disk's own light-travel-time delay surface
   `tau(r, phi) = r(1 + cos(phi) sin(inclination))` (their eq. 5), and a
   Planck-derivative response weighting -- genuine physics, not an assumed
   shape, unlike the default skew-normal.

   **It is now an exact analytic reduction to a 1-D integral over azimuth,
   not a 2-D radius/azimuth quadrature (the original implementation, and
   what the Fortran/Starkey+2016 both do numerically).** At fixed `phi`,
   `tau(r, phi)` is linear in `r`, so the disk integral's delta function
   collapses the radial integral exactly: `r*(phi, tau) = tau / (1 +
   cos(phi) sin(inclination))`, giving `psi_raw(tau) = tau * integral
   weight(r*(phi,tau)) / (1 + cos(phi) sin(inclination))**2 dphi`. No
   radial grid, no domain cutoff -- `docs/thin_disk_response.md` section 4
   has the full derivation. This was a direct response to two things: a
   user report that high-inclination curves still looked wrong even after
   the `r_max` cap fix below (turned out to be real residual radial-grid
   artefacts, now gone entirely), and a direct request to derive an
   analytic form for speed. Removing the radial dimension cut a real
   `O(n_r * n_phi * n_tau)` compute cost down to `O(n_phi * n_tau)` --
   an exact multiple wasn't re-verified under `jax.jit` before the old
   grid version was deleted (see decision #9's "a methodology correction"
   paragraph for why an eager-mode number here would be unreliable
   regardless), `jax.grad` still non-zero w.r.t. `log_mdot` and
   `inclination` (checked including at `inclination=89`, near the `sin=1`
   edge). With smoothing off (`smoothing_days=0.0`, see below), the
   mean-lag inclination-independence from decision #4 is exact to <1%
   drift face-on to 80 degrees (was flat to only 4 significant figures,
   with drift purely from quadrature error, before this rewrite) --
   **with the default smoothing on, this loosens to ~10%, a deliberate
   trade-off, not a regression** (see the smoothing paragraph below).

   **Gaussian smoothing is back, but for a different reason and by a
   different mechanism than before.** A second user report -- the
   high-inclination curves still didn't look like Starkey+2016 Figure 3's
   skewed-Gaussian-like shapes, showing an abrupt change in decay rate
   (a "shoulder") near tau~1-2 days instead -- led to checking the Fortran
   again, at the user's prompting, for a smoothing step. Confirmed present
   in *both* response-function subroutines: `tfbx`'s `sig_gaus = dtau` and
   the older `tfb`'s `sig_g = dtau/2`, each applied not as artefact
   clean-up but as part of the physical model itself -- every individual
   disk element's contribution is deposited as a Gaussian spread across
   several tau bins (`tfbx` lines ~12604-12714), not a single delta-function
   spike at its own exact delay. Because that smoothing is linear and
   `sig_gaus` is one constant for the whole disk, smoothing each element's
   contribution and summing is mathematically identical to computing the
   exact unsmoothed integral once and convolving *that* with the same
   Gaussian -- confirmed directly, and implemented as the latter (cheaper).
   The Fortran's literal `sig_gaus = dtau` convention doesn't transplant
   directly, though: it only smooths meaningfully because the Fortran's
   typical output grids were coarse (`dtau` incidentally ~0.3-1 day); for a
   finer `tau_grid` (confirmed with 800 points: `dtau` ~0.01 days) it's
   negligible, and ties the disk's own physical smoothing to an unrelated
   resolution choice regardless. `smoothing_frac` (default 0.4 then; 0.1 since decision #22) instead
   scales the width with `tau_ref` (from `lag_scaling`), chosen by directly
   comparing rendered curves against Starkey+2016 Figure 3 -- see
   `docs/thin_disk_response.md` section 4.

   **This reintroduces a real, quantified trade-off, and it was resolved
   with the user directly rather than picked unilaterally**: enough
   smoothing to match the published shape (`smoothing_frac=0.4`) costs
   ~10% mean-lag drift face-on to 80 degrees (0% unsmoothed, ~3% at 0.1,
   ~6% at 0.2 -- monotonic in `smoothing_frac`). A reflecting-boundary
   kernel (folding smoothed-away mass at `tau=0` back in, rather than
   discarding it) was tried and does *not* fix this -- the drift is
   inherent to any causally-respecting smoothing near a boundary that
   high-inclination responses sit much closer to than face-on ones, not an
   artefact of discarding vs redistributing leaked mass. Presented this
   quantified trade-off to the user directly; they chose to keep
   `smoothing_frac=0.4` as the default (matching the published look) over
   a smaller value or defaulting to exact. `smoothing_days=0.0` remains
   available for anyone who wants the decision-#4-exact behaviour back.
   See `tests/test_thin_disk_response.py`'s
   `test_thin_disk_response_smoothing_days_zero_gives_exact_mean_lag_independence`
   and `test_thin_disk_response_default_smoothing_mean_lag_drift_is_bounded`.

   The **irradiation term** (`include_irradiation=True`) was also corrected
   while cross-checking against Starkey+2016 eq. 2: it's now the true
   lamppost geometry `T_irr**4(r) ~ h_x / (r**2 + h_x**2)**1.5`
   (`lamppost_height_rs`, default 3.0 Schwarzschild radii, matching the
   paper's own illustrative value and the Fortran's `x_height=-3.0`), not
   the `~1/r**3` far-field approximation used before -- the two only agree
   for `r >> h_x`, and `h_x` is the same order of magnitude as `r_in`
   (also 3 Schwarzschild radii, confirmed against both the paper's explicit
   statement and the Fortran's actual default, `rinsch=1.0`, not the
   commented-out `=3.0` that would give a different value), i.e. exactly
   the regime this function spends the most time in. `include_irradiation`
   still defaults to `False` (pure viscous), so this only affects opt-in
   use.

   Two adaptation choices still worth knowing:
   - It reuses `lag_scaling(log_mdot, wavelength, M_BH)` for its absolute
     lag scale (the "Wien radius"), rather than independently deriving an
     Eddington-ratio-to-Mdot conversion from scratch. This was deliberate:
     an independent derivation would give `log_mdot` a second, incompatible
     meaning depending which response a band used. Only the inner (ISCO)
     radius and the lamppost height use real physical constants (G, c,
     M_sun) directly, since that conversion needs no extra accretion-rate
     calibration.
   - The `r* < r_in` (inside the ISCO) mask is a smooth sigmoid, not a hard
     `jnp.where` -- same zero-gradient trap as decision #7's "already hit
     once", since `r*` depends on `inclination`, a sampled parameter.

   The superseded (pre-analytic-rewrite) `n_r`/`r_max_factor` machinery,
   and the debugging trail that led to it (a `40x24` default that passed
   every numeric test but looked jagged once plotted; then an `r_max`
   geometric-formula bug that made high-inclination curves ~66x-100x wider
   in radial domain than face-on ones on the same grid), is preserved in
   git history and `docs/thin_disk_response.md`'s section 4 as a record of
   what was tried, not as current guidance -- neither exists in the
   function any more. `smoothing_days` does still exist, but reintroduced
   for a completely different reason (see above) -- don't assume it's the
   same numerical-artefact-management parameter it used to be.

9. **`build_thin_disk_response_table`/`build_thin_disk_response_fast`
   trade a little more accuracy for MCMC-usable speed, the same way the
   author's PhD-era CREAM Fortran code did (confirmed directly in
   `cream_f90.f90`: a `psistore(:,ist)` array precomputed once across an
   inclination grid at one fixed reference `umdotref`/`wavref`, exactly
   this mechanism).** These precompute a table of responses across an
   inclination grid *once* (at one reference `log_mdot`/`wavelength`), then
   get any other inclination via linear interpolation and any other
   `log_mdot`/`wavelength` by *stretching* the lag axis according to
   `lag_scaling`'s own scaling law (`s = lag_scaling(...) /
   tau_ref_reference`, evaluate the template at `tau_grid / s`, divide by
   `s` to keep the area normalised). `jax.grad` still non-zero w.r.t.
   `log_mdot` and `inclination` (checked on and off the precomputed
   inclination grid points, same discipline as decision #7's gradient
   trap).

   **A methodology correction, worth flagging explicitly:** every speed
   number in this file and `docs/thin_disk_response.md` up to this point
   was measured with plain, eager (non-`jax.jit`) repeated Python calls --
   which is *not* what happens inside a real NUTS fit, where NumPyro
   `jax.jit`-compiles the whole log-density function once and every
   leapfrog step reuses that compiled executable with none of the
   per-call Python dispatch overhead eager timing includes. Checked
   directly: at `n_tau=600`, an eager call to `thin_disk_response` took
   ~134ms; the *same* call through `jax.jit` took ~2.6ms -- a ~50x gap
   that has nothing to do with the function's real cost. This means the
   ~90x/~4x/~25x/~1.3-1.5x figures this file used to carry for the fast
   path's speedup at various points in its history are not reliable, and
   in fact went the *wrong direction*: re-measured under `jax.jit` at the
   point right after decision #8's analytic rewrite but before this
   section's smoothing was reintroduced, the fast path was actually
   **~13x faster** (not the ~4x the eager measurement had claimed).
   **Properly measured now** (`jax.jit`, `n_tau=400`, `build_grid()`'s
   own default): `thin_disk_response` (with default smoothing) costs
   ~1.5ms per call, the templated fast path ~0.55ms -- **~2.8x faster**,
   with the range ~2x-5x depending on `n_tau` (checked at 150/400/600).
   Smoothing itself adds real cost when jitted (confirmed: not negligible
   the way it looked eagerly), ~1.5x over `smoothing_days=0.0`. Both
   `thin_disk_response` variants remain substantially more expensive than
   `response_function` even fast and jitted -- ~12x for the templated
   path, ~34x for the plain one, at the same settings -- confirming this
   is an inherent cost of doing a real disk integral (any form) rather
   than an assumed closed-form shape, not something either optimisation
   removes; still a real, essentially-free win at call time over the
   plain version, worth reaching for on long runs where every bit of
   per-step cost compounds, but not a way to make `thin_disk_response`
   competitive with the skew-normal's cost.

   A banded/truncated smoothing kernel (only summing over the `~5 sigma`
   nearest tau bins instead of the full dense `n_tau x n_tau` matrix) was
   prototyped when investigating this and found genuinely faster under
   `jax.jit` (~1.65x on the smoothing step alone) but with a real
   edge-handling bug (clipping out-of-range neighbour indices to the
   boundary over-weights the last few bins rather than properly excluding
   them) that would need fixing before it's trustworthy -- not implemented,
   given the smoothing step's absolute cost is already small once
   properly jitted and this fit's overall per-step cost has other
   components (Fourier transfer, per-band likelihoods) not profiled here
   that may dominate regardless. Worth revisiting if profiling a real fit
   shows the disk response specifically as the bottleneck.

   **Templates are built unsmoothed (`smoothing_days=0.0`
   internally), and smoothing is applied once, after stretching, at each
   query's own `tau_ref` -- not baked into the table.** Smoothing the
   templates *before* stretching them onto a different `log_mdot`/
   `wavelength` would stretch the smoothing bandwidth right along with
   everything else, silently coupling two things that should be
   independent -- confirmed directly to matter a lot: doing it the wrong
   way around made even the *near-reference-point* case (which should be
   almost exact) off by ~13% at the peak; doing it the right way (as
   implemented) brings that back under 1%, and the previously-hardest
   tested case (`inclination=85`, `wavelength=7000`, table built with the
   default `reference_wavelength=5000`) down to ~0.7% (was ~35%
   pre-analytic-rewrite, ~4% immediately after it, before smoothing was
   reintroduced). `thin_disk_response_from_table`/
   `build_thin_disk_response_fast` both take their own
   `smoothing_frac`/`smoothing_days`, independent of whatever
   `build_thin_disk_response_table`'s `**table_kwargs` were.

   The stretch is still a genuine approximation, not an identity: the ISCO
   (`r_in`) is a fixed absolute length that doesn't stretch along with
   everything else, so the ratio `r_in / tau_ref` -- and with it, how much
   the inner-boundary term shapes the response -- differs between the
   table's reference point and wherever a fit actually queries it. This
   still bites hardest exactly where the response is sharpest: high
   inclination, far from the reference wavelength/`log_mdot` -- now a much
   smaller effect in practice than the smoothing-order bug above was, but
   the underlying asymmetry hasn't gone away, and is worth remembering if
   accuracy at extreme parameter combinations matters -- see
   `docs/thin_disk_response.md` section 7 and
   `tests/test_thin_disk_response_fast.py`'s
   `test_fast_response_approximation_degrades_away_from_reference`. This
   is a real, documented tradeoff to make deliberately (build the table
   with a `reference_wavelength` close to the run's actual bands if the
   posterior is expected to favour high inclination), not a bug to chase.

10. **`plot_corner`/`plot_corner_bands`/`plot_corner_free_lag`/
    `plot_fourier_correlation` (`plotting.py`) were added directly in
    response to looking at Starkey+2016's own diagnostic figures (Figure 6:
    a `log_mdot`/inclination corner plot, 3 chains overlaid, crosshair at
    the true value; Figure 4: individual-frequency power spectrum points,
    not just a smoothed line -- the latter is why `plot_power_spectrum`'s
    median line also got marker dots at each frequency).** `plot_corner`
    is intentionally a small, hand-rolled matplotlib implementation, not
    `arviz.plot_pair` (already a project dependency, used elsewhere for
    `chains.nc`): `arviz.plot_pair` doesn't distinguish chains by colour in
    scatter mode, and per-chain colouring is exactly the point here --
    chains landing in visibly different places is the same thing a
    Gelman-Rubin R-hat check would flag (see the `lag_mode="free"`
    rough-edge below), made visible in a plot instead of a single number.
    `S_band`/`C_band`/`tau_{band}` are individually-named scalar sites, so
    `plot_corner_bands`/`plot_corner_free_lag` just auto-detect names from
    `self.bands` and call the general `plot_corner`.

    `S`/`C` (the driver's Fourier coefficients) are different in kind --
    one vector-valued site each, from `numpyro.plate("freq", n_freq)`, not
    `n_freq` individually-named scalars -- and `n_freq` is often in the
    tens, where an `n_freq x n_freq` scatter-matrix corner plot would be
    both unreadable and slow. `plot_fourier_correlation` shows the
    posterior correlation matrix as a heatmap instead (chains pooled, not
    coloured separately, since a correlation matrix is already a
    per-sample summary): the same "is the posterior geometry sane"
    question a corner plot would answer, `O(n_freq^2)` pixels instead of
    `O(n_freq^2)` subplots, so it scales to any `n_freq` without changing
    shape. All four are wired into `reporting.generate_report` --
    `plot_corner_bands`/`plot_fourier_correlation` unconditionally,
    `plot_corner`/`plot_corner_free_lag` only when the fit actually has
    `log_mdot`/`inclination` or any `lag_mode="free"` band respectively
    (checked via `ef.samples`/`ef.bands`, not assumed), so a report never
    errors on a fit that doesn't have the relevant parameters.

11. **Badness-of-Fit (`plot_bof`) piggybacks on NUTS's own
    `potential_energy`, and doesn't need any new computation.** Starkey,
    Horne & Villforth (2016) eq. 12 defines `BOF = chi**2 +
    sum(ln(sigma_i**2)) - 2*ln(P(Theta)) + const`; NumPyro's NUTS already
    computes `potential_energy = -log(likelihood * prior)` on every
    leapfrog step to decide whether to accept a proposal, and for a
    standard Gaussian log-likelihood `-2*ln(likelihood)` is exactly
    `chi**2 + sum(ln(sigma_i**2)) + const`, so `BOF = 2 *
    potential_energy` up to that additive constant -- confirmed by direct
    derivation, not assumed. `inference.run_mcmc`/`run_mcmc_chunked` both
    request `extra_fields=("potential_energy",)` (NumPyro only returns
    `"diverging"` unless asked), and `EchoFit` now tracks
    `self._extra_fields_by_chain` the same way it already tracked
    `self._samples_by_chain`. `EchoFit.plot_bof()` /
    `plotting.plot_bof` draw one BOF trace per chain; wired into
    `reporting.generate_report` conditionally on `"potential_energy" in
    ef.extra_fields`, since a checkpoint resumed from before this feature
    existed (`_merge_dicts` only iterates the first chunk's keys, so an
    old `prev_extra` without `potential_energy` silently drops it from the
    merge rather than erroring) won't have it -- a narrow, accepted edge
    case, not one worth extra machinery for.

12. **`EchoFit.fit(title=..., report_every=...)` can refresh
    `report.html` partway through a long checkpointed run, not just once
    at the end.** `report_every` (in samples, like `checkpoint_every`) is
    checked inside the existing `_on_chunk_done` callback
    (`inference.run_mcmc_chunked`'s per-chunk hook, previously used only
    for on-disk persistence): once enough new samples have accumulated
    since the last refresh, it updates `self.samples`/`self.extra_fields`/
    `self._samples_by_chain`/`self._extra_fields_by_chain` from the
    merged-so-far checkpoint data and calls `reporting.generate_report`
    again, so the report reflects genuinely fresh state rather than a
    stale write. Off by default (`None`): re-rendering the full plot set
    (multiple corner plots, posterior-predictive fits with credible
    bands, ...) has real cost, and most callers won't be watching a run
    live. `self.mcmc` stays `None` throughout the checkpointed path
    regardless (decision #6) -- `reporting.py` never reads it, only
    `ef.samples`/`ef.extra_fields`/`ef._samples_by_chain`/
    `ef._extra_fields_by_chain`/`ef.bands`, which is what makes updating
    those mid-fit sufficient for a mid-fit report to work at all.

13. **`sigma_drw`'s prior scale is anchored to the registered data's own
    amplitude (`EchoFit._sigma_drw_prior_scale`), not a fixed constant --
    found via `scripts/make_fit_animation.py`'s GIF, which made a real,
    previously-unnoticed degeneracy visible.** Rescaling the driver by any
    `lambda != 0` (`S, C -> lambda*S, lambda*C`), compensated by rescaling
    every band's gain by `1/lambda` (`S_band -> S_band/lambda` for every
    band), leaves every predicted light curve -- and hence the
    likelihood -- exactly unchanged. Unlike the shift degeneracy in
    decision #7, this holds regardless of `lag_mode` or band count: it's a
    property of the model's linear driver-amplitude/per-band-gain
    separability, not something the disk physics (shared `log_mdot`, etc.)
    can break. With a fixed `HalfNormal(2.0)` prior on `sigma_drw`
    (unrelated to the actual light curves' units) and `S_band ~
    LogNormal(0, 1)`, this direction is only weakly regularised by the
    priors -- visible directly in the animation as the inferred driver
    growing/shrinking with little constraint early in a chain, before the
    (still fairly broad) prior eventually reins it in.

    Checked directly against the author's PhD-era CREAM Fortran code
    (`cream_f90.f90`) for how it handled this at the time: it has an
    explicit, optional Gaussian prior directly on `P0`, the power-spectrum
    normalisation (`sigma_drw`'s counterpart), gated by
    `sigp0square`/`siglogp0` (the `bof4` term, off by default). Same fix,
    same mechanism, applied here: `EchoFit._model_kwargs()` now computes a
    data-anchored `sigma_drw_prior_scale` (prefers a registered driver
    light curve's own std if one exists, via `add_driver_lightcurve`,
    since that's the most direct available observation of the driver;
    otherwise the largest std across the registered bands, since the
    least-reprocessed band is the closest available proxy for the driver's
    own amplitude -- a reprocessed echo is usually damped relative to what
    drives it, not amplified) and passes it to
    `reverberation_model(..., sigma_drw_prior_scale=...)`, which now takes
    that as a parameter instead of hardcoding `2.0`. The old fixed value
    remains the function's own default, for direct/standalone calls to
    `reverberation_model` outside `EchoFit`.

14. **`scripts/make_fit_animation.py` and `forward_model.disk_temperature_profile`
    are a demo/visualisation aid, not part of the fitting pipeline itself.**
    The GIF embedded in README.md's opener renders driver/echo/response-
    function panels (response-function x-axis capped at 30 days -- the
    response itself is always much narrower than the full lag grid it's
    evaluated on -- and the light-curve column given more width via
    `gridspec_kw={"width_ratios": [3, 1, 1]}`, since that's the panel
    worth the most screen space) plus two disk panels, evolving sample by
    sample early in a deliberately short-warmup chain -- the same
    short-warmup chain whose driver amplitude swings motivated decision
    #13 above. `disk_temperature_profile` and `thin_disk_response`'s
    viscous term now share the one actual formula, via `_viscous_t4_shape`
    (same formula, same Wien's-law reference point -- see
    `tests/test_thin_disk_response.py`'s
    `test_disk_temperature_profile_matches_wien_law_at_the_reference_radius`).
    Originally deliberately *not* shared, to avoid regression risk to the
    already-validated `thin_disk_response` for what was then a purely
    cosmetic reuse -- revisited and done anyway after a direct "best not to
    have duplication of code" request, once there was a second real
    consumer of the same physics (this decision, the animation) and the
    existing `thin_disk_response` test suite (20 tests, all passing
    unchanged after the refactor) to catch a regression if the shared
    helper got it wrong. `_viscous_t4_shape` takes the two callers'
    already-clipped radius and the constants they'd already computed
    (`tau_ref`, `r_in`) rather than clipping internally, since the two
    callers need genuinely different clipping strategies at `r <= r_in`
    (`thin_disk_response`'s `r_star` depends on a sampled parameter and
    needs a smooth cutoff -- handled separately via its own sigmoid mask,
    same zero-gradient concern as decision #7; `disk_temperature_profile`
    doesn't sample `r`, so a plain hard clip is fine there) -- the formula
    is shared, the clipping decision deliberately isn't. Always uses the
    viscous-only profile at a fixed illustrative reference wavelength
    (5000 Angstrom), regardless of which response function the fit itself
    used or what wavelengths its bands are at -- only the thin-disk
    response family has a literal disk geometry to show, and the picture
    is meant to convey the physics qualitatively, not represent the
    specific fit's own bands.

    **The disk view is split into two panels, not one, after a direct
    user correction:** an earlier single-panel version squashed the
    face-on disk vertically by `cos(inclination)` *and* moved the
    observer icon's position to suggest the viewing angle, which read as
    the observer itself moving rather than the disk tilting. Now:
    face-on temperature panel (fixed geometry, colour-only updates via
    `set_array` -- cheaper than the old per-frame `pcolormesh` rebuild,
    titled each frame with the current `log_mdot`/`inclination` values)
    and a separate side-on schematic panel where a line through the
    centre rotates with inclination (vertical at `inclination=0`/face-on,
    rotating toward horizontal/aligned with the fixed observer's line of
    sight as inclination approaches edge-on) while the eye-on-a-sphere
    observer and its dashed sightline are drawn once and never move.
    Both remain a simplified 2-D schematic, not a 3-D/raytraced render.

    **The face-on panel is coloured in true blackbody colour, not an arbitrary colour map (another direct
    request).** `plotting.blackbody_to_colour` integrates the Planck spectrum against the CIE 1931
    colour-matching functions (Wyman, Sloan & Shirley 2013 analytic fit), converts XYZ to sRGB and
    normalises brightness out (chromaticity only), so the disk runs blue-white at the hot centre, through
    white near 6500 K, to orange-red outside. One fixed log-temperature scale (a Kelvin colour bar) is
    shared by every frame, floored at 1000 K, because the zero-torque inner boundary sends T to ~0 exactly
    at the ISCO, and an unfloored scale reached ~10 K there and produced NaN colours. Rings are coloured at
    their geometric midpoint temperature on a geometric radial grid, so the small hot core is resolved
    while the face-on geometry stays linear in radius. The hot (>10^4 K) zone genuinely is small (within
    ~0.2 of the Wien radius); that is the physics, not a rendering fault.

    **Every band/driver panel also shows an expanding-window 68%/95%
    credible envelope** (`_expanding_percentiles`): frame `i`'s envelope
    is the percentiles of samples `0..i` only, not the full run, so it
    starts at zero width (one sample has no spread), can widen as the
    chain starts genuinely exploring, and narrows towards the converged
    posterior's own width as more samples dilute the influence of any
    early, still-unconverged ones -- the same shaded-band convention as
    the standard (non-animated) `plot_lightcurve_fits`, just computed
    cumulatively per frame instead of once over the whole chain. Percentiles
    of a samples-so-far prefix are mathematically bounded by the full run's
    own min/max, so the existing axis limits (set from the full arrays)
    already contain every frame's envelope with no extra padding needed.
    `fill_between` has no in-place update method, so these artists are
    removed and redrawn every frame (unlike the disk-temperature mesh's
    `set_array`).

    **Bands are coloured by real-world wavelength
    (`plotting.wavelength_to_colour`), not an arbitrary per-plot palette,
    in both this animation and every standard light-curve plot
    (`plot_raw_lightcurves`/`plot_lightcurve_fits`) -- another direct user
    request.** X-ray (< 100 A) black, UV (100-3800 A) a strong violet,
    the visible range (3800-7500 A) its actual approximate spectral
    colour (`_visible_wavelength_to_rgb`, the standard Bruton-style
    piecewise-linear wavelength-to-RGB approximation), and IR and beyond
    (> 7500 A) a reddish colour -- matching the lamppost picture of a
    short-wavelength driver reprocessed into progressively redder bands
    further out in the disk. The driving light curve's own line is plain
    `"black"` throughout, consistent with "X-ray driving light curves are
    black" regardless of whether a real driver light curve was registered.
    `_band_colours`'s old behaviour (a `plasma_r` colormap normalised to
    whatever wavelength range happened to be in the current fit) is gone
    entirely -- colours are now absolute and comparable across different
    fits/reports, not just internally consistent within one.

    **Higher resolution (`--dpi`, default 150, was a fixed 80) after a
    direct "looks blurry when zoomed in" correction** -- also bumps the
    committed GIF from ~3-4MB to ~10MB; accepted as the direct cost of
    that request rather than silently re-compressed back down, since
    colour-count reduction was checked and barely helps (96 to 64 colours
    saved well under 1MB) without visibly hurting the smooth gradients.

    **Gridlines on every light-curve/psi/BOF panel (`alpha=0.6`, both here
    and in `plotting.py`) after direct feedback that the charts "don't
    have gridlines"** -- they technically did (`alpha=0.25`), but
    `grid.color`'s default (`#b0b0b0`, already a light grey) at that alpha
    on a white background turned out to be essentially invisible once
    actually rendered, not just subtle; confirmed by rendering comparison
    swatches at several alpha values before picking `0.6`. Also added a
    sixth panel, Badness of Fit vs. sample (`2 * potential_energy`,
    matching `plotting.plot_bof`) as a growing trace extending one point
    per frame -- reuses `ef.extra_fields["potential_energy"]`, already
    collected by every fit by default (decision #11), so no extra
    computation, same as the standalone `plot_bof`.

15. **`inclination` is sampled uniform in `cos(inclination)`, not
    `inclination` itself, and `EchoFit(fixed_params={...})` generalises
    decision #3 ("`M_BH` is always fixed") to any scalar site.** Both
    direct user requests, both in `model.py` -- see its "inclination note"
    and "fixed-parameter note" docstring sections for the full mechanism
    (`_param`: substitute a `numpyro.deterministic` constant for the
    `numpyro.sample` call when a name is in `fixed_params`; `inclination`'s
    non-fixed path samples `cos_inclination ~ Uniform(cos(80deg), 1)` and
    derives `inclination = arccos(...)` as a deterministic). Two things
    worth knowing that aren't obvious from the mechanism alone:

    - **"Step in cos(i), uniform there" and "an explicit prior on
      inclination" are the same thing, not two different options.**
      Directly clarified with the user after they asked whether stepping
      in cos-space could "mimic the effect of" a prior without one being
      declared: it can't, and doesn't need to try to -- sampling
      `cos_inclination` uniformly and setting `inclination =
      arccos(cos_inclination)` *is*, via the transform's Jacobian, exactly
      the standard isotropic-orientation prior `p(inclination) ∝
      sin(inclination)`. There's no version of this that changes only how
      NUTS steps without also changing the marginal prior on inclination.
    - **`EchoFit._init_strategy()`** (data-anchored starting guesses for
      each band's `S_{band}`/`C_{band}` -- `C_band` the band's own mean,
      `S_band` its std relative to `_sigma_drw_prior_scale`, the same
      CREAM-Fortran-inspired idea as decision #13's prior anchoring, via
      `numpyro.infer.init_to_value`) **must stay disabled whenever
      `num_chains != 1`.** Found the hard way, not anticipated in advance:
      applying it unconditionally made
      `tests/test_free_lag_mode.py::test_free_lag_recovery_with_driver_anchor`'s
      4-chain R-hat check on the free-lag `tau_{band}` sites blow up to
      ~1000 (down from ~1.0 with it disabled) -- `init_to_value` gives
      *every* chain the identical starting point, which defeats the
      independently-initialised-chains premise that whole R-hat-based
      convergence-checking methodology depends on (see the "rough edges"
      note below on why that independence matters for this model's
      free-lag multimodality risk specifically). `_init_strategy(num_chains=1)`
      is safe (and the checkpointed path is always single-chain regardless,
      decision #6); anything else returns `None` (NumPyro's own
      `init_to_uniform` default) so multi-chain fits keep genuinely
      independent starts. `fixed_params` is persisted to/restored from
      `manifest.json` across `EchoFit.resume()` -- forgetting this would
      silently change the model structure (which sites get sampled at all)
      partway through a checkpointed run's chunks.

16. **Test coverage is measured (`pytest-cov`), enforced locally via
    pre-commit hooks, and run in full by CI -- all added directly in
    response to "how is our test coverage".** See README's "Tests and
    coverage" for the commands; the summary here is what changed and why.
    Coverage sat at ~90% line coverage before any of this was added (the
    codebase already had substantial tests throughout this file's earlier
    decisions), concentrated weakest in `run_manager.py` (68%) and
    `synthetic.py` (75%) -- their less-common paths (default arguments,
    observing-gap handling, resume edge cases) rather than anything
    central to the model. `tests/test_validation_and_utils.py` adds a
    first pass at those, plus EchoFit's own input-validation/guard-clause
    paths (bad `lag_mode`, `build_grid()` with no bands, every `plot_*`
    method's "call `.fit()` first" check) and a couple of branches nothing
    else happened to exercise (`max_tree_depth` actually reaching NUTS,
    `EchoFit.resume()` with a registered driver light curve, the
    `chains.nc` best-effort fallback documented above actually working
    when netCDF writing fails, not just having a `try`/`except` around it).

    **A genuine bug found this way, not a hypothetical one:**
    `grid_utils.estimate_dt_min`'s "no band has 2+ points" fallback
    (`dt_min = t_span if t_span is not None else 1.0`) was unreachable
    dead code -- `np.concatenate([])` on an empty list of per-band gap
    arrays raises `ValueError` immediately, before the code ever reaches
    the check that was supposed to handle exactly that case. Fixed by
    checking the list of per-band diffs directly, before concatenating,
    instead of checking the (never successfully computed) concatenated
    result's size. This is exactly the value a coverage report is for --
    "line never executed" flagged a branch that could never execute given
    how it was reached, not just one nobody had gotten around to testing.

    **Pre-commit hooks run file hygiene only (trailing whitespace,
    large-file checks, valid YAML/TOML), deliberately *not* any tests** --
    see `.pre-commit-config.yaml`, and its own comment for why, since this
    was a real course-correction worth recording rather than the first
    design that shipped. `@pytest.mark.slow` (registered in
    `pyproject.toml`) exists and marks the handful of tests actually
    responsible for the bulk of the full suite's ~15-minute runtime
    (decided from real `pytest --durations=0` data, not a guess -- one of
    them, the free-lag 4-chain recovery test, is a large fraction of it by
    itself), and the plan going in was a local `pytest -m "not slow"` hook
    at roughly that time saved back. Measured directly instead of assumed:
    that "fast" subset still took **~10 minutes**, because almost every
    other test in this suite also fits a real (if small) NUTS chain, each
    paying its own one-time JAX JIT-compilation cost -- there's no large
    genuinely-fast subset available short of cutting down to only the
    handful of pure-unit (no-fit) tests, which would defeat the point of
    a pre-commit smoke check. Large-file checks exclude
    `docs/images/fit_animation.gif` and `smoke_test_output/`, both
    intentional, regenerated-on-purpose binary assets, not accidents.
    Tests instead run automatically via CI (`.github/workflows/tests.yml`,
    full suite, no marker filter, every push/PR), where a few extra
    minutes doesn't block anyone's local `git commit`; run
    `poetry run pytest -m "not slow"` manually before pushing for that
    feedback sooner, accepting the ~10 minutes.

17. **`dense_mass=True` (a NUTS kernel option, now exposed on `EchoFit.fit()`) is the single
    highest-leverage, accuracy-preserving speed lever found so far, discovered directly from
    `scripts/profile_pipeline.py`'s own numbers, not guessed.** That script's per-iteration chart showed
    every individual forward-model component (response function, `transfer_coeffs`+`compute_echo`,
    potential-energy eval/grad) costing well under a millisecond, while NUTS's own measured per-sample cost
    was ~47ms, a ~120x gap that only makes sense if a single NUTS sample is taking on the order of 100+
    leapfrog steps. Checked directly against NUTS's own `num_steps` extra field (not inferred, measured):
    with the default diagonal mass matrix, this model's NUTS chain spends essentially every sample pinned at
    `max_tree_depth`'s default ceiling (2^10-1 = 1023 steps) -- mean 857.9, median exactly 1023, minimum 511,
    with **zero divergences**. That "maxed-out trajectory length with zero divergences" pattern is the
    textbook signature of NUTS being cut off by the step budget before it can find a genuine U-turn, not
    genuinely difficult/multimodal geometry -- exactly the kind of correlated-parameter geometry a diagonal
    mass matrix (which only rescales each parameter independently) can't correct for, but a full
    covariance-based one can.

    `dense_mass=True` on NumPyro's `NUTS` kernel (estimates a full covariance matrix during warmup instead of
    per-parameter variances) fixed it directly: mean leapfrog steps per sample dropped **~7.5x** (857.9 ->
    113.6), with `log_mdot` recovery unchanged (if anything marginally tighter: posterior std 0.023 ->
    0.021). The one real cost is that a dense mass matrix has O(P^2) entries to estimate during warmup
    instead of O(P), so it needs more warmup than the default to adapt properly -- confirmed directly: 200
    warmup samples gave a real, if modest, divergence rate (12/200, 6%), which 800 warmup samples brought
    down to a healthy 1.5% (3/200) with the same steps-per-sample improvement. This is a warmup-phase
    (one-off) cost, not a per-sample (recurring) one, so it doesn't erode the wall-time win on any run long
    enough for the sampling phase to dominate, which `scripts/profile_pipeline.py`'s own pipeline breakdown
    shows is true for any run past a few hundred samples.

    Off by default (`dense_mass=False`) to keep existing behaviour/results reproducible for anyone already
    relying on it; exposed as a plain passthrough on `EchoFit.fit()` (both the in-memory and `title=`
    checkpointed paths -- persisted in the checkpointed path's `_fit_config`/`manifest.json` the same way
    `max_tree_depth` already was, so a resumed run keeps using it rather than silently reverting to diagonal
    partway through) and on `scripts/fit_lightcurves.py`'s `--dense-mass` flag. Not applied automatically,
    because the extra warmup it needs is a real, user-facing tradeoff (more warmup samples means more one-off
    wall time before sampling starts) that's better left as an explicit choice than a silent default change.
    `docs/mcmc_implementation.md` covers the whole mechanism (the mass matrix, why NUTS needs gradients at
    all, JIT compilation) with real before/after charts (`scripts/plot_dense_mass_comparison.py` regenerates
    them), and a comparison to the original CREAM Fortran implementation checked directly against
    `cream_f90.f90`'s own sampling loop, not assumed: its default (`mcmcmulti_iteration`) is single-site
    random-scan Metropolis-Hastings (a Gaussian random-walk proposal on one parameter at a time, standard
    Metropolis accept/reject, crude accept/reject-streak step-size doubling/halving), and it turns out to
    have a genuine, independently-arrived-at analogue to `dense_mass` (`affine_step`, opt-in via a
    `cream_affine.par` file, only active every other iteration, only for a hand-picked parameter subset):
    eigendecompose that subset's empirical covariance and propose a joint step along its principal axes,
    the same "align the proposal with correlations, not just per-parameter scale" idea `dense_mass` applies
    automatically to every parameter. The difference that remains is gradients: `affine_step` is still a
    blind random draw within the right-shaped geometry, while NUTS's leapfrog dynamics move along the
    log-posterior's gradient at every step, which is the standard, well-established reason HMC/NUTS-family
    samplers need far fewer posterior evaluations than Metropolis-Hastings-family ones in a correlated,
    moderate-to-high-dimensional posterior like this one's (tens of driver Fourier coefficients alone). No
    head-to-head `pycream2`-vs-`pycecream` wall-clock benchmark has been run; the comparison above is of the
    two mechanisms, verified against the Fortran source, not a benchmark of the two actual codebases.

18. **Per-light-curve error rescaling (`fit_error_model`) is a direct, checked adaptation of the author's
    PhD-era CREAM Fortran code's `sigexpand`/`varexpand` nuisance parameters, found by reading
    `cream_f90.f90` while answering "is there anything to learn from the Fortran implementation" (see
    `docs/mcmc_implementation.md`'s comparison section).** Confirmed directly at `cream_f90.f90:4284`:
    `ernew2 = (er(it)*fnow)**2 + varnow`, i.e. the reported error is treated as only approximately correct
    and combined with a multiplicative rescale (`fnow`) and an additive jitter variance (`varnow`), both
    themselves fitted nuisance parameters. `pycream2` previously had no equivalent: `model.py`'s likelihood
    used `d["yerr"]` verbatim (`dist.Normal(y_pred, d["yerr"])`), which silently assumes every quoted
    uncertainty is exactly right -- a real risk on actual (not synthetic) data, where an overconfident
    likelihood from underestimated errors makes everything else (the BOF, the other posteriors) look more
    constrained than it should.

    `EchoFit.add_lightcurve(..., fit_error_model=True)` / `add_driver_lightcurve(..., fit_error_model=True)`
    turn this on **per light curve**, off by default (`False`) so existing/synthetic-data fits keep exactly
    today's likelihood unless explicitly opted in -- a deliberate, explicit request when this was
    implemented, not an incidental default. When on, that light curve's `sigma_scale_{name}`
    (`LogNormal(0, 0.5)`, median 1, "no rescaling" is the prior's own centre) and `sigma_jitter_{name}`
    (`HalfNormal(mean(yerr))`, anchored to that light curve's own typical quoted error, the same
    data-anchoring philosophy as `sigma_drw`'s prior, decision #13) combine as `sigma_eff =
    sqrt((sigma_scale*yerr)**2 + sigma_jitter**2)` in place of the raw `yerr`. Both new sites go through the
    existing `_param`/`fixed_params` mechanism (decision #15) automatically, so e.g.
    `fixed_params={"sigma_jitter_g": 0.0}` pins just the jitter term while still fitting the rescale factor
    for that band, without any new plumbing.

    **A real bug found and fixed while wiring this up, not a hypothetical one:** `run_manager.save_bands_npz`/
    `save_driver_npz` were hardcoded to a fixed set of keys (`t`, `y`, `yerr`, `wavelength`, `lag_mode`) and
    would have silently dropped `fit_error_model` across a checkpointed run's resume cycle -- the exact same
    class of bug decision #15 hit once already with `fixed_params` not persisting. Fixed by adding
    `fit_error_model` to both save/load functions (backward-compatible: an old checkpoint file without the
    key loads as `False`, not an error). A *second*, separate instance of the same class of bug was caught by
    the resume-persistence test itself: `EchoFit.resume()` correctly loaded `fit_error_model` via the fixed
    loader but never passed it through to the `add_lightcurve()`/`add_driver_lightcurve()` calls that
    reconstruct `self.bands`/`self.driver_data` -- two independent places the same value had to flow through
    correctly, both needed fixing, both are covered by
    `tests/test_error_model.py::test_error_model_persists_across_resume`.

19. **The driver's prior defaults to a pure random-walk (RW) power law, not the original damped random walk
    (DRW), a direct, explicit request.** `EchoFit(drw_prior=False)` (the new default) gives the driver's
    Fourier coefficients a plain `1/w**2` power-law power spectrum via the new `model.rw_prior_scale`, only
    `sigma_drw` inferred, no `tau_drw` site at all. `EchoFit(drw_prior=True)` restores the original DRW
    prior (`model.drw_prior_scale`, a Lorentzian, `tau_drw` inferred alongside `sigma_drw`). Motivation: for
    the short, irregularly-sampled campaigns this package targets, `tau_drw` is often weakly identified
    anyway (the "Known rough edges" note below already flagged this), and a plain power law is one fewer
    hyperparameter to identify for the same qualitative "smooth stochastic variability" driver.

    **This is deliberately not implemented as "fix `tau_drw` to a large constant inside `drw_prior_scale`"**,
    the seemingly obvious way to remove the DRW's turnover (push it to `w -> 0`) -- confirmed directly, not
    assumed, that this doesn't work: the Lorentzian's large-`tau_drw` limit at *fixed* `sigma_drw` is
    `sigma_drw**2 * tau_drw / (w*tau_drw)**2 = sigma_drw**2 / (tau_drw * w**2) -> 0` as `tau_drw -> infinity`,
    i.e. the power spectrum collapses to *zero* everywhere, not to a finite power law. Getting a genuine,
    non-trivial `1/w**2` law that way needs `sigma_drw**2 / tau_drw` held fixed as `tau_drw` grows, meaning
    `sigma_drw`'s own prior would need rescaling to compensate for whatever constant `tau_drw` was fixed
    to -- fragile, and coupled to the specific constant chosen. `rw_prior_scale` writes the `1/w**2` form
    directly instead: exact, no large-constant tuning, no `tau_drw` site or rescaling needed at all, and
    `sigma_drw` keeps the *same* site name and the *same* data-anchored prior (`EchoFit._sigma_drw_prior_scale`,
    decision #13) in both modes -- though its physical meaning differs (a saturating asymptotic variability
    scale for the DRW; a non-saturating power-law amplitude for the RW, since a pure random walk's variance
    grows without bound rather than saturating).

    Threaded through the same way `fixed_params` was (decision #15): a plain constructor argument
    (`EchoFit(drw_prior=...)`), persisted across checkpointed runs via `manifest.json`/`EchoFit.resume()` (a
    fresh instance of the exact bug class decisions #15/#18 already hit twice -- caught this time by writing
    the resume-persistence test *before* declaring the feature done, not after a report of it silently
    reverting to the default), and exposed as `scripts/fit_lightcurves.py --drw-prior`.
    `_valid_fixed_param_names()` only includes `"tau_drw"` when `drw_prior=True`, so
    `fixed_params={"tau_drw": ...}` correctly raises rather than silently doing nothing when there's no such
    site to fix. `plotting.plot_power_spectrum` takes `tau_drw_samples` as `Optional` now: given (DRW fits)
    it overlays the fitted Lorentzian as before; omitted (RW fits, the default) it overlays the fitted
    `sigma_drw**2/w**2` power law instead, both against the same empirical `P(w) = (S**2+C**2)/(2*dw)`
    estimate and the same `w**-2` reference line.

    **A real, wide ripple effect from flipping the default, not a self-contained change:** every existing
    call site that unconditionally read `ef.samples["tau_drw"]` assuming it always existed would have broken
    the moment the default flipped -- found and fixed across `tests/test_recovery.py` (now explicitly passes
    `drw_prior=True`, since that test specifically validates DRW hyperparameter recovery),
    `tests/test_fixed_params_and_priors.py`, `scripts/smoke_test.py`'s printed diagnostic table (now skips
    `tau_drw` if absent instead of guarding nothing), and `scripts/plot_dense_mass_comparison.py`/
    `scripts/profile_pipeline.py` (both explicitly pinned to `drw_prior=True`, since their whole point is
    reproducing the *exact* numbers already written up in decisions #9/#17, measured before this decision
    existed -- not a claim DRW is still the default). `tests/test_rw_prior.py` adds dedicated coverage for
    the new default path itself (site presence, `fixed_params` validation, resume persistence, plotting,
    and a `log_mdot` recovery check at the same rigour `test_recovery.py` already held the DRW path to).

20. **The fixed trig matrices are precomputed once per fit (`forward_model.transfer_matrices`/
    `fourier_basis`), not rebuilt on every gradient evaluation: ~34x faster per NUTS leapfrog step,
    identical result.** Prompted by a direct question about porting the CREAM Fortran code's `itaumax`
    early cut-off (checked in `cream_f90.f90`: it stops the real-space `(n_t, n_tau)` convolution
    lookback at the first lag where `psigrid` returns to zero after being positive). Measured before
    deciding, not assumed: in `pycream2` the lag grid only enters via `transfer_coeffs`, and its cost was
    almost entirely re-evaluating `cos`/`sin` of the fixed `(n_freq, n_tau)` phase matrix, per band, on
    every step, not the length of the lookback. `tau_grid`, `freqs` and each light curve's observation
    times never change during a fit, so `EchoFit._model_kwargs()` now precomputes the trapezoid-weighted
    `(Wc, Ws)` (in float64, then cast) and each band's/driver's `(sin(w t), cos(w t))` basis, and
    `reverberation_model` uses them via `transfer_mats=`/each dict's `"basis"` entry. Measured under
    `jax.jit` (5 bands x 100 obs, `n_freq=60`, `n_tau=400`): potential+gradient 8.7ms -> 0.26ms; a
    600-sample fit at `max_tree_depth=6` 48s -> 15s wall time, where JIT compilation now dominates;
    `log_mdot` posterior unchanged. A cut-off on top would only save part of a ~0.03ms matvec, and could
    only be static anyway, since `jax.jit` needs fixed shapes (a `psi`-dependent cut-off would still
    multiply the same zeros). All the new arguments are optional (`transfer_coeffs(..., matrices=None)`,
    `compute_echo(..., basis=None)`, `driver_at(..., basis=None)`, `reverberation_model(...,
    transfer_mats=None)`), so plotting code and direct standalone calls keep the on-the-fly path
    unchanged; `tests/test_precomputed_transfer.py` checks both paths agree, including the full model's
    potential energy and gradient. Decision #2's closed-form Fourier convolution is untouched: this
    only caches its fixed ingredients.

21. **Linear-parameter marginalisation (`EchoFit(marginalise_linear=True)`), the direct solve
    built on it (`EchoFit.optimise()`), and `dense_mass=True` as the default: all three came out of one
    direct question, "is there a way to directly solve this with linear algebra instead of MCMC?", and
    the answer changed once it was measured rather than predicted.** Full maths, charts and tables:
    `docs/performance_improvements.md` (regenerated by `scripts/plot_performance_analysis.py`).

    - **The mechanism.** With every nonlinear parameter fixed, every prediction is linear in the
      driver's Fourier coefficients and every `C_{band}`/`C_driver` offset, all Gaussian-prior, so
      they integrate out exactly: `y ~ N(0, D + M M^T)` with `M` the prior-whitened design matrix
      (`model._linear_marginal`). NUTS then sees ~8-10 parameters instead of ~130; the linear ones come
      back afterwards as exact conditional-Gaussian draws (`EchoFit._add_linear_draws`, `draw_linear=True`
      through `Predictive`) under their usual site names, so every plot, report and saved chain works
      unchanged. Verified against brute-force dense float64 integration to ~1e-5 relative
      (`tests/test_linear_marginalisation.py`).
    - **The numerical trap, found the hard way.** The first version used a float32 Cholesky of
      `P = I + M^T D^-1 M`. `P`'s condition number was ~3e7 on a 5-band, 500-point set, the gradients
      came out ~9% wrong, NUTS's adapted step size collapsed, and the "8-dimensional" posterior needed a
      median 143-207 leapfrog steps per sample. The same fit in float64 needed 7. Jacobi scaling didn't
      help (condition number only ~3x lower). Fixed with QR of the stacked `[D^-1/2 M; I]` system
      (`cond(R) = sqrt(cond(P))`): 7 steps in float32 too, log-likelihood within ~1e-4 of float64, at
      ~3.5x the cost of the Cholesky. Enabling `jax_enable_x64` instead was rejected because it is
      process-global: a library shouldn't flip it for the user. The quadratic form is evaluated at its
      minimiser (`||resid||^2 + ||theta_hat||^2`), never as the textbook `y^T D^-1 y - b^T P^-1 b`,
      which cancels catastrophically in float32.
    - **The honest result: for NUTS, marginalising does not pay any more.** Decision #20 made each
      sampled-model gradient ~30x cheaper (8.3ms -> 0.28ms), and a dense mass matrix fixes the sampled model's geometry, so
      on the benchmark (5 bands x 100 obs, 500+500) sampled/dense reached ~39 min-ESS per sampling
      second against ~19 for marginalised/dense (~7.4 vs ~3.6 overall), with identical `log_mdot`
      posteriors. Marginalised steps are ~7 per sample, but each costs ~21x more (a `(N+p) x p` QR and its
      gradient). `marginalise_linear` therefore stays opt-in and off by default, documented as *not*
      faster for NUTS; it may win where the sampled geometry is genuinely hard (many more Fourier modes,
      poorly adapted mass matrix), which is untested.
    - **The real payoff is `EchoFit.optimise()`**: L-BFGS on the marginal posterior (multi-start,
      `scipy.optimize.minimize` with JAX gradients), then a Laplace Gaussian from the Hessian in
      NumPyro's unconstrained space, then exact linear draws. On the same benchmark the whole first
      call took ~27s (~5s L-BFGS over 4 restarts, 2 Newton iterations; the rest mostly one-off JIT
      compilation of the Hessian and the draws) against ~56s for 500+500 dense NUTS, and its `log_mdot` posterior
      (0.194 +/- 0.012) matched NUTS (0.193 +/- 0.011). **L-BFGS alone is not enough, found on real
      data:** on NGC 5548 (4 bands) it stopped 6.5 units of potential above the optimum, with the implied
      Newton step 3.5 posterior standard deviations long, because it crawls along the prior-dominated
      driver-amplitude/band-gain ridge (decision #13; all four `S_band` had identical posterior widths,
      the prior's). `optimise()` therefore polishes with Newton steps using the exact Hessian, each
      backtracked until the potential falls and only accepted if it does, until the implied step is under
      0.01 standard deviations (NGC 5548: 2 iterations, potential exactly at the optimum, ~1s beyond the
      Hessian's compilation). Convergence is reported as `optimise_timings["newton_offset_in_sd"]`, with a
      warning above 0.25. An earlier version used that offset as a diagnostic only, having concluded
      that taking the step "made inclination worse"; that conclusion was wrong. It compared the Laplace
      *mode* with NUTS's posterior *mean*, which legitimately differ for a skewed, bounded parameter; the
      correct test is whether the potential falls, and with a line search it always does. L-BFGS's
      tolerances are also matched to float32 (its default `ftol` is below float32's resolution).
      Inclination is the weak spot generally: its posterior piles up against the 80 degree prior bound,
      which a Gaussian in logit space can't reproduce, so the Laplace mode sits a few degrees below
      NUTS's mean. It always uses the marginalised model regardless of `marginalise_linear`: its
      target is the peak of the *marginal* posterior, which is what Laplace needs; a joint peak over the
      Fourier coefficients too would be a different, biased estimator. Laplace is only as good as the
      posterior is Gaussian in unconstrained space: not for `lag_mode="free"` multimodality (rough-edges
      note below), where `.fit()` remains the tool. **Multi-start reproducibility** (a direct request, as the
      optimiser's counterpart of multi-chain MCMC checks): every restart is Newton-polished to its *own*
      optimum, not just the lowest (otherwise a restart L-BFGS merely stopped short on looks like a
      different answer), then compared with the best in Laplace posterior standard deviations
      (`ef.optimise_restarts`, `optimise_timings["restarts_agreeing"]`, `RESTART_AGREEMENT_SD = 0.5`,
      a report section and `plot_optimise_restarts`); it warns if any disagree. On pure-noise test data
      restarts genuinely land on a second mode 15 units higher, which the old keep-the-lowest-L-BFGS
      version could have returned. `fit(init_from_optimum=True)` starts NUTS at the
      peak (single-chain only, same R-hat reason as decision #15's `_init_strategy`); in the sampled
      model it also initialises `S_raw`/`C_raw`/`C_{band}` at their conditional mean, via
      `draw_linear=True` with `linear_eps` pinned to zero.
    - **`dense_mass=True` is now the default** (user decision, taken with these numbers in hand):
      ~17x more min-ESS per wall-clock second overall than diagonal on the benchmark, same posterior. A
      checkpointed run resumed from before `dense_mass` was persisted keeps diagonal rather than switching
      mid-run (`tests/test_validation_and_utils.py`). `scripts/make_fit_animation.py` pins
      `dense_mass=False` explicitly, because its deliberately tiny warmup can't adapt a dense matrix and
      the point there is an early, still-searching chain; `smoke_test.py`/`fit_lightcurves.py` gained
      `--diagonal-mass` (`fit_lightcurves.py --dense-mass` is kept as a no-op so old command files still
      run) and `fit_lightcurves.py --optimise` for the direct solve.

22. **September 2026 physics and numerics fixes, found while writing the NGC 5548 paper (an
    independent referee review found the first one too).** Each is covered by a test; together they
    change every thin-disk result, so earlier NGC 5548 numbers in this file are superseded.

    - **Missing lamppost dilution in `thin_disk_response`'s weight.** The weight was
      `X**5 e**X/(e**X-1)**2` (Planck derivative times `T**-3`) without the `h_x/x**3` geometric
      dilution of the lamppost flux (`dT/dL_x ~ h_x/(x**3 T**3)`). The outer disk responded far too
      strongly: mean delays were ~1.9 Wien radii and the implied temperature ~3x the model's own `T_1`.
      Fixed; the unsmoothed response now obeys `<tau> = (X k lambda T_1/hc)**(4/3)` with X ~ 3.2
      (`test_thin_disk_mean_lag_obeys_the_standard_lag_temperature_relation`).
    - **The temperature profile depended on the observing wavelength whenever `viscous_slope != 0.75`**
      (it was anchored at `lag_scaling`'s `lambda**(4/3)` radius for every slope). Now
      `T = T_1 r**-alpha (1 - sqrt(r_in/r))**(1/4)` with `T_1 = disk_t1_kelvin(log_mdot, M_BH)`
      independent of slope and wavelength; `wien_radius()` gives the slope-aware characteristic radius.
    - **`EchoFit(fit_temperature_slope=True)`** samples `temperature_slope` (alpha,
      `Uniform(*model.TEMPERATURE_SLOPE_PRIOR)`, `(0.5, 2.5)`; it was `(0.5, 1.5)` until NGC 5548's
      posterior piled up against 1.5) and passes it as `viscous_slope`, the analogue of Starkey et al.
      (2017) Model 2. Persisted across `resume()` (tested, the recurring bug class of decisions
      #15/#18/#19).
    - **Three float32 overflows gave NaN gradients or Hessians**: radii clipped at 1e-12 (NaN inclination
      gradient), `e**X/(e**X-1)**2` (now the `e**-X` form), and the fourth root of a clipped `T**4`
      (NaN second derivative in the slope; now a double `jnp.where`).
    - **A pole in the potential from the zero-torque edge.** Between the ISCO and the temperature peak
      (`r_in ((4 alpha + 0.5)/(4 alpha))**2`, 1.36 r_in) T rises from zero, so every band's Wien
      temperature is crossed in a ring ~1e-6 r_in wide next to the lamppost, where `h_x/x**3` is huge.
      Whenever a quadrature point landed in it, psi spiked (0.03 -> 14 at one tau) and the NGC 5548
      potential had a pole at one inclination (54.144 degrees) that trapped every optimiser restart.
      The response mask now starts at the temperature peak instead of the ISCO (negligible emission at
      UV-optical wavelengths for AGN temperatures). Found with the multi-start check: restarts disagreed,
      the potential fell monotonically along lines between them, and a per-parameter scan isolated
      `cos_inclination`. (Beware: `ravel_pytree` orders dict keys *sorted*, not by insertion; label
      flattened coordinates accordingly.)
    - **Causal smoothing and the exact lamppost delay (`docs/cream_response_comparison.md`, a direct
      comparison with CREAM's own `tfbx`, compiled from `cream_f90.f90` by
      `scripts/compare_cream_response.py`).** The user noticed pycream2's responses were non-zero at
      tau = 0 while Starkey+2016's all started at zero. Found: CREAM hard-codes `psi(1) = 0.0` (and
      normalises to the peak); with CREAM's smoothing and delay pycream2 reproduces CREAM's shapes
      (mean/median delays within ~4%); pycream2's Gaussian-in-tau smoothing (0.1 r_Wien ~ 0.5 d) spread
      the near-vertical onset across tau = 0 and peaked there for inclined discs. Now: the default
      smoothing is a 5% Gaussian in ln tau (`smoothing_log`, `DEFAULT_SMOOTHING_LOG`), causal by
      construction (the Gaussian in tau remains via `smoothing_days`/`smoothing_frac`); and the delay
      includes the lamppost height, tau = sqrt(r**2 + h_x**2) + h_x cos i + r sin i cos phi, whose
      delta-function roots at fixed phi are a quadratic's (two on the near side, each weighted by
      r/|dtau/dr|; `delay_lamppost_height=False` restores CREAM's delay). Decision #4 now holds for
      <tau> - h_x cos i (exact), not <tau>; the shift is ~0.01 d for NGC 5548. Unsmoothed, the peak of the
      response sits at the disc's inner edge at every wavelength, so tests use median delays, not peaks.
      Every earlier thin-disc fit (including the NGC 5548 paper's) used the old smoothing.
    - **The default `smoothing_frac` is now 0.1, not 0.4** (superseded by the causal default above) (the author's decision, as decision #8's
      was): 0.4 was tuned against the pre-fix, too-broad responses and now smooths over more than the
      mean delay, raising mean delays ~25% and erasing most inclination information. At 0.1 the mean
      delay drifts ~7% face-on to 80 degrees (`test_thin_disk_response_default_smoothing_...`).
    - **Filon quadrature for the transfer coefficients** (`_filon_weights`): the trapezoid rule aliased
      at high driver frequency on the coarse tail of the graded lag grid (w = 11 rad/day against
      0.3-day spacing on NGC 5548). Exact for the piecewise-linear psi, identical to the trapezoid as
      `w dtau -> 0`; precomputed and on-the-fly paths share one implementation. Side effect:
      `test_free_lag_recovery_with_driver_anchor` needed 1000 warmup steps instead of 400, because one of
      its 4 chains otherwise stayed in a local mode ~660 units of potential worse (the aliasing had
      happened to smooth it away); the lags themselves were recovered by every chain either way.
    - **Data-anchored offset and gain priors** (`model._offset_prior`, `_gain_prior_loc`):
      `C ~ Normal(mean(y), 10 std(y))`, `S ~ LogNormal(log(std(y)/sigma_drw_prior_scale), 1)`, replacing
      `Normal(0, 5)`/`LogNormal(0, 1)` in absolute flux units (the 1158 A offset sat ~9 sigma from its
      prior mean). The width is 10 std, not 5: at 5 NUTS diverged 1-24% on `test_rw_prior.py`.
    - **The Laplace curvature is measured over the posterior's width, not at a point
      (`_posterior_scale_curvature`, `CURVATURE_TOLERANCE`).** The thin-disc potential has tiny ripples in
      inclination (~1e-4 deep in float64, ~1e-3 in float32; not quadrature error: unchanged by 4x `n_phi`
      or 2x `n_tau`). At some optima they dominated the exact Hessian (-1.4, or -1170 with a finer
      quadrature, where the curvature over one posterior sd was +1.0): `optimise()` then reported "one
      more Newton step would move it 190 sd", clipped that direction to a ~1e4 sd variance, and, since
      restart agreement is measured in those sd, reported every restart as agreeing. Each eigenvalue is now
      checked against the finite-difference curvature one sd along its eigenvector and replaced when not
      positive or off by more than a factor of 4 (`CURVATURE_RATIO_LIMIT`; `timings["curvature_corrections"]`
      counts them). A first version replaced anything off by 50%, which also caught merely anharmonic
      directions (jitters near zero), where the pointwise Hessian is the better Gaussian: NGC 5548 Model 1's
      PSIS k-hat went from 0.49 to 0.74. The probe step
      starts at no less than 0.05 unconstrained units and is iterated to the measured curvature's own sd:
      a first version probed at the pointwise eigenvalue's sd, so a spurious *huge* eigenvalue shrank the
      probe inside the very feature that caused it and gave one synthetic fit an inclination sd of 0.015
      degrees. The check runs once per polished restart (~3 s on 13-band NGC 5548), not every Newton step.
    - **The Laplace mode is found in unconstrained coordinates**, where a bounded uniform prior's
      Jacobian favours the middle of its range. Where the data constrain a parameter weakly the mode is
      pulled there (on NGC 5548-cadence synthetic data: inclination modes 39-72 degrees for a true 30,
      alpha 0.69-1.06 for a true 0.75, mean +0.17); PSIS-reweighted draws (`ngc5548_paper/psis.py`)
      target the exact posterior and correct it.
    - **`optimise()` multi-start reproducibility**: every restart is polished to its own optimum
      (damped, saddle-free Newton, Levenberg-Marquardt style) and compared with the best
      (`ef.optimise_restarts`, `plot_optimise_restarts`, a report section). It is a check for a unique
      optimum, not for Gaussianity; `ngc5548_paper/psis.py` shows the PSIS check for the latter.

23. **Optional extra components, both off by default (October 2026, a direct request; full write-up
    `docs/extra_components.md`, figures from `scripts/plot_extra_components.py`).**
    - **Diffuse continuum** (`add_lightcurve(..., diffuse_continuum=True)`): a second reprocessor with a
      log-normal delay distribution (`forward_model.lognormal_response`), mixed into the band's response,
      `psi -> (1 - f) psi + f psi_LN` (`mix_diffuse_continuum`), after Cackett, Zoghbi & Ulrich (2022,
      ApJ 925, 29). Sites `dce_fraction_{band}` (Uniform(0, 1)), `dce_delay_{band}` (median, days,
      LogUniform(`DCE_DELAY_MIN`, tau_max)), `dce_width_{band}` (dex, Uniform(`DCE_WIDTH_PRIOR_DEX`)).
      Both parts are area-normalised, so the gain still multiplies the whole response and nothing
      downstream changes (closed-form transfer, marginalisation, plots).
    - **Slow background** (`add_lightcurve`/`add_driver_lightcurve(..., background_order=K)`): Legendre
      `P_1..P_K` in time over the whole campaign (`legendre_background_basis`; `EchoFit._background_t_range`
      is shared by every light curve and the plots), coefficients `bg_{name}` ~ Normal(0,
      `BACKGROUND_PRIOR_WIDTH` std(y)). Linear, so `_linear_marginal` integrates them out with the Fourier
      terms and offsets (extra columns after the offsets; brute-force-checked in
      `tests/test_extra_components.py`). `_optimum_init_values` and `_add_linear_draws` had hard-coded
      lists of linear sites and had to learn about them: any new linear component must be added there too.
    - Persisted across `resume()` (`run_manager` band/driver npz keys; tested, the recurring bug class).
    - **`EchoFit.log_evidence`** (Laplace, set by `optimise()`) was added so users can test whether a
      component is warranted; the recovery test checks it prefers components that are really present.
    - **`optimise()` NaN restarts fixed**: L-BFGS's first, gradient-scaled step from a start far up a
      steep slope could overflow (sigma_drw = inf, log_mdot = 33) to a NaN potential, from which SciPy's line
      search cannot backtrack, so the restart aborted with a NaN (seen on NGC 5548 and in the extra-components
      tests). The objective now returns `_NONFINITE_POTENTIAL` (1e10) with a zero gradient there.

## Known rough edges / things to check before trusting results on real data

- `synthetic.py`'s ground truth is generated with the *same* forward model
  used for fitting: good for verifying the code is self-consistent
  end-to-end, but it is not a substitute for testing on independently
  simulated or real light curves.
- The DRW-Fourier prior (`drw_prior_scale`) is an approximation to an exact
  DRW process. If you need exact DRW likelihoods, consider swapping in a
  Kalman-filter/celerite-style likelihood instead: that's a bigger change
  and would touch `model.py` more than `forward_model.py`.
- `n_freq` / `n_tau` / `tau_max` in `EchoFit.build_grid()` are still simple
  heuristics (log-spaced frequencies from the baseline to a Nyquist-style
  estimate; `tau_max` defaults to half the time baseline). The frequency
  upper bound (`w_max = pi / dt_min`) now comes from
  `grid_utils.estimate_dt_min`, a robust (5th-percentile) estimate of
  observation gaps, shared with `synthetic.py`'s ground-truth grid. This
  replaced an earlier version that used the single *tightest* observed gap,
  which for irregular sampling could blow up `w_max` and put the fit on a
  completely different frequency basis than the data actually supports;
  caught by `tests/test_recovery.py`. Still revisit if fitting real
  campaigns with very different cadences per band; pass `dt_min` explicitly
  to `build_grid()` if the data-driven estimate looks off.
- The pipeline has now been run end-to-end (`pytest`, including an MCMC
  recovery test on synthetic data in `tests/test_recovery.py`), so it's no
  longer purely `py_compile`-checked. One finding from that: NUTS can spend
  most samples pinned at the max-tree-depth ceiling on this model even after
  non-centred reparameterising the driver's `S`/`C` coefficients
  (`model.py`): `inclination`, `sigma_drw`, `tau_drw` recover only loosely
  in the tested synthetic setup even with zero divergences. `log_mdot` (the
  mean-lag-setting parameter) recovers well. Treat the weaker parameters'
  posteriors with appropriate scepticism until this is investigated further;
  `inference.run_mcmc`/`EchoFit.fit` now expose `max_tree_depth` and
  `chain_method` if you want to bound worst-case cost or add cheap
  diagnostic chains (`chain_method="vectorized"`) while digging in.
  Largely resolved since: the diagonal mass matrix was the cause (decision
  #17), and `dense_mass=True` is now the default (decision #21).
- `pyproject.toml` gained `h5netcdf`/`h5py` (a netCDF backend for ArviZ,
  which was already a listed dependency but had no working backend
  installed) so `EchoFit(title=...)` can write `chains.nc`. That write is
  wrapped in a try/except in `echofit.py::_save_chains` and only warns on
  failure -- `chains.npz` (plain numpy) is the dependency-free guaranteed
  artifact, don't remove it even if the netCDF path seems reliable.
- **A single NUTS chain is not sufficient evidence for `lag_mode="free"`
  recovery, even with a driver and zero divergences.** A broad
  `Uniform(0, tau_max)` prior on each `tau_{band}`, combined with the
  driver's own stochastic autocorrelation structure, gives a genuinely
  multimodal likelihood: confirmed directly by running the same fit from 3
  different single-chain seeds at a sparse data budget (35 obs/line,
  `noise_level=0.05`) -- 2 of 3 converged (confidently, 0 divergences,
  tight posterior) to the true lags almost exactly, and the third converged
  just as confidently to values 3-4x too large. Don't trust a single
  chain's point estimate for free-lag bands; run several chains
  (`num_chains=4, chain_method="vectorized"` is cheap on top of a single
  chain) and check Gelman-Rubin R-hat (`numpyro.diagnostics.summary`)
  before trusting any of them -- see
  `tests/test_free_lag_mode.py::test_free_lag_recovery_with_driver_anchor`,
  which does exactly this and documents the sparse-data failure mode in
  its docstring. With more/cleaner data (60 obs/line, `noise_level=0.02`)
  4 chains converge cleanly (R-hat ~1.0) to the true lags -- multimodality
  risk trades off against how constraining the data actually is.
- **Running `pytest` in a background/headless shell can crash on exit if
  matplotlib's default backend is interactive.** `tests/test_response_function_swap.py`
  calls `EchoFit.plot_lightcurve_fits()` without ever closing the returned
  figure; on a machine where matplotlib's default backend is `TkAgg` (true
  on at least one contributor's Mac), this creates real Tk windows against
  the active display even from a non-interactive test run. If that process
  is then killed (e.g. a `timeout` wrapper cutting off a background run
  before the ~10-minute full suite finishes), Python's interpreter teardown
  can hit a Tcl/Tk finalisation bug (`PyEval_RestoreThread: NULL tstate`)
  and abort with `SIGABRT`, a visible crash dialog with no connection to
  whatever test was actually running. Run `pytest` with `MPLBACKEND=Agg`
  set (and give it enough time to finish) in any headless/background
  context to avoid this; it's an invocation-time fix, not a reason to
  force a non-interactive backend inside `plotting.py` itself, which would
  break interactive use from the notebook.
- **Real multi-band data needs `fit_error_model=True`: NGC 5548's full 13
  bands don't fit without it.** Measured (not assumed) on all 13 AGN STORM
  bands: dense NUTS pinned at 1023 leapfrog steps, 86/300 divergences,
  minimum ESS 4/300, and `.optimise()` failed on a non-positive-definite
  Hessian (from 7 bands onwards). With the error model on every band: 2/300
  divergences, minimum ESS 32/300, NUTS and `.optimise()` agree on
  `log_mdot` (1.62 +/- 0.05 vs 1.63). The quoted errors (0.2-0.7% of flux)
  are consistent with within-night scatter; the extra variance is model
  mismatch on longer timescales. See `docs/fitting_guide.md`'s
  `fit_error_model` section. Separately, the download script used to write
  `r_band.txt`/`R_band.txt` and `i_band.txt`/`I_band.txt`, which are one file
  each on macOS's case-insensitive filesystem, so "13-band" fits silently
  read R and I twice and never saw r or i. The Johnson/Cousins files are now
  `B_johnson`/`V_johnson`/`R_cousins`/`I_cousins`, and
  `fit_lightcurves.py` refuses two bands that resolve to the same file.
- **Dependency versions matter more than they look like they should for
  this stack.** `jax`/`jaxlib` are pinned `>=0.4.28,<0.5` (not just
  floored) because an unconstrained range let `poetry install` resolve to
  `jaxlib==0.10.2`, which has no published wheel for this machine's
  platform/Python combination and fails outright; `pip install` happened
  to land on `0.4.38` instead via an indirect constraint from `numpyro`,
  but that's not something to rely on. A `poetry.lock` is committed so
  `poetry install` is fully reproducible regardless -- regenerate it
  (`poetry lock`) if `pyproject.toml`'s dependencies change, don't hand-edit
  it. Separately, `np.trapz` (plain NumPy, not `jnp.trapz`) was removed in
  NumPy 2.x; every plain-NumPy trapezoidal-integral call site now uses the
  same `np.trapezoid if hasattr(np, "trapezoid") else np.trapz` fallback
  already established for `jnp` in `forward_model.py` -- keep using that
  pattern for any new one rather than calling `np.trapz`/`jnp.trapz` bare.

## Useful commands

Poetry-managed (see README.md's "Install" for a from-scratch walkthrough);
`poetry install --extras dev` once, then prefix commands with `poetry run`,
or `poetry shell` (needs the `shell` plugin in Poetry 2.x) to avoid
repeating it. Plain `pip install -e ".[dev]"` also works without Poetry, it
just skips the `poetry.lock` version pinning.

```bash
poetry install --extras dev
poetry run pytest             # full suite, ~15 min (see decision #16 for
                               # why -- pytest -m "not slow" for ~2 min)
poetry run pytest --cov=pycream2 --cov-report=term-missing  # + coverage
poetry run pre-commit install  # one-time: run the fast subset on every commit
poetry run python scripts/smoke_test.py  # quick visual check: fit + save
                                          # plots to smoke_test_output/report.html (~30-50s)
poetry run python scripts/profile_pipeline.py  # one-off perf snapshot: where
                                                # wall time goes across the whole
                                                # pipeline -> profiling_output/
                                                # (gitignored, ~a few minutes)
poetry run jupyter notebook notebooks/demo.ipynb
poetry install --with docs && poetry run mkdocs serve  # docs site preview at
                                                        # http://127.0.0.1:8000;
                                                        # CI deploys it to GitHub Pages
                                                        # (.github/workflows/docs.yml)
```

## Style notes

- Keep files minimal / avoid unnecessary abstraction, per the original spec.
- Prefer `jax.numpy` inside anything that runs under NumPyro's model
  function; plain `numpy` is fine in `synthetic.py` and `plotting.py`, which
  never run inside `jax.jit`/NUTS.
- Physical parameters should stay interpretable (days, Angstrom, degrees,
  solar masses) rather than unit-less/rescaled internally, so priors and
  posterior summaries are directly readable.
