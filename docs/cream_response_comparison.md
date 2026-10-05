# pycream2 and CREAM: comparing the thin-disc response functions

This page compares pycream2's thin-disc response function,
`forward_model.thin_disk_response`, with the one in the original CREAM
Fortran code (subroutine `tfbx` in `cream_f90.f90`, Starkey, Horne & Villforth
2016). The question behind it is why CREAM's published response functions all
start at zero at zero lag, while pycream2's, especially at high inclination,
start at a finite value.

In short:

- **CREAM forces the first lag bin of every response to zero.** It is a
  hard-coded line at the end of `tfbx`, not a property of the disc. The
  response rises almost vertically in both codes.
- **With the same smoothing, pycream2 reproduces CREAM's response shapes
  closely at every inclination tested.** Mean and median delays agree to
  within about 4 per cent, pycream2's being about 0.1 days shorter.
- **The remaining differences come from deliberate modelling choices.** They
  are the treatment of the inner edge of the disc and the exact form of the
  lamppost dilution.
- **CREAM does not put the lamppost height into the delay,** and nor did
  pycream2 until this comparison.
- **The finite response at zero lag in pycream2 came from its default
  smoothing.** That smoothing was about ten times wider than CREAM's on a
  typical lag grid.
- **All three follow-ups are now implemented** (September 2026; see
  [What changed](#what-changed)):
  - the default smoothing is causal and much narrower, so the response is
    zero at zero lag;
  - the delay surface includes the lamppost height.

## How the comparison was made

`scripts/compare_cream_response.py` does the following:

1. It extracts `tfbx` and the helpers it calls verbatim from `cream_f90.f90`.
   The helpers are `f_rs2ld`, `tr4visc`, `tr4irad`, `myedrat2mdot`, `ranu`
   and `ran3`.
2. It compiles them with gfortran next to a small driver program.
3. It runs them on the same lag grid as `thin_disk_response`.

Both codes get the same disc:

| Quantity | Value |
|---|---|
| Black-hole mass | $10^{7.51}\mkern3mu M_\odot$ (NGC 5548; Pancoast et al. 2014) |
| Temperature at one light-day, $T_1$ | 19,400 K (pycream2's `log_mdot` = 3.079) |
| Temperature slope | 3/4 (CREAM: `slope_v` = `slope_i` = 0.75) |
| Inner radius | $3R_\mathrm{S}$ (CREAM: `rinsch` = 1) |
| Wavelength | 5000 Å |
| Inclinations | 0, 30, 60 and 80 degrees |
| Lag grids | 0 to 10 days: coarse (201 points, 0.05 d) and fine (2001 points, 0.005 d) |

CREAM's temperature is given by its viscous term alone (irradiation
temperature `T1i` = 0). Its response weight still carries the lamppost
dilution, as pycream2's does.

For the like-for-like comparison, pycream2 uses CREAM's smoothing convention,
a Gaussian as wide as the lag-grid spacing (`smoothing_days` = the grid
spacing), and CREAM's delay surface, with no lamppost height
(`delay_lamppost_height=False`). pycream2's old and new defaults are shown
separately:

- **old default:** a Gaussian in τ, 0.1 of the Wien radius wide, with no
  lamppost height in the delay;
- **new default:** a 5 per cent Gaussian in ln τ, with the exact lamppost
  delay.

The figure normalises each response to its peak, as CREAM does.

Reproduce it with:

```bash
MPLBACKEND=Agg poetry run python scripts/compare_cream_response.py \
    --cream-source ../pycecream/pycecream/cream_f90.f90
```

This writes `docs/images/cream_response_comparison.png` and the numbers in
`docs/images/cream_response_comparison.json`.

![CREAM and pycream2 thin-disc responses](images/cream_response_comparison.png)

*Each response normalised to its peak. Black: CREAM's `tfbx`. Red dashed:
pycream2 with CREAM's smoothing and delay. Blue dotted: pycream2's old
default. Green dash-dotted: pycream2's new default. Top row: 0.05-day lag grid.
Bottom row: 0.005-day lag grid. At 80 degrees on the coarse grid the new
default's tail sits lower only because its sharper peak is resolved; its
mean and median delays match the red curve's.*

## Finding 1: CREAM forces the response to zero at zero lag

The last lines of `tfbx` are:

```fortran
psi(1) = 0.0
psimax = maxval(psi)
do it = 1,Ntau
psi(it) = psi(it)/psimax
enddo
! how to deal with any negative time lag bins
if (ibzero .gt. 1) psi(1:ibzero-1) = 0
```

So CREAM makes three adjustments after computing the response:

1. It sets the first lag bin to zero.
2. It normalises the response to its peak.
3. It zeroes every bin at negative lag.

A plotted CREAM response therefore always starts at zero at the first grid
point, whatever the disc's parameters. The next bin shows what the disc is
actually doing. On a 0.05-day grid that bin is already at 46 per cent of the
peak face-on, and at 100 per cent at 80 degrees:

| Grid | Inclination (deg) | CREAM: mean / median delay (d) | pycream2, CREAM's smoothing and delay | pycream2, old default | pycream2, new default | CREAM second bin / peak | pycream2 ψ(0) / peak, old / new | Largest shape difference |
|---|---|---|---|---|---|---|---|---|
| 0.05 d | 0 | 2.82 / 2.30 | 2.72 / 2.20 | 2.73 / 2.25 | 2.73 / 2.20 | 0.46 | 0.90 / 0.00 | 0.14 |
| 0.05 d | 30 | 2.68 / 2.10 | 2.59 / 2.00 | 2.61 / 2.00 | 2.60 / 2.00 | 0.51 | 0.94 / 0.00 | 0.12 |
| 0.05 d | 60 | 2.47 / 1.65 | 2.38 / 1.55 | 2.44 / 1.65 | 2.39 / 1.55 | 0.68 | 1.00 / 0.00 | 0.04 |
| 0.05 d | 80 | 2.44 / 1.60 | 2.34 / 1.50 | 2.46 / 1.60 | 2.36 / 1.50 | 1.00 | 1.00 / 0.00 | 0.06 |
| 0.005 d | 0 | 2.81 / 2.31 | 2.71 / 2.21 | 2.73 / 2.23 | 2.73 / 2.22 | 0.05 | 0.94 / 0.00 | 0.28 |
| 0.005 d | 30 | 2.67 / 2.08 | 2.58 / 1.98 | 2.61 / 2.02 | 2.60 / 2.00 | 0.10 | 0.97 / 0.00 | 0.25 |
| 0.005 d | 60 | 2.44 / 1.62 | 2.37 / 1.54 | 2.44 / 1.63 | 2.38 / 1.55 | 0.29 | 1.00 / 0.00 | 0.15 |
| 0.005 d | 80 | 2.36 / 1.50 | 2.28 / 1.41 | 2.43 / 1.57 | 2.29 / 1.41 | 0.64 | 1.00 / 0.00 | 0.03 |

*Delays are computed over the 0 to 10-day grid for all three responses. "Largest shape
difference" is the largest absolute difference between CREAM and pycream2 (same
smoothing and delay), both normalised to their peaks, after CREAM's forced first bin.*

pycream2's unsmoothed response is also exactly zero at zero lag: its 1-D
reduction (section 4 of [Thin-disk response function](thin_disk_response.md))
carries a factor of τ. But it rises within about 0.003 to
0.015 days, because the lamppost's $h/d^3$ dilution weights the response
towards the inner disc. In the Rayleigh–Jeans limit the response grows as
$\tau^{1/4}$ near zero lag. At high inclination, the near side of every ring
also arrives at a delay of only $r(1-\sin i)$. So neither code's disc produces
a gradual rise from zero. CREAM's zero is imposed on the first grid point.

## Finding 2: with the same smoothing, the shapes agree

With CREAM's smoothing and delay, pycream2's response follows CREAM's closely
at every inclination and on both grids (red dashed against black in the
figure). That
includes the sharp near-side spike at 80 degrees and the long far-side tail.
Mean and median delays agree to within about 4 per cent. On the fine grid,
CREAM's response shows scatter from bin to bin, because `tfbx` places each
disc element at a random azimuth (`ranu`). pycream2's integral over azimuth is
evaluated exactly, so its response is smooth.

## Finding 3: the remaining differences are modelling choices

The largest shape differences (0.25 to 0.28 of the peak) are at lags below
about 0.5 days for a nearly face-on disc. There pycream2's response rises
faster, which also makes its mean delay about 0.1 days shorter. The codes
differ in two ways that affect the innermost disc, where these lags come from:

- **The inner edge.**
  - pycream2's temperature profile includes the zero-torque factor
    $(1-\sqrt{r_\mathrm{in}}/\sqrt{r})^{1/4}$.
  - Its response starts at the temperature peak, $1.36\mkern3mu r_\mathrm{in}$ for
    slope 3/4 (see [Thin-disk response function](thin_disk_response.md)).
  - In CREAM the zero-torque factor is commented out of the temperature used
    by `tfbx`, and the radial integral starts at $r_\mathrm{in}$.
  - pycream2's inner disc is therefore cooler. At 5000 Å a cooler inner ring
    responds more strongly: in the Rayleigh–Jeans limit the weight scales as
    $T^{-3}$.
  - This is the most likely cause of the faster rise. We have not isolated it
    by switching the factor off.
  - Since October 2026 pycream2's default response mixes in lamppost
    irradiation (`include_irradiation=True`), which keeps the inner disc hot
    and removes this inner-edge response. The comparison on this page uses the
    viscous-only response (`include_irradiation=False`), as CREAM is run here
    with no irradiation temperature.
- **The dilution.**
  - pycream2 uses the exact lamppost geometry, $h/d^3$ with
    $d=\sqrt{r^2+h^2}$.
  - CREAM uses its far-field form, $(r_0/r)^{4\alpha_\mathrm{irr}}$, which is
    $r^{-3}$ for the standard irradiation slope.
  - The two differ only for $r\lesssim h$. With $h = 3R_\mathrm{S}$ that is
    about 0.01 light-days for NGC 5548, so it matters little here.

**CREAM does not include the lamppost height in the delay,** and pycream2's
delay was the same until this comparison. CREAM's delay is
`taulag = rnow*(1.+ sininc*cosaz)`, as in Starkey, Horne & Villforth (2016)
eq. 5. `tfbx` computes the lamppost height in
light-days (`xhld`) but never uses it. With the height included, the delay
becomes $\tau = \sqrt{r^2+h^2} + h\cos i + r\sin i\cos\phi$ (over $c$), and no
light arrives before roughly $h(1+\cos i)/c$. For NGC 5548 that is about
0.02 days, far too small to change the response's appearance on a scale of
days, but it is the correct geometry, and pycream2 now uses it (see
[What changed](#what-changed)).

## Finding 4: pycream2's old default smoothing was what departed from CREAM

pycream2's default smoothing width was `smoothing_frac` = 0.1 of the Wien
radius. That is about 0.5 days at 5000 Å for this disc, ten times CREAM's
width on a 0.05-day grid and a hundred times it on a 0.005-day grid.

The Gaussian spreads the near-vertical onset of the response across zero lag.
The part that falls below zero is discarded, so the smoothed response has a
finite value at zero lag. At high inclination, where the unsmoothed response
peaks within a few hundredths of a day, the smoothed response peaks at zero
lag. That is the behaviour that prompted this comparison.

By mean and median delay, that smoothing changed little: 2.73 against
2.82 days face-on, and 2.46 against 2.44 at 80 degrees. But it changes the
shape near the peak a lot. The broad blue curves at 60 and 80 degrees in the
figure partly reflect normalising to a peak that the smoothing lowers. The
response's shape near the peak is what carries the information on
inclination. In the NGC 5548 analysis, switching the smoothing off moved the
thin-disc inclination from about 40 to about 34 degrees.

## What changed

All three follow-ups are implemented in `forward_model.thin_disk_response`
(and the table-based fast path, `thin_disk_response_from_table`), with tests
in `tests/test_thin_disk_response.py`.

1. **Causal smoothing by default.** The default is now a local Gaussian
   average in ln τ, of width `smoothing_log` = 0.05 (5 per cent in delay at
   every delay).
   - Nothing is moved below zero lag, so the response is exactly zero at, and
     before, τ = 0 (ψ(0)/peak = 0.00 in the table, against 0.90 to 1.00 with
     the old default).
   - It is also much narrower where it matters: near the onset its width is a
     small fraction of a day, and in the tail it is comparable to the old
     one's.
   - It shifts the mean delay by only $e^{\sigma^2/2}-1 \approx 0.1$ per cent.
   - The Gaussian in τ is still available: pass `smoothing_days` (CREAM's
     convention is the grid spacing) or `smoothing_frac`.
2. **The lamppost height is in the delay surface.** The delay is now
   $\tau = \sqrt{r^2+h^2} + h\cos i + r\sin i\cos\phi$ (over $c$).
   - At fixed azimuth the radii with a given delay are the roots of a
     quadratic in $r$. On the far side there is one root; on the near side
     there can be two, because the delay first falls and then rises with
     radius. Each root is weighted by $r/|d\tau/dr|$. So the 1-D reduction
     remains exact and analytic.
   - Nothing responds before the shortest lamppost–disc–observer path.
   - The mean delay becomes $\langle d\rangle + h\cos i$ rather than being
     exactly independent of inclination: a shift of about 0.01 days for
     NGC 5548. $\langle\tau\rangle - h\cos i$ stays independent of
     inclination to within 1 per cent.
   - `delay_lamppost_height=False` recovers CREAM's delay surface for
     comparisons.
3. **Much less smoothing.** This comes from item 1 by construction. The new
   default's mean and median delays match the unsmoothed, CREAM-like response
   to within 0.02 days at every inclination tested.

One consequence worth knowing: without the old smoothing, the peak of a
lamppost-irradiated thin disc's response sits at its inner edge at every
wavelength, where the zero-torque cooling and the $h/d^3$ dilution make the
innermost ring respond most strongly. It is the mean and median delays, not
the peak, that grow with wavelength. The old smoothing only moved the peak
outwards. This applies to the viscous-only response: with irradiation, the
default since October 2026, the peak moves outwards with wavelength too (see
[Thin-disk response function](thin_disk_response.md)).

Every thin-disc fit made before these changes, including the NGC 5548
analysis, used the old default and needs redoing.
