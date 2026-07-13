# approx-mll

`approx-mll` is a Mojo implementation of a physical-measure affine
stochastic-volatility jump (SVJ) model, its Bates-style approximate maximum
likelihood (AML) filter, and a parameter artifact shared with the sibling
`simulations` project.

The code is research software. It is suitable for deterministic synthetic-data
experiments and numerical validation, but it is not a production trading or
risk engine.

## Model

Let `X = log(S)`. Under the physical measure the implemented dynamics are

```text
dV = beta * (alpha - V) dt + volvol * sqrt(V) dWv
dX = [mu0 + mu1*V - V/2 - k*(lambda0 + lambda1*V)] dt
     + sqrt(V) dWs + J dN
corr(dWs, dWv) = rho
J ~ Normal(jump_mean, jump_std^2)
intensity(N) = lambda0 + lambda1*V
k = exp(jump_mean + jump_std^2/2) - 1
```

Thus `mu0 + mu1*V` is the instantaneous expected proportional-return drift.
The jump compensator `k` is derived; it is never fitted or stored.

The canonical parameter order is:

| Index | Field | Units / constraint |
| ---: | --- | --- |
| 0 | `mu0` | return / year, finite |
| 1 | `mu1` | return per variance-year, finite |
| 2 | `lambda0` | jumps / year, nonnegative |
| 3 | `lambda1` | jumps per variance-year, nonnegative |
| 4 | `jump_mean` | log return per jump, finite |
| 5 | `jump_std` | log return per jump, nonnegative |
| 6 | `rho` | correlation, strictly between -1 and 1 |
| 7 | `alpha` | long-run annualized variance, positive |
| 8 | `beta` | mean-reversion rate / year, positive |
| 9 | `volvol` | variance diffusion scale / sqrt(year), positive |
| 10 | `dt` | years per observation, positive |

`dt` is fixed sampling metadata. The optimizer transforms and fits the first
ten fields only.

## Shared parameter artifact

Both projects import the dependency-light `affine_svj` package owned by this
repository. Parameter files are exactly 104 bytes and use network (big-endian)
byte order:

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 8 | ASCII magic `AMLSVJ1\0` |
| 8 | 2 | schema version `1` (`UInt16`) |
| 10 | 2 | model ID `1` (`UInt16`) |
| 12 | 2 | physical-measure ID `1` (`UInt16`) |
| 14 | 2 | field count `11` (`UInt16`) |
| 16 | 88 | eleven canonical `Float64` values |

Readers reject the wrong length (including trailing bytes), unsupported
headers, non-finite values, and invalid parameter domains.

## Build and fast tests

The pinned environment is Linux x86-64 with Mojo `1.0.0b1`:

```bash
pixi install --locked
pixi run check
pixi run test
```

Run the sibling project fast suite separately:

```bash
cd ../simulations
pixi run test
```

The fast suite covers artifact bytes and rejection paths, scalar/SIMD
characteristic-function invariants, Fourier density and Gamma updates,
likelihood and central-gradient comparisons, deterministic simulation,
stationary CIR and jump moments, and big-endian output layout.

Verified on 2026-07-12: the approx-mll check, example, and 26 fast tests
passed; the simulations GBM/FFI smoke paths and nine SVJ tests passed; both
CLI help paths and a simulator-to-fit-to-simulator artifact round trip passed.

The intentionally expensive recovery experiment is separate:

```bash
pixi run recovery
```

It uses 20 fixed seeds, 4,000 daily returns per seed, 50 intraday CIR steps,
and deterministic perturbations of all ten fitted coordinates. It checks
finite likelihood improvement and aggregate parameter bias against three
empirical Monte Carlo standard errors.

The executable was compile-checked on 2026-07-12 but the full experiment
was not run during this implementation pass. Recovery validation is therefore
not yet claimed, and the release gate remains open.

## Generate synthetic SVJ data

From `../simulations`, run the affine simulator command with a parameter
artifact created by the codec or fitter:

```bash
pixi run simulate-svj -- \
  --params /path/to/physical_params.bin \
  --out-dir data/svj \
  --paths 32 --steps 4000 --seed 1729 --s0 100 \
  --intraday-steps 50
```

Add `--v0 0.04` to give every path the same initial variance. Without it,
each path receives an independent draw from the stationary CIR Gamma law. Add
`--force` to replace an existing complete output set; otherwise the command
refuses to touch the directory if any target file already exists.

The command writes headerless, path-major, big-endian `Float64` arrays:

| File | Shape |
| --- | --- |
| `prices.bin` | `(paths, steps + 1)` |
| `log_returns.bin` | `(paths, steps)` |
| `variance.bin` | `(paths, steps + 1)` |

The simulator leaves the existing compile-time `GeomBM` and Brownian-bridge
APIs intact. It advances variance with exact CIR transition draws using the
Poisson-Gamma representation of the noncentral chi-square law, integrates
variance by the trapezoidal rule over the intraday grid, samples jumps
conditional on that integrated variance, and uses the conditional orthogonal
Gaussian return equation.

## Fit an artifact

The fitter reads a headerless big-endian `Float64` log-return series and an
initial parameter artifact:

```bash
pixi run fit-svj -- \
  --returns ../simulations/data/svj/log_returns.bin \
  --count 4000 \
  --initial /path/to/initial_params.bin \
  --out fitted_params.bin \
  --max-iterations 40 --grid-count 2049 --grid-du 0.29296875
```

Grid and optimizer flags override defaults without changing `dt`. The output
uses the same 104-byte artifact format and can be passed directly back to the
simulator.

The filter performs positive-frequency Fourier inversion of the affine joint
characteristic function, including explicit frequency spacing, trapezoidal
endpoint weights, padded-lane masks, symmetry, and `1/pi` normalization. After
each observed log-return increment it matches the first two conditional
variance moments to a Gamma prior represented as `[kappa, nu]` (scale, shape).

Optimization occurs in unconstrained coordinates: logarithms enforce positive
parameters, `tanh` enforces the correlation interval, and unconstrained drift
and jump-mean coordinates remain linear. Gradients use central differences and
accepted steps use Armijo likelihood ascent. The structured fit result reports
parameters, likelihood, gradient norm, iterations, convergence reason, and the
final filtered Gamma parameters.

The fit command derives the stationary starting `[kappa, nu]` from the initial
artifact once and intentionally freezes that starting prior across every
candidate-parameter likelihood evaluation. Each candidate still performs the
sequential Gamma updates over the observed returns; only its starting prior is
shared. Candidate-specific stationary initialization remains a numerical
follow-up.

## Numerical limitations

- Fourier inversion is a truncated uniform-grid quadrature. Very small `dt`,
  extreme jump tails, and parameters near the admissible boundary may require
  a wider or denser grid.
- Gamma moment matching is an approximation to the filtered variance law and
  can lose accuracy after unusually large returns.
- Central differences are robust and transparent but expensive: one gradient
  needs two likelihood evaluations for each of ten free coordinates.
- Exact CIR transitions do not remove discretization error from integrated
  variance; that integral still uses the requested intraday trapezoidal grid.
- The slow recovery task is a statistical regression, not a proof of global
  identifiability or convergence from arbitrary starting values.

See [TODO.md](TODO.md) for remaining engineering and numerical work.
