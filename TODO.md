# approx-mll TODO

## Required pipeline contracts

- [ ] Keep the canonical `AffineSvjParams` order and the 104-byte artifact
  stable; introduce a new schema version before any incompatible change.
- [ ] Keep physical dynamics, compensator convention, characteristic function,
  simulator, and documentation covered by one cross-project consistency test.
- [ ] Keep `dt` immutable throughout every optimizer evaluation.
- [ ] Keep the existing simulations `GeomBM` API and FFI smoke path passing.

## Numerical follow-up

- [ ] Add adaptive frequency-range selection based on `dt` and the current
  predictive variance, while retaining the explicit fixed-grid option for
  reproducibility.
- [ ] Investigate candidate-specific stationary `[kappa, nu]` initialization
  against the current deliberately frozen initial-artifact prior, including
  effects on objective smoothness and recovery bias.
- [ ] Compare uniform trapezoidal inversion with higher-order quadrature and an
  FFT batch implementation.
- [ ] Add stable analytic complex-Riccati evaluation as a benchmark against the
  current correctness-first implementation.
- [ ] Quantify Gamma moment-matching error against particle filtering for jump
  and leverage regimes outside the recovery fixture.
- [ ] Add common-random-number and parallel-seed modes to reduce the cost and
  variance of recovery studies.
- [ ] Evaluate automatic or complex-step differentiation after Mojo exposes a
  suitable generic path; retain central differences as the reference gradient.

## Engineering follow-up

- [ ] Add versioned release fixtures for at least one parameter artifact and
  one complete small simulator output set.
- [ ] Add performance benchmarks by SIMD width, Fourier grid size, observation
  count, path count, and intraday step count.
- [ ] Add transactional temporary-file rename support to the simulator command
  when the standard library offers a portable atomic rename workflow.
- [ ] Add Windows and macOS artifact-byte CI when those Mojo targets are
  supported by the pinned environment.
- [ ] Decide on a public license and stability policy before publishing the
  package for downstream use.

## Release gate

- [x] `pixi run check` and `pixi run test` pass from the locked environment.
- [x] The sibling simulations fast suite passes with cross-project imports.
- [ ] `pixi run recovery` passes the 20-seed finite-convergence, likelihood-
  improvement, and three-standard-error aggregate-bias checks.
  The executable is compile-checked; the full statistical run remains pending.
- [x] README command lines, default values, validation status, and remaining
  limitations match the shipped implementation.
