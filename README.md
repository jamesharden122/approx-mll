# approx-mll

Approximate models and utilities for stochastic characteristic functions in Mojo.

## Overview
- Implements an SVJ-style joint characteristic function (`Svj1JointCF`) and SIMD-friendly complex utilities (`ComplexSIMD`, `CFGrid`).
- Intended for Fourier-domain methods (e.g., inverse Fourier transforms) and grid-based quadrature.

## Project Layout
- `stochcharfunc/`: Stochastic characteristic functions.
  - `svj1.mojo`: Parameters and accessors for the SVJ model interface.
  - `__init__.mojo`: Re-exports `Svj1JointCF` plus complex SIMD helpers.
- `complexsimd/`: Complex SIMD data types and grids.
  - `complex_simd.mojo`: `ComplexSIMD[T, L]` with add/sub/mul, conjugate, and `exp(i·θ)` helpers.
  - `cf_grid.mojo`: `CFGrid[T, L]` tiled storage with lane load/store.
- `papers/`: Primary method reference(s) used in this module.

## Run
- Enter the environment and run the example:
  - `cd approx-mll && pixi shell && mojo run main.mojo`
- Use from code:
  - `from stochcharfunc import Svj1JointCF, ComplexSIMD, CFGrid`

## Build
- Install env: `cd approx-mll && pixi install`
- Run via Pixi: `pixi run mojo main.mojo` (or `pixi shell && mojo run main.mojo`).
- Optimized run: `mojo run -O main.mojo` inside the Pixi shell.
- Tip: run from the `approx-mll` folder so module imports (`stochcharfunc`, `complexsimd`) resolve.

### Build an executable (mojo build)
- Basic: `cd approx-mll && pixi shell && mojo build main.mojo`
- Optimized and named output: `mojo build -O main.mojo -o build/approx-mll`
- Include paths (if building from elsewhere): `mojo build -I . main.mojo`
- Other outputs: `--emit shared-lib | object | asm | llvm | py-ext`

Run the built binary: `./build/approx-mll`

Note: Python libraries used by Mojo aren’t bundled; they must exist in the runtime environment.

## Reference
- Bates, D. (2006), Review of Financial Studies — Maximum-likelihood estimation approach for latent affine/jump-diffusion models. See `papers/Bates mle RFS 06.pdf`.
  - This repository follows the Bates-style method as the baseline reference for the characteristic-function workflow (calibration and inference context), adapted to Mojo and SIMD primitives.

## Notes
- Mojo and MAX versions are pinned in `pixi.toml`.
- Keep large datasets and calibration files out of the repo; document local paths in PRs.
