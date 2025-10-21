from complexsimd.complex_simd import ComplexSIMD
from complexsimd.cf_grid_simd import CFGridSIMD
from stochcharfunc.svj1 import Params, SvSpec
from math import sin, cos

# SIMD inverse Fourier helpers:
# - Keeps SvSpec bound for the CF provider type.
# - Computes phi per tile/lane from (u0, du).
# - Uses reduce_sum across lanes.


# Note: Generic trait declarations with parameters are limited; we omit a trait here.


struct UniformGridInverterSIMDGrid[L: Int]:
    var grid: CFGridSIMD[DType.float64, L]

    fn __init__(out self, grid: CFGridSIMD[DType.float64, L]):
        self.grid = grid.copy()

    fn _inverse_transform_tile_simd[T: SvSpec](
        self,
        cb: T,
        x: Float64,
        u0: Float64,
        du: Float64,
        prior_params: List[Float64],  # [y, kappa, nu]
        tile_idx: Int,
    ) -> ComplexSIMD[DType.float64, L]:
        # Build phi lanes: phi_j = u0 + du * (tile_idx * L + j)
        var phi = SIMD[DType.float64, L]()
        for j in range(L):
            var idx = tile_idx * L + j
            phi[j] = u0 + du * Float64(idx)

        # Model CF (real and imaginary parts) evaluated at phi
        var re = cb.predictive_cf_latent_simd_re[L](phi, prior_params)
        var im = cb.predictive_cf_latent_simd_im[L](phi, prior_params)

        # Multiply by e^{-i phi x}
        var theta = phi * SIMD[DType.float64, L](-x)
        var c = cos(theta)
        var s = sin(theta)
        var tile_re = re * c - im * s
        var tile_im = re * s + im * c
        return ComplexSIMD[DType.float64, L](tile_re, tile_im)

    fn inverse_at[T: SvSpec](
        self,
        x: Float64,
        u0: Float64,
        du: Float64,
        cb: T,
        prior_params: List[Float64],  # [y, kappa, nu]
        normalize: Bool = True,
    ) -> ComplexSIMD[DType.float64, 1]:
        var two_pi: Float64 = 6.283185307179586
        var tiles: Int = self.grid.num_tiles()
        var acc_re: Float64 = 0.0
        var acc_im: Float64 = 0.0
        var scale: Float64 = (1.0 / two_pi) if normalize else 1.0

        for t in range(tiles):
            var z_tile = self._inverse_transform_tile_simd[T](cb, x, u0, du, prior_params, t)
            var sum_re = z_tile.re.reduce_add()[0]
            var sum_im = z_tile.im.reduce_add()[0]
            acc_re += Float64(sum_re) * scale
            acc_im += Float64(sum_im) * scale

        return ComplexSIMD[DType.float64, 1].from_scalars(acc_re, acc_im)
