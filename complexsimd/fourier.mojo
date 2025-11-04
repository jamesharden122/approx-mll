from complexsimd.complex_simd import ComplexSIMD
from complexsimd.cf_grid_simd import CFGridSIMD
from stochcharfunc.svj1 import Params, SvSpec
from math import sin, cos

# SIMD inverse Fourier helpers:
# - Keeps SvSpec bound for the CF provider type.
# - Computes phi per tile/lane from (u0, du).
# - Uses reduce_sum across lanes.


# Note: Generic trait declarations with parameters are limited; we omit a trait here.


struct UniformGridInverterSIMDGrid[L: Int](Copyable, Movable):
    var grid: CFGridSIMD[DType.float64, L]

    fn __init__(out self, grid: CFGridSIMD[DType.float64, L]):
        self.grid = grid.copy()

    fn _inverse_transform_tile_simd[S: SvSpec, T: Int](
        self,
        cb: S,
        x: Float64,
        psi: SIMD[DType.float64, 1],
        prior_params: InlineArray[Float64, T],  # [y, kappa, nu]
        tile_idx: Int,
    ) -> ComplexSIMD[DType.float64, L]:
        # Build phi lanes: phi_j = u0 + du * (tile_idx * L + j)
        var phi = self.grid.re_tiles[tile_idx] - self.grid.re_tiles[tile_idx].shift_left[1]()
        # Model CF (real and imaginary parts) evaluated at phi
        var re = cb.predictive_cf_latent_simd_re[L, T](phi, psi, prior_params)
        var im = cb.predictive_cf_latent_simd_im[L, T](phi, psi, prior_params)

        # Multiply by e^{-i phi x}
        var theta = phi * SIMD[DType.float64, L](-x)
        var c = cos(theta)
        var s = sin(theta)
        var tile_re = re * c - im * s
        var tile_im = re * s + im * c.shift_left[1]()
        return ComplexSIMD[DType.float64, L](tile_re, tile_im)

    fn inverse_at[S: SvSpec, T: Int](
        self,
        x: Float64,
        cb: S,
        psi: SIMD[DType.float64, 1],
        prior_params: InlineArray[Float64, T],  # [y, kappa, nu]
        normalize: Bool = True,
    ) -> ComplexSIMD[DType.float64, 1]:
        var two_pi: Float64 = 6.283185307179586
        var tiles: Int = self.grid.num_tiles()
        var acc_re: Float64 = 0.0
        var acc_im: Float64 = 0.0
        var scale: Float64 = (1.0 / two_pi) if normalize else 1.0

        for t in range(tiles):
            var z_tile = self._inverse_transform_tile_simd[S, T](cb, x, psi, prior_params, t)
            var sum_re = z_tile.re.reduce_add()[0]
            var sum_im = z_tile.im.reduce_add()[0]
            acc_re += Float64(sum_re) * scale
            acc_im += Float64(sum_im) * scale

        return ComplexSIMD[DType.float64, 1].from_scalars(acc_re, acc_im)

    # inverse_at_with_psi intentionally omitted; use inverse_at with SIMD psi.

    # Optimized: compute ψ = {-h, 0, +h} in ONE pass over tiles
    fn inverse_three_psi[S: SvSpec, T: Int](
        self,
        u0: Float64,
        du: Float64,
        cb: S,
        prior_params: InlineArray[Float64, T],  # [y, kappa, nu]
        h: Float64,
        normalize: Bool = True,
    ) -> (Float64, Float64, Float64):  # returns Re at (-h, 0, +h)
        var two_pi: Float64 = 6.283185307179586
        var tiles: Int = self.grid.num_tiles()
        var scale: Float64 = (1.0 / two_pi) if normalize else 1.0
        var acc_mh: Float64 = 0.0
        var acc_0:  Float64 = 0.0
        var acc_ph: Float64 = 0.0
        for t in range(tiles):
            # Build phi lanes once
            var phi = self.grid.re_tiles[t] - self.grid.re_tiles[t].shift_left[1]()
            # Precompute exp(-i phi x) parts once
            var theta = phi * SIMD[DType.float64, L](-prior_params[0])
            var c = cos(theta)
            var s = sin(theta)

            # ψ = -h, 0, +h evaluations
            var re_mh = cb.predictive_cf_latent_simd_re[L, T](phi, SIMD[DType.float64, 1](-h), prior_params)
            var im_mh = cb.predictive_cf_latent_simd_im[L, T](phi, SIMD[DType.float64, 1](-h), prior_params)
            var re_0  = cb.predictive_cf_latent_simd_re[L, T](phi, SIMD[DType.float64, 1](0.0), prior_params)
            var im_0  = cb.predictive_cf_latent_simd_im[L, T](phi, SIMD[DType.float64, 1](0.0), prior_params)
            var re_ph = cb.predictive_cf_latent_simd_re[L, T](phi, SIMD[DType.float64, 1](h), prior_params)
            var im_ph = cb.predictive_cf_latent_simd_im[L, T](phi, SIMD[DType.float64, 1](h), prior_params)

            # rotate by e^{-i phi x}
            var mh_re = re_mh * c - im_mh * s
            var zero_re = re_0  * c - im_0  * s
            var ph_re = re_ph * c - im_ph * s

            acc_mh += Float64(mh_re.reduce_add()) * scale
            acc_0  += Float64(zero_re.reduce_add()) * scale
            acc_ph += Float64(ph_re.reduce_add()) * scale

        return (acc_mh, acc_0, acc_ph)
