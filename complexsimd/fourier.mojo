from complexsimd.complex_simd import ComplexSIMD
from complexsimd.cf_grid_simd import CFGridSIMD
from stochcharfunc.svj1 import Params, SvSpec, Svj1Params, SVj1JointCF
from math import sin, cos

# SIMD-only inverse Fourier helpers with no per-lane loops.
# Use two SIMD CF callbacks (real and imaginary) and reduce across lanes.


trait FourierGridSIMD[L: UInt,T: SvSpec, P: Params]:
    fn inverse_at(
        self,
        x: Float64,
        u0: Float64,
        du: Float64,
        cb: T,
        params: P,
        normalize: Bool = True
    ) -> ComplexSIMD[DType.float64, 1]

    fn _inverse_transform_tile_simd(
        self,
        cb: T,
        lane_idx: UInt
    )

struct GridInverterSIMDGrid[L: UInt](FourierGridSIMD[L]):
    var grid: CFGridSIMD[DType.float64, L]

    fn __init__(out self, grid: CFGridSIMD[DType.float64, L]):
        self.grid = ^grid

    fn _inverse_transform_tile_simd(
        self,
        cb: Svj1VecCF,
        y: Float64,
        lane_idx: UInt,
    ) -> ComplexSIMD[DType.float64, L]:
        let re = cb.predictive_cf_latent_simd_re[L](grid.re_tiles[lane_idx])
        let im = cb.predictive_cf_latent_simd_im[L](grid.im_tiles[lane_idx])
        let phi_tile = grid.re_tiles[lane_idx]
        let theta = -y * phi_tile          
        let c = cos(theta)
        let s = sin(theta)
        let tile_re = re * c - im * s
        let tile_im = re * s + im * c
        return ComplexSIMD[DType.float64, L](tile_re, tile_im)

   fn inverse_at(
        self,
        x: Float64,               # y* at which to evaluate p(y*)
        cb: Svj1VecCF,
        cf_params: AnyType,       # if your cb needs params; pass through or remove if not used
        normalize: Bool = True,
    ) -> ComplexSIMD[DType.float64, 1]:
        let two_pi = 6.283185307179586
        let tiles = self.grid.num_tiles()
        var acc_re: Float64 = 0.0
        var acc_im: Float64 = 0.0

        for t in range(tiles):
            var z_tile = self._inverse_transform_tile_simd(cb, x, t)

            var re = (z_tile.re * t.re_tiles[t])/(two_pi if normalize else 1.0)
            var im = (z_tile.im * t.im_tiles[t])/ (two_pi if normalize else 1.0)
            # sum across the 8 lanes in this tile
            acc_re += re.reduce_add()
            acc_im += im.reduce_add()

        # Optional: if you know symmetry should kill the imaginary part:
        # acc_im = 0.0  # or just keep the tiny residue

        return ComplexSIMD[DType.float64, 1].from_scalars(acc_re, acc_im) 
