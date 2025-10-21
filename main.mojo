from stochcharfunc import Svj1JointCF
from complexsimd import (
    ComplexSIMD,
    CFGridSIMD,
    UniformGridInverterSIMDGrid,
)

fn main():
    # Model params (example values)
    var y: Float64 = 0.01
    var kappa: Float64 = 0.5
    var nu: Float64 = 2.0
    var u0: Float64 = 0.0
    var du: Float64 = 0.01
    var K: Int = 1024            # frequency samples (multiple of L)
    var Ltiles: Int = 8          # SIMD width
    var tiles: Int = Int(K / Ltiles)  # number of SIMD tiles

    var model = Svj1JointCF()

    # Build a dummy CFGridSIMD with the desired tile shape (content is sampled via callbacks);
    # here it's only used to convey tile count and layout to the inverter.
    var re_tiles = List[SIMD[DType.float64, 8]]()
    var im_tiles = List[SIMD[DType.float64, 8]]()
    for _ in range(tiles):
        re_tiles.append(SIMD[DType.float64, 8]())
        im_tiles.append(SIMD[DType.float64, 8]())
    var k_valid = List[Int]()
    k_valid.append(tiles * 8)
    var grid = CFGridSIMD[DType.float64, 8](re_tiles, im_tiles, k_valid)

    # Vectorized CF callbacks (split into real and imaginary parts)
    # Vectorized CF callbacks (split into real and imaginary parts)
    var cf_re = fn(phi: SIMD[DType.float64, 8]) capturing -> SIMD[DType.float64, 8]:
        return model.predictive_cf_latent_simd_re[8](phi, y, kappa, nu)
    var cf_im = fn(phi: SIMD[DType.float64, 8]) capturing -> SIMD[DType.float64, 8]:
        return model.predictive_cf_latent_simd_im[8](phi, y, kappa, nu)

    # Invert at a single x (time/space) point
    var x: Float64 = 0.1
    var inverter = UniformGridInverterSIMDGrid[8](grid)
    var out = inverter.inverse_at(x, u0, du, cf_re, cf_im, normalize=True)

    print("Inverse Fourier at x=", x, ": re=", out.re[0], ", im=", out.im[0])
