from stochcharfunc import Svj1JointCF
from stochcharfunc.svj1 import Svj1Params
from complexsimd import (
    ComplexSIMD,
    CFGridSIMD,
    UniformGridInverterSIMDGrid,
)
from optimizers import OptimFiniteDiffConfig

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

    var model = Svj1JointCF(Svj1Params())

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

    # Basic construction test of OptimFiniteDiffConfig
    var cfg = OptimFiniteDiffConfig[Svj1Params, Svj1JointCF, 8](
        Svj1Params(),
        1e-3, 1e-4, 1.0, 0.5, 0.1,
        grid,
        model
    )

    # Prior params as [y, kappa, nu]
    var prior = List[Float64]()
    prior.append(y)
    prior.append(kappa)
    prior.append(nu)

    # Invert at a single x (time/space) point
    var x: Float64 = 0.1
    var inverter = UniformGridInverterSIMDGrid[8](grid)
    var out = inverter.inverse_at(x, u0, du, model, prior, normalize=True)

    print("OptimFiniteDiffConfig init ok; h=", cfg.h)
    print("Inverse Fourier at x=", x, ": re=", out.re[0], ", im=", out.im[0])
