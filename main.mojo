from stochcharfunc import Svj1JointCF
from stochcharfunc.svj1 import Svj1Params
from complexsimd import (
    ComplexSIMD,
    CFGridSIMD,
    UniformGridInverterSIMDGrid,
)
from optimizers import OptimFiniteDiffConfig, DirMethod, DirectionMat

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
    print("TODO")
