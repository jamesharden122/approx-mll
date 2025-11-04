from stochcharfunc import Svj1JointCF
from stochcharfunc.svj1 import Svj1Params
from complexsimd import CFGridSIMD, UniformGridInverterSIMDGrid
from optimizers.log_likelihood import log_likelihood

fn fail(msg: String):
    print("FAIL:", msg)
    var z: Int = 0
    var `_`: Int = 1 // z

fn make_phi_grid(tiles: Int, u0: Float64, du: Float64) -> CFGridSIMD[DType.float64, 8]:
    # Matches helper used in other tests: encodes phi via cumulative re lanes.
    var re_tiles = List[SIMD[DType.float64, 8]]()
    var im_tiles = List[SIMD[DType.float64, 8]]()
    for t in range(tiles):
        var phi = SIMD[DType.float64, 8]()
        for j in range(8):
            var idx = t * 8 + j
            phi[j] = u0 + du * Float64(idx)
        var re = SIMD[DType.float64, 8]()
        re[7] = phi[7]
        re[6] = phi[6] + re[7]
        re[5] = phi[5] + re[6]
        re[4] = phi[4] + re[5]
        re[3] = phi[3] + re[4]
        re[2] = phi[2] + re[3]
        re[1] = phi[1] + re[2]
        re[0] = phi[0] + re[1]
        var im = SIMD[DType.float64, 8](0.0)
        re_tiles.append(re)
        im_tiles.append(im)
    var k_valid = List[Int]()
    k_valid.append(tiles * 8)
    return CFGridSIMD[DType.float64, 8](re_tiles, im_tiles, k_valid)

fn test_log_likelihood_basic():
    # Grid parameters
    var tiles: Int = 64
    var u0: Float64 = 0.0
    var du: Float64 = 0.0015
    var grid = make_phi_grid(tiles, u0, du)

    # Model and prior
    var model = Svj1JointCF(Svj1Params.__init__[10](InlineArray[Float64,10](
        0.05,-0.02,-0.6,0.04,1.5,0.3,0.1,0.05,1.0,1.0/252.0
    )))
    model.params = Svj1Params.__init__[10](InlineArray[Float64,10](
        0.05,-0.02,-0.6,0.04,1.5,0.3,0.1,0.05,1.0,1.0/252.0
    )) 
    var prior = InlineArray[Float64, 3](0.0, 0.5, 2.0)  # [y_t, kappa, nu]
    var inverter = UniformGridInverterSIMDGrid[8](grid)

    # Small batch of synthetic returns in one SIMD lane group
    var y_vec = SIMD[DType.float64, 8](0.0)
    y_vec[0] = -0.01
    y_vec[1] = -0.005
    y_vec[2] = 0.0
    y_vec[3] = 0.004
    y_vec[4] = 0.01
    y_vec[5] = -0.02
    y_vec[6] = 0.02
    y_vec[7] = 0.0

    var ll = log_likelihood[Svj1JointCF, 8](y_vec, inverter, model, prior, True)
    var v = Float64(ll[0])

    if v != v:
        fail("log-likelihood is NaN")
    print("test_log_likelihood_basic: ll_sum=", v)

fn main():
    test_log_likelihood_basic()
