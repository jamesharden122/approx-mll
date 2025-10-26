from stochcharfunc import Svj1JointCF
from stochcharfunc.svj1 import Svj1Params
from complexsimd import CFGridSIMD, UniformGridInverterSIMDGrid

fn fail(msg: String):
    print("FAIL:", msg)
    var z: Int = 0
    var crash: Int = 1 // z

fn make_small_grid(tiles: Int) -> CFGridSIMD[DType.float64, 8]:
    var re_tiles = List[SIMD[DType.float64, 8]]()
    var im_tiles = List[SIMD[DType.float64, 8]]()
    for _ in range(tiles):
        re_tiles.append(SIMD[DType.float64, 8]())
        im_tiles.append(SIMD[DType.float64, 8]())
    var k_valid = List[Int]()
    k_valid.append(tiles * 8)
    return CFGridSIMD[DType.float64, 8](re_tiles, im_tiles, k_valid)

fn test_uniform_grid_inverter_inverse_at():
    var tiles: Int = 8
    var grid = make_small_grid(tiles)
    var model = Svj1JointCF(Svj1Params())

    # Prior params as [y, kappa, nu]
    var prior = List[Float64]()
    prior.append(0.01)
    prior.append(0.5)
    prior.append(2.0)

    var inverter = UniformGridInverterSIMDGrid[8](grid)
    var out = inverter.inverse_at(0.1, 0.0, 0.01, model, prior, normalize=True)

    # Basic sanity: results should be real numbers (not NaN)
    if out.re[0] != out.re[0]:
        fail("out.re is NaN")
    if out.im[0] != out.im[0]:
        fail("out.im is NaN")
    print("test_uniform_grid_inverter_inverse_at: ok; re=", out.re[0], ", im=", out.im[0])

fn main():
    test_uniform_grid_inverter_inverse_at()
