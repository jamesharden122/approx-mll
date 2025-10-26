from stochcharfunc import Svj1JointCF
from stochcharfunc.svj1 import Svj1Params
from complexsimd import CFGridSIMD
from optimizers import OptimFiniteDiffConfig

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

fn test_optim_config_construction():
    var tiles: Int = 8
    var grid = make_small_grid(tiles)
    var model = Svj1JointCF(Svj1Params())

    var cfg = OptimFiniteDiffConfig[Svj1Params, Svj1JointCF, 8, 8](
        Svj1Params(),
        1e-3, 1e-4, 1.0, 0.5, 0.1,
        grid,
        model
    )

    if cfg.h != 1e-3:
        fail("cfg.h should equal 1e-3")
    if cfg.armijo_c != 1e-4:
        fail("cfg.armijo_c should equal 1e-4")
    print("test_optim_config_construction: ok; h=", cfg.h)

fn main():
    test_optim_config_construction()
    
