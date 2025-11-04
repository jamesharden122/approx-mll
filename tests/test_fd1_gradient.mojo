from stochcharfunc import Svj1JointCF
from stochcharfunc.svj1 import Svj1Params
from complexsimd import CFGridSIMD, UniformGridInverterSIMDGrid
from optimizers import OptimFiniteDiffConfig, FD1, DirMethod
from optimizers.log_likelihood import log_likelihood
from math import sqrt

fn fail(msg: String):
    print("FAIL:", msg)
    var z: Int = 0
    var `_`: Int = 1 // z

fn make_phi_grid(tiles: Int, u0: Float64, du: Float64) -> CFGridSIMD[DType.float64, 8]:
    # Build φ lanes such that phi = re - re.shift_left(1)
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

fn test_fd1_gradient_basic():
    # Grid
    var tiles: Int = 8
    var grid = make_phi_grid(tiles, 0.0, 0.0015)
    var inverter = UniformGridInverterSIMDGrid[8](grid)
    print("inverter and grid defined")
    # Spec and params (10-dim theta)
    var theta0 = Svj1Params(InlineArray[Float64, 16](
        0.05, -0.02, 0.1, 0.05, 1.0, -0.6, 0.04, 1.5, 0.3, 1.0/252.0, 0.0, 0.0 ,0.0, 0.0,0.0,0.0
    ))
    var spec = Svj1JointCF(theta0)

    # Config with identity directions
    var cfg = OptimFiniteDiffConfig[Svj1Params, Svj1JointCF, 8, 16](
        theta0, 1e-3, 1e-4, 1e-1, 1e-6, grid, spec, DirMethod.Identity()
    )

    var opt = FD1[Svj1Params, Svj1JointCF, 8, 16](cfg)
    print("FD1 Defined")
    var prior = InlineArray[Float64, 3](0.0, 0.5, 2.0)
    var y_vec = SIMD[DType.float64, 8](0.0)
    y_vec[0] = -0.01
    y_vec[1] = 0.0
    y_vec[2] = 0.01

    # Simple gradient-descent loop using FD1.gradient (which updates theta internally)
    var iters: Int = 5
    var tol: Float64 = 1e-6
    for it in range(iters):
        _ = opt.gradient(prior, y_vec, log_likelihood[Svj1JointCF, 8])
        # Read back current theta from optimizer config and report progress
        var theta_cur = opt.config.theta.get_params[16]()
        print("iter=", it, ", theta0..3=", theta_cur[0], ",", theta_cur[1], ",", theta_cur[2], ",", theta_cur[3])

    # End of gradient-descent demo; if needed, add assertions here

fn main():
    _ = test_fd1_gradient_basic()
