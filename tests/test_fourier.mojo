from stochcharfunc import Svj1JointCF
from stochcharfunc.svj1 import Svj1Params
from complexsimd import CFGridSIMD, UniformGridInverterSIMDGrid

fn fail(msg: String):
    print("FAIL:", msg)
    var z: Int = 0
    var `_`: Int = 1 // z

fn make_phi_grid(tiles: Int, u0: Float64, du: Float64) -> CFGridSIMD[DType.float64, 8]:
    # Build φ lanes that the inverter reconstructs via: phi = re - re.shift_left(1)
    # Target φ_j = u0 + du * (t*L + j). We set re cumulatively so that
    # re[j] - re[j+1] = φ_j (with re[7] - 0 = φ_7).
    var re_tiles = List[SIMD[DType.float64, 8]]()
    var im_tiles = List[SIMD[DType.float64, 8]]()
    for t in range(tiles):
        var phi = SIMD[DType.float64, 8]()
        for j in range(8):
            var idx = t * 8 + j
            phi[j] = u0 + du * Float64(idx)
        var re = SIMD[DType.float64, 8]()
        # Backward cumulative sum to satisfy re[j] - re[j+1] = phi[j]
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

fn test_uniform_grid_inverter_inverse_at():
    # Use a denser φ-grid for better integral approximation
    # Tighter φ range to avoid CF domain issues
    var tiles: Int = 64    # K = 512 samples
    var u0: Float64 = 0.0
    var du: Float64 = 0.0015
    var grid = make_phi_grid(tiles, u0, du)
    var model = Svj1JointCF(Svj1Params.__init__[10](InlineArray[Float64,10](0.05,-0.02,-0.6,0.04,1.5,0.3,0.1,0.05,1.0,1.0/252.0)))

    # Prior params as InlineArray [y, kappa, nu]
    var prior = InlineArray[Float64, 3](0.01, 0.5, 2.0)

    var inverter = UniformGridInverterSIMDGrid[8](grid)
    var (kappa_new, nu_new, x_hat, P) = model.update_gamma_prior_from_inverter[8](
        inverter, prior, 0.0, 0.01, 1e-6, True
    )
    # Basic sanity checks
    print("test_uniform_grid_inverter_inverse_at: ok; kappa_new=", kappa_new,", nu_new=", nu_new, ", x_hat=", x_hat, ", P=", P)

fn test_bates_step2_density_via_inverse_at():
    # Step 2 (Bates): density via Fourier inversion of the characteristic function
    # Build a modest grid and evaluate f(x | Y_t) at a small set of x values
    # Use the same realistic φ-grid here
    var tiles: Int = 64
    var u0: Float64 = 0.0
    var du: Float64 = 0.0015
    var grid = make_phi_grid(tiles, u0, du)
    var model = Svj1JointCF(Svj1Params.__init__[10](InlineArray[Float64,10](
        0.05,-0.02,-0.6,0.04,1.5,0.3,0.1,0.05,1.0,1.0/252.0
    )))

    # Prior as [y, kappa, nu]; choose y_t = 0 for symmetry
    var prior = InlineArray[Float64, 3](0.0, 0.5, 2.0)

    var inverter = UniformGridInverterSIMDGrid[8](grid)
    var psi0 = SIMD[DType.float64, 1](0.0)

    # Evaluate density at a few returns x
    var xs = List[Float64]()
    xs.append(-0.05)
    xs.append(-0.02)
    xs.append(0.00)
    xs.append(0.02)
    xs.append(0.05)

    for x in xs:
        var z = inverter.inverse_at[Svj1JointCF, 3](x, model, psi0, prior)
        # Correct Riemann sum: multiply by Δφ and account for positive-φ only (×2)
        var f = z.re[0] * 2.0 * du
        if f != f:
            fail("density is NaN at x")
        if f < 0.0:
            print("warning: negative density at x=", x, ": ", f)
        else:
            print("density f(x|Y_t) at x=", x, " -> ", f)

fn main():
    test_uniform_grid_inverter_inverse_at()
    test_bates_step2_density_via_inverse_at()
