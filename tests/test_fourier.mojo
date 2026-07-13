from affine_svj import AffineSvjParams
from complexsimd.fourier import UniformGridInverterSIMDGrid
from stochcharfunc.svj1 import AffineSvjJointCF
from std.math import cos, exp, sin
from std.testing import assert_almost_equal, assert_true, TestSuite
from std.utils.numerics import isfinite


def make_params() -> AffineSvjParams:
    return AffineSvjParams(
        0.03,
        -0.15,
        0.12,
        0.35,
        -0.04,
        0.12,
        -0.55,
        0.04,
        1.5,
        0.3,
        0.1,
    )


def stationary_prior() -> InlineArray[Float64, 2]:
    var params = make_params()
    return [
        params.volvol * params.volvol / (2.0 * params.beta),
        2.0 * params.beta * params.alpha
            / (params.volvol * params.volvol),
    ]

def scalar_joint_density(
    inverter: UniformGridInverterSIMDGrid[8],
    x: Float64,
    psi: Float64,
    model: AffineSvjJointCF,
    prior: InlineArray[Float64, 2],
) -> Float64:
    var total = 0.0
    for index in range(inverter.count):
        var u = inverter.u0 + inverter.du * Float64(index)
        var value = model.predictive_cf_scalar(
            u, psi, prior[0], prior[1]
        )
        var angle = -u * x
        var weight = 1.0
        if index == 0 or index == inverter.count - 1:
            weight = 0.5
        total += weight * (
            value[0] * cos(angle) - value[1] * sin(angle)
        )
    return total * inverter.du / 3.14159265358979323846




def test_scalar_and_simd_cf_match() raises:
    var model = AffineSvjJointCF(make_params())
    var prior = stationary_prior()
    var frequencies = SIMD[DType.float64, 8](
        -4.0, -1.5, -0.25, 0.0, 0.5, 1.25, 3.0, 7.0
    )
    var simd = model.predictive_cf_simd[8](
        frequencies, 0.01, prior[0], prior[1]
    )
    for lane in range(8):
        var scalar = model.predictive_cf_scalar(
            Float64(frequencies[lane]), 0.01, prior[0], prior[1]
        )
        assert_almost_equal(
            Float64(simd.re[lane]),
            scalar[0],
            atol=2e-11,
            msg="scalar and SIMD CF real parts must match",
        )
        assert_almost_equal(
            Float64(simd.im[lane]),
            scalar[1],
            atol=2e-11,
            msg="scalar and SIMD CF imaginary parts must match",
        )


def test_cf_invariants_and_cir_moments() raises:
    var params = make_params()
    var model = AffineSvjJointCF(params)
    var prior = stationary_prior()
    var at_zero = model.predictive_cf_scalar(
        0.0, 0.0, prior[0], prior[1]
    )
    assert_almost_equal(at_zero[0], 1.0, atol=1e-12)
    assert_almost_equal(at_zero[1], 0.0, atol=1e-12)

    for index in range(1, 9):
        var u = 0.5 * Float64(index)
        var positive = model.predictive_cf_scalar(
            u, 0.0, prior[0], prior[1]
        )
        var negative = model.predictive_cf_scalar(
            -u, 0.0, prior[0], prior[1]
        )
        assert_true(
            isfinite(positive[0])[0] and isfinite(positive[1])[0],
            msg="CF must remain finite",
        )
        assert_almost_equal(
            positive[0], negative[0], atol=2e-11,
            msg="CF real part must have conjugate symmetry",
        )
        assert_almost_equal(
            positive[1], -negative[1], atol=2e-11,
            msg="CF imaginary part must have conjugate symmetry",
        )

    var h = 1e-4
    var minus = model.predictive_cf_scalar(
        0.0, -h, prior[0], prior[1]
    )[0]
    var zero = model.predictive_cf_scalar(
        0.0, 0.0, prior[0], prior[1]
    )[0]
    var plus = model.predictive_cf_scalar(
        0.0, h, prior[0], prior[1]
    )[0]
    var estimated_mean = (plus - minus) / (2.0 * h * zero)
    var estimated_second = (plus - 2.0 * zero + minus) / (h * h * zero)
    var estimated_variance = (
        estimated_second - estimated_mean * estimated_mean
    )

    var initial_mean = prior[0] * prior[1]
    var initial_variance = prior[0] * prior[0] * prior[1]
    var decay = exp(-params.beta * params.dt)
    var expected_mean = (
        params.alpha + (initial_mean - params.alpha) * decay
    )
    var expected_variance = (
        initial_variance * decay * decay
        + initial_mean
            * params.volvol
            * params.volvol
            * decay
            * (1.0 - decay)
            / params.beta
        + params.alpha
            * params.volvol
            * params.volvol
            * (1.0 - decay)
            * (1.0 - decay)
            / (2.0 * params.beta)
    )
    assert_almost_equal(
        estimated_mean,
        expected_mean,
        atol=2e-7,
        msg="CF terminal-variance mean must match CIR moments",
    )
    assert_almost_equal(
        estimated_variance,
        expected_variance,
        atol=2e-6,
        msg="CF terminal-variance variance must match CIR moments",
    )


def test_constant_and_state_jump_terms_are_active() raises:
    var base = make_params()
    var prior = stationary_prior()
    var no_jumps = AffineSvjParams(
        base.mu0, base.mu1, 0.0, 0.0, base.jump_mean, base.jump_std,
        base.rho, base.alpha, base.beta, base.volvol, base.dt
    )
    var constant_jumps = AffineSvjParams(
        base.mu0, base.mu1, 0.5, 0.0, base.jump_mean, base.jump_std,
        base.rho, base.alpha, base.beta, base.volvol, base.dt
    )
    var state_jumps = AffineSvjParams(
        base.mu0, base.mu1, 0.0, 0.5, base.jump_mean, base.jump_std,
        base.rho, base.alpha, base.beta, base.volvol, base.dt
    )
    var zero_value = AffineSvjJointCF(no_jumps).predictive_cf_scalar(
        2.0, 0.0, prior[0], prior[1]
    )
    var constant_value = AffineSvjJointCF(
        constant_jumps
    ).predictive_cf_scalar(2.0, 0.0, prior[0], prior[1])
    var state_value = AffineSvjJointCF(
        state_jumps
    ).predictive_cf_scalar(2.0, 0.0, prior[0], prior[1])
    var constant_delta = (
        (constant_value[0] - zero_value[0])
            * (constant_value[0] - zero_value[0])
        + (constant_value[1] - zero_value[1])
            * (constant_value[1] - zero_value[1])
    )
    var state_delta = (
        (state_value[0] - zero_value[0])
            * (state_value[0] - zero_value[0])
        + (state_value[1] - zero_value[1])
            * (state_value[1] - zero_value[1])
    )
    assert_true(
        constant_delta > 1e-12,
        msg="constant Gaussian jump exponent must affect the CF",
    )
    assert_true(
        state_delta > 1e-12,
        msg="variance-dependent Gaussian jump exponent must affect the CF",
    )



def test_density_mass_and_gamma_update() raises:
    var model = AffineSvjJointCF(make_params())
    var prior = stationary_prior()
    var inverter = UniformGridInverterSIMDGrid[8](0.0, 0.2, 401)

    var density_zero = inverter.density(0.0, model, prior)
    assert_true(
        density_zero > 0.0 and isfinite(density_zero)[0],
        msg="density must be positive and finite near its center",
    )
    var scalar_density = scalar_joint_density(
        inverter, 0.0, 0.0, model, prior
    )
    assert_almost_equal(
        density_zero,
        scalar_density,
        atol=2e-11,
        msg="tiled SIMD density must match scalar Fourier reference",
    )


    var mass = 0.0
    var dx = 0.015
    var points = 201
    for index in range(points):
        var x = -1.5 + dx * Float64(index)
        var weight = 1.0
        if index == 0 or index == points - 1:
            weight = 0.5
        mass += weight * inverter.density(x, model, prior)
    mass *= dx
    assert_almost_equal(
        mass,
        1.0,
        atol=0.025,
        msg="Fourier density must integrate to approximately one",
    )

    var update = inverter.update_gamma_prior(0.0, model, prior, 1e-4)
    assert_true(update.valid, msg="Gamma moment update must be valid")
    assert_true(
        update.kappa > 0.0
        and update.nu > 0.0
        and update.conditional_mean > 0.0
        and update.conditional_variance > 0.0,
        msg="updated Gamma parameters and moments must be positive",
    )
    var h = 1e-4
    var scalar_minus = scalar_joint_density(
        inverter, 0.0, -h, model, prior
    )
    var scalar_zero = scalar_joint_density(
        inverter, 0.0, 0.0, model, prior
    )
    var scalar_plus = scalar_joint_density(
        inverter, 0.0, h, model, prior
    )
    var scalar_mean = (
        (scalar_plus - scalar_minus) / (2.0 * h * scalar_zero)
    )
    var scalar_second = (
        (scalar_plus - 2.0 * scalar_zero + scalar_minus)
        / (h * h * scalar_zero)
    )
    var scalar_variance = scalar_second - scalar_mean * scalar_mean
    assert_almost_equal(
        update.conditional_mean,
        scalar_mean,
        atol=2e-9,
        msg="SIMD Gamma mean update must match scalar Fourier reference",
    )
    assert_almost_equal(
        update.conditional_variance,
        scalar_variance,
        atol=2e-8,
        msg="SIMD Gamma variance update must match scalar Fourier reference",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
