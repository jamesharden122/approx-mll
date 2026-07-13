from affine_svj import AffineSvjParams
from complexsimd.fourier import UniformGridInverterSIMDGrid
from optimizers.log_likelihood import filter_log_returns
from stochcharfunc.svj1 import AffineSvjJointCF
from std.collections import List
from std.math import cos, log, sin
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


def scalar_filter(
    returns: List[Float64],
    inverter: UniformGridInverterSIMDGrid[8],
    model: AffineSvjJointCF,
    initial_prior: InlineArray[Float64, 2],
) -> Tuple[Float64, Float64, Float64]:
    var prior = initial_prior
    var likelihood = 0.0
    var h = 1e-3
    for index in range(len(returns)):
        var x = returns[index]
        var p_minus = scalar_joint_density(
            inverter, x, -h, model, prior
        )
        var p_zero = scalar_joint_density(
            inverter, x, 0.0, model, prior
        )
        var p_plus = scalar_joint_density(
            inverter, x, h, model, prior
        )
        likelihood += log(p_zero)
        var mean = (p_plus - p_minus) / (2.0 * h * p_zero)
        var second = (
            (p_plus - 2.0 * p_zero + p_minus) / (h * h * p_zero)
        )
        var variance = second - mean * mean
        prior[0] = variance / mean
        prior[1] = mean * mean / variance
    return (likelihood, prior[0], prior[1])




def test_runtime_return_filter_is_finite() raises:
    var params = make_params()
    var model = AffineSvjJointCF(params)
    var inverter = UniformGridInverterSIMDGrid[8](0.0, 0.2, 401)
    var prior: InlineArray[Float64, 2] = [
        params.volvol * params.volvol / (2.0 * params.beta),
        2.0 * params.beta * params.alpha
            / (params.volvol * params.volvol),
    ]
    var returns: List[Float64] = [0.01, -0.015, 0.004]
    var result = filter_log_returns[8](
        returns, inverter, model, prior
    )
    assert_true(result.valid, msg="valid returns must produce a valid filter")
    assert_true(
        isfinite(result.likelihood)[0],
        msg="filtered likelihood must be finite",
    )
    assert_true(
        result.final_kappa > 0.0 and result.final_nu > 0.0,
        msg="final Gamma parameters must be positive",
    )
    var scalar_result = scalar_filter(
        returns, inverter, model, prior
    )
    assert_almost_equal(
        result.likelihood,
        scalar_result[0],
        atol=2e-9,
        msg="SIMD sequential likelihood must match scalar reference",
    )
    assert_almost_equal(
        result.final_kappa,
        scalar_result[1],
        atol=2e-8,
        msg="SIMD final Gamma scale must match scalar reference",
    )
    assert_almost_equal(
        result.final_nu,
        scalar_result[2],
        atol=2e-7,
        msg="SIMD final Gamma shape must match scalar reference",
    )


    var first_only: List[Float64] = [returns[0]]
    var first_result = filter_log_returns[8](
        first_only, inverter, model, prior
    )
    var first_density = inverter.density(returns[0], model, prior)
    assert_almost_equal(
        first_result.likelihood,
        log(first_density),
        atol=1e-11,
        msg="filter must consume the supplied log-return increment directly",
    )


def test_invalid_prior_is_rejected() raises:
    var model = AffineSvjJointCF(make_params())
    var inverter = UniformGridInverterSIMDGrid[8](0.0, 0.2, 401)
    var invalid_prior: InlineArray[Float64, 2] = [0.0, 1.0]
    var returns: List[Float64] = [0.0]
    var result = filter_log_returns[8](
        returns, inverter, model, invalid_prior
    )
    assert_true(not result.valid, msg="invalid Gamma prior must be rejected")
    assert_true(
        result.likelihood <= -1e299,
        msg="invalid filtered path must return the likelihood sentinel",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
