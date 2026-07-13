from affine_svj import AffineSvjParams
from complexsimd.fourier import UniformGridInverterSIMDGrid
from optimizers.finitediff import (
    AffineSvjOptimConfig,
    AffineSvjOptimizer,
    params_to_unconstrained,
    unconstrained_to_params,
)
from stochcharfunc.svj1 import AffineSvjJointCF
from std.collections import List
from std.math import cos, log, sin
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_true,
    TestSuite,
)
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


def make_prior() -> InlineArray[Float64, 2]:
    var params = make_params()
    return [
        params.volvol * params.volvol / (2.0 * params.beta),
        2.0 * params.beta * params.alpha
            / (params.volvol * params.volvol),
    ]


def scalar_density(
    inverter: UniformGridInverterSIMDGrid[8],
    x: Float64,
    model: AffineSvjJointCF,
    prior: InlineArray[Float64, 2],
) -> Float64:
    var total = 0.0
    for index in range(inverter.count):
        var u = inverter.u0 + inverter.du * Float64(index)
        var value = model.predictive_cf_scalar(
            u, 0.0, prior[0], prior[1]
        )
        var angle = -u * x
        var weight = 1.0
        if index == 0 or index == inverter.count - 1:
            weight = 0.5
        total += weight * (
            value[0] * cos(angle) - value[1] * sin(angle)
        )
    return total * inverter.du / 3.14159265358979323846


def test_transforms_preserve_domains_and_dt() raises:
    var params = make_params()
    var coordinates = params_to_unconstrained(params)
    var roundtrip = unconstrained_to_params(coordinates, params.dt)
    var expected = params.free_values()
    var actual = roundtrip.free_values()
    for index in range(10):
        assert_almost_equal(
            actual[index],
            expected[index],
            atol=2e-12,
            msg="unconstrained transform must round-trip parameters",
        )
    assert_equal(roundtrip.dt, params.dt)
    assert_true(
        roundtrip.lambda0 > 0.0
        and roundtrip.lambda1 > 0.0
        and roundtrip.jump_std > 0.0
        and roundtrip.alpha > 0.0
        and roundtrip.beta > 0.0
        and roundtrip.volvol > 0.0
        and roundtrip.rho > -1.0
        and roundtrip.rho < 1.0,
        msg="parameter transforms must enforce model domains",
    )


def test_central_gradient_matches_scalar_reference() raises:
    var params = make_params()
    var prior = make_prior()
    var inverter = UniformGridInverterSIMDGrid[8](0.0, 0.4, 201)
    var config = AffineSvjOptimConfig[8](
        inverter,
        prior,
        difference_step=1e-4,
        max_iterations=1,
    )
    var optimizer = AffineSvjOptimizer[8](config)
    var returns: List[Float64] = [0.012]
    var gradient = optimizer.central_gradient(returns, params)
    var center = params_to_unconstrained(params)

    for coordinate in range(10):
        var plus = InlineArray[Float64, 10](fill=0.0)
        var minus = InlineArray[Float64, 10](fill=0.0)
        for index in range(10):
            plus[index] = center[index]
            minus[index] = center[index]
        plus[coordinate] += config.difference_step
        minus[coordinate] -= config.difference_step
        var plus_model = AffineSvjJointCF(
            unconstrained_to_params(plus, params.dt)
        )
        var minus_model = AffineSvjJointCF(
            unconstrained_to_params(minus, params.dt)
        )
        var reference = (
            log(scalar_density(inverter, returns[0], plus_model, prior))
            - log(scalar_density(inverter, returns[0], minus_model, prior))
        ) / (2.0 * config.difference_step)
        assert_true(
            isfinite(gradient[coordinate])[0],
            msg="central gradient must be finite",
        )
        assert_almost_equal(
            gradient[coordinate],
            reference,
            atol=2e-6,
            msg="SIMD likelihood gradient must match scalar reference",
        )


def test_armijo_fit_ascends_likelihood() raises:
    var params = make_params()
    var prior = make_prior()
    var inverter = UniformGridInverterSIMDGrid[8](0.0, 0.4, 201)
    var config = AffineSvjOptimConfig[8](
        inverter,
        prior,
        difference_step=1e-4,
        initial_step=1e-3,
        gradient_tolerance=1e-10,
        likelihood_tolerance=1e-12,
        max_iterations=1,
    )
    var optimizer = AffineSvjOptimizer[8](config)
    var returns: List[Float64] = [0.012]
    var initial = optimizer.evaluate(returns, params)
    var result = optimizer.fit(returns, params)
    assert_true(initial.valid, msg="initial likelihood must be valid")
    assert_true(
        result.likelihood >= initial.likelihood,
        msg="Armijo likelihood ascent must not reduce the objective",
    )
    assert_true(
        isfinite(result.likelihood)[0]
        and isfinite(result.gradient_norm)[0],
        msg="fit result likelihood and final gradient norm must be finite",
    )
    assert_equal(result.params.dt, params.dt)
    result.params.validate()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
