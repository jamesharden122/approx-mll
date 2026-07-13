from complexsimd.fourier import UniformGridInverterSIMDGrid
from optimizers.finitediff import AffineSvjOptimConfig
from std.math import sqrt
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_true,
    TestSuite,
)


def test_explicit_grid_metadata_and_lane_mask() raises:
    var inverter = UniformGridInverterSIMDGrid[8](0.0, 0.25, 10)
    assert_equal(inverter.num_tiles(), 2)
    var first = inverter.frequency_tile(0)
    var final = inverter.frequency_tile(1)
    for lane in range(8):
        assert_almost_equal(
            Float64(first[lane]), 0.25 * Float64(lane), atol=0.0
        )
    assert_almost_equal(Float64(final[0]), 2.0, atol=0.0)
    assert_almost_equal(Float64(final[1]), 2.25, atol=0.0)
    for lane in range(2, 8):
        assert_almost_equal(
            Float64(final[lane]),
            0.0,
            atol=0.0,
            msg="invalid final-tile lanes must be masked",
        )


def test_optimizer_config_and_invalid_metadata() raises:
    var inverter = UniformGridInverterSIMDGrid[8](0.0, 0.25, 65)
    var prior: InlineArray[Float64, 2] = [0.03, 4.0 / 3.0]
    var config = AffineSvjOptimConfig[8](
        inverter,
        prior,
        difference_step=2e-4,
        armijo_c=2e-4,
        shrink=0.4,
        initial_step=0.05,
        gradient_tolerance=2e-5,
        likelihood_tolerance=2e-8,
        max_iterations=7,
        min_step=2e-8,
    )
    assert_almost_equal(config.difference_step, 2e-4, atol=0.0)
    assert_almost_equal(config.armijo_c, 2e-4, atol=0.0)
    assert_almost_equal(config.shrink, 0.4, atol=0.0)
    assert_almost_equal(config.initial_step, 0.05, atol=0.0)
    assert_equal(config.max_iterations, 7)

    var rejected_origin = False
    try:
        _ = UniformGridInverterSIMDGrid[8](0.1, 0.25, 65)
    except:
        rejected_origin = True
    assert_true(
        rejected_origin,
        msg="nonzero Fourier grid origin must be rejected",
    )

    var negative = -1.0
    var nan_value = sqrt(negative)
    var rejected_nonfinite = False
    try:
        _ = UniformGridInverterSIMDGrid[8](0.0, nan_value, 65)
    except:
        rejected_nonfinite = True
    assert_true(
        rejected_nonfinite,
        msg="non-finite Fourier spacing must be rejected",
    )

    var rejected_prior = False
    var invalid_prior: InlineArray[Float64, 2] = [0.0, 1.0]
    try:
        _ = AffineSvjOptimConfig[8](inverter, invalid_prior)
    except:
        rejected_prior = True
    assert_true(
        rejected_prior,
        msg="invalid optimizer Gamma prior must be rejected",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
