from affine_svj import (
    AffineSvjParams,
    decode_big_endian_f64_returns,
    encode_big_endian_f64_returns,
    parse_affine_svj_fit_args,
    read_big_endian_f64_returns,
)
from affine_svj.fit_command import run_affine_svj_fit, stationary_gamma_prior
from std.collections import List
from std.testing import assert_almost_equal
from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
    assert_true,
)


def test_big_endian_returns_golden_and_file_roundtrip() raises:
    var values: List[Float64] = [0.0, 1.0, -2.0, 0.5]
    var expected: List[UInt8] = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        63,
        240,
        0,
        0,
        0,
        0,
        0,
        0,
        192,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        63,
        224,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    var encoded = encode_big_endian_f64_returns(values)
    assert_equal(len(encoded), 32)
    for i in range(32):
        assert_equal(encoded[i], expected[i])

    var decoded = decode_big_endian_f64_returns(encoded, 4)
    for i in range(4):
        assert_equal(decoded[i], values[i])

    var path = String("/tmp/approx_mll_fit_returns.bin")
    var file = open(path, "w")
    file.write_all(encoded)
    file.close()
    var from_file = read_big_endian_f64_returns(path, 4)
    for i in range(4):
        assert_equal(from_file[i], values[i])


def test_big_endian_returns_reject_invalid_inputs() raises:
    var one: List[Float64] = [1.0]
    var encoded = encode_big_endian_f64_returns(one)

    var short = List[UInt8]()
    for i in range(7):
        short.append(encoded[i])
    with assert_raises(contains="exactly 8 bytes"):
        _ = decode_big_endian_f64_returns(short, 1)

    var trailing = encoded.copy()
    trailing.append(0)
    with assert_raises(contains="exactly 8 bytes"):
        _ = decode_big_endian_f64_returns(trailing, 1)

    with assert_raises(contains="greater than zero"):
        _ = decode_big_endian_f64_returns(encoded, 0)

    var infinity: List[UInt8] = [127, 240, 0, 0, 0, 0, 0, 0]
    with assert_raises(contains="finite"):
        _ = decode_big_endian_f64_returns(infinity, 1)


def required_args() -> List[String]:
    return [
        "--returns",
        "returns.bin",
        "--count",
        "64",
        "--initial",
        "initial.bin",
        "--out",
        "fitted.bin",
    ]


def test_fit_cli_required_and_defaults() raises:
    var config = parse_affine_svj_fit_args(required_args())
    assert_equal(config.returns_path, String("returns.bin"))
    assert_equal(config.count, 64)
    assert_equal(config.initial_path, String("initial.bin"))
    assert_equal(config.out_path, String("fitted.bin"))
    assert_equal(config.grid_u0, 0.0)
    assert_equal(config.grid_du, 0.25)
    assert_equal(config.grid_count, 2048)
    assert_equal(config.max_iterations, 100)
    assert_true(not config.show_help)


def test_fit_cli_optimizer_and_grid_overrides() raises:
    var args = required_args()
    args.append("--grid-u0")
    args.append("0.0")
    args.append("--grid-du")
    args.append("0.5")
    args.append("--grid-count")
    args.append("513")
    args.append("--max-iterations")
    args.append("7")
    args.append("--fd-step")
    args.append("0.002")
    args.append("--initial-step")
    args.append("0.125")
    args.append("--gradient-tolerance")
    args.append("0.003")
    args.append("--likelihood-tolerance")
    args.append("0.004")
    args.append("--armijo-c")
    args.append("0.005")
    args.append("--armijo-shrink")
    args.append("0.25")
    args.append("--min-step")
    args.append("0.000001")

    var config = parse_affine_svj_fit_args(args)
    assert_equal(config.grid_u0, 0.0)
    assert_equal(config.grid_du, 0.5)
    assert_equal(config.grid_count, 513)
    assert_equal(config.max_iterations, 7)
    assert_equal(config.fd_step, 0.002)
    assert_equal(config.initial_step, 0.125)
    assert_equal(config.gradient_tolerance, 0.003)
    assert_equal(config.likelihood_tolerance, 0.004)
    assert_equal(config.armijo_c, 0.005)
    assert_equal(config.armijo_shrink, 0.25)
    assert_equal(config.min_step, 0.000001)


def test_fit_cli_rejects_malformed_options() raises:
    var missing: List[String] = ["--returns", "returns.bin"]
    with assert_raises(contains="--count"):
        _ = parse_affine_svj_fit_args(missing)

    var unknown = required_args()
    unknown.append("--mystery")
    unknown.append("1")
    with assert_raises(contains="unknown option"):
        _ = parse_affine_svj_fit_args(unknown)

    var nonzero_origin = required_args()
    nonzero_origin.append("--grid-u0")
    nonzero_origin.append("0.125")
    with assert_raises(contains="exactly zero"):
        _ = parse_affine_svj_fit_args(nonzero_origin)

    var duplicate = required_args()
    duplicate.append("--count")
    duplicate.append("2")
    with assert_raises(contains="duplicate --count"):
        _ = parse_affine_svj_fit_args(duplicate)

    var bad_number = required_args()
    bad_number.append("--grid-du")
    bad_number.append("not-a-number")
    with assert_raises(contains="invalid number"):
        _ = parse_affine_svj_fit_args(bad_number)

    var bad_range = required_args()
    bad_range.append("--armijo-shrink")
    bad_range.append("1.0")
    with assert_raises(contains="strictly between"):
        _ = parse_affine_svj_fit_args(bad_range)

    var missing_value = required_args()
    missing_value.append("--fd-step")
    with assert_raises(contains="missing value"):
        _ = parse_affine_svj_fit_args(missing_value)


def test_fit_cli_help_needs_no_other_options() raises:
    var help_args: List[String] = ["--help"]
    var config = parse_affine_svj_fit_args(help_args)
    assert_true(config.show_help)


def test_stationary_gamma_prior_parameterization() raises:
    var params = AffineSvjParams(
        0.05,
        -0.02,
        0.1,
        0.05,
        -0.1,
        0.2,
        -0.6,
        0.04,
        1.5,
        0.3,
        1.0 / 252.0,
    )
    var prior = stationary_gamma_prior(params)
    assert_almost_equal(prior[0], 0.03, atol=1e-15)
    assert_almost_equal(prior[1], 4.0 / 3.0, atol=1e-14)


def test_fit_command_preserves_output_on_input_failure() raises:
    var returns_path = String("/tmp/approx_mll_fit_bad_returns.bin")
    var out_path = String("/tmp/approx_mll_fit_untouched.bin")
    var one_return: List[Float64] = [0.0]
    var encoded = encode_big_endian_f64_returns(one_return)
    var returns_file = open(returns_path, "w")
    returns_file.write_all(encoded)
    returns_file.close()

    var sentinel: InlineArray[UInt8, 3] = [7, 8, 9]
    var out_file = open(out_path, "w")
    out_file.write_all(sentinel)
    out_file.close()

    var args: List[String] = [
        "--returns",
        returns_path,
        "--count",
        "2",
        "--initial",
        "/tmp/does_not_need_to_exist.bin",
        "--out",
        out_path,
        "--grid-count",
        "2",
        "--max-iterations",
        "1",
    ]
    var config = parse_affine_svj_fit_args(args)
    with assert_raises(contains="exactly 16 bytes"):
        run_affine_svj_fit[2](config)

    var preserved_file = open(out_path, "r")
    var preserved = preserved_file.read_bytes()
    assert_equal(len(preserved), 3)
    for i in range(3):
        assert_equal(preserved[i], sentinel[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
