from affine_svj import (
    AFFINE_SVJ_ARTIFACT_SIZE,
    AffineSvjParams,
    decode_affine_svj_params,
    encode_affine_svj_params,
    read_affine_svj_params,
    write_affine_svj_params,
)
from std.collections import List
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)


def make_params() -> AffineSvjParams:
    return AffineSvjParams(
        0.0, 1.0, 0.0, 1.0, -2.0, 0.5, -0.5, 2.0, 4.0, 8.0, 0.25
    )


def test_canonical_values_and_free_values() raises:
    var params = make_params()
    params.validate()
    var values = params.values()
    var free = params.free_values()
    for i in range(10):
        assert_equal(free[i], values[i])
    assert_equal(values[10], params.dt)

    var rebuilt = AffineSvjParams.with_free_values(free, params.dt)
    var rebuilt_values = rebuilt.values()
    for i in range(11):
        assert_equal(rebuilt_values[i], values[i])

    assert_almost_equal(
        params.jump_compensator(),
        -0.8466450331550716,
        atol=1e-15,
    )


def test_artifact_golden_bytes_and_roundtrip() raises:
    var expected: InlineArray[UInt8, 104] = [
        65,
        77,
        76,
        83,
        86,
        74,
        49,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        11,
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
        191,
        224,
        0,
        0,
        0,
        0,
        0,
        0,
        64,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        64,
        16,
        0,
        0,
        0,
        0,
        0,
        0,
        64,
        32,
        0,
        0,
        0,
        0,
        0,
        0,
        63,
        208,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert_equal(AFFINE_SVJ_ARTIFACT_SIZE, 104)
    var encoded = encode_affine_svj_params(make_params())
    for i in range(104):
        assert_equal(encoded[i], expected[i])

    var decoded = decode_affine_svj_params(encoded)
    var actual_values = decoded.values()
    var expected_values = make_params().values()
    for i in range(11):
        assert_equal(actual_values[i], expected_values[i])


def test_artifact_rejects_length_magic_and_headers() raises:
    var encoded = encode_affine_svj_params(make_params())
    var short = List[UInt8]()
    for i in range(103):
        short.append(encoded[i])
    with assert_raises(contains="exactly 104 bytes"):
        _ = decode_affine_svj_params(short)

    var trailing = List[UInt8]()
    for i in range(104):
        trailing.append(encoded[i])
    trailing.append(0)
    with assert_raises(contains="exactly 104 bytes"):
        _ = decode_affine_svj_params(trailing)

    var bad = encoded.copy()
    bad[0] = 0
    with assert_raises(contains="magic"):
        _ = decode_affine_svj_params(bad)

    bad = encoded.copy()
    bad[9] = 2
    with assert_raises(contains="schema"):
        _ = decode_affine_svj_params(bad)

    bad = encoded.copy()
    bad[11] = 2
    with assert_raises(contains="model"):
        _ = decode_affine_svj_params(bad)

    bad = encoded.copy()
    bad[13] = 2
    with assert_raises(contains="measure"):
        _ = decode_affine_svj_params(bad)

    bad = encoded.copy()
    bad[15] = 10
    with assert_raises(contains="field count"):
        _ = decode_affine_svj_params(bad)


def test_artifact_rejects_nonfinite_and_invalid_domains() raises:
    var encoded = encode_affine_svj_params(make_params())
    var bad = encoded.copy()
    # +Infinity in the first Float64 field.
    bad[16] = 127
    bad[17] = 240
    with assert_raises(contains="finite"):
        _ = decode_affine_svj_params(bad)

    var overflowing_compensator = AffineSvjParams(
        0.0, 1.0, 0.0, 1.0, 1000.0, 0.5, -0.5, 2.0, 4.0, 8.0, 0.25
    )
    with assert_raises(contains="jump compensator"):
        overflowing_compensator.validate()
    with assert_raises(contains="jump compensator"):
        _ = encode_affine_svj_params(overflowing_compensator)

    with assert_raises(contains="intensities"):
        AffineSvjParams(
            0.0, 1.0, -0.1, 1.0, -2.0, 0.5, -0.5, 2.0, 4.0, 8.0, 0.25
        ).validate()
    with assert_raises(contains="jump_std"):
        AffineSvjParams(
            0.0, 1.0, 0.0, 1.0, -2.0, -0.5, -0.5, 2.0, 4.0, 8.0, 0.25
        ).validate()
    with assert_raises(contains="rho"):
        AffineSvjParams(
            0.0, 1.0, 0.0, 1.0, -2.0, 0.5, 1.0, 2.0, 4.0, 8.0, 0.25
        ).validate()
    with assert_raises(contains="alpha"):
        AffineSvjParams(
            0.0, 1.0, 0.0, 1.0, -2.0, 0.5, -0.5, 0.0, 4.0, 8.0, 0.25
        ).validate()
    with assert_raises(contains="beta"):
        AffineSvjParams(
            0.0, 1.0, 0.0, 1.0, -2.0, 0.5, -0.5, 2.0, 0.0, 8.0, 0.25
        ).validate()
    with assert_raises(contains="volvol"):
        AffineSvjParams(
            0.0, 1.0, 0.0, 1.0, -2.0, 0.5, -0.5, 2.0, 4.0, 0.0, 0.25
        ).validate()
    with assert_raises(contains="dt"):
        AffineSvjParams(
            0.0, 1.0, 0.0, 1.0, -2.0, 0.5, -0.5, 2.0, 4.0, 8.0, 0.0
        ).validate()


def test_file_codec_roundtrip_and_trailing_rejection() raises:
    var path = String("/tmp/approx_mll_affine_svj_params.bin")
    write_affine_svj_params(path, make_params())
    var decoded = read_affine_svj_params(path)
    var actual = decoded.values()
    var expected = make_params().values()
    for i in range(11):
        assert_equal(actual[i], expected[i])

    var data = encode_affine_svj_params(make_params())
    var file = open(path, "w")
    file.write_all(data)
    var extra: InlineArray[UInt8, 1] = [0]
    file.write_all(extra)
    file.close()
    with assert_raises(contains="exactly 104 bytes"):
        _ = read_affine_svj_params(path)

    # Leave the shared temporary fixture valid for sibling CLI smoke tests.
    write_affine_svj_params(path, make_params())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
