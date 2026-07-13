from std.collections import List
from std.memory import bitcast
from std.utils.numerics import isfinite


def _load_f64_be(data: List[UInt8], offset: Int) -> Float64:
    var bits = UInt64(0)
    for i in range(8):
        bits = (bits << 8) | UInt64(data[offset + i])
    return bitcast[DType.float64](bits)


def _store_f64_be(mut output: List[UInt8], value: Float64):
    var bits = bitcast[DType.uint64](value)
    for i in range(8):
        var shift = UInt64(56 - 8 * i)
        output.append(UInt8((bits >> shift) & UInt64(0xFF)))


def decode_big_endian_f64_returns(
    data: List[UInt8], count: Int
) raises -> List[Float64]:
    """Decode exactly count headerless big-endian Float64 log returns."""
    if count <= 0:
        raise Error("return count must be greater than zero")
    if count > Int.MAX // 8:
        raise Error("return count is too large")
    var expected_size = count * 8
    if len(data) != expected_size:
        raise Error(
            String(
                "returns file must contain exactly ",
                expected_size,
                " bytes for ",
                count,
                " values",
            )
        )

    var values = List[Float64](capacity=count)
    for i in range(count):
        var value = _load_f64_be(data, 8 * i)
        if not isfinite(value)[0]:
            raise Error("returns must contain only finite Float64 values")
        values.append(value)
    return values^


def read_big_endian_f64_returns(
    path: String, count: Int
) raises -> List[Float64]:
    """Read exactly count headerless big-endian Float64 log returns."""
    var file = open(path, "r")
    var data = file.read_bytes()
    return decode_big_endian_f64_returns(data, count)


def encode_big_endian_f64_returns(values: List[Float64]) raises -> List[UInt8]:
    """Encode finite Float64 values for Mojo-only codec and CLI tests."""
    var output = List[UInt8](capacity=8 * len(values))
    for value in values:
        if not isfinite(value)[0]:
            raise Error("returns must contain only finite Float64 values")
        _store_f64_be(output, value)
    return output^
