from std.collections import List
from std.math import exp
from std.memory import bitcast
from std.utils.numerics import isfinite


comptime AFFINE_SVJ_ARTIFACT_SIZE = 104


struct AffineSvjParams(ImplicitlyCopyable):
    """Physical-measure affine SVJ parameters in their canonical order.

    `dt` is sampling metadata. The preceding ten values are the free model
    parameters used by estimation.
    """

    var mu0: Float64
    var mu1: Float64
    var lambda0: Float64
    var lambda1: Float64
    var jump_mean: Float64
    var jump_std: Float64
    var rho: Float64
    var alpha: Float64
    var beta: Float64
    var volvol: Float64
    var dt: Float64

    def __init__(
        out self,
        mu0: Float64,
        mu1: Float64,
        lambda0: Float64,
        lambda1: Float64,
        jump_mean: Float64,
        jump_std: Float64,
        rho: Float64,
        alpha: Float64,
        beta: Float64,
        volvol: Float64,
        dt: Float64,
    ):
        self.mu0 = mu0
        self.mu1 = mu1
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.volvol = volvol
        self.dt = dt

    def __init__(out self, values: InlineArray[Float64, 11]):
        self.mu0 = values[0]
        self.mu1 = values[1]
        self.lambda0 = values[2]
        self.lambda1 = values[3]
        self.jump_mean = values[4]
        self.jump_std = values[5]
        self.rho = values[6]
        self.alpha = values[7]
        self.beta = values[8]
        self.volvol = values[9]
        self.dt = values[10]

    def values(self) -> InlineArray[Float64, 11]:
        return [
            self.mu0,
            self.mu1,
            self.lambda0,
            self.lambda1,
            self.jump_mean,
            self.jump_std,
            self.rho,
            self.alpha,
            self.beta,
            self.volvol,
            self.dt,
        ]

    def free_values(self) -> InlineArray[Float64, 10]:
        """Return the ten estimable fields, excluding fixed sampling `dt`."""
        return [
            self.mu0,
            self.mu1,
            self.lambda0,
            self.lambda1,
            self.jump_mean,
            self.jump_std,
            self.rho,
            self.alpha,
            self.beta,
            self.volvol,
        ]

    @staticmethod
    def with_free_values(
        values: InlineArray[Float64, 10], dt: Float64
    ) raises -> Self:
        """Build and validate parameters while keeping `dt` out of optimization.
        """
        var params = Self(
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            values[8],
            values[9],
            dt,
        )
        params.validate()
        return params

    def jump_compensator(self) -> Float64:
        """Return E[exp(J)] - 1 for J ~ Normal(jump_mean, jump_std^2)."""
        return exp(self.jump_mean + 0.5 * self.jump_std * self.jump_std) - 1.0

    def validate(self) raises:
        """Reject non-finite parameters and values outside the model domain."""
        var vals = self.values()
        for i in range(11):
            if not isfinite(vals[i])[0]:
                raise Error("affine SVJ parameters must be finite")

        if not isfinite(self.jump_compensator())[0]:
            raise Error("affine SVJ jump compensator must be finite")

        if self.dt <= 0.0:
            raise Error("affine SVJ dt must be greater than zero")
        if self.alpha <= 0.0:
            raise Error("affine SVJ alpha must be greater than zero")
        if self.beta <= 0.0:
            raise Error("affine SVJ beta must be greater than zero")
        if self.volvol <= 0.0:
            raise Error("affine SVJ volvol must be greater than zero")
        if self.lambda0 < 0.0 or self.lambda1 < 0.0:
            raise Error("affine SVJ intensities must be nonnegative")
        if self.jump_std < 0.0:
            raise Error("affine SVJ jump_std must be nonnegative")
        if self.rho <= -1.0 or self.rho >= 1.0:
            raise Error("affine SVJ rho must lie strictly between -1 and 1")


def _store_u16_be(
    mut output: InlineArray[UInt8, AFFINE_SVJ_ARTIFACT_SIZE],
    offset: Int,
    value: UInt16,
):
    output[offset] = UInt8(value >> 8)
    output[offset + 1] = UInt8(value & UInt16(0xFF))


def _load_u16_be(
    data: InlineArray[UInt8, AFFINE_SVJ_ARTIFACT_SIZE], offset: Int
) -> UInt16:
    return (UInt16(data[offset]) << 8) | UInt16(data[offset + 1])


def _store_f64_be(
    mut output: InlineArray[UInt8, AFFINE_SVJ_ARTIFACT_SIZE],
    offset: Int,
    value: Float64,
):
    var bits = bitcast[DType.uint64](value)
    for i in range(8):
        var shift = 56 - 8 * i
        output[offset + i] = UInt8((bits >> UInt64(shift)) & UInt64(0xFF))


def _load_f64_be(
    data: InlineArray[UInt8, AFFINE_SVJ_ARTIFACT_SIZE], offset: Int
) -> Float64:
    var bits = UInt64(0)
    for i in range(8):
        bits = (bits << 8) | UInt64(data[offset + i])
    return bitcast[DType.float64](bits)


def encode_affine_svj_params(
    params: AffineSvjParams,
) raises -> InlineArray[UInt8, AFFINE_SVJ_ARTIFACT_SIZE]:
    """Encode a validated parameter value in the exact 104-byte wire format."""
    params.validate()
    var output = InlineArray[UInt8, AFFINE_SVJ_ARTIFACT_SIZE](fill=0)

    # "AMLSVJ1\0"
    output[0] = 65
    output[1] = 77
    output[2] = 76
    output[3] = 83
    output[4] = 86
    output[5] = 74
    output[6] = 49
    output[7] = 0

    _store_u16_be(output, 8, 1)  # schema version
    _store_u16_be(output, 10, 1)  # affine SVJ model ID
    _store_u16_be(output, 12, 1)  # physical-measure ID
    _store_u16_be(output, 14, 11)  # field count

    var vals = params.values()
    for i in range(11):
        _store_f64_be(output, 16 + 8 * i, vals[i])
    return output


def decode_affine_svj_params(
    data: InlineArray[UInt8, AFFINE_SVJ_ARTIFACT_SIZE]
) raises -> AffineSvjParams:
    """Decode and validate one exact-size affine SVJ parameter artifact."""
    if (
        data[0] != 65
        or data[1] != 77
        or data[2] != 76
        or data[3] != 83
        or data[4] != 86
        or data[5] != 74
        or data[6] != 49
        or data[7] != 0
    ):
        raise Error("invalid affine SVJ artifact magic")
    if _load_u16_be(data, 8) != 1:
        raise Error("unsupported affine SVJ artifact schema version")
    if _load_u16_be(data, 10) != 1:
        raise Error("unsupported affine SVJ artifact model ID")
    if _load_u16_be(data, 12) != 1:
        raise Error("unsupported affine SVJ artifact measure ID")
    if _load_u16_be(data, 14) != 11:
        raise Error("unsupported affine SVJ artifact field count")

    var values = InlineArray[Float64, 11](uninitialized=True)
    for i in range(11):
        values[i] = _load_f64_be(data, 16 + 8 * i)
    var params = AffineSvjParams(values)
    params.validate()
    return params


def decode_affine_svj_params(data: List[UInt8]) raises -> AffineSvjParams:
    """Decode runtime bytes, rejecting truncation and trailing data."""
    if len(data) != AFFINE_SVJ_ARTIFACT_SIZE:
        raise Error("affine SVJ artifact must be exactly 104 bytes")
    var exact = InlineArray[UInt8, AFFINE_SVJ_ARTIFACT_SIZE](fill=0)
    for i in range(AFFINE_SVJ_ARTIFACT_SIZE):
        exact[i] = data[i]
    return decode_affine_svj_params(exact)


def read_affine_svj_params(path: String) raises -> AffineSvjParams:
    """Read, decode, and validate an affine SVJ parameter artifact."""
    var file = open(path, "r")
    var data = file.read_bytes()
    return decode_affine_svj_params(data)


def write_affine_svj_params(path: String, params: AffineSvjParams) raises:
    """Validate and write exactly one affine SVJ parameter artifact."""
    var data = encode_affine_svj_params(params)
    var file = open(path, "w")
    file.write_all(data)
