from stochcharfunc.svj1 import AffineSvjJointCF
from std.math import cos, sin
from std.utils.numerics import isfinite


struct GammaUpdate(ImplicitlyCopyable):
    var kappa: Float64
    var nu: Float64
    var conditional_mean: Float64
    var conditional_variance: Float64
    var density: Float64
    var valid: Bool

    def __init__(
        out self,
        kappa: Float64,
        nu: Float64,
        conditional_mean: Float64,
        conditional_variance: Float64,
        density: Float64,
        valid: Bool,
    ):
        self.kappa = kappa
        self.nu = nu
        self.conditional_mean = conditional_mean
        self.conditional_variance = conditional_variance
        self.density = density
        self.valid = valid


struct UniformGridInverterSIMDGrid[L: Int](Copyable, Movable):
    """Positive-frequency trapezoidal Fourier inverter.

    Positive-half inversion requires u0 = 0 so no frequency interval is
    omitted. The origin remains explicit metadata for grid construction.

    Frequencies are explicit metadata: u_j = u0 + j * du for j in
    [0, count). The final SIMD tile is lane-masked when count is not divisible
    by L. Conjugate symmetry supplies the negative-frequency half and the
    normalization is therefore du / pi.
    """

    var u0: Float64
    var du: Float64
    var count: Int

    def __init__(out self, u0: Float64, du: Float64, count: Int) raises:
        if not isfinite(u0)[0] or u0 != 0.0:
            raise Error("Fourier grid u0 must be exactly zero")
        if not isfinite(du)[0] or du <= 0.0:
            raise Error("Fourier grid du must be finite and positive")
        if count < 2:
            raise Error("Fourier grid requires at least two frequencies")
        self.u0 = u0
        self.du = du
        self.count = count

    def num_tiles(self) -> Int:
        return (self.count + Self.L - 1) // Self.L

    def frequency_tile(self, tile: Int) -> SIMD[DType.float64, Self.L]:
        var frequencies = SIMD[DType.float64, Self.L](0.0)
        for lane in range(Self.L):
            var index = tile * Self.L + lane
            if index < self.count:
                frequencies[lane] = self.u0 + self.du * Float64(index)
        return frequencies

    def joint_density(
        self,
        x: Float64,
        psi: Float64,
        model: AffineSvjJointCF,
        prior: InlineArray[Float64, 2],
    ) -> Float64:
        if (
            not isfinite(x)[0]
            or not isfinite(psi)[0]
            or not isfinite(prior[0])[0]
            or not isfinite(prior[1])[0]
            or prior[0] <= 0.0
            or prior[1] <= 0.0
        ):
            return -1.0

        var integral = 0.0
        for tile in range(self.num_tiles()):
            var frequencies = self.frequency_tile(tile)
            var transform = model.predictive_cf_simd[Self.L](
                frequencies, psi, prior[0], prior[1]
            )
            for lane in range(Self.L):
                var index = tile * Self.L + lane
                if index < self.count:
                    var u = Float64(frequencies[lane])
                    var angle = -u * x
                    var rotated_real = Float64(transform.re[lane]) * cos(
                        angle
                    ) - Float64(transform.im[lane]) * sin(angle)
                    var weight = 1.0
                    if index == 0 or index == self.count - 1:
                        weight = 0.5
                    integral += weight * rotated_real
        var result = integral * self.du / 3.14159265358979323846
        if not isfinite(result)[0]:
            return -1.0
        return result

    def density(
        self,
        x: Float64,
        model: AffineSvjJointCF,
        prior: InlineArray[Float64, 2],
    ) -> Float64:
        return self.joint_density(x, 0.0, model, prior)

    def update_gamma_prior(
        self,
        x: Float64,
        model: AffineSvjJointCF,
        prior: InlineArray[Float64, 2],
        h: Float64 = 1e-3,
    ) -> GammaUpdate:
        if not isfinite(h)[0] or h <= 0.0:
            return GammaUpdate(prior[0], prior[1], 0.0, 0.0, -1.0, False)
        var p_minus = self.joint_density(x, -h, model, prior)
        var p_zero = self.joint_density(x, 0.0, model, prior)
        var p_plus = self.joint_density(x, h, model, prior)
        if (
            p_zero <= 1e-300
            or not isfinite(p_zero)[0]
            or not isfinite(p_minus)[0]
            or not isfinite(p_plus)[0]
        ):
            return GammaUpdate(
                prior[0],
                prior[1],
                prior[0] * prior[1],
                prior[0] * prior[0] * prior[1],
                p_zero,
                False,
            )

        var conditional_mean = (p_plus - p_minus) / (2.0 * h * p_zero)
        var conditional_second = (p_plus - 2.0 * p_zero + p_minus) / (
            h * h * p_zero
        )
        var conditional_variance = (
            conditional_second - conditional_mean * conditional_mean
        )
        if (
            conditional_mean <= 0.0
            or conditional_variance <= 1e-16
            or not isfinite(conditional_mean)[0]
            or not isfinite(conditional_variance)[0]
        ):
            return GammaUpdate(
                prior[0],
                prior[1],
                conditional_mean,
                conditional_variance,
                p_zero,
                False,
            )

        var kappa = conditional_variance / conditional_mean
        var nu = conditional_mean * conditional_mean / conditional_variance
        if (
            kappa <= 0.0
            or nu <= 0.0
            or not isfinite(kappa)[0]
            or not isfinite(nu)[0]
        ):
            return GammaUpdate(
                prior[0],
                prior[1],
                conditional_mean,
                conditional_variance,
                p_zero,
                False,
            )
        return GammaUpdate(
            kappa,
            nu,
            conditional_mean,
            conditional_variance,
            p_zero,
            True,
        )
