from affine_svj import AffineSvjParams
from complexsimd.complex_simd import ComplexSIMD
from std.math import atan2, cos, exp, log, sin, sqrt


struct Complex64(ImplicitlyCopyable):
    """Small dependency-free complex scalar used by the reference CF kernel."""

    var re: Float64
    var im: Float64

    def __init__(out self, re: Float64, im: Float64 = 0.0):
        self.re = re
        self.im = im

    def add(self, other: Self) -> Self:
        return Self(self.re + other.re, self.im + other.im)

    def sub(self, other: Self) -> Self:
        return Self(self.re - other.re, self.im - other.im)

    def scale(self, value: Float64) -> Self:
        return Self(self.re * value, self.im * value)

    def mul(self, other: Self) -> Self:
        return Self(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    def div(self, other: Self) -> Self:
        var denominator = other.re * other.re + other.im * other.im
        if denominator < 1e-300:
            denominator = 1e-300
        return Self(
            (self.re * other.re + self.im * other.im) / denominator,
            (self.im * other.re - self.re * other.im) / denominator,
        )

    def exp(self) -> Self:
        var magnitude = exp(self.re)
        return Self(magnitude * cos(self.im), magnitude * sin(self.im))

    def log(self) -> Self:
        var magnitude_squared = self.re * self.re + self.im * self.im
        if magnitude_squared < 1e-300:
            magnitude_squared = 1e-300
        return Self(0.5 * log(magnitude_squared), atan2(self.im, self.re))

    def sqrt(self) -> Self:
        var magnitude = sqrt(self.re * self.re + self.im * self.im)
        var real_squared = 0.5 * (magnitude + self.re)
        var imag_squared = 0.5 * (magnitude - self.re)
        if real_squared < 0.0:
            real_squared = 0.0
        if imag_squared < 0.0:
            imag_squared = 0.0
        var real_part = sqrt(real_squared)
        var imag_part = sqrt(imag_squared)
        if self.im < 0.0:
            imag_part = -imag_part
        return Self(real_part, imag_part)


struct AffineSvjJointCF(Copyable, Movable):
    """Physical-measure affine Bates characteristic function.

    The state dynamics are

      dX = (mu0 + mu1 V - (lambda0 + lambda1 V) k - V/2) dt
           + sqrt(V) dW_x + J dN,
      dV = beta (alpha - V) dt + volvol sqrt(V) dW_v,

    with corr(dW_x, dW_v) = rho and Gaussian J. psi is the
    terminal-variance transform argument. Integrating over a Gamma prior with
    scale kappa and shape nu gives the predictive transform used by the AML
    filter.
    """

    var params: AffineSvjParams

    def __init__(out self, params: AffineSvjParams):
        self.params = params.copy()

    def _riccati_scalar(
        self, u: Float64, psi: Float64
    ) -> Tuple[Complex64, Complex64]:
        var sigma = self.params.volvol
        var a = 0.5 * sigma * sigma
        var k = self.params.jump_compensator()

        var jump_magnitude = exp(
            -0.5 * self.params.jump_std * self.params.jump_std * u * u
        )
        var q = Complex64(
            jump_magnitude * cos(self.params.jump_mean * u),
            jump_magnitude * sin(self.params.jump_mean * u),
        )
        var jump_exponent = q.sub(Complex64(1.0))

        var b = Complex64(-self.params.beta, self.params.rho * sigma * u)
        var c = Complex64(
            -0.5 * u * u + self.params.lambda1 * jump_exponent.re,
            u
                * (
                    self.params.mu1
                    - 0.5
                    - self.params.lambda1 * k
                )
                + self.params.lambda1 * jump_exponent.im,
        )
        var discriminant = b.mul(b).sub(c.scale(4.0 * a))
        var g = discriminant.sqrt()
        var two_a = 2.0 * a
        var y1 = Complex64(-b.re, -b.im).add(g).scale(1.0 / two_a)
        var y2 = Complex64(-b.re, -b.im).sub(g).scale(1.0 / two_a)
        var psi_complex = Complex64(psi)
        var z0 = psi_complex.sub(y2).div(y1.sub(psi_complex))
        var zdt = z0.mul(g.scale(-self.params.dt).exp())
        var one = Complex64(1.0)
        var D = y2.add(y1.mul(zdt)).div(one.add(zdt))

        var g_safe = g
        if g_safe.re * g_safe.re + g_safe.im * g_safe.im < 1e-300:
            g_safe = Complex64(1e-150)
        var log_ratio = one.add(zdt).log().sub(one.add(z0).log())
        var integral_D = y2.scale(self.params.dt).sub(
            y1.sub(y2).div(g_safe).mul(log_ratio)
        )

        var constant_jump = jump_exponent.scale(self.params.lambda0)
        var constant_return = Complex64(
            constant_jump.re,
            u * (self.params.mu0 - self.params.lambda0 * k)
                + constant_jump.im,
        ).scale(self.params.dt)
        var C = constant_return.add(
            integral_D.scale(self.params.beta * self.params.alpha)
        )
        return (C, D)

    def predictive_cf_scalar(
        self,
        u: Float64,
        psi: Float64,
        kappa: Float64,
        nu: Float64,
    ) -> Tuple[Float64, Float64]:
        var riccati = self._riccati_scalar(u, psi)
        var C = riccati[0]
        var D = riccati[1]
        var gamma_base = Complex64(1.0).sub(D.scale(kappa))
        var gamma_mgf = gamma_base.log().scale(-nu).exp()
        var result = C.exp().mul(gamma_mgf)
        return (result.re, result.im)

    def predictive_cf_simd[L: Int](
        self,
        u: SIMD[DType.float64, L],
        psi: Float64,
        kappa: Float64,
        nu: Float64,
    ) -> ComplexSIMD[DType.float64, L]:
        comptime scalar = SIMD[DType.float64, L]
        var sigma = self.params.volvol
        var a = 0.5 * sigma * sigma
        var k = self.params.jump_compensator()
        var one = ComplexSIMD[DType.float64, L].from_scalars(1.0, 0.0)

        var jump_magnitude = exp(
            -scalar(0.5 * self.params.jump_std * self.params.jump_std)
                * u
                * u
        )
        var jump_phase = scalar(self.params.jump_mean) * u
        var q = ComplexSIMD[DType.float64, L](
            jump_magnitude * cos(jump_phase),
            jump_magnitude * sin(jump_phase),
        )
        var jump_exponent = q.sub(one)
        var b = ComplexSIMD[DType.float64, L](
            scalar(-self.params.beta),
            scalar(self.params.rho * sigma) * u,
        )
        var c = ComplexSIMD[DType.float64, L](
            scalar(-0.5) * u * u
                + scalar(self.params.lambda1) * jump_exponent.re,
            u
                * scalar(
                    self.params.mu1
                    - 0.5
                    - self.params.lambda1 * k
                )
                + scalar(self.params.lambda1) * jump_exponent.im,
        )
        var g = b.mul(b).sub(
            c.scale(scalar(4.0 * a))
        ).sqrt()
        var negative_b = b.scale(scalar(-1.0))
        var y1 = negative_b.add(g).scale(scalar(1.0 / (2.0 * a)))
        var y2 = negative_b.sub(g).scale(scalar(1.0 / (2.0 * a)))
        var psi_complex = ComplexSIMD[DType.float64, L].from_scalars(
            psi, 0.0
        )
        var z0 = psi_complex.sub(y2).div(y1.sub(psi_complex))
        var zdt = z0.mul(
            g.scale(scalar(-self.params.dt)).exp()
        )
        var D = y2.add(y1.mul(zdt)).div(one.add(zdt))
        var log_ratio = one.add(zdt).log().sub(one.add(z0).log())
        var integral_D = y2.scale(scalar(self.params.dt)).sub(
            y1.sub(y2).div(g).mul(log_ratio)
        )

        var constant_jump = jump_exponent.scale(
            scalar(self.params.lambda0)
        )
        var constant_return = ComplexSIMD[DType.float64, L](
            constant_jump.re,
            u * scalar(self.params.mu0 - self.params.lambda0 * k)
                + constant_jump.im,
        ).scale(scalar(self.params.dt))
        var C = constant_return.add(
            integral_D.scale(
                scalar(self.params.beta * self.params.alpha)
            )
        )
        var gamma_base = one.sub(D.scale(scalar(kappa)))
        var gamma_mgf = gamma_base.log().scale(scalar(-nu)).exp()
        return C.exp().mul(gamma_mgf)

    def predictive_cf_latent_simd_re[L: Int, T: Int](
        self,
        u: SIMD[DType.float64, L],
        psi: SIMD[DType.float64, 1],
        prior: InlineArray[Float64, T],
    ) -> SIMD[DType.float64, L]:
        return self.predictive_cf_simd[L](
            u, Float64(psi[0]), prior[0], prior[1]
        ).re

    def predictive_cf_latent_simd_im[L: Int, T: Int](
        self,
        u: SIMD[DType.float64, L],
        psi: SIMD[DType.float64, 1],
        prior: InlineArray[Float64, T],
    ) -> SIMD[DType.float64, L]:
        return self.predictive_cf_simd[L](
            u, Float64(psi[0]), prior[0], prior[1]
        ).im
