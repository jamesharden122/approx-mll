from std.math import atan2, cos, exp, log, sin, sqrt


struct ComplexSIMD[T: DType, L: Int]:
    var re: SIMD[Self.T, Self.L]
    var im: SIMD[Self.T, Self.L]

    def __init__(out self, re: SIMD[Self.T, Self.L], im: SIMD[Self.T, Self.L]):
        self.re = re
        self.im = im

    @staticmethod
    def zero() -> Self:
        return Self(SIMD[Self.T, Self.L](), SIMD[Self.T, Self.L]())

    @staticmethod
    def from_scalars(r: Scalar[Self.T], i: Scalar[Self.T]) -> Self:
        return Self(SIMD[Self.T, Self.L](r), SIMD[Self.T, Self.L](i))

    def add(self, other: Self) -> Self:
        return Self(self.re + other.re, self.im + other.im)

    def sub(self, other: Self) -> Self:
        return Self(self.re - other.re, self.im - other.im)

    def scale(self, value: SIMD[Self.T, Self.L]) -> Self:
        return Self(self.re * value, self.im * value)

    def mul(self, other: Self) -> Self:
        var ac = self.re * other.re
        var bd = self.im * other.im
        var ad = self.re * other.im
        var bc = self.im * other.re
        return Self(ac - bd, ad + bc)

    def div(self, other: Self) -> Self where Self.T.is_floating_point():
        var denominator = (
            other.re * other.re
            + other.im * other.im
            + SIMD[Self.T, Self.L](1e-300)
        )
        return Self(
            (self.re * other.re + self.im * other.im) / denominator,
            (self.im * other.re - self.re * other.im) / denominator,
        )

    def conj(self) -> Self:
        return Self(self.re, -self.im)

    def exp(self) -> Self where Self.T.is_floating_point():
        var magnitude = exp(self.re)
        return Self(magnitude * cos(self.im), magnitude * sin(self.im))

    def log(self) -> Self where Self.T.is_floating_point():
        var magnitude_squared = (
            self.re * self.re
            + self.im * self.im
            + SIMD[Self.T, Self.L](1e-300)
        )
        return Self(
            SIMD[Self.T, Self.L](0.5) * log(magnitude_squared),
            atan2(self.im, self.re),
        )

    def sqrt(self) -> Self where Self.T.is_floating_point():
        var magnitude = sqrt(self.re * self.re + self.im * self.im)
        var real_squared = (
            SIMD[Self.T, Self.L](0.5) * (magnitude + self.re)
        )
        var imag_squared = (
            SIMD[Self.T, Self.L](0.5) * (magnitude - self.re)
        )
        for lane in range(Self.L):
            if real_squared[lane] < 0:
                real_squared[lane] = 0
            if imag_squared[lane] < 0:
                imag_squared[lane] = 0
        var real_part = sqrt(real_squared)
        var imag_part = sqrt(imag_squared)
        for lane in range(Self.L):
            if self.im[lane] < 0:
                imag_part[lane] = -imag_part[lane]
        return Self(real_part, imag_part)

    @staticmethod
    def exp_i(
        theta: SIMD[Self.T, Self.L]
    ) -> Self where Self.T.is_floating_point():
        return Self(cos(theta), sin(theta))

    def mul_exp_i(
        self, theta: SIMD[Self.T, Self.L]
    ) -> Self where Self.T.is_floating_point():
        var c = cos(theta)
        var s = sin(theta)
        return Self(
            self.re * c - self.im * s,
            self.re * s + self.im * c,
        )
