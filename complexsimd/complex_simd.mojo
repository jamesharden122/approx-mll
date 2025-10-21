from math import sin, cos

struct ComplexSIMD[T: DType, L: Int]:
    var re: SIMD[T, L]
    var im: SIMD[T, L]

    fn __init__(out self, re: SIMD[T, L], im: SIMD[T, L]):
        self.re = re
        self.im = im

    @staticmethod
    fn zero() -> Self:
        return Self(SIMD[T, L](), SIMD[T, L]())

    @staticmethod
    fn from_scalars(r: Scalar[T], i: Scalar[T]) -> Self:
        return Self(SIMD[T, L](r), SIMD[T, L](i))

    fn add(self, other: Self) -> Self:
        return Self(self.re + other.re, self.im + other.im)

    fn sub(self, other: Self) -> Self:
        return Self(self.re - other.re, self.im - other.im)

    fn mul(self, other: Self) -> Self:
        var ac = self.re * other.re
        var bd = self.im * other.im
        var ad = self.re * other.im
        var bc = self.im * other.re
        return Self(ac - bd, ad + bc)

    fn conj(self) -> Self:
        return Self(self.re, -self.im)

    @staticmethod
    fn exp_i(theta: SIMD[T, L]) -> Self:
        var c = cos(theta)
        var s = sin(theta)
        return Self(c, s)

    fn mul_exp_i(self, theta: SIMD[T, L]) -> Self:
        var c = cos(theta)
        var s = sin(theta)
        var re = self.re * c - self.im * s
        var im = self.re * s + self.im * c
        return Self(re, im)
