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
        let ac = self.re * other.re
        let bd = self.im * other.im
        let ad = self.re * other.im
        let bc = self.im * other.re
        return Self(ac - bd, ad + bc)

    fn conj(self) -> Self:
        return Self(self.re, -self.im)

    @staticmethod
    fn exp_i(theta: SIMD[T, L]) -> Self:
        let c = cos(theta)
        let s = sin(theta)
        return Self(c, s)

    fn mul_exp_i(self, theta: SIMD[T, L]) -> Self:
        let c = cos(theta)
        let s = sin(theta)
        let re = self.re * c - self.im * s
        let im = self.re * s + self.im * c
        return Self(re, im)
