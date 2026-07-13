from std.random import randn


struct DirMethod(Copyable, Movable):
    var raw: Int32

    def __init__(out self, raw: Int32):
        self.raw = raw

    @staticmethod
    def Gaussian() -> Self:
        return Self(Int32(0))

    @staticmethod
    def Identity() -> Self:
        return Self(Int32(1))


struct DirectionMat[D: Int](Copyable, Movable):
    """Owns a row-major D-by-D finite-difference direction matrix."""

    var p_mat: InlineArray[Float64, Self.D * Self.D]

    def __init__(out self, method: DirMethod):
        if method.raw == DirMethod.Gaussian().raw:
            self.p_mat = Self._gaussian()
        elif method.raw == DirMethod.Identity().raw:
            self.p_mat = Self._identity()
        else:
            self.p_mat = InlineArray[Float64, Self.D * Self.D](fill=0.0)

    @staticmethod
    def _gaussian(
        mu: Float64 = 0.0, sigma: Float64 = 1.0
    ) -> InlineArray[Float64, Self.D * Self.D]:
        var storage = InlineArray[Float64, Self.D * Self.D](uninitialized=True)
        randn[DType.float64](
            storage.unsafe_ptr(), Self.D * Self.D, mu, sigma
        )
        return storage

    @staticmethod
    def _identity() -> InlineArray[Float64, Self.D * Self.D]:
        var storage = InlineArray[Float64, Self.D * Self.D](fill=0.0)
        for i in range(Self.D):
            storage[i * Self.D + i] = 1.0
        return storage
