from complexsimd.complex_simd import ComplexSIMD

struct CFGrid[T: DType, L: Int]:
    var re: List[Scalar[T]]
    var im: List[Scalar[T]]
    var K: Int

    fn __init__(out self, re: List[Scalar[T]], im: List[Scalar[T]]):
        if re.size != im.size:
            print("CFGrid: re/im size mismatch; truncating to min size")
        self.K = re.size if re.size <= im.size else im.size
        self.re = re
        self.im = im

    fn lane(self, tile: Int) -> ComplexSIMD[T, L]:
        let off = tile * L
        var r = SIMD[T, L]()
        var i = SIMD[T, L]()
        for j in range(L):
            r[j] = self.re[off + j]
            i[j] = self.im[off + j]
        return ComplexSIMD[T, L](r, i)

    fn store_lane(mut self, tile: Int, z: ComplexSIMD[T, L]):
        let off = tile * L
        for j in range(L):
            self.re[off + j] = z.re[j]
            self.im[off + j] = z.im[j]
