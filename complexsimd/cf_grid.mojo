from complexsimd.complex_simd import ComplexSIMD
from complexsimd.cf_grid_simd import CFGridSIMD

struct CFGrid[T: DType, N: Int]:
    var re_components: List[List[Scalar[T]]]
    var im_components: List[List[Scalar[T]]]

    fn __init__(out self, re_components: List[List[Scalar[T]]], im_components: List[List[Scalar[T]]]):
        if len(re_components) != N or len(im_components) != N:
            print("CFGrid: component list size must equal N")
        self.re_components = re_components
        self.im_components = im_components

    fn num_components(self) -> Int:
        return N

    fn k_valid(self, comp: Int) -> Int:
        var r = len(self.re_components[comp])
        var i = len(self.im_components[comp])
        return r if r <= i else i

    fn get(self, comp: Int, idx: Int) -> (Scalar[T], Scalar[T]):
        return (self.re_components[comp][idx], self.im_components[comp][idx])

    fn set(mut self, comp: Int, idx: Int, re: Scalar[T], im: Scalar[T]):
        self.re_components[comp][idx] = re
        self.im_components[comp][idx] = im

    @staticmethod
    fn from_k_per_comp(k_per_comp: List[Int]) -> Self:
        if len(k_per_comp) != N:
            print("CFGrid: k_per_comp size must equal N")
        var re = List[List[Scalar[T]]]()
        var im = List[List[Scalar[T]]]()
        re.reserve(N)
        im.reserve(N)
        for c in range(N):
            var rc = List[Scalar[T]]()
            var ic = List[Scalar[T]]()
            rc.reserve(k_per_comp[c])
            ic.reserve(k_per_comp[c])
            for _ in range(k_per_comp[c]):
                rc.append(Scalar[T](0))
                ic.append(Scalar[T](0))
            re.append(rc)
            im.append(ic)
        return Self(re, im)

    fn to_simd[L: Int](self) -> CFGridSIMD[T, L]:
        return CFGridSIMD[T, L].from_components(self.re_components, self.im_components)

    @staticmethod
    fn from_simd[L: Int](grid: CFGridSIMD[T, L]) -> Self:
        var (re_c, im_c) = grid.to_components()
        return Self(re_c, im_c)
