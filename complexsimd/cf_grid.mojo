from std.collections import List
from complexsimd.cf_grid_simd import CFGridSIMD

struct CFGrid[T: DType, N: Int]:
    var re_components: List[List[Scalar[Self.T]]]
    var im_components: List[List[Scalar[Self.T]]]

    def __init__(out self, re_components: List[List[Scalar[Self.T]]], im_components: List[List[Scalar[Self.T]]]):
        if len(re_components) != Self.N or len(im_components) != Self.N:
            print("CFGrid: component list size must equal N")
        self.re_components = re_components.copy()
        self.im_components = im_components.copy()

    def num_components(self) -> Int:
        return Self.N

    def k_valid(self, comp: Int) -> Int:
        var r = len(self.re_components[comp])
        var i = len(self.im_components[comp])
        return r if r <= i else i

    def get(self, comp: Int, idx: Int) -> Tuple[Scalar[Self.T], Scalar[Self.T]]:
        return (self.re_components[comp][idx], self.im_components[comp][idx])

    def set(mut self, comp: Int, idx: Int, re: Scalar[Self.T], im: Scalar[Self.T]):
        self.re_components[comp][idx] = re
        self.im_components[comp][idx] = im

    @staticmethod
    def from_k_per_comp(k_per_comp: List[Int]) -> Self:
        if len(k_per_comp) != Self.N:
            print("CFGrid: k_per_comp size must equal N")
        var re = List[List[Scalar[Self.T]]]()
        var im = List[List[Scalar[Self.T]]]()
        re.reserve(Self.N)
        im.reserve(Self.N)
        for c in range(Self.N):
            var rc = List[Scalar[Self.T]]()
            var ic = List[Scalar[Self.T]]()
            rc.reserve(k_per_comp[c])
            ic.reserve(k_per_comp[c])
            for _ in range(k_per_comp[c]):
                rc.append(Scalar[Self.T](0))
                ic.append(Scalar[Self.T](0))
            re.append(rc^)
            im.append(ic^)
        return Self(re, im)

    def to_simd[L: Int](self) -> CFGridSIMD[Self.T, L]:
        return CFGridSIMD[Self.T, L].from_components(self.re_components, self.im_components)

    @staticmethod
    def from_simd[L: Int](grid: CFGridSIMD[Self.T, L]) -> Self:
        var components = grid.to_components()
        return Self(components[0], components[1])
