from std.collections import List
from complexsimd.complex_simd import ComplexSIMD

struct CFGridSIMD[T: DType, L: Int](Copyable, Movable):
    # SIMD-tiled complex CF grid with N components (dimensions), where N is
    # inferred at runtime from the size of k_valid_per_comp.
    # Lane layout per tile: lanes are split into N contiguous components.
    # Component c occupies lanes [c*W, (c+1)*W), where W = L / N.
    # Requires L % N == 0. If not, packing pads the last component with zeros.

    var re_tiles: List[SIMD[Self.T, Self.L]]
    var im_tiles: List[SIMD[Self.T, Self.L]]
    var k_valid_per_comp: List[Int]   # logical scalar count per component

    def __init__(out self, re_tiles: List[SIMD[Self.T, Self.L]], im_tiles: List[SIMD[Self.T, Self.L]], k_valid_per_comp: List[Int]):
        self.re_tiles = re_tiles.copy()
        self.im_tiles = im_tiles.copy()
        self.k_valid_per_comp = k_valid_per_comp.copy()

    def num_components(self) -> Int:
        return len(self.k_valid_per_comp)

    def lanes_per_component(self) -> Int:
        # W = L / N
        var n = self.num_components()
        return Self.L / n

    def num_tiles(self) -> Int:
        return len(self.re_tiles)

    def lane(self, tile: Int) -> ComplexSIMD[Self.T, Self.L]:
        return ComplexSIMD[Self.T, Self.L](self.re_tiles[tile], self.im_tiles[tile])

    def store_lane(mut self, tile: Int, z: ComplexSIMD[Self.T, Self.L]):
        self.re_tiles[tile] = z.re
        self.im_tiles[tile] = z.im

    def component_tile_count(self, comp: Int) -> Int:
        var W = self.lanes_per_component()
        var k = self.k_valid_per_comp[comp]
        return (k + W - 1) / W

    # Map a component scalar index -> (tile, lane_idx)
    def _map_index(self, comp: Int, idx: Int) -> Tuple[Int, Int]:
        var W = self.lanes_per_component()
        var tile = idx / W
        var off  = idx % W
        var lane = comp * W + off
        return (tile, lane)

    def get_component_value(self, comp: Int, idx: Int) -> Tuple[Scalar[Self.T], Scalar[Self.T]]:
        var (tile, lane) = self._map_index(comp, idx)
        var r = self.re_tiles[tile]
        var i = self.im_tiles[tile]
        return (r[lane], i[lane])

    def set_component_value(mut self, comp: Int, idx: Int, re: Scalar[Self.T], im: Scalar[Self.T]):
        var (tile, lane) = self._map_index(comp, idx)
        var r = self.re_tiles[tile]
        var i = self.im_tiles[tile]
        r[lane] = re
        i[lane] = im
        self.re_tiles[tile] = r
        self.im_tiles[tile] = i

    @staticmethod
    def from_components(
        re_components: List[List[Scalar[Self.T]]],
        im_components: List[List[Scalar[Self.T]]]
    ) -> Self:
        # Build a CFGridSIMD by packing N component vectors into tiles.
        var N = len(re_components)
        if len(im_components) != N:
            print("CFGridSIMD: component list size mismatch")

        var W = Self.L / N
        if W * N != Self.L:
            print("CFGridSIMD: L % N != 0; last lanes will be zero-padded")

        var k_valid = List[Int]()
        k_valid.reserve(N)
        var max_tiles = 0
        for c in range(N):
            var rc_sz = len(re_components[c])
            var ic_sz = len(im_components[c])
            var kc = rc_sz if rc_sz <= ic_sz else ic_sz
            k_valid.append(kc)
            var tiles_c = (kc + W - 1) / W
            if tiles_c > max_tiles:
                max_tiles = tiles_c

        var re_tiles = List[SIMD[Self.T, Self.L]]()
        var im_tiles = List[SIMD[Self.T, Self.L]]()
        re_tiles.reserve(max_tiles)
        im_tiles.reserve(max_tiles)
        for t in range(max_tiles):
            var r = SIMD[Self.T, Self.L]()
            var i = SIMD[Self.T, Self.L]()
            # fill all components for tile t
            for c in range(N):
                for off in range(W):
                    var lane = c * W + off
                    var idx  = t * W + off
                    if idx < k_valid[c]:
                        r[lane] = re_components[c][idx]
                        i[lane] = im_components[c][idx]
                    else:
                        r[lane] = Scalar[Self.T](0)
                        i[lane] = Scalar[Self.T](0)
            re_tiles.append(r)
            im_tiles.append(i)

        return Self(re_tiles, im_tiles, k_valid)

    def to_components(self) -> Tuple[List[List[Scalar[Self.T]]], List[List[Scalar[Self.T]]]]:
        # Unpack into N component vectors (drop padded tails)
        var W = self.lanes_per_component()
        var N = self.num_components()
        var out_re = List[List[Scalar[Self.T]]]()
        var out_im = List[List[Scalar[Self.T]]]()
        out_re.reserve(N)
        out_im.reserve(N)
        for c in range(N):
            var re_c = List[Scalar[Self.T]]()
            var im_c = List[Scalar[Self.T]]()
            re_c.reserve(self.k_valid_per_comp[c])
            im_c.reserve(self.k_valid_per_comp[c])
            var kc = self.k_valid_per_comp[c]
            var tiles_c = (kc + W - 1) / W
            for t in range(tiles_c):
                var r = self.re_tiles[t]
                var i = self.im_tiles[t]
                for off in range(W):
                    var idx = t * W + off
                    if idx < kc:
                        var lane = c * W + off
                        re_c.append(r[lane])
                        im_c.append(i[lane])
            out_re.append(re_c^)
            out_im.append(im_c^)
        return (out_re^, out_im^)
