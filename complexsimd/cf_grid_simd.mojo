from complexsimd.complex_simd import ComplexSIMD

struct CFGridSIMD[T: DType, L: Int]:
    # SIMD-tiled complex CF grid with N components (dimensions), where N is
    # inferred at runtime from the size of k_valid_per_comp.
    # Lane layout per tile: lanes are split into N contiguous components.
    # Component c occupies lanes [c*W, (c+1)*W), where W = L / N.
    # Requires L % N == 0. If not, packing pads the last component with zeros.

    var re_tiles: List[SIMD[T, L]]
    var im_tiles: List[SIMD[T, L]]
    var k_valid_per_comp: List[Int]   # logical scalar count per component

    fn __init__(out self, re_tiles: List[SIMD[T, L]], im_tiles: List[SIMD[T, L]], k_valid_per_comp: List[Int]):
        self.re_tiles = ^re_tiles
        self.im_tiles = ^im_tiles
        self.k_valid_per_comp = ^k_valid_per_comp

    fn num_components(self) -> Int:
        return self.k_valid_per_comp.size

    fn lanes_per_component(self) -> Int:
        # W = L / N
        let n = self.num_components()
        return L / n

    fn num_tiles(self) -> Int:
        return self.re_tiles.size

    fn lane(self, tile: Int) -> ComplexSIMD[T, L]:
        return ComplexSIMD[T, L](self.re_tiles[tile], self.im_tiles[tile])

    fn store_lane(mut self, tile: Int, z: ComplexSIMD[T, L]):
        self.re_tiles[tile] = z.re
        self.im_tiles[tile] = z.im

    fn component_tile_count(self, comp: Int) -> Int:
        let W = self.lanes_per_component()
        let k = self.k_valid_per_comp[comp]
        return (k + W - 1) / W

    # Map a component scalar index -> (tile, lane_idx)
    fn _map_index(self, comp: Int, idx: Int) -> (Int, Int):
        let W = self.lanes_per_component()
        let tile = idx / W
        let off  = idx % W
        let lane = comp * W + off
        return (tile, lane)

    fn get_component_value(self, comp: Int, idx: Int) -> (Scalar[T], Scalar[T]):
        let (tile, lane) = self._map_index(comp, idx)
        let z = self.tiles[tile]
        return (z.re[lane], z.im[lane])

    fn set_component_value(mut self, comp: Int, idx: Int, re: Scalar[T], im: Scalar[T]):
        let (tile, lane) = self._map_index(comp, idx)
        var z = self.tiles[tile]
        z.re[lane] = re
        z.im[lane] = im
        self.tiles[tile] = z

    @staticmethod
    fn from_components(
        re_components: List[List[Scalar[T]]],
        im_components: List[List[Scalar[T]]]
    ) -> Self:
        # Build a CFGridSIMD by packing N component vectors into tiles.
        let N = re_components.size
        if im_components.size != N:
            print("CFGridSIMD: component list size mismatch")

        let W = L / N
        if W * N != L:
            print("CFGridSIMD: L % N != 0; last lanes will be zero-padded")

        var k_valid = List[Int]()
        k_valid.reserve(N)
        var max_tiles = 0
        for c in range(N):
            let kc = re_components[c].size if re_components[c].size <= im_components[c].size else im_components[c].size
            k_valid.push_back(kc)
            let tiles_c = (kc + W - 1) / W
            if tiles_c > max_tiles:
                max_tiles = tiles_c

        var re_tiles = List[SIMD[T, L]]()
        var im_tiles = List[SIMD[T, L]]()
        re_tiles.reserve(max_tiles)
        im_tiles.reserve(max_tiles)
        for t in range(max_tiles):
            var r = SIMD[T, L]()
            var i = SIMD[T, L]()
            # fill all components for tile t
            for c in range(N):
                for off in range(W):
                    let lane = c * W + off
                    let idx  = t * W + off
                    if idx < k_valid[c]:
                        r[lane] = re_components[c][idx]
                        i[lane] = im_components[c][idx]
                    else:
                        r[lane] = Scalar[T](0)
                        i[lane] = Scalar[T](0)
            re_tiles.push_back(r)
            im_tiles.push_back(i)

        return Self(re_tiles, im_tiles, k_valid)

    fn to_components(self) -> (List[List[Scalar[T]]], List[List[Scalar[T]]]):
        # Unpack into N component vectors (drop padded tails)
        let W = self.lanes_per_component()
        let N = self.num_components()
        var out_re = List[List[Scalar[T]]]()
        var out_im = List[List[Scalar[T]]]()
        out_re.reserve(N)
        out_im.reserve(N)
        for c in range(N):
            var re_c = List[Scalar[T]]()
            var im_c = List[Scalar[T]]()
            re_c.reserve(self.k_valid_per_comp[c])
            im_c.reserve(self.k_valid_per_comp[c])
            let kc = self.k_valid_per_comp[c]
            let tiles_c = (kc + W - 1) / W
            for t in range(tiles_c):
                let r = self.re_tiles[t]
                let i = self.im_tiles[t]
                for off in range(W):
                    let idx = t * W + off
                    if idx < kc:
                        let lane = c * W + off
                        re_c.push_back(r[lane])
                        im_c.push_back(i[lane])
            out_re.push_back(re_c)
            out_im.push_back(im_c)
        return (out_re, out_im)
