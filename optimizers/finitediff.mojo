from stochcharfunc.svj1 import Params, SvSpec
from complexsimd import CFGridSIMD, UniformGridInverterSIMDGrid
from layout import Layout, LayoutTensor
from math import sqrt

alias F = DType.float32

struct OptimFiniteDiffConfig[P: Params, S: SvSpec, I: Int, D: Int]:
    var theta: P
    var h: Float64
    var armijo_c: Float64
    var eta_init: Float64
    var eta_shrink: Float64
    var min_thresh: Float64
    var inverter: UniformGridInverterSIMDGrid[I]
    var spec: S
    # Working matrix for parameter updates (D x D)
    var p_mat: LayoutTensor[DType.float64, Layout.row_major(D, D), MutableAnyOrigin]
    fn __init__(
        out self,
        theta: P,
        h: Float64,
        armijo_c: Float64,
        eta_init: Float64,
        eta_shrink: Float64,
        min_thresh: Float64,
        grid: CFGridSIMD[DType.float64, I],
        spec: S
    ):
        self.theta = theta.copy()
        self.h = h
        self.armijo_c = armijo_c
        self.eta_init = eta_init
        self.eta_shrink = eta_shrink
        self.min_thresh = min_thresh
        self.inverter = UniformGridInverterSIMDGrid[I](grid)
        self.spec = spec.copy()
        # Allocate storage and materialize the tensor with a row-major layout
        var storage = InlineArray[Float64, D * D](uninitialized=True)
        self.p_mat = LayoutTensor[DType.float64, Layout.row_major(D, D), MutableAnyOrigin](storage)
