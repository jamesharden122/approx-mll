from stochcharfunc.svj1 import Params, SvSpec
from complexsimd import CFGridSIMD, UniformGridInverterSIMDGrid

struct OptimFiniteDiffConfig[P: Params, S: SvSpec, I: Int]:
    var theta: P
    var h: Float64
    var armijo_c: Float64
    var eta_init: Float64
    var eta_shrink: Float64
    var min_thresh: Float64
    var inverter: UniformGridInverterSIMDGrid[I]
    var spec: S

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
