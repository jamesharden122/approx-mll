from stochcharfunc.svj1 import Params, SvSpec
from .directions import DirMethod, DirectionMat
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
        spec: S,
        dir_meth: DirMethod
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
        self.p_mat = DirectionMat[D](dir_meth).p_mat


struct FD1[S: SvSpec, params: P, D: Int]:
    var config: OptimFiniteDiffConfig
    var g: SIMD[DType.float64,D]
    fn __init__(out self, config: OptimFiniteDiffConfig, g_init):
        self.config = config
        self.g = 0
    fn gradient(self, theta):
        #function old parameters value
        var theta_old_ll = self.spec(theta)
        self.config.inverter.inverse_at[S](x,u0,du,cb,prior_params
        for i in range(D):
                                    #function new parameters value 
            #fuction differnce divided by stepsize multiplied by direction
