from stochcharfunc.svj1 import Params, SvSpec
from .directions import DirMethod, DirectionMat
from complexsimd import CFGridSIMD, UniformGridInverterSIMDGrid
from layout import Layout, LayoutTensor
from math import sqrt

alias F = DType.float32

struct OptimFiniteDiffConfig[P: Params, S: SvSpec, I: Int, D: Int](Copyable, Movable):
    var theta: P
    var h: Float64
    var armijo_c: Float64
    var eta: Float64
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
        eta: Float64,
        min_thresh: Float64,
        grid: CFGridSIMD[DType.float64, I],
        spec: S,
        dir_meth: DirMethod
    ):
        self.theta = theta.copy()
        self.h = h
        self.armijo_c = armijo_c
        self.eta = eta
        self.min_thresh = min_thresh
        self.inverter = UniformGridInverterSIMDGrid[I](grid)
        self.spec = spec.copy()
        # Allocate storage and materialize the tensor with a row-major layout
        self.p_mat = DirectionMat[D](dir_meth).p_mat


struct FD1[P: Params, S: SvSpec & Copyable, I: Int, D: Int]:
    var config: OptimFiniteDiffConfig[P, S, I, D]
    var g: LayoutTensor[DType.float64, Layout.row_major(D, 1), MutableAnyOrigin]
    var theta:  LayoutTensor[DType.float64, Layout.row_major(D, 1), MutableAnyOrigin]

    fn __init__(
            out self, 
            config: OptimFiniteDiffConfig[P, S, I, D]
        ):
        self.config = config.copy()
        var g_init = InlineArray[Float64,D](0.0)
        self.g = LayoutTensor[DType.float64, Layout.row_major(D, 1), MutableAnyOrigin](g_init)
        # Initialize theta tensor from parameter object
        self.theta = LayoutTensor[DType.float64, Layout.row_major(D, 1), MutableAnyOrigin](self.config.theta.get_params[D]())
        

    fn gradient(
            mut self,
            prior_params: InlineArray[Float64, 3],
            y_vec: SIMD[DType.float64, I],
            ll: fn(
                SIMD[DType.float64, I],
                UniformGridInverterSIMDGrid[I],
                S,
                InlineArray[Float64, 3],
                Bool,
            ) -> SIMD[DType.float64, 1]
        ) -> LayoutTensor[DType.float64, Layout.row_major(D, 1), MutableAnyOrigin]:
        self.__clear_gradient()
        # Vectorize convenience views
        var pmat_vec = self.config.p_mat.vectorize[D, 1]()
        var theta_vec = self.theta.vectorize[D, 1]()
        var g_vec = self.g.vectorize[D, 1]()
        var theta0 = theta_vec.load[D](0, 0)
        var h = self.config.h

        # Baseline objective at theta0
        var ftheta0 = ll(
            y_vec,
            self.config.inverter,
            self.config.spec,
            prior_params,
            False,
        )

        var g_update = SIMD[DType.float64, D](0.0)
        var spec_hp: S = self.config.spec.copy()

        for i in range(D):
            var d_i = pmat_vec.load[D](i, 0)
            var theta_hp = Self.__simd_step[D](d_i, h, theta0)

            # Update params from theta_hp and rebuild spec
            spec_hp = self.config.spec.copy()
            spec_hp.set_params_simd[D](theta_hp)

            var ftheta_hp = ll(y_vec, self.config.inverter, spec_hp, prior_params, False)

            # Forward-diff directional derivative and accumulate
            var df = Float64(ftheta_hp[0] - ftheta0[0]) / h
            g_update = d_i * SIMD[DType.float64, D](df)
            var acc = g_vec.load[D](0, 0)
            acc = acc + g_update
            g_vec.store[D](0, 0, acc)

        # Gradient descent step on theta
        self.__update_theta()
        return self.g

    @staticmethod
    fn __simd_step[W: Int](d: SIMD[DType.float64, W], h: Float64, t: SIMD[DType.float64, W]) -> SIMD[DType.float64, W]:
        return t + d.__mul__(SIMD[DType.float64, W](h))

    fn __update_theta(mut self):
        var theta_vec = self.theta.vectorize[D, 1]()
        var g_vec = self.g.vectorize[D, 1]()
        var theta0 = theta_vec.load[D](0, 0)
        var gsimd = g_vec.load[D](0, 0)
        var eta = self.config.eta
        var new_theta = theta0 - gsimd * SIMD[DType.float64, D](eta)
        theta_vec.store[D](0, 0, new_theta)

        # Reflect the update back into the parameter object
        var p_upd = self.config.theta.copy()
        p_upd.set_params_simd[D](new_theta)
        self.config.theta = p_upd.copy()

    # Optional helper not used currently; kept for parity
    fn __update_gradient(
            mut self,
            ftheta: SIMD[DType.float64, 1],
            ftheta_hp: SIMD[DType.float64, 1],
            direc: SIMD[DType.float64, D]
        ):
        var g_vec = self.g.vectorize[D, 1]()
        var df = Float64(ftheta_hp[0] - ftheta[0]) / self.config.h
        var upd = direc * SIMD[DType.float64, D](df)
        var acc = g_vec.load[D](0, 0)
        acc = acc + upd
        g_vec.store[D](0, 0, acc)

    fn __clear_gradient(mut self):
        var g_init = InlineArray[Float64, D](0.0)
        self.g = LayoutTensor[DType.float64, Layout.row_major(D, 1), MutableAnyOrigin](g_init)

        

    



        #function old parameters value
