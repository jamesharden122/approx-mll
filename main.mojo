from stochcharfunc import Svj1JointCF
from stochcharfunc.svj1 import Svj1Params
from complexsimd import (
    ComplexSIMD,
    CFGridSIMD,
    UniformGridInverterSIMDGrid,
)
from optimizers import OptimFiniteDiffConfig, DirMethod, DirectionMat
from layout import Layout, LayoutTensor


fn simd_step[W: Int](d: SIMD[DType.float64, W], h: Float64, t: SIMD[DType.float64, W]) -> SIMD[DType.float64, W]:
    return t + d.__mul__(SIMD[DType.float64, W](h))

fn main():
    var pmat_storage   = InlineArray[Float64, 8 * 8](1.0)
    var g_storage      = InlineArray[Float64, 8 * 1](0.0)   # match 8x1
    var theta_storage  = InlineArray[Float64, 8 * 1](1.0)

    var pmat  = LayoutTensor[DType.float64, Layout.row_major(8, 8)](pmat_storage)
    var g     = LayoutTensor[DType.float64, Layout.row_major(8, 1)](g_storage)
    var theta = LayoutTensor[DType.float64, Layout.row_major(8, 1)](theta_storage)
    pmat.slice[Slice(0,8),Slice(0,1)]().__imul__(1.5)
    var h: Float64 = 0.4
    pmat *= 3.0
    print("Directions Matrix:\n", pmat)
    print("Theta Matrix:\n", theta)
    print("Gradient Matrix:\n", g)
    var pmat_vec = pmat.vectorize[8,1]()
    var d0 = pmat_vec.load[8](0, 0)
    print("Directions SIMD first row:\n", d0)
    var theta_vec = theta.vectorize[8,1]()
    var th0 = theta_vec.load[8](0, 0)
    print("Theta SIMD:\n", th0)
    var g_vec = g.vectorize[8,1]()
    var t = simd_step[8](pmat_vec.load[8](0, 0), h, theta_vec.load[8](0, 0)) 
    print("Product tile:\n", t)
    for j in range(1,8):
        t = simd_step[8](pmat_vec.load[8](j, 0), h, theta_vec.load[8](0, 0))
        var acc = g_vec.load[8](0, 0)
        acc = acc + t
        g_vec.store[8](0, 0, acc)
        print("Accumulated g SIMD:\n", acc)
    
