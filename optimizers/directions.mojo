from layout import Layout, LayoutTensor
from random import randn
from memory.unsafe_pointer import UnsafePointer
struct DirMethod:
    var raw: Int32

    fn __init__(out self, raw: Int32):
        self.raw = raw

    @staticmethod
    fn Gaussian() -> DirMethod:
        return DirMethod(Int32(0))

    @staticmethod
    fn Identity() -> DirMethod:
        return DirMethod(Int32(1))
    

struct DirectionMat[D: Int](Copyable, Movable):
    alias array = InlineArray[Float64, D * D]
    alias matrix =  LayoutTensor[DType.float64, Layout.row_major(D, D), MutableAnyOrigin] 
    var p_mat: Self.matrix

    fn __init__(out self, method: DirMethod):
        if method.raw == 0:
            print("method: Gaussian")
            var storage = self._gaussian()
            self.p_mat = Self.matrix(storage)
        elif method.raw == 1:
            print("method: Identity")
            var storage = self._identity()
            self.p_mat = Self.matrix(storage)
        else: 
            print("method: Other")
            var storage = Self.array(uninitialized=True)
            self.p_mat = Self.matrix(storage)
        print(self.p_mat)

        
    @staticmethod
    fn _gaussian(mu: Float64 = 0.0, sigma: Float64 = 1.0) -> Self.array:
        var storage = Self.array(uninitialized=True)
        ptr = storage.unsafe_ptr()
        randn[DType.float64](ptr,D*D,mu,sigma)
        return storage
    
    @staticmethod
    fn _identity() -> Self.array:
        var storage = Self.array(0)
        for i in range(D): storage[i * D + i] = 1.0
        return storage
        

