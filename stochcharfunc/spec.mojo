trait Params(Copyable, Movable, ImplicitlyDestructible):
    # Keep method-level generics for callers that provide InlineArray[L].
    def __init__[L: Int](out self, vals: InlineArray[Float64, L]):
        ...

    def get_params[L: Int](self) -> InlineArray[Float64, L]:
        ...

    def set_params_simd[L: Int](mut self, vals: SIMD[DType.float64, L]):
        ...

trait SvSpec(Copyable, Movable, ImplicitlyDestructible):
    def predictive_cf_latent_simd_re[L: Int, T: Int](
        self,
        phi: SIMD[DType.float64, L],
        psi: SIMD[DType.float64, 1],
        prior_params: InlineArray[Float64, T],
    ) -> SIMD[DType.float64, L]:
        ...
    def predictive_cf_latent_simd_im[L: Int, T: Int](
        self,
        phi: SIMD[DType.float64, L],
        psi: SIMD[DType.float64, 1],
        prior_params: InlineArray[Float64, T],
    ) -> SIMD[DType.float64, L]:
        ...
    def set_params_simd[L: Int](mut self, vals: SIMD[DType.float64, L]):
        ...

    def set_params_array[L: Int](mut self, vals: InlineArray[Float64, L]):
        ...