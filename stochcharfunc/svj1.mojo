from complexsimd import ComplexSIMD
from math import sqrt, exp, log, cos, sin

# Marker trait for parameter structs (kept per request).
trait Params(Copyable, Movable):
    pass


struct Svj1Params(Params, Copyable, Movable):
    var mu0: Float64
    var mu1: Float64
    var lambda0: Float64
    var lambda1: Float64
    var k: Float64
    var rho: Float64
    var alpha: Float64
    var beta: Float64
    var volvol: Float64
    var dt: Float64

    fn __init__(
        out self,
        mu0: Float64 = 0.05,        # drift of returns (≈5% annual)
        mu1: Float64 = -0.02,       # jump mean (negative for downward jumps)
        rho: Float64 = -0.6,        # correlation between returns and volatility (leverage effect)
        alpha: Float64 = 0.04,      # long-run variance (4%)
        beta: Float64 = 1.5,        # mean reversion speed
        volvol: Float64 = 0.3,      # vol of vol (moderate stochastic volatility)
        lambda0: Float64 = 0.1,     # base jump intensity (jumps per unit time)
        lambda1: Float64 = 0.05,    # jump sensitivity to volatility
        k: Float64 = 1.0,           # scaling / risk-neutral drift adjustment
        dt: Float64 = 1.0 / 252.0   # one trading day in years
    ):
        self.mu0 = mu0
        self.mu1 = mu1
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.volvol = volvol
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.k = k
        self.dt = dt

    fn get_params(self) -> Self:
        return self

    fn set_params(mut self, params: Self):
        self.mu0 = params.mu0
        self.mu1 = params.mu1
        self.rho = params.rho
        self.alpha = params.alpha
        self.beta = params.beta
        self.volvol = params.volvol
        self.lambda0 = params.lambda0
        self.lambda1 = params.lambda1
        self.k = params.k
        self.dt = params.dt
 
trait SvSpec(Copyable, Movable):
    fn predictive_cf_latent_simd_re[L: Int](
        self,
        phi: SIMD[DType.float64, L],
        prior_params: List[Float64]
    ) -> SIMD[DType.float64, L]:
        pass
    fn predictive_cf_latent_simd_im[L: Int](
        self,
        phi: SIMD[DType.float64, L],
        prior_params: List[Float64]
    ) -> SIMD[DType.float64, L]:
        pass

struct Svj1JointCF(SvSpec,Copyable, Movable):
    var params: Svj1Params

    fn __init__(out self, params: Svj1Params):
        self.params = params.copy()

    fn _riccati_sv1_simd[L: Int](self, Phi: SIMD[DType.float64, L], psi0: Float64) -> (SIMD[DType.float64, L], SIMD[DType.float64, L]):
        var dt = self.params.dt
        var sig = self.params.volvol
        var a: Float64 = 0.5 * sig * sig
        var b = SIMD[DType.float64, L](self.params.rho * sig) * Phi - SIMD[DType.float64, L](self.params.beta)
        var c = SIMD[DType.float64, L](0.5) * (Phi * Phi) + SIMD[DType.float64, L](self.params.mu1 - 0.5) * Phi
        var disc = b * b - SIMD[DType.float64, L](4.0 * a) * c
        var g = sqrt(disc)

        var two_a = 2.0 * a
        var y1 = (-b + g) / SIMD[DType.float64, L](two_a if two_a != 0.0 else 1.0)
        var y2 = (-b - g) / SIMD[DType.float64, L](two_a if two_a != 0.0 else 1.0)
        if two_a == 0.0:
            y1 = -c / (b + SIMD[DType.float64, L](1e-15))
            y2 = SIMD[DType.float64, L](0.0)

        var denom0 = y1 - SIMD[DType.float64, L](psi0)
        var z0 = (SIMD[DType.float64, L](psi0) - y2) / (denom0 + SIMD[DType.float64, L](1e-15))
        var zdt = z0 * exp(-g * SIMD[DType.float64, L](dt))

        var D = (y2 + y1 * zdt) / (SIMD[DType.float64, L](1.0) + zdt)
        var log1p_z0  = log(SIMD[DType.float64, L](1.0) + z0)
        var log1p_zdt = log(SIMD[DType.float64, L](1.0) + zdt)
        var g_safe = g + SIMD[DType.float64, L](1e-15)
        var integral_D =
            SIMD[DType.float64, L](self.params.alpha) * (
                y2 * SIMD[DType.float64, L](dt)
                - ((y1 - y2) / g_safe) * (log1p_zdt - log1p_z0)
            )
        var C = SIMD[DType.float64, L](self.params.mu0 * dt) * Phi + integral_D
        return (C, D)

    fn _gamma_mgf_complex_simd[L: Int](self, D: SIMD[DType.float64, L], kappa: Float64, nu: Float64) -> SIMD[DType.float64, L]:
        var x = SIMD[DType.float64, L](1.0) - SIMD[DType.float64, L](kappa) * D
        return exp(SIMD[DType.float64, L](-nu) * log(x))

    fn predictive_cf_latent_simd[L: Int](
        self,
        phi: SIMD[DType.float64, L],
        y: Float64,
        kappa: Float64,
        nu: Float64
    ) -> ComplexSIMD[DType.float64, L]:
        var (C, D) = self._riccati_sv1_simd[L](phi, 0.0)
        var eC = exp(C)
        var G  = self._gamma_mgf_complex_simd[L](D, kappa, nu)
        var c = cos(phi * SIMD[DType.float64, L](y))
        var s = sin(phi * SIMD[DType.float64, L](y))
        var mag = eC * G
        return ComplexSIMD[DType.float64, L](mag * c, mag * s)

    # SIMD real/imag parts separately (no lane loops)
    fn predictive_cf_latent_simd_re[L: Int](
        self,
        phi: SIMD[DType.float64, L],
        prior_params: List[Float64]
    ) -> SIMD[DType.float64, L]:
        var (C, D) = self._riccati_sv1_simd[L](phi, 0.0)
        var y = prior_params[0]
        var kappa = prior_params[1]
        var nu = prior_params[2]
        var mag = exp(C) * self._gamma_mgf_complex_simd[L](D, kappa, nu)
        var c = cos(phi * SIMD[DType.float64, L](y))
        return mag * c

    fn predictive_cf_latent_simd_im[L: Int](
        self,
        phi: SIMD[DType.float64, L],
        prior_params: List[Float64]
    ) -> SIMD[DType.float64, L]:
        var (C, D) = self._riccati_sv1_simd[L](phi, 0.0)
        var y = prior_params[0]
        var kappa = prior_params[1]
        var nu = prior_params[2]
        var mag = exp(C) * self._gamma_mgf_complex_simd[L](D, kappa, nu)
        var s = sin(phi * SIMD[DType.float64, L](y))
        return mag * s
