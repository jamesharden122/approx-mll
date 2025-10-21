from complexsimd import ComplexSIMD
from math import sqrt, exp, log, cos, sin

trait Params:
    fn get_params(self) -> Self
    fn set_params(mut self, params: Self)



struct Svj1Params(Params):
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
        print(
            "mu0:", self.mu0,
            "mu1:", self.mu1,
            "rho:", self.rho,
            "alpha:", self.alpha,
            "beta:", self.beta,
            "volvol:", self.volvol,
            "lambda0:", self.lambda0,
            "lambda1:", self.lambda1,
            "k:", self.k,
            "dt:", self.dt
        )
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
 
trait SvSpec[T: Params]:
    fn __init__(out self, params: T)

struct Svj1JointCF(SvSpec[Svj1Params]):
    var params: Svj1Params

    fn __init__(out self, params: Svj1Params):
        self.params = ^params

    fn _riccati_sv1_simd[L: Int](self, Phi: SIMD[DType.float64, L], psi0: Float64) -> (SIMD[DType.float64, L], SIMD[DType.float64, L]):
        let dt = self.params.dt
        let sig = self.params.volvol
        let a: Float64 = 0.5 * sig * sig
        let b = SIMD[DType.float64, L](self.params.rho * sig) * Phi - SIMD[DType.float64, L](self.params.beta)
        let c = SIMD[DType.float64, L](0.5) * (Phi * Phi) + SIMD[DType.float64, L](self.params.mu1 - 0.5) * Phi
        let disc = b * b - SIMD[DType.float64, L](4.0 * a) * c
        let g = sqrt(disc)

        let two_a = 2.0 * a
        var y1 = (-b + g) / SIMD[DType.float64, L](two_a if two_a != 0.0 else 1.0)
        var y2 = (-b - g) / SIMD[DType.float64, L](two_a if two_a != 0.0 else 1.0)
        if two_a == 0.0:
            y1 = -c / (b + SIMD[DType.float64, L](1e-15))
            y2 = SIMD[DType.float64, L](0.0)

        let denom0 = y1 - SIMD[DType.float64, L](psi0)
        let z0 = (SIMD[DType.float64, L](psi0) - y2) / (denom0 + SIMD[DType.float64, L](1e-15))
        let zdt = z0 * exp(-g * SIMD[DType.float64, L](dt))

        let D = (y2 + y1 * zdt) / (SIMD[DType.float64, L](1.0) + zdt)
        let log1p_z0  = log(SIMD[DType.float64, L](1.0) + z0)
        let log1p_zdt = log(SIMD[DType.float64, L](1.0) + zdt)
        let g_safe = g + SIMD[DType.float64, L](1e-15)
        let integral_D =
            SIMD[DType.float64, L](self.params.alpha) * (
                y2 * SIMD[DType.float64, L](dt)
                - ((y1 - y2) / g_safe) * (log1p_zdt - log1p_z0)
            )
        let C = SIMD[DType.float64, L](self.params.mu0 * dt) * Phi + integral_D
        return (C, D)

    fn _gamma_mgf_complex_simd[L: Int](self, D: SIMD[DType.float64, L], kappa: Float64, nu: Float64) -> SIMD[DType.float64, L]:
        let x = SIMD[DType.float64, L](1.0) - SIMD[DType.float64, L](kappa) * D
        return exp(SIMD[DType.float64, L](-nu) * log(x))

    fn predictive_cf_latent_simd[L: Int](
        self,
        phi: SIMD[DType.float64, L],
        y: Float64,
        kappa: Float64,
        nu: Float64
    ) -> ComplexSIMD[DType.float64, L]:
        let (C, D) = self._riccati_sv1_simd[L](phi, 0.0)
        let eC = exp(C)
        let G  = self._gamma_mgf_complex_simd[L](D, kappa, nu)
        let c = cos(phi * SIMD[DType.float64, L](y))
        let s = sin(phi * SIMD[DType.float64, L](y))
        let mag = eC * G
        return ComplexSIMD[DType.float64, L](mag * c, mag * s)

    # SIMD real/imag parts separately (no lane loops)
    fn predictive_cf_latent_simd_re[L: Int](
        self,
        phi: SIMD[DType.float64, L],
        prior_params: Slice[Float64]# [y: Float64, kappa: Float64, nu: Float64]
    ) -> SIMD[DType.float64, L]:
        let (C, D) = self._riccati_sv1_simd[L](phi, 0.0)
        let mag = exp(C) * self._gamma_mgf_complex_simd[L](D, prior_params[1], prior_params[2])
        let c = cos(phi * SIMD[DType.float64, L](prior_params[0]))
        return mag * c

    fn predictive_cf_latent_simd_im[L: Int](
        self,
        phi: SIMD[DType.float64, L],
        prior_params: Slice[Float64]# [y: Float64, kappa: Float64, nu: Float64]
    ) -> SIMD[DType.float64, L]:
        let (C, D) = self._riccati_sv1_simd[L](phi, 0.0)
        let mag = exp(C) * self._gamma_mgf_complex_simd[L](D, prior_params[1], prior_params[2])
        let s = sin(phi * SIMD[DType.float64, L](prior_params[0]))
        return mag * s


