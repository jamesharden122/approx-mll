from complexsimd import ComplexSIMD
from math import sqrt, exp, log, cos, sin
from collections import Set
from complexsimd import CFGridSIMD, UniformGridInverterSIMDGrid

# Marker trait for parameter structs (kept per request).
trait Params(Copyable, Movable):
    # Keep method-level generic to satisfy callers that provide InlineArray[L].
    fn __init__[L: Int](out self, vals: InlineArray[Float64, L]):
        pass
    fn get_params[L: Int](self) -> InlineArray[Float64, L]:
        pass
    fn set_params_simd[L: Int](mut self, vals: SIMD[DType.float64, L]):
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

    #mu0: Float64 = 0.05,        # drift of returns (≈5% annual)
    #mu1: Float64 = -0.02,       # jump mean (negative for downward jumps)
    #rho: Float64 = -0.6,        # correlation between returns and volatility (leverage effect)
    #alpha: Float64 = 0.04,      # long-run variance (4%)
    #beta: Float64 = 1.5,        # mean reversion speed
    #volvol: Float64 = 0.3,      # vol of vol (moderate stochastic volatility)
    #lambda0: Float64 = 0.1,     # base jump intensity (jumps per unit time)
    #lambda1: Float64 = 0.05,    # jump sensitivity to volatility
    #k: Float64 = 1.0,           # scaling / risk-neutral drift adjustment
    #dt: Float64 = 1.0 / 252.0   # one trading day in years
    @always_inline
    fn __init__[L: Int](
            out self, 
            vals: InlineArray[Float64, L]
            ):
        # One pass, no branches, no string work.
        self.mu0    = vals[0]
        self.mu1    = vals[1]
        self.lambda0= vals[2]
        self.lambda1= vals[3]
        self.k      = vals[4]
        self.rho    = vals[5]
        self.alpha  = vals[6]
        self.beta   = vals[7]
        self.volvol = vals[8]
        self.dt     = vals[9]

    # Convenience constructor for the fixed 10-parameter case.
    # Keeps the same index mapping as the generic __init__ above.
    @always_inline
    fn __init__(out self, vals: InlineArray[Float64, 10]):
        self.mu0    = vals[0]
        self.mu1    = vals[1]
        self.lambda0= vals[2]
        self.lambda1= vals[3]
        self.k      = vals[4]
        self.rho    = vals[5]
        self.alpha  = vals[6]
        self.beta   = vals[7]
        self.volvol = vals[8]
        self.dt     = vals[9]

    fn get_params(self) -> InlineArray[Float64, 10]:
        var vals = InlineArray[Float64, 10](uninitialized=True)
        vals[0] = self.mu0 
        vals[1] = self.mu1
        vals[2] = self.lambda0
        vals[3] = self.lambda1
        vals[4] = self.k
        vals[5] = self.rho
        vals[6] = self.alpha
        vals[7] = self.beta
        vals[8] = self.volvol
        vals[9] = self.dt
        return vals

    # Generic adapter: callers should use L >= 10 for this model
    fn get_params[L: Int](self) -> InlineArray[Float64, L]:
        var out = InlineArray[Float64, L](uninitialized=True)
        var base = self.get_params()
        out[0] = base[0]
        out[1] = base[1]
        out[2] = base[2]
        out[3] = base[3]
        out[4] = base[4]
        out[5] = base[5]
        out[6] = base[6]
        out[7] = base[7]
        out[8] = base[8]
        out[9] = base[9]
        return out

    fn set_params_simd[L: Int](mut self, vals: SIMD[DType.float64, L]):
        # Mirror set_params() field order using SIMD lane values (fixed width)
        self.mu0     = Float64(vals[0])
        self.mu1     = Float64(vals[1])
        self.rho     = Float64(vals[2])
        self.alpha   = Float64(vals[3])
        self.beta    = Float64(vals[4])
        self.volvol  = Float64(vals[5])
        self.lambda0 = Float64(vals[6])
        self.lambda1 = Float64(vals[7])
        self.k       = Float64(vals[8])
        self.dt      = Float64(vals[9])

    fn set_params(mut self, vals: InlineArray[Float64, 10]):
        self.mu0 = vals[0]#params.mu0
        self.mu1 = vals[1]#params.mu1
        self.rho = vals[2]#params.rho
        self.alpha = vals[3]#params.alpha
        self.beta = vals[4]#params.beta
        self.volvol = vals[5] #params.volvol
        self.lambda0 = vals[6] #params.lambda0
        self.lambda1 = vals[7] #params.lambda1
        self.k = vals[8] #params.k
        self.dt = vals[9] #params.dt
 
trait SvSpec(Copyable, Movable):
    fn predictive_cf_latent_simd_re[L: Int, T: Int](
        self,
        phi: SIMD[DType.float64, L],
        psi: SIMD[DType.float64, 1],
        prior_params: InlineArray[Float64, T]
    ) -> SIMD[DType.float64, L]:
        pass
    fn predictive_cf_latent_simd_im[L: Int, T: Int](
        self,
        phi: SIMD[DType.float64, L],
        psi: SIMD[DType.float64, 1],
        prior_params: InlineArray[Float64, T]
    ) -> SIMD[DType.float64, L]:
        pass

    fn update_gamma_prior_from_inverter[L: Int](
        self,
        inverter: UniformGridInverterSIMDGrid[L],
        prior_params: InlineArray[Float64, 3],#y,nu,kappa
        u0: Float64,                         # grid start in φ-space
        du: Float64,                         # grid step in φ-space
        h:  Float64 = 1e-6,                  # ψ step (central diff)
        normalize: Bool = True               # include 1/(2π)
    ) -> (Float64, Float64, Float64, Float64):
        pass
    fn set_params_simd[L: Int](mut self, vals: SIMD[DType.float64, L]):
        pass
    # New: allow setting parameters from InlineArray to avoid non-power-of-two SIMD widths
    fn set_params_array[L: Int](mut self, vals: InlineArray[Float64, L]):
        pass

struct Svj1JointCF(SvSpec,Copyable, Movable):
    var params: Svj1Params

    fn __init__(out self, params: Svj1Params):
        self.params = params.copy()

    fn set_params_simd[L: Int](mut self, vals: SIMD[DType.float64, L]):
        self.params.set_params_simd[L](vals)

    fn set_params_array[L: Int](mut self, vals: InlineArray[Float64, L]):
        # Construct params directly from InlineArray (expects L >= 10)
        self.params = Svj1Params.__init__[L](vals)

    fn _riccati_sv1_simd[L: Int](self, Phi: SIMD[DType.float64, L], psi0: SIMD[DType.float64, 1]) -> (SIMD[DType.float64, L], SIMD[DType.float64, L]):
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
        psi: SIMD[DType.float64, 1],
        y: Float64,
        kappa: Float64,
        nu: Float64
    ) -> ComplexSIMD[DType.float64, L]:
        var (C, D) = self._riccati_sv1_simd[L](phi, psi)
        var eC = exp(C)
        var G  = self._gamma_mgf_complex_simd[L](D, kappa, nu)
        var c = cos(phi * SIMD[DType.float64, L](y))
        var s = sin(phi * SIMD[DType.float64, L](y))
        var mag = eC * G
        return ComplexSIMD[DType.float64, L](mag * c, mag * s)

    # SIMD real/imag parts separately (no lane loops)
    fn predictive_cf_latent_simd_re[L: Int,T: Int](
        self,
        phi: SIMD[DType.float64, L],
        psi: SIMD[DType.float64, 1],
        prior_params:InlineArray[Float64, T]
    ) -> SIMD[DType.float64, L]:
        var (C, D) = self._riccati_sv1_simd[L](phi, psi)
        var y = prior_params[0]
        var kappa = prior_params[1]
        var nu = prior_params[2]
        var mag = exp(C) * self._gamma_mgf_complex_simd[L](D, kappa, nu)
        var c = cos(phi * SIMD[DType.float64, L](y))
        return mag * c

    fn predictive_cf_latent_simd_im[L: Int,T: Int](
        self,
        phi: SIMD[DType.float64, L],
        psi: SIMD[DType.float64, 1],
        prior_params: InlineArray[Float64, T]
    ) -> SIMD[DType.float64, L]:
        var (C, D) = self._riccati_sv1_simd[L](phi, psi)
        var y = prior_params[0]
        var kappa = prior_params[1]
        var nu = prior_params[2]
        var mag = exp(C) * self._gamma_mgf_complex_simd[L](D, kappa, nu)
        var s = sin(phi * SIMD[DType.float64, L](y))
        return mag * s

    # ---- NEW: update Gamma prior from CF on a single SIMD φ-tile ----
    fn update_gamma_prior_from_inverter[L: Int](
        self,
        inverter: UniformGridInverterSIMDGrid[L],
        prior_params: InlineArray[Float64, 3],#y,nu,kappa
        u0: Float64,                         # grid start in φ-space
        du: Float64,                         # grid step in φ-space
        h:  Float64 = 1e-6,                  # ψ step (central diff)
        normalize: Bool = True               # include 1/(2π)
    ) -> (Float64, Float64, Float64, Float64):
        """
        Returns: (kappa_new, nu_new, x_hat, P)
        with x_hat = E[x_{t+1}|Y_{t+1}], P = Var[x_{t+1}|Y_{t+1}].

        Uses inverse Fourier to compute:
          P0 = Re ∫ F(φ, 0 | Y) e^{-i φ y} dφ
          S1 = ∂/∂ψ Re ∫ F(φ, ψ | Y) e^{-i φ y} dφ |_{ψ=0}
          S2 = ∂²/∂ψ² Re ∫ F(φ, ψ | Y) e^{-i φ y} dφ |_{ψ=0}
        then moment-matches a Gamma(ν, κ).
        """
        # Compute Re integrals at ψ = {-h, 0, +h} in ONE pass
        var (P_mh, P0, P_ph) = inverter.inverse_three_psi[Self,3](u0, du, self.copy(), prior_params, h, normalize)

        # Central differences in ψ of the (already integrated) real part
        var inv_2h = 1.0 / (2.0 * h)
        var inv_h2 = 1.0 / (h * h)
        var S1 = (P_ph - P_mh) * inv_2h
        var S2 = (P_ph - 2.0 * P0 + P_mh) * inv_h2

        # Ratios: if normalize==True, common 1/(2π) cancels; if False, it still cancels.
        var eps   = 1e-14
        var denom = P0 + eps

        var x_hat = S1 / denom
        var EX2   = S2 / denom
        var P     = EX2 - x_hat * x_hat
        if P < 0.0:
            P = 0.0   # guard tiny negatives due to roundoff

        # Moment-match Gamma(ν, κ):  E=νκ, Var=νκ²  => κ = P/x̂, ν = x̂²/P
        var kappa_new: Float64
        if x_hat != 0.0:
            kappa_new = P / x_hat
        else:
            kappa_new = 0.0

        var nu_new: Float64
        if P > eps:
            nu_new = (x_hat * x_hat) / P
        else:
            # fallback to prior ν (third element)
            nu_new = prior_params[2]

        return (kappa_new, nu_new, x_hat, P)
