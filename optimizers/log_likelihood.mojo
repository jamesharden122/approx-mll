from stochcharfunc.svj1 import Params, SvSpec
from complexsimd.fourier import UniformGridInverterSIMDGrid
from math import log

# Bates AML step: sum log densities via inverse Fourier and update Gamma prior.
#
# - y_vec: SIMD batch of observations (e.g., returns) [y_{t+1}^{(i)}]
# - inverter: uniform-phi grid inverter (holds CF tiles)
# - spec: model implementing SvSpec (e.g., Svj1JointCF)
# - prior_params: InlineArray[3] = [y_t, kappa_t, nu_t]
# - u0, du: φ-grid start and step (used by the prior update routine)
# - normalize: pass-through to inverter/update to include 1/(2π)
fn log_likelihood[S: SvSpec, L: Int](
    y_vec: SIMD[DType.float64, L],
    inverter: UniformGridInverterSIMDGrid[L],
    spec: S,
    prior_params: InlineArray[Float64, 3],
    u0: Float64,
    du: Float64,
    normalize: Bool = True,
) -> SIMD[DType.float64, 1]:
    var ll_sum: Float64 = 0.0
    var prior = prior_params  # [y_t, kappa_t, nu_t]
    var psi0 = SIMD[DType.float64, 1](0.0)
    var eps: Float64 = 1e-300

    for i in range(L):
        var x = Float64(y_vec[i])
        # Step 2 (Bates): density via inverse Fourier at x = y_{t+1}
        var z = inverter.inverse_at[S, 3](x, spec, psi0, prior, normalize)
        var f = Float64(z.re[0]) * 2.0 * du
        print(f)
        if f <= 0.0:
            f = eps
        ll_sum += log(f)

        # Update Gamma prior parameters using integrated ψ-derivatives
        var (kappa_new, nu_new, _, _) = spec.update_gamma_prior_from_inverter[L](
            inverter, prior, u0, du, 1e-6, normalize
        )
        # Roll prior forward: Y_t <- y_{t+1}, (kappa, nu) <- updated
        prior[0] = x
        prior[1] = kappa_new
        prior[2] = nu_new

    return SIMD[DType.float64, 1](ll_sum)
