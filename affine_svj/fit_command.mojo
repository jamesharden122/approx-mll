from affine_svj.fit_cli import AffineSvjFitCliConfig
from affine_svj.params import (
    AffineSvjParams,
    read_affine_svj_params,
    write_affine_svj_params,
)
from affine_svj.returns_io import read_big_endian_f64_returns
from complexsimd.fourier import UniformGridInverterSIMDGrid
from optimizers.finitediff import AffineSvjOptimConfig, AffineSvjOptimizer
from std.utils.numerics import isfinite


def stationary_gamma_prior(
    params: AffineSvjParams,
) raises -> InlineArray[Float64, 2]:
    """Return the stationary CIR Gamma prior as [scale kappa, shape nu]."""
    params.validate()
    var variance_scale = params.volvol * params.volvol
    var kappa = variance_scale / (2.0 * params.beta)
    var nu = 2.0 * params.beta * params.alpha / variance_scale
    if (
        not isfinite(kappa)[0]
        or not isfinite(nu)[0]
        or kappa <= 0.0
        or nu <= 0.0
    ):
        raise Error("stationary Gamma prior is not finite and positive")
    return [kappa, nu]


def _print_fit_summary(
    likelihood: Float64,
    gradient_norm: Float64,
    iterations: Int,
    convergence_reason: String,
    final_kappa: Float64,
    final_nu: Float64,
):
    print("AffineSvjFitResult {")
    print("  likelihood:", likelihood)
    print("  gradient_norm:", gradient_norm)
    print("  iterations:", iterations)
    print("  convergence_reason:", convergence_reason)
    print("  final_kappa:", final_kappa)
    print("  final_nu:", final_nu)
    print("}")


def run_affine_svj_fit[L: Int](config: AffineSvjFitCliConfig) raises:
    """Run a fit and write an artifact only after validating the full result."""
    var returns = read_big_endian_f64_returns(config.returns_path, config.count)
    var initial = read_affine_svj_params(config.initial_path)
    var prior = stationary_gamma_prior(initial)
    var inverter = UniformGridInverterSIMDGrid[L](
        config.grid_u0, config.grid_du, config.grid_count
    )
    var optim_config = AffineSvjOptimConfig[L](
        inverter,
        prior,
        config.fd_step,
        config.armijo_c,
        config.armijo_shrink,
        config.initial_step,
        config.gradient_tolerance,
        config.likelihood_tolerance,
        config.max_iterations,
        config.min_step,
    )
    var optimizer = AffineSvjOptimizer[L](optim_config)
    var result = optimizer.fit(returns, initial)

    _print_fit_summary(
        result.likelihood,
        result.gradient_norm,
        result.iterations,
        result.convergence_reason,
        result.final_kappa,
        result.final_nu,
    )

    result.params.validate()
    if result.params.dt != initial.dt:
        raise Error("fit changed fixed dt sampling metadata")
    if not isfinite(result.likelihood)[0]:
        raise Error("fit produced a non-finite likelihood")
    if not isfinite(result.gradient_norm)[0] or result.gradient_norm < 0.0:
        raise Error("fit produced an invalid gradient norm")
    if (
        not isfinite(result.final_kappa)[0]
        or not isfinite(result.final_nu)[0]
        or result.final_kappa <= 0.0
        or result.final_nu <= 0.0
    ):
        raise Error("fit produced invalid final Gamma parameters")
    if result.iterations < 0:
        raise Error("fit produced an invalid iteration count")
    if result.convergence_reason == "invalid_initial_likelihood":
        raise Error("fit failed because the initial likelihood was invalid")
    if result.convergence_reason == "line_search_failed":
        raise Error("fit failed because Armijo line search found no step")
    if (
        result.convergence_reason != "gradient_tolerance"
        and result.convergence_reason != "likelihood_tolerance"
        and result.convergence_reason != "max_iterations"
    ):
        raise Error(
            String(
                "fit returned an unsupported convergence reason: ",
                result.convergence_reason,
            )
        )

    write_affine_svj_params(config.out_path, result.params)
    print("wrote fitted parameters:", config.out_path)
