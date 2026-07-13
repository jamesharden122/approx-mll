from complexsimd.fourier import UniformGridInverterSIMDGrid
from stochcharfunc.svj1 import AffineSvjJointCF
from std.collections import List
from std.math import log
from std.utils.numerics import isfinite


struct AffineSvjFilterResult(ImplicitlyCopyable):
    var likelihood: Float64
    var final_kappa: Float64
    var final_nu: Float64
    var valid: Bool

    def __init__(
        out self,
        likelihood: Float64,
        final_kappa: Float64,
        final_nu: Float64,
        valid: Bool,
    ):
        self.likelihood = likelihood
        self.final_kappa = final_kappa
        self.final_nu = final_nu
        self.valid = valid


def filter_log_returns[L: Int](
    returns: List[Float64],
    inverter: UniformGridInverterSIMDGrid[L],
    model: AffineSvjJointCF,
    initial_prior: InlineArray[Float64, 2],
) -> AffineSvjFilterResult:
    """Filter log-return increments using a Gamma variance prior [kappa, nu]."""

    var prior = initial_prior
    if (
        prior[0] <= 0.0
        or prior[1] <= 0.0
        or not isfinite(prior[0])[0]
        or not isfinite(prior[1])[0]
    ):
        return AffineSvjFilterResult(-1e300, prior[0], prior[1], False)

    var likelihood = 0.0
    for index in range(len(returns)):
        var increment = returns[index]
        var density = inverter.density(increment, model, prior)
        if density <= 1e-300 or not isfinite(density)[0]:
            return AffineSvjFilterResult(
                -1e300, prior[0], prior[1], False
            )
        likelihood += log(density)

        var update = inverter.update_gamma_prior(increment, model, prior)
        if not update.valid:
            return AffineSvjFilterResult(
                -1e300, prior[0], prior[1], False
            )
        prior[0] = update.kappa
        prior[1] = update.nu

    if not isfinite(likelihood)[0]:
        return AffineSvjFilterResult(
            -1e300, prior[0], prior[1], False
        )
    return AffineSvjFilterResult(
        likelihood, prior[0], prior[1], True
    )


def log_likelihood[L: Int](
    returns: List[Float64],
    inverter: UniformGridInverterSIMDGrid[L],
    model: AffineSvjJointCF,
    initial_prior: InlineArray[Float64, 2],
) -> Float64:
    return filter_log_returns[L](
        returns, inverter, model, initial_prior
    ).likelihood
