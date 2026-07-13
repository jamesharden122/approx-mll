from affine_svj import AffineSvjParams
from complexsimd.fourier import UniformGridInverterSIMDGrid
from optimizers.log_likelihood import (
    AffineSvjFilterResult,
    filter_log_returns,
)
from stochcharfunc.svj1 import AffineSvjJointCF
from std.collections import List
from std.math import exp, log, sqrt, tanh
from std.utils.numerics import isfinite


struct AffineSvjOptimConfig[L: Int](Copyable, Movable):
    var inverter: UniformGridInverterSIMDGrid[Self.L]
    var initial_prior: InlineArray[Float64, 2]
    var difference_step: Float64
    var armijo_c: Float64
    var shrink: Float64
    var initial_step: Float64
    var gradient_tolerance: Float64
    var likelihood_tolerance: Float64
    var max_iterations: Int
    var min_step: Float64

    def __init__(
        out self,
        inverter: UniformGridInverterSIMDGrid[Self.L],
        initial_prior: InlineArray[Float64, 2],
        difference_step: Float64 = 1e-4,
        armijo_c: Float64 = 1e-4,
        shrink: Float64 = 0.5,
        initial_step: Float64 = 0.1,
        gradient_tolerance: Float64 = 1e-5,
        likelihood_tolerance: Float64 = 1e-8,
        max_iterations: Int = 100,
        min_step: Float64 = 1e-8,
    ) raises:
        if (
            not isfinite(initial_prior[0])[0]
            or not isfinite(initial_prior[1])[0]
            or not isfinite(difference_step)[0]
            or not isfinite(armijo_c)[0]
            or not isfinite(shrink)[0]
            or not isfinite(initial_step)[0]
            or not isfinite(gradient_tolerance)[0]
            or not isfinite(likelihood_tolerance)[0]
            or not isfinite(min_step)[0]
        ):
            raise Error("optimizer configuration must be finite")
        if initial_prior[0] <= 0.0 or initial_prior[1] <= 0.0:
            raise Error("initial Gamma kappa and nu must be positive")
        if difference_step <= 0.0:
            raise Error("central-difference step must be positive")
        if armijo_c <= 0.0 or armijo_c >= 1.0:
            raise Error("Armijo c must lie in (0, 1)")
        if shrink <= 0.0 or shrink >= 1.0:
            raise Error("line-search shrink must lie in (0, 1)")
        if initial_step <= 0.0 or min_step <= 0.0:
            raise Error("line-search steps must be positive")
        if gradient_tolerance <= 0.0 or likelihood_tolerance <= 0.0:
            raise Error("convergence tolerances must be positive")
        if max_iterations < 1:
            raise Error("max_iterations must be positive")
        self.inverter = inverter.copy()
        self.initial_prior = initial_prior
        self.difference_step = difference_step
        self.armijo_c = armijo_c
        self.shrink = shrink
        self.initial_step = initial_step
        self.gradient_tolerance = gradient_tolerance
        self.likelihood_tolerance = likelihood_tolerance
        self.max_iterations = max_iterations
        self.min_step = min_step


struct AffineSvjFitResult(Movable):
    var params: AffineSvjParams
    var likelihood: Float64
    var gradient_norm: Float64
    var iterations: Int
    var convergence_reason: String
    var final_kappa: Float64
    var final_nu: Float64

    def __init__(
        out self,
        params: AffineSvjParams,
        likelihood: Float64,
        gradient_norm: Float64,
        iterations: Int,
        convergence_reason: String,
        final_kappa: Float64,
        final_nu: Float64,
    ):
        self.params = params
        self.likelihood = likelihood
        self.gradient_norm = gradient_norm
        self.iterations = iterations
        self.convergence_reason = convergence_reason
        self.final_kappa = final_kappa
        self.final_nu = final_nu


def _bounded_exp(value: Float64) -> Float64:
    var bounded = value
    if bounded < -40.0:
        bounded = -40.0
    if bounded > 40.0:
        bounded = 40.0
    return exp(bounded)


def _positive_log(value: Float64) -> Float64:
    var bounded = value
    if bounded < 1e-12:
        bounded = 1e-12
    return log(bounded)


def params_to_unconstrained(
    params: AffineSvjParams,
) -> InlineArray[Float64, 10]:
    var values = params.free_values()
    var result = InlineArray[Float64, 10](fill=0.0)
    result[0] = values[0]
    result[1] = values[1]
    result[2] = _positive_log(values[2])
    result[3] = _positive_log(values[3])
    result[4] = values[4]
    result[5] = _positive_log(values[5])
    var rho = values[6]
    if rho < -0.999999999:
        rho = -0.999999999
    if rho > 0.999999999:
        rho = 0.999999999
    result[6] = 0.5 * log((1.0 + rho) / (1.0 - rho))
    result[7] = _positive_log(values[7])
    result[8] = _positive_log(values[8])
    result[9] = _positive_log(values[9])
    return result


def unconstrained_to_params(
    values: InlineArray[Float64, 10], dt: Float64
) raises -> AffineSvjParams:
    var free = InlineArray[Float64, 10](fill=0.0)
    free[0] = values[0]
    free[1] = values[1]
    free[2] = _bounded_exp(values[2])
    free[3] = _bounded_exp(values[3])
    free[4] = values[4]
    free[5] = _bounded_exp(values[5])
    free[6] = tanh(values[6])
    free[7] = _bounded_exp(values[7])
    free[8] = _bounded_exp(values[8])
    free[9] = _bounded_exp(values[9])
    return AffineSvjParams.with_free_values(free, dt)


struct AffineSvjOptimizer[L: Int](Copyable, Movable):
    var config: AffineSvjOptimConfig[Self.L]

    def __init__(out self, config: AffineSvjOptimConfig[Self.L]):
        self.config = config.copy()

    def evaluate(
        self, returns: List[Float64], params: AffineSvjParams
    ) -> AffineSvjFilterResult:
        return filter_log_returns[Self.L](
            returns,
            self.config.inverter,
            AffineSvjJointCF(params),
            self.config.initial_prior,
        )

    def central_gradient(
        self, returns: List[Float64], params: AffineSvjParams
    ) raises -> InlineArray[Float64, 10]:
        var center = params_to_unconstrained(params)
        var gradient = InlineArray[Float64, 10](fill=0.0)
        for coordinate in range(10):
            var plus = InlineArray[Float64, 10](fill=0.0)
            var minus = InlineArray[Float64, 10](fill=0.0)
            for index in range(10):
                plus[index] = center[index]
                minus[index] = center[index]
            plus[coordinate] += self.config.difference_step
            minus[coordinate] -= self.config.difference_step
            var plus_params = unconstrained_to_params(plus, params.dt)
            var minus_params = unconstrained_to_params(minus, params.dt)
            var plus_result = self.evaluate(returns, plus_params)
            var minus_result = self.evaluate(returns, minus_params)
            if (
                not plus_result.valid
                or not minus_result.valid
                or not isfinite(plus_result.likelihood)[0]
                or not isfinite(minus_result.likelihood)[0]
                or plus_result.likelihood <= -1e299
                or minus_result.likelihood <= -1e299
            ):
                raise Error("invalid central-difference likelihood")
            gradient[coordinate] = (
                (plus_result.likelihood - minus_result.likelihood)
                / (2.0 * self.config.difference_step)
            )
            if not isfinite(gradient[coordinate])[0]:
                raise Error("non-finite central-difference gradient")
        return gradient

    def fit(
        self, returns: List[Float64], initial: AffineSvjParams
    ) raises -> AffineSvjFitResult:
        initial.validate()
        var coordinates = params_to_unconstrained(initial)
        var current_params = initial
        var current = self.evaluate(returns, current_params)
        if (
            not current.valid
            or not isfinite(current.likelihood)[0]
            or current.likelihood <= -1e299
        ):
            return AffineSvjFitResult(
                current_params,
                current.likelihood,
                0.0,
                0,
                "invalid_initial_likelihood",
                current.final_kappa,
                current.final_nu,
            )

        var gradient_norm: Float64
        var completed_iterations = 0
        var reason = String("max_iterations")
        for iteration in range(self.config.max_iterations):
            var gradient = self.central_gradient(returns, current_params)
            var norm_squared = 0.0
            for coordinate in range(10):
                norm_squared += gradient[coordinate] * gradient[coordinate]
            gradient_norm = sqrt(norm_squared)
            if gradient_norm <= self.config.gradient_tolerance:
                reason = "gradient_tolerance"
                break

            var step = self.config.initial_step
            var accepted = False
            var candidate_params = current_params
            var candidate = current
            while step >= self.config.min_step:
                var trial = InlineArray[Float64, 10](fill=0.0)
                for coordinate in range(10):
                    trial[coordinate] = (
                        coordinates[coordinate]
                        + step * gradient[coordinate]
                    )
                var trial_params = unconstrained_to_params(
                    trial, current_params.dt
                )
                var trial_result = self.evaluate(returns, trial_params)
                if (
                    trial_result.valid
                    and isfinite(trial_result.likelihood)[0]
                    and trial_result.likelihood
                    >= current.likelihood
                        + self.config.armijo_c * step * norm_squared
                ):
                    coordinates = trial
                    candidate_params = trial_params
                    candidate = trial_result
                    accepted = True
                    break
                step *= self.config.shrink

            if not accepted:
                reason = "line_search_failed"
                break

            var improvement = candidate.likelihood - current.likelihood
            current_params = candidate_params
            current = candidate
            completed_iterations = iteration + 1
            if improvement <= self.config.likelihood_tolerance:
                reason = "likelihood_tolerance"
                break

        var final_gradient = self.central_gradient(returns, current_params)
        var final_norm_squared = 0.0
        for coordinate in range(10):
            final_norm_squared += (
                final_gradient[coordinate] * final_gradient[coordinate]
            )
        gradient_norm = sqrt(final_norm_squared)

        return AffineSvjFitResult(
            current_params,
            current.likelihood,
            gradient_norm,
            completed_iterations,
            reason,
            current.final_kappa,
            current.final_nu,
        )
