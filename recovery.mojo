from affine_svj import AffineSvjParams
from affine_svj.fit_command import stationary_gamma_prior
from affine_svj_simulator import AffineSvjSimulator
from complexsimd.fourier import UniformGridInverterSIMDGrid
from optimizers.finitediff import (
    AffineSvjOptimConfig,
    AffineSvjOptimizer,
    params_to_unconstrained,
    unconstrained_to_params,
)
from std.math import sqrt
from std.utils.numerics import isfinite


comptime RECOVERY_RUNS = 20
comptime OBSERVATIONS_PER_RUN = 4000
comptime INTRADAY_STEPS = 50
comptime FREE_PARAMETER_COUNT = 10
comptime MAX_ITERATIONS = 40


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def true_parameters() -> AffineSvjParams:
    return AffineSvjParams(
        0.06,
        -0.15,
        0.8,
        1.5,
        -0.025,
        0.08,
        -0.65,
        0.04,
        1.5,
        0.30,
        1.0 / 252.0,
    )


def recovery_seeds() -> InlineArray[Int, RECOVERY_RUNS]:
    return [
        1729,
        2718,
        31415,
        16180,
        57721,
        14142,
        17320,
        22360,
        24494,
        26457,
        31622,
        33166,
        34641,
        36055,
        37416,
        38729,
        40001,
        41231,
        42427,
        43589,
    ]


def parameter_name(index: Int) -> String:
    if index == 0:
        return "mu0"
    if index == 1:
        return "mu1"
    if index == 2:
        return "lambda0"
    if index == 3:
        return "lambda1"
    if index == 4:
        return "jump_mean"
    if index == 5:
        return "jump_std"
    if index == 6:
        return "rho"
    if index == 7:
        return "alpha"
    if index == 8:
        return "beta"
    if index == 9:
        return "volvol"
    return "unknown"


def deterministic_initial(
    truth: AffineSvjParams, replicate: Int
) raises -> AffineSvjParams:
    """Perturb every free parameter in unconstrained coordinates."""
    var coordinates = params_to_unconstrained(truth)
    var scales: InlineArray[Float64, FREE_PARAMETER_COUNT] = [
        0.008,
        0.040,
        0.100,
        0.100,
        0.008,
        0.080,
        0.080,
        0.080,
        0.100,
        0.080,
    ]
    var magnitude_multiplier = 0.8 + 0.05 * Float64(replicate % 5)
    for coordinate in range(FREE_PARAMETER_COUNT):
        var sign = 1.0
        if (replicate + coordinate) % 2 == 0:
            sign = -1.0
        coordinates[coordinate] += (
            sign * magnitude_multiplier * scales[coordinate]
        )
    var initial = unconstrained_to_params(coordinates, truth.dt)
    initial.validate()
    return initial


def acceptable_convergence_reason(reason: String) -> Bool:
    return reason == "gradient_tolerance" or reason == "likelihood_tolerance"


def validate_filter_result(
    likelihood: Float64,
    kappa: Float64,
    nu: Float64,
    valid: Bool,
    label: String,
) raises:
    require(valid, String(label, " filter state is invalid"))
    require(
        isfinite(likelihood)[0],
        String(label, " likelihood is non-finite"),
    )
    require(
        isfinite(kappa)[0] and kappa > 0.0,
        String(label, " final kappa is not finite and positive"),
    )
    require(
        isfinite(nu)[0] and nu > 0.0,
        String(label, " final nu is not finite and positive"),
    )


def main() raises:
    var truth = true_parameters()
    truth.validate()
    var truth_values = truth.free_values()
    var seeds = recovery_seeds()
    var estimates = InlineArray[Float64, RECOVERY_RUNS * FREE_PARAMETER_COUNT](
        fill=0.0
    )

    var simulator = AffineSvjSimulator(truth, intraday_steps=INTRADAY_STEPS)
    # Positive-frequency grid: u = 0, 2, ..., 512.
    var inverter = UniformGridInverterSIMDGrid[8](0.0, 2.0, 257)

    for replicate in range(RECOVERY_RUNS):
        var simulation = simulator.simulate(
            1,
            OBSERVATIONS_PER_RUN,
            seeds[replicate],
            100.0,
        )
        require(
            simulation.paths == 1,
            "recovery simulation must contain exactly one path",
        )
        require(
            simulation.steps == OBSERVATIONS_PER_RUN,
            "recovery simulation must contain exactly 4000 observations",
        )
        require(
            len(simulation.log_returns) == OBSERVATIONS_PER_RUN,
            "recovery return vector has the wrong size",
        )
        for observation in range(OBSERVATIONS_PER_RUN):
            require(
                isfinite(simulation.log_returns[observation])[0],
                String("non-finite simulated return in replicate ", replicate),
            )

        var initial = deterministic_initial(truth, replicate)
        require(
            initial.dt == truth.dt,
            "deterministic perturbation changed fixed dt",
        )
        var initial_prior = stationary_gamma_prior(initial)
        var optim_config = AffineSvjOptimConfig[8](
            inverter,
            initial_prior,
            2e-4,
            1e-4,
            0.5,
            0.01,
            1e-4,
            1e-7,
            MAX_ITERATIONS,
            1e-10,
        )
        var optimizer = AffineSvjOptimizer[8](optim_config)
        var initial_filter = optimizer.evaluate(simulation.log_returns, initial)
        validate_filter_result(
            initial_filter.likelihood,
            initial_filter.final_kappa,
            initial_filter.final_nu,
            initial_filter.valid,
            "initial",
        )

        var result = optimizer.fit(simulation.log_returns, initial)
        result.params.validate()
        validate_filter_result(
            result.likelihood,
            result.final_kappa,
            result.final_nu,
            True,
            "fitted",
        )
        require(
            isfinite(result.gradient_norm)[0] and result.gradient_norm >= 0.0,
            "fitted gradient norm is invalid",
        )
        require(
            result.iterations >= 0 and result.iterations <= MAX_ITERATIONS,
            "fitted iteration count is invalid",
        )
        require(
            result.convergence_reason.byte_length() > 0,
            "fitted convergence reason is empty",
        )
        require(
            acceptable_convergence_reason(result.convergence_reason),
            String(
                "unacceptable recovery convergence reason: ",
                result.convergence_reason,
            ),
        )
        require(
            result.params.dt == truth.dt,
            "optimizer changed fixed dt",
        )

        require(
            result.likelihood > initial_filter.likelihood,
            String(
                "fitted likelihood did not strictly improve in replicate ",
                replicate,
                ": initial=",
                initial_filter.likelihood,
                " final=",
                result.likelihood,
            ),
        )

        var fitted_values = result.params.free_values()
        for coordinate in range(FREE_PARAMETER_COUNT):
            require(
                isfinite(fitted_values[coordinate])[0],
                String(
                    "non-finite fitted ",
                    parameter_name(coordinate),
                    " in replicate ",
                    replicate,
                ),
            )
            estimates[
                replicate * FREE_PARAMETER_COUNT + coordinate
            ] = fitted_values[coordinate]

        print(
            "recovery replicate",
            replicate + 1,
            "seed",
            seeds[replicate],
            "initial_ll",
            initial_filter.likelihood,
            "final_ll",
            result.likelihood,
            "iterations",
            result.iterations,
            "reason",
            result.convergence_reason,
        )

    for coordinate in range(FREE_PARAMETER_COUNT):
        var estimate_sum = 0.0
        for replicate in range(RECOVERY_RUNS):
            estimate_sum += estimates[
                replicate * FREE_PARAMETER_COUNT + coordinate
            ]
        var estimate_mean = estimate_sum / Float64(RECOVERY_RUNS)

        var squared_deviation_sum = 0.0
        for replicate in range(RECOVERY_RUNS):
            var deviation = (
                estimates[replicate * FREE_PARAMETER_COUNT + coordinate]
                - estimate_mean
            )
            squared_deviation_sum += deviation * deviation
        var sample_variance = squared_deviation_sum / Float64(RECOVERY_RUNS - 1)
        require(
            isfinite(sample_variance)[0] and sample_variance >= 0.0,
            String(
                "invalid empirical variance for ",
                parameter_name(coordinate),
            ),
        )
        var monte_carlo_se = sqrt(sample_variance / Float64(RECOVERY_RUNS))
        var bias = estimate_mean - truth_values[coordinate]
        var absolute_bias = abs(bias)
        var roundoff_tolerance = 1e-10 * (1.0 + abs(truth_values[coordinate]))
        var bias_bound = roundoff_tolerance
        if monte_carlo_se > 0.0:
            bias_bound += 3.0 * monte_carlo_se

        require(
            isfinite(estimate_mean)[0]
            and isfinite(monte_carlo_se)[0]
            and isfinite(absolute_bias)[0],
            String(
                "non-finite aggregate statistic for ",
                parameter_name(coordinate),
            ),
        )
        require(
            absolute_bias <= bias_bound,
            String(
                "aggregate bias exceeds three Monte Carlo SE for ",
                parameter_name(coordinate),
                ": mean=",
                estimate_mean,
                " truth=",
                truth_values[coordinate],
                " bias=",
                bias,
                " mc_se=",
                monte_carlo_se,
                " bound=",
                bias_bound,
            ),
        )
        print(
            "aggregate",
            parameter_name(coordinate),
            "mean",
            estimate_mean,
            "truth",
            truth_values[coordinate],
            "bias",
            bias,
            "mc_se",
            monte_carlo_se,
            "bound",
            bias_bound,
        )

    print(
        "PASS: 20-seed affine SVJ recovery met finite convergence,",
        "likelihood improvement, fixed-dt, and three-Monte-Carlo-SE checks",
    )
