from std.collections import List
from std.utils.numerics import isfinite


struct AffineSvjFitCliConfig(Copyable, Movable):
    var returns_path: String
    var count: Int
    var initial_path: String
    var out_path: String
    var grid_u0: Float64
    var grid_du: Float64
    var grid_count: Int
    var max_iterations: Int
    var fd_step: Float64
    var gradient_tolerance: Float64
    var likelihood_tolerance: Float64
    var armijo_c: Float64
    var armijo_shrink: Float64
    var min_step: Float64
    var initial_step: Float64
    var show_help: Bool

    def __init__(out self):
        self.returns_path = String()
        self.count = 0
        self.initial_path = String()
        self.out_path = String()
        self.grid_u0 = 0.0
        self.grid_du = 0.25
        self.grid_count = 2048
        self.max_iterations = 100
        self.fd_step = 1e-4
        self.gradient_tolerance = 1e-5
        self.likelihood_tolerance = 1e-8
        self.armijo_c = 1e-4
        self.armijo_shrink = 0.5
        self.min_step = 1e-8
        self.initial_step = 0.1
        self.show_help = False


def affine_svj_fit_usage() -> String:
    return String(
        "Usage: fit_svj --returns FILE --count N --initial FILE --out FILE",
        "\n  [--grid-u0 0] [--grid-du X] [--grid-count N]",
        "\n  [--max-iterations N] [--fd-step X] [--initial-step X]",
        "\n  [--gradient-tolerance X] [--likelihood-tolerance X]",
        "\n  [--armijo-c X] [--armijo-shrink X] [--min-step X]",
    )


def _parse_int_option(text: String, option: String) raises -> Int:
    try:
        return Int(text)
    except:
        raise Error(String("invalid integer for ", option, ": ", text))


def _parse_float_option(text: String, option: String) raises -> Float64:
    try:
        var value = Float64(text)
        if not isfinite(value)[0]:
            raise Error(String("non-finite value for ", option))
        return value
    except:
        raise Error(String("invalid number for ", option, ": ", text))


def parse_affine_svj_fit_args(
    args: List[String],
) raises -> AffineSvjFitCliConfig:
    """Parse option/value CLI tokens, excluding the executable name."""
    var config = AffineSvjFitCliConfig()
    var seen_returns = False
    var seen_count = False
    var seen_initial = False
    var seen_out = False
    var seen_grid_u0 = False
    var seen_grid_du = False
    var seen_grid_count = False
    var seen_max_iterations = False
    var seen_fd_step = False
    var seen_gradient_tolerance = False
    var seen_likelihood_tolerance = False
    var seen_armijo_c = False
    var seen_armijo_shrink = False
    var seen_min_step = False
    var seen_initial_step = False

    var i = 0
    while i < len(args):
        var option = String(args[i])
        if option == "--help" or option == "-h":
            config.show_help = True
            i += 1
            continue
        if i + 1 >= len(args):
            raise Error(String("missing value for ", option))
        var value = String(args[i + 1])

        if option == "--returns":
            if seen_returns:
                raise Error("duplicate --returns option")
            seen_returns = True
            config.returns_path = value.copy()
        elif option == "--count":
            if seen_count:
                raise Error("duplicate --count option")
            seen_count = True
            config.count = _parse_int_option(value, option)
        elif option == "--initial":
            if seen_initial:
                raise Error("duplicate --initial option")
            seen_initial = True
            config.initial_path = value.copy()
        elif option == "--out":
            if seen_out:
                raise Error("duplicate --out option")
            seen_out = True
            config.out_path = value.copy()
        elif option == "--grid-u0":
            if seen_grid_u0:
                raise Error("duplicate --grid-u0 option")
            seen_grid_u0 = True
            config.grid_u0 = _parse_float_option(value, option)
        elif option == "--grid-du":
            if seen_grid_du:
                raise Error("duplicate --grid-du option")
            seen_grid_du = True
            config.grid_du = _parse_float_option(value, option)
        elif option == "--grid-count":
            if seen_grid_count:
                raise Error("duplicate --grid-count option")
            seen_grid_count = True
            config.grid_count = _parse_int_option(value, option)
        elif option == "--max-iterations":
            if seen_max_iterations:
                raise Error("duplicate --max-iterations option")
            seen_max_iterations = True
            config.max_iterations = _parse_int_option(value, option)
        elif option == "--fd-step":
            if seen_fd_step:
                raise Error("duplicate --fd-step option")
            seen_fd_step = True
            config.fd_step = _parse_float_option(value, option)
        elif option == "--gradient-tolerance":
            if seen_gradient_tolerance:
                raise Error("duplicate --gradient-tolerance option")
            seen_gradient_tolerance = True
            config.gradient_tolerance = _parse_float_option(value, option)
        elif option == "--likelihood-tolerance":
            if seen_likelihood_tolerance:
                raise Error("duplicate --likelihood-tolerance option")
            seen_likelihood_tolerance = True
            config.likelihood_tolerance = _parse_float_option(value, option)
        elif option == "--armijo-c":
            if seen_armijo_c:
                raise Error("duplicate --armijo-c option")
            seen_armijo_c = True
            config.armijo_c = _parse_float_option(value, option)
        elif option == "--armijo-shrink":
            if seen_armijo_shrink:
                raise Error("duplicate --armijo-shrink option")
            seen_armijo_shrink = True
            config.armijo_shrink = _parse_float_option(value, option)
        elif option == "--initial-step":
            if seen_initial_step:
                raise Error("duplicate --initial-step option")
            seen_initial_step = True
            config.initial_step = _parse_float_option(value, option)
        elif option == "--min-step":
            if seen_min_step:
                raise Error("duplicate --min-step option")
            seen_min_step = True
            config.min_step = _parse_float_option(value, option)
        else:
            raise Error(String("unknown option: ", option))
        i += 2

    if config.show_help:
        return config^

    if not seen_returns or config.returns_path.byte_length() == 0:
        raise Error("missing required --returns option")
    if not seen_count:
        raise Error("missing required --count option")
    if not seen_initial or config.initial_path.byte_length() == 0:
        raise Error("missing required --initial option")
    if not seen_out or config.out_path.byte_length() == 0:
        raise Error("missing required --out option")
    if config.count <= 0:
        raise Error("--count must be greater than zero")
    if config.grid_u0 != 0.0:
        raise Error("--grid-u0 must be exactly zero")
    if config.grid_du <= 0.0:
        raise Error("--grid-du must be greater than zero")
    if config.grid_count < 2:
        raise Error("--grid-count must be at least two")
    if config.max_iterations <= 0:
        raise Error("--max-iterations must be greater than zero")
    if config.fd_step <= 0.0:
        raise Error("--fd-step must be greater than zero")
    if config.gradient_tolerance <= 0.0:
        raise Error("--gradient-tolerance must be greater than zero")
    if config.likelihood_tolerance <= 0.0:
        raise Error("--likelihood-tolerance must be greater than zero")
    if config.armijo_c <= 0.0 or config.armijo_c >= 1.0:
        raise Error("--armijo-c must lie strictly between zero and one")
    if config.armijo_shrink <= 0.0 or config.armijo_shrink >= 1.0:
        raise Error("--armijo-shrink must lie strictly between zero and one")
    if config.initial_step <= 0.0:
        raise Error("--initial-step must be greater than zero")
    if config.min_step <= 0.0:
        raise Error("--min-step must be greater than zero")
    if config.min_step > config.initial_step:
        raise Error("--min-step must not exceed --initial-step")
    return config^
