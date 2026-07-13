from affine_svj import AffineSvjParams
from stochcharfunc import AffineSvjJointCF


def main():
    var params = AffineSvjParams(
        0.03,
        -0.15,
        0.12,
        0.35,
        -0.04,
        0.12,
        -0.55,
        0.04,
        1.5,
        0.3,
        1.0 / 252.0,
    )
    var model = AffineSvjJointCF(params)
    var frequencies = SIMD[DType.float64, 8](
        0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75
    )
    var kappa = (
        params.volvol * params.volvol / (2.0 * params.beta)
    )
    var nu = (
        2.0 * params.beta * params.alpha
        / (params.volvol * params.volvol)
    )
    var characteristic = model.predictive_cf_simd[8](
        frequencies, 0.0, kappa, nu
    )
    print("frequencies:", frequencies)
    print("characteristic function (real):", characteristic.re)
    print("characteristic function (imag):", characteristic.im)
