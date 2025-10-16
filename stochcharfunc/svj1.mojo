struct Svj1JointCF:
    var phi: Float64
    var psi: Float64
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
        phi: Float64 = 0.0,         # characteristic function frequency (real part)
        psi: Float64 = 0.0,         # latent characteristic function frequency (imag part)
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
        self.phi = phi
        self.psi = psi
        self.k = k
        self.dt = dt

    fn get_params(self) -> (Float64, Float64, Float64, Float64, 
                            Float64, Float64, Float64, Float64, 
                            Float64, Float64, Float64, Float64):
        print(
            "mu0:", self.mu0,
            "mu1:", self.mu1,
            "rho:", self.rho,
            "alpha:", self.alpha,
            "beta:", self.beta,
            "volvol:", self.volvol,
            "lambda0:", self.lambda0,
            "lambda1:", self.lambda1,
            "phi:", self.phi,
            "psi:", self.psi,
            "k:", self.k,
            "dt:", self.dt
        )
        return (
            self.mu0, self.mu1, self.rho, self.alpha, self.beta,
            self.volvol, self.lambda0, self.lambda1,
            self.phi, self.psi, self.k, self.dt
        )

    fn joint_cf_cond_y(self, phi, psi) ->

    fn latent_cond_cf(self, psi) ->
    fn latent_uncond_cf(self, phi,psi) ->



