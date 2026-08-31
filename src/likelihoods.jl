# Likelihood families for LAML/IRLS
#
# Each family provides:
#   log_likelihood(fam, y, mu, w) — total log-likelihood
#   irls_weights(fam, y, mu, w)  — IRLS working weights W̃
#
# Reference: Wood, Pya & Säfken (2016), Section 2.

# ─── Standard normal helpers (avoid Distributions.jl dependency) ────

const _INV_SQRT_2PI = 1.0 / sqrt(2π)

"""Standard normal PDF φ(x)."""
_normpdf(x::Real) = _INV_SQRT_2PI * exp(-0.5 * x^2)

"""Standard normal CDF Φ(x) via rational approximation (Abramowitz & Stegun 26.2.17)."""
function _normcdf(x::Real)
    # Accuracy: |ε| < 7.5e-8
    if x >= 0
        t = 1.0 / (1.0 + 0.2316419 * x)
        poly = t * (0.319381530 + t * (-0.356563782 +
               t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        1.0 - _normpdf(x) * poly
    else
        # Compute the small tail DIRECTLY: 1 − Φ(−x) cancels catastrophically
        # below x ≈ −7.5 (Φ underflows against 1), which inflated the
        # TruncatedNormal information ~300× in the far tail.
        t = 1.0 / (1.0 - 0.2316419 * x)
        poly = t * (0.319381530 + t * (-0.356563782 +
               t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        _normpdf(x) * poly
    end
end

"""
Standard normal log-CDF log Φ(x), stable and accurate in BOTH tails.

Replaces a truncated asymptotic expansion that dropped the Mills-series
correction, causing a ~0.022 jump at the old x = −6 branch switch and
0.3–0.7% log error across the tail (the A&S rational `_normcdf` is accurate
ABSOLUTELY to 7.5e-8, so its log error also grows in the tail). All
branches below are accurate to ~1e-15 relative, so the seams are smooth to
machine precision — second differences across them (e.g. finite-difference
information checks) stay clean.

The positive half is computed from the negative half by complement,
`log Φ(x) = log1p(−Φ(−x))`: `log(0.5 + φ(x)Σ…)` loses all relative
precision as Φ → 1 (only ~7 correct digits at x = 6, none past x ≈ 9,
where it returns exactly 0), and a branch switch there made the function
DECREASE by 3.5e-12 across x = 6 — the only non-monotone point of the
whole function. Going through the (tiny, fully accurate) upper tail
instead keeps ~1e-16 relative accuracy for every x > 0.
"""
function _normlogcdf(x::Real)
    if x > 0.0
        # Φ(x) = 1 − Φ(−x) with Φ(−x) ≤ ½ computed to full relative
        # precision by the branches below; log1p is exact for tiny inputs.
        log1p(-exp(_normlogcdf(-x)))
    elseif x > -1.0
        # Exact small-|x| series (Abramowitz & Stegun 26.2.11):
        #   Φ(x) = ½ + φ(x) Σ_{n≥0} x^{2n+1}/(2n+1)!!
        # All terms share x's sign — no cancellation for x > −1 — and the
        # ratio x²/(2n+3) → 0 guarantees fast convergence on this range.
        term = x
        s = x
        n = 0
        while abs(term) > 1e-17 * abs(s) && n < 200
            n += 1
            term *= x * x / (2n + 1)
            s += term
        end
        log(0.5 + _normpdf(x) * s)
    else
        # Mills-ratio Laplace continued fraction: for t = −x ≥ 1,
        #   Φ(x) = φ(t)·R(t),  R(t) = 1/(t + 1/(t + 2/(t + 3/(t + ⋯)))),
        # evaluated by backward recurrence (depth 400: machine precision
        # for t ≥ 1, and trivially cheap).
        t = -x
        r = zero(t)
        for k in 400:-1:1
            r = k / (t + r)
        end
        -0.5 * x^2 - 0.5 * log(2π) - log(t + r)
    end
end

# ─── log Γ (Lanczos; avoids a SpecialFunctions dependency) ──────────

"""Log-gamma via the Lanczos approximation (g=7), accurate to ~1e-13 for x>0."""
function _loggamma(x::Real)
    g = 7.0
    c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
    if x < 0.5
        return log(π / abs(sin(π * x))) - _loggamma(1 - x)
    end
    x -= 1
    a = c[1]
    t = x + g + 0.5
    for i in 2:9
        a += c[i] / (x + (i - 1))
    end
    0.5 * log(2π) + (x + 0.5) * log(t) - t + log(a)
end

# ─── Cell usability (the package-wide masking convention) ───────────

"""
    _usable(y, w) -> Bool

A data cell counts toward an objective iff its weight is strictly
positive AND its value is FINITE. This is the single convention shared by
`weighted_data_loss`, `usable_cell`, `n_usable`, `_reject_masked_data`,
the LAML/GCV data flattening, the bootstrap, ABC, MAGI, VI and MCMC.

`isfinite`, not `!isnan`: `±Inf` is no more usable as an observation than
`NaN` is. The two predicates coexisted for a while — `weighted_data_loss`
gated on `isfinite` while the cell predicates gated on `!isnan` — so an
`Inf` datum was counted in every denominator, excluded from every
numerator, and waved through by `_reject_masked_data` into the Kalman
solvers that cannot mask at all.

Enforcing it *inside* the likelihood families is what makes the
convention hold for the OPTIMIZED objective and not merely the reported
one. `w = 0` is not sufficient protection on its own: IEEE arithmetic
gives `0 * NaN = NaN`, so a single masked-out missing observation would
otherwise turn the whole log-likelihood — and every gradient derived
from it — into NaN, and the optimizer would reject every step and
return its initialization without raising an error.

For complete data (every weight positive, every value finite) the guard
never fires, so all values below are bit-for-bit unchanged.
"""
@inline _usable(y::Real, w::Real) = w > 0 && isfinite(y)

"""
    _n_usable(y, w) -> Int

Number of usable cells in a flattened `(y, w)` pair — the sample size
that every per-observation denominator (REML `n`, GCV `n`, σ̂²'s
`n − edf`) must use. Equals `length(y)` for complete data.
"""
function _n_usable(y::AbstractVector, w::AbstractVector)
    c = 0
    for i in eachindex(y)
        _usable(y[i], w[i]) && (c += 1)
    end
    c
end

# ─── Log-likelihood functions ───────────────────────────────────────

"""
    log_likelihood(fam, y, mu, w)

Total weighted log-likelihood: Σ_i w_i ℓ(y_i, μ_i), summed over the
USABLE cells only (`w_i > 0` and `y_i` finite — see `_usable`).

Poisson, NegativeBinomial, and TruncatedNormal include their full
normalizing constants and are mutually comparable (e.g. for AIC). The
**Gaussian** value is the kernel −½Σw(y−μ)² only — σ² is profiled out
elsewhere, so the −(n/2)log(2πσ²) term is omitted and Gaussian values are
NOT comparable across families.
"""
function log_likelihood(::Gaussian, y::AbstractVector,
                        mu::AbstractVector, w::AbstractVector)
    ll = 0.0
    for i in eachindex(y)
        _usable(y[i], w[i]) || continue
        ll -= 0.5 * w[i] * (y[i] - mu[i])^2
    end
    ll
end

function log_likelihood(::Poisson, y::AbstractVector,
                        mu::AbstractVector, w::AbstractVector)
    ll = 0.0
    for i in eachindex(y)
        _usable(y[i], w[i]) || continue
        # Convention (shared with irls_weights): μ ≤ 0 is regularized as
        # max(|μ|, 1e-6) so the objective and the IRLS curvature see the
        # SAME effective mean when identity-link iterates transiently go
        # nonpositive. (Flooring at 1e-10 here while the weights used
        # max(|μ|, 1e-6) put a flat cliff in the objective exactly where
        # the curvature was still finite.) Identity for healthy μ > 1e-6.
        mu_i = max(abs(mu[i]), 1e-6)
        kern = y[i] > 0 ? y[i] * log(mu_i) - mu_i : -mu_i
        ll += w[i] * (kern - _loggamma(y[i] + 1))   # − log(y!)
    end
    ll
end

function log_likelihood(fam::NegativeBinomial, y::AbstractVector,
                        mu::AbstractVector, w::AbstractVector)
    θ = fam.theta
    lgθ = _loggamma(θ)
    ll = 0.0
    for i in eachindex(y)
        _usable(y[i], w[i]) || continue
        mu_i = max(mu[i], 1e-10)
        kern = y[i] * log(mu_i / (mu_i + θ)) + θ * log(θ / (mu_i + θ))
        norm = _loggamma(y[i] + θ) - lgθ - _loggamma(y[i] + 1)
        ll += w[i] * (kern + norm)
    end
    ll
end

function log_likelihood(fam::TruncatedNormal, y::AbstractVector,
                        mu::AbstractVector, w::AbstractVector)
    σ = fam.sigma
    a = fam.lower
    ll = 0.0
    for i in eachindex(y)
        _usable(y[i], w[i]) || continue
        z = (y[i] - mu[i]) / σ
        # log f(y|μ,σ,a) = -½z² - log(σ) - ½log(2π) - log Φ((μ-a)/σ)
        ll += w[i] * (-0.5 * z^2 - log(σ) - 0.5 * log(2π) -
                       _normlogcdf((mu[i] - a) / σ))
    end
    ll
end

function log_likelihood(fam::CustomLikelihood, y::AbstractVector,
                        mu::AbstractVector, w::AbstractVector)
    ll = 0.0
    for i in eachindex(y)
        _usable(y[i], w[i]) || continue
        ll += w[i] * fam.loglik_scalar(y[i], mu[i])
    end
    ll
end

# ─── Pointwise log-likelihood (single observation) ──────────────────

"""
    loglik_pointwise(fam, y, mu)

Pointwise log-likelihood ℓ(y, μ) for a single observation; the caller
multiplies in any observation weight. Term-for-term identical to what
`log_likelihood` accumulates, so summing `w_i · loglik_pointwise(fam,
y_i, μ_i)` reproduces `log_likelihood(fam, y, mu, w)` exactly. The
**Gaussian** method is the σ-free kernel −½(y−μ)² (σ² is a nuisance
parameter handled by each solver), matching the `log_likelihood`
convention. AD-safe: `μ` may be a `ForwardDiff.Dual`.

A non-finite `y` (NaN or ±Inf) is a masked (missing) observation and
contributes exactly zero, matching `log_likelihood`'s `_usable` guard. The zero is built as
`zero(y * mu)` so it carries `μ`'s type — a `ForwardDiff.Dual` stays a
Dual (with zero partials), keeping the accumulation type-stable and the
gradient contribution correctly zero rather than NaN. Zero-weight cells
need no guard here: the caller multiplies by `w = 0`, and the value
returned for a finite `y` is finite.
"""
loglik_pointwise(::Gaussian, y::Real, mu::Real) =
    !isfinite(y) ? zero(y * mu) : -0.5 * (y - mu)^2

function loglik_pointwise(::Poisson, y::Real, mu::Real)
    !isfinite(y) && return zero(y * mu)
    # Same μ ≤ 0 convention as log_likelihood/irls_weights: max(|μ|, 1e-6).
    mu_c = max(abs(mu), 1e-6)
    kern = y > 0 ? y * log(mu_c) - mu_c : -mu_c
    kern - _loggamma(y + 1)
end

function loglik_pointwise(fam::NegativeBinomial, y::Real, mu::Real)
    !isfinite(y) && return zero(y * mu)
    θ = fam.theta
    mu_c = max(mu, 1e-10)
    kern = y * log(mu_c / (mu_c + θ)) + θ * log(θ / (mu_c + θ))
    kern + _loggamma(y + θ) - _loggamma(θ) - _loggamma(y + 1)
end

function loglik_pointwise(fam::TruncatedNormal, y::Real, mu::Real)
    !isfinite(y) && return zero(y * mu)
    σ = fam.sigma
    a = fam.lower
    z = (y - mu) / σ
    -0.5 * z^2 - log(σ) - 0.5 * log(2π) - _normlogcdf((mu - a) / σ)
end

loglik_pointwise(fam::CustomLikelihood, y::Real, mu::Real) =
    !isfinite(y) ? zero(y * mu) : fam.loglik_scalar(y, mu)

# ─── IRLS working weights ──────────────────────────────────────────

"""
    irls_weights(fam, y, mu, w)

Compute IRLS working weights for identity-link Fisher scoring.

The PSM solver operates on the response scale (identity link), so the
working weight is the observed/Fisher information of the family:

    W̃_i = w_i × I(μ_i),   I(μ) = −∂²ℓ/∂μ²

For `Gaussian`, `Poisson` and `NegativeBinomial` — identity-link
exponential families whose mean parameter IS the mean — this coincides
with the inverse variance, `W̃_i = w_i / V(μ_i)`, and those methods are
written that way.

`TruncatedNormal` is the exception, and its method is NOT `w/V(μ)`: there
`μ` is the latent normal location, and
`I(μ) = (1 − ξλ(ξ) − λ(ξ)²)/σ² = V(μ)/σ⁴`, which is not `1/V(μ)` (the two
differ by a factor `V(μ)²/σ⁴`; measured at μ = 0.2, σ = 0.15,
`I = 32.30232` against `1/V = 61.15068`). The information is the correct
curvature and is what the method uses — only this generic summary was
wrong. `CustomLikelihood` obtains `−∂²ℓ/∂μ²` by automatic differentiation,
so it follows the general rule directly.

Unusable cells (`_usable` false: zero weight or non-finite datum) get working
weight exactly 0, so they drop out of `J'W̃J`, out of the EDF trace and
out of every PCLS step. For a zero-weight cell that is already what the
formulae give whenever `y` is finite; the guard matters when `y` is NaN
— notably for `CustomLikelihood`, whose curvature is obtained by
differentiating `loglik_scalar(y, ·)`, which returns NaN for a NaN `y`
and would then be multiplied by `w = 0` to give NaN, not 0.
"""
function irls_weights(::Gaussian, y::AbstractVector,
                      mu::AbstractVector, w::AbstractVector)
    wt = copy(w)
    for i in eachindex(wt)
        _usable(y[i], w[i]) || (wt[i] = zero(eltype(wt)))
    end
    wt
end

function irls_weights(::Poisson, y::AbstractVector,
                      mu::AbstractVector, w::AbstractVector)
    # Identity link, V(μ) = μ → W̃ = w / μ
    wt = similar(w)
    for i in eachindex(w)
        if !_usable(y[i], w[i])
            wt[i] = zero(eltype(wt)); continue
        end
        mu_i = max(abs(mu[i]), 1e-6)
        wt[i] = w[i] / mu_i
    end
    wt
end

function irls_weights(fam::NegativeBinomial, y::AbstractVector,
                      mu::AbstractVector, w::AbstractVector)
    # Identity link, V(μ) = μ + μ²/θ → W̃ = w / V(μ)
    θ = fam.theta
    wt = similar(w)
    for i in eachindex(w)
        if !_usable(y[i], w[i])
            wt[i] = zero(eltype(wt)); continue
        end
        mu_i = max(abs(mu[i]), 1e-6)
        wt[i] = w[i] / (mu_i + mu_i^2 / θ)
    end
    wt
end

function irls_weights(fam::TruncatedNormal, y::AbstractVector,
                      mu::AbstractVector, w::AbstractVector)
    # For TruncatedNormal(a, σ), the observed (= Fisher) information is:
    #   I(μ) = (1/σ²)(1 - ξ·λ(ξ) - λ(ξ)²)
    # where ξ = (μ-a)/σ, λ(ξ) = φ(ξ)/Φ(ξ) (inverse Mills ratio), from
    # -∂²ℓ/∂μ² = (1 + λ'(ξ))/σ² with λ'(ξ) = -ξλ - λ².
    # Analytically I(μ) ∈ (0, 1/σ²); the floor guards roundoff at ξ ≪ 0.
    # Working weight: W̃ = w × I(μ)
    σ = fam.sigma
    a = fam.lower
    wt = similar(w)
    for i in eachindex(w)
        if !_usable(y[i], w[i])
            wt[i] = zero(eltype(wt)); continue
        end
        ξ = (mu[i] - a) / σ
        Φξ = max(_normcdf(ξ), 1e-15)
        λξ = _normpdf(ξ) / Φξ          # inverse Mills ratio
        info = (1.0 - ξ * λξ - λξ^2) / σ^2
        wt[i] = w[i] * max(info, 1e-10)
    end
    wt
end

function irls_weights(fam::CustomLikelihood, y::AbstractVector,
                      mu::AbstractVector, w::AbstractVector)
    # Derive via ForwardDiff: w̃_i = w_i × (-∂²ℓ/∂μ²)
    wt = similar(w)
    for i in eachindex(w)
        if !_usable(y[i], w[i])
            wt[i] = zero(eltype(wt)); continue
        end
        yi = y[i]
        neg_d2l = -ForwardDiff.derivative(
            μ -> ForwardDiff.derivative(μ2 -> fam.loglik_scalar(yi, μ2), μ),
            mu[i]
        )
        wt[i] = w[i] * max(neg_d2l, 1e-10)
    end
    wt
end

# ─── Working residual (z − η) for Fisher scoring ────────────────────

"""
    _working_residual_scalar(fam, y, mu) -> Real
    _working_residual(fam, y, mu, w) -> Vector   (broadcast of the above)

The PIRLS/Fisher-scoring working residual `z − η`, i.e. the quantity that
pairs with `irls_weights` so that

    W̃ᵢ · (z − η)ᵢ  ≡  wᵢ · ∂ℓᵢ/∂μᵢ                                    (★)

holds exactly. (★) is the whole point: the normal equations `J'W̃(z − Jβ)`
are then the true penalized score, so the fixed point of the iteration is
the actual MLE and not some nearby quantity.

`irls_weights` uses the convention `W̃ = w · I(μ)` with `I` the Fisher
information for μ, so (★) forces `(z − η) = (∂ℓ/∂μ) / I(μ)`.

For every family with an identity link and an exponential-family kernel
this collapses to the textbook `y − μ`, because the score is `(y − μ)/V(μ)`
and `I = 1/V(μ)`:

| family          | ∂ℓ/∂μ      | I(μ)   | (z − η) |
|-----------------|------------|--------|---------|
| Gaussian        | (y−μ)/σ²   | 1/σ²   | y − μ   |
| Poisson         | (y−μ)/μ    | 1/μ    | y − μ   |
| NegativeBinomial| (y−μ)/V    | 1/V    | y − μ   |

`TruncatedNormal` is the family that does NOT collapse, and it is the
reason this function exists — see its method below.

The `AbstractLikelihood` fallback is `y .- mu`, so every family that has
not opted in keeps its pre-existing arithmetic bit-for-bit. NaN/masked
observations are passed through untouched here (they are neutralized by
the zero weight at the call site, exactly as before).
"""
_working_residual_scalar(::AbstractLikelihood, y::Real, mu::Real) = y - mu

"""
    _working_residual_scalar(fam::TruncatedNormal, y, mu)

For `TruncatedNormal(a, σ)` with `ξ = (μ-a)/σ` and `λ(ξ) = φ(ξ)/Φ(ξ)`:

    ∂ℓ/∂μ = (y − μ)/σ² − λ(ξ)/σ = (y − E[Y|μ]) / σ²
    I(μ)  = (1 − ξλ − λ²) / σ²                        (see `irls_weights`)
    ⇒ (z − η) = (y − μ − σλ) / (1 − ξλ − λ²)

so BOTH corrections matter and they are independent:

1. **Centring.** The numerator is `y − E[Y|μ]`, not `y − μ`. μ is the
   LATENT location; the observable's mean is `E[Y|μ] = μ + σλ` (that is
   `_family_mean`, added for the Pearson dispersion in the same campaign).
2. **Scaling.** The `1/(1 − ξλ − λ²)` factor. It is identically 1 for the
   families in the table above, which is why the plain `y − μ` form
   survived so long — nothing else in the package needed it.

Using `y − μ` here does not merely bias the fit, it points the step the
WRONG WAY wherever the bound bites. Measured against the ForwardDiff score
(relative error of `W̃·(z−η)` vs `w·∂ℓ/∂μ`, invariant in σ and a):

| ξ    | W̃·(y−μ) | w·∂ℓ/∂μ  | rel. err |
|------|----------|----------|----------|
| −1.0 | +0.0597  | −1.225   | 1.05     |
| 0.0  | +0.109   | −0.498   | 1.22     |
| 0.5  | +0.146   | −0.209   | 1.70     |
| 1.0  | +0.189   | +0.0124  | 14.2     |
| 2.0  | +0.266   | +0.245   | 0.0866   |
| 4.0  | +0.300   | +0.300   | 8.93e-5  |

The sign is inverted for ξ ≲ 1 — PIRLS pushed μ̂ away from the optimum
and stopped where the two errors happened to cancel. The corrected form
reproduces the score to machine precision (≤1.11e-16) at every ξ above.
The error decays as the truncation stops binding (8.93e-5 by ξ = 4), so
this is a no-op on weakly truncated fits and only bites where the bound
actually bites — the same "only bites where it bites" property
`_family_mean` has.

The `Φ` and information floors are the SAME guards `irls_weights` applies
(`max(Φ(ξ), 1e-15)`, `max(I, 1e-10)`). They have to be, or (★) would fail
in the far tail where one side is clamped and the other is not.
"""
function _working_residual_scalar(fam::TruncatedNormal, y::Real, mu::Real)
    σ = fam.sigma
    a = fam.lower
    ξ = (mu - a) / σ
    Φξ = max(_normcdf(ξ), 1e-15)
    λξ = _normpdf(ξ) / Φξ
    # Same guarded information as `irls_weights`, so W̃·(z−η) = w·score holds
    # even where the floor is active.
    info = max((1.0 - ξ * λξ - λξ^2) / σ^2, 1e-10)
    score = (y - mu) / σ^2 - λξ / σ            # = (y − E[Y|μ])/σ²
    score / info
end

"""
    _working_residual_scalar(fam::CustomLikelihood, y, mu)

A user-supplied kernel has no reason to satisfy `∂ℓ/∂μ = (y−μ)/V(μ)`, so
`y − μ` is not its working residual either. `irls_weights` already derives
this family's curvature by ForwardDiff; this derives the matching score the
same way, with the SAME `max(·, 1e-10)` floor, so (★) holds by construction.

Measured: on a Gaussian-shaped kernel `-(y-μ)²/2` the two forms agree to
0.0 exactly (the derivatives are exact for a quadratic), so every custom
kernel of that shape — which is what the package's own fixtures use — is
bit-identical. On a skewed kernel `-(y-μ) - exp(-(y-μ))` the old `y − μ`
form was off by 45.3%.

The cost is one extra first-derivative sweep alongside the second-derivative
sweep `irls_weights` already performs.
"""
function _working_residual_scalar(fam::CustomLikelihood, y::Real, mu::Real)
    score = ForwardDiff.derivative(m -> fam.loglik_scalar(y, m), mu)
    neg_d2l = -ForwardDiff.derivative(
        m -> ForwardDiff.derivative(m2 -> fam.loglik_scalar(y, m2), m), mu)
    score / max(neg_d2l, 1e-10)
end

# The vector form is a broadcast of the scalar one — deliberately ONE
# implementation. Two parallel implementations of this arithmetic (a vector
# one for the LAML/GCV pseudodata and a scalar one for the collocation
# residual rows) is precisely how a fix lands in one solver and not its
# sibling, which is the defect class this package keeps rediscovering.
_working_residual(fam::AbstractLikelihood, y::AbstractVector,
                  mu::AbstractVector, w::AbstractVector) =
    _working_residual_scalar.(Ref(fam), y, mu)

