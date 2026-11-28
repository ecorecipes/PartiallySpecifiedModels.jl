module PartiallySpecifiedModelsDistributionsExt

using PartiallySpecifiedModels
const PSM = PartiallySpecifiedModels
using Distributions

# Map Distributions.jl families/instances directly to PSM likelihood types so
# users can write `PSMProblem(...; likelihood=Distributions.Poisson())`,
# `likelihood=Distributions.NegativeBinomial(5.0, 0.5)`,
# `likelihood=Distributions.Normal()`, `likelihood=Distributions.Truncated(Normal(0, 0.5), 0.0, Inf)`.

PSM._coerce_likelihood(::Distributions.Normal) = PSM.GaussianLikelihood()
PSM._coerce_likelihood(::Type{<:Distributions.Normal}) = PSM.GaussianLikelihood()

PSM._coerce_likelihood(::Distributions.Poisson) = PSM.PoissonLikelihood()
PSM._coerce_likelihood(::Type{<:Distributions.Poisson}) = PSM.PoissonLikelihood()

function PSM._coerce_likelihood(d::Distributions.NegativeBinomial)
    PSM.NegativeBinomialLikelihood(Float64(d.r))
end

function PSM._coerce_likelihood(d::Distributions.Truncated{<:Distributions.Normal})
    if !(isfinite(d.lower) && isinf(d.upper) && d.upper > 0)
        throw(ArgumentError(
            "PSMProblem: Truncated(Normal(...), lower, upper) requires finite `lower` " *
            "and `upper = Inf` (got [$(d.lower), $(d.upper)])."))
    end
    PSM.TruncatedNormalLikelihood(Float64(d.lower), Float64(d.untruncated.σ))
end

PSM._coerce_likelihood(d::Distributions.Gamma) =
    PSM.GammaLikelihood(1.0 / Float64(Distributions.shape(d)))

PSM._coerce_likelihood(d::Distributions.LogNormal) =
    PSM.LogNormalLikelihood(Float64(d.σ))

PSM._coerce_likelihood(d::Distributions.Binomial) =
    PSM.BinomialLikelihood(Int(d.n))

PSM._coerce_likelihood(d::Distributions.TDist) =
    PSM.StudentTLikelihood(Float64(d.ν), 1.0)

function PSM._coerce_likelihood(d::Distributions.LocationScale{<:Real, Distributions.Continuous, <:Distributions.TDist})
    PSM.StudentTLikelihood(Float64(d.ρ.ν), Float64(d.σ))
end

end # module
