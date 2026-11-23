# ─── Shared neural-network evaluator infrastructure ──────────────
#
# One home for the ForwardDiff-compatible MLP path (used by AdamSolver,
# BNGSolver, MCMCSolver, …) and the single `build_neural_evaluator`
# helper every solver uses to turn a NeuralApproximator parameter slice
# into a callable `x -> f(x)`.
#
# Included before solver.jl so `build_param_struct` can use it.

# ─── ForwardDiff-compatible MLP ──────────────────────────────────

"""
    MLPSpec

Specification for a simple MLP (Multi-Layer Perceptron) that can be evaluated
with ForwardDiff Dual numbers. Stores layer sizes and activation functions.
"""
struct MLPSpec
    layer_sizes::Vector{Tuple{Int,Int}}  # [(n_in, n_out), ...]
    activations::Vector{Function}         # activation per layer
    n_params::Int
end

"""
Extract MLP specification from a Lux Chain model.
"""
function mlp_spec_from_lux(model::Lux.Chain)
    layers = Tuple{Int,Int}[]
    activations = Function[]
    n_params = 0

    for layer in model.layers
        if layer isa Lux.Dense
            n_in = layer.in_dims
            n_out = layer.out_dims
            push!(layers, (n_in, n_out))
            act = layer.activation
            push!(activations, act)
            n_params += n_in * n_out + n_out  # weights + bias
        else
            # Refuse ANY non-Dense layer. A parameterized one (Scale,
            # SkipConnection, nested Chain…) would leave its parameters
            # silently untrained, and even a zero-parameter one
            # (WrappedFunction, Dropout…) transforms the signal, so
            # skipping it would evaluate a different function than the
            # model defines. build_neural_evaluator catches this error
            # and falls back to Lux.apply.
            error("The Dual-safe MLP evaluator supports Chains of " *
                  "Lux.Dense layers only; the given model contains a " *
                  "$(typeof(layer)) layer. Restructure the network or " *
                  "use a solver that evaluates through Lux.apply.")
        end
    end
    # Belt-and-braces: the Dense layers must account for every parameter.
    n_params == Lux.parameterlength(model) ||
        error("The Dual-safe MLP evaluator supports Chains of Lux.Dense layers " *
              "only; the given model has $(Lux.parameterlength(model)) " *
              "parameters but its Dense layers account for $n_params. " *
              "Restructure the network or use a different solver.")
    MLPSpec(layers, activations, n_params)
end

# A bare Dense layer is a one-layer Chain for our purposes.
mlp_spec_from_lux(layer::Lux.Dense) = mlp_spec_from_lux(Lux.Chain(layer))

"""
    mlp_evaluate(spec, params, x)

Evaluate MLP with given parameter vector. Works with any numeric type
including ForwardDiff.Dual, in both the input and the parameters.

Parameters are packed as: [W1..., b1..., W2..., b2..., ...]
where Wi is column-major (n_out × n_in) — the same layout as a flattened
Lux `ComponentArray`, so slices of the global β vector plug in directly.
"""
function mlp_evaluate(spec::MLPSpec, params, x_scalar)
    T = promote_type(eltype(params), typeof(x_scalar))
    w = _mlp_width(spec)
    _mlp_forward!(Vector{T}(undef, w), Vector{T}(undef, w), spec, params, T(x_scalar))
end

_mlp_width(spec::MLPSpec) =
    maximum(((n_in, n_out),) -> max(n_in, n_out), spec.layer_sizes; init=1)

# The forward pass over two ping-pong buffers, each at least `_mlp_width`
# long. It used to build fresh arrays per layer (`act.(W * x .+ b)`): about
# 900 bytes per call for a Dense(1,4)→Dense(4,1) net, i.e. per ODE
# right-hand-side evaluation, which dominated the cost of neural fits
# (measured: 178M calls and 152 GiB in one uncapped LAML fit).
function _mlp_forward!(a::AbstractVector, b::AbstractVector, spec::MLPSpec,
                       params, x)
    a[1] = x
    offset = 0
    n = 1
    for (i, (n_in, n_out)) in enumerate(spec.layer_sizes)
        _mlp_layer!(spec.activations[i], b, a, params, offset, n_in, n_out)
        offset += n_in * n_out + n_out
        a, b = b, a
        n = n_out
    end
    n == 1 ? a[1] : a[1:n]
end

# One Dense layer, out = act.(W * inp .+ bias), with W column-major
# (n_out × n_in) at params[offset+1 : offset+n_in*n_out] and the bias after
# it. A function barrier, so the loop is specialized on the activation
# (MLPSpec stores them as abstract `Function`s).
function _mlp_layer!(act::F, out, inp, params, offset::Int, n_in::Int,
                     n_out::Int) where {F}
    boff = offset + n_in * n_out
    @inbounds for r in 1:n_out
        s = params[offset + r] * inp[1]
        for c in 2:n_in
            s += params[offset + (c - 1) * n_out + r] * inp[c]
        end
        out[r] = act(s + params[boff + r])
    end
    out
end

# Float64 scratch owned by one evaluator closure. A closure can be shared
# across threads (a parameter struct from `to_ode_problem` used in a
# threaded ensemble), so the buffers are claimed with an atomic flag and a
# concurrent or non-Float64 (Dual) call takes the allocating path instead.
mutable struct _MLPScratch
    a::Vector{Float64}
    b::Vector{Float64}
    @atomic busy::Bool
end
_MLPScratch(spec::MLPSpec) =
    (w = _mlp_width(spec); _MLPScratch(zeros(w), zeros(w), false))

function mlp_evaluate(spec::MLPSpec, params, x_scalar, sc::_MLPScratch)
    if eltype(params) === Float64 && x_scalar isa Float64 &&
       (@atomicreplace sc.busy false => true).success
        try
            return _mlp_forward!(sc.a, sc.b, spec, params, x_scalar)
        finally
            @atomic sc.busy = false
        end
    end
    mlp_evaluate(spec, params, x_scalar)
end

"""
Initialize MLP parameters matching Lux's Glorot uniform initialization.
Returns Float64 vector.
"""
function init_mlp_params(spec::MLPSpec, rng::AbstractRNG)
    params = Float64[]
    for (n_in, n_out) in spec.layer_sizes
        # Glorot uniform
        scale = sqrt(24.0 / (n_in + n_out))
        W = (rand(rng, n_out * n_in) .- 0.5) .* scale
        b = zeros(n_out)
        append!(params, W)
        append!(params, b)
    end
    params
end

# ─── Architecture probe shared by every solver ───────────────────

"""
    neural_mlp_spec(approx::NeuralApproximator) -> Union{MLPSpec, Nothing}

The Dual-safe `MLPSpec` for `approx.model`, or `nothing` when
`mlp_spec_from_lux` cannot describe the architecture (anything other than
a `Lux.Chain` of `Lux.Dense` layers — `WrappedFunction`, `Dropout`,
`SkipConnection`, `use_bias=false`, …).

`mlp_spec_from_lux` *errors* on those models by design, so every caller
must go through this probe rather than calling it directly: an
unguarded call turns an exotic-but-valid architecture into a hard error
at `solve()`, when the correct behaviour is to fall back to the
`Lux.apply` path in `build_neural_evaluator`.
"""
function neural_mlp_spec(approx::NeuralApproximator)
    try
        mlp_spec_from_lux(approx.model)
    catch e
        e isa InterruptException && rethrow()
        # Not silent: without this the only symptom of a genuine bug (a
        # typo or a Lux rename inside mlp_spec_from_lux) would be every
        # model quietly taking the slower, less Dual-safe fallback.
        @debug("Dual-safe MLP evaluator unavailable for approximator " *
               ":$(approx.name); using the Lux.apply fallback.",
               exception = e)
        nothing
    end
end

"""
    neural_init_params(approx::NeuralApproximator, rng) -> Vector{Float64}

Initial parameter vector for a `NeuralApproximator`, always
`nparams(approx)` long: Glorot draws via `init_mlp_params` for a
Dense-only chain, and Lux's own initialisation (driven by the same `rng`)
for architectures the MLP spec cannot describe. Solvers use this instead
of `init_mlp_params(mlp_spec_from_lux(...), rng)` so an exotic
architecture initialises correctly and is then evaluated through the
`build_neural_evaluator` fallback, rather than erroring out of `solve`.
"""
function neural_init_params(approx::NeuralApproximator, rng::AbstractRNG)
    spec = neural_mlp_spec(approx)
    spec === nothing || return init_mlp_params(spec, rng)
    ps, _ = Lux.setup(rng, approx.model)
    Float64.(collect(ComponentArray(ps)))
end

# ─── Unified NeuralApproximator evaluator ────────────────────────

"""
    build_neural_evaluator(approx::NeuralApproximator, params_k) -> Function

Build the callable `x -> f(x)` for a `NeuralApproximator` with parameter
slice `params_k` (flattened-`ComponentArray` layout, `[W; b]` per layer).
Applies the approximator's `domain` normalization to `[0, 1]` when set.

For a `Lux.Chain` of `Lux.Dense` layers (or a bare `Lux.Dense`) the
evaluator uses the hand-rolled `mlp_evaluate` path, which is generic in
the element types of both the input and the parameters — ForwardDiff
Dual numbers propagate through it, so stiff ODE solvers with autodiff
Jacobians work, and objectives can be differentiated w.r.t. β.

Architectures `mlp_spec_from_lux` cannot describe fall back to
`Lux.apply` with input and parameters promoted to a common element type
(no Float32 truncation); Dual-safety of that path depends on the layers
involved.
"""
function build_neural_evaluator(approx::NeuralApproximator, params_k)
    lo = approx.domain === nothing ? nothing : approx.domain[1]
    span = approx.domain === nothing ? nothing :
           (approx.domain[2] - approx.domain[1])

    # `nothing` ⇒ exotic architecture; use the Lux.apply fallback below.
    spec = neural_mlp_spec(approx)

    if spec !== nothing
        let pk = params_k, s = spec, lo_ = lo, span_ = span,
            sc = _MLPScratch(spec)
            return x -> begin
                xval = x isa AbstractArray ? x[1] : x
                xn = if lo_ !== nothing && span_ !== nothing && span_ > 0
                    (xval - lo_) / span_
                else
                    xval
                end
                mlp_evaluate(s, pk, xn, sc)
            end
        end
    end

    # Fallback: eltype-generic Lux.apply. The state and the ComponentArray
    # axes are built once; each call rebuilds the parameter ComponentArray
    # in the promoted element type so Duals in either x or params_k survive
    # (for layers whose Lux kernels accept them).
    rng = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) :
          Random.default_rng()
    _, st = Lux.setup(rng, approx.model)
    rng2 = approx.rng_seed !== nothing ? Random.Xoshiro(approx.rng_seed) :
           Random.default_rng()
    ax = getaxes(ComponentArray(Lux.initialparameters(rng2, approx.model)))
    # Cache the parameter ComponentArray for the common non-Dual case;
    # only rebuild per call when the promoted eltype differs (Dual x
    # and/or Dual params).
    ps_cached = eltype(params_k) <: AbstractFloat ?
                ComponentArray(collect(params_k), ax) : nothing
    let pk = params_k, model = approx.model, st_ = st, ax_ = ax,
        psc = ps_cached, lo_ = lo, span_ = span
        return x -> begin
            xval = x isa AbstractArray ? x[1] : x
            xn = if lo_ !== nothing && span_ !== nothing && span_ > 0
                (xval - lo_) / span_
            else
                xval
            end
            T = promote_type(typeof(xn), eltype(pk))
            ps = (psc !== nothing && T === eltype(psc)) ? psc :
                 ComponentArray(convert(Vector{T}, collect(pk)), ax_)
            out, _ = Lux.apply(model, reshape(T[xn], :, 1), ps, st_)
            length(out) == 1 ? out[1] : out
        end
    end
end
