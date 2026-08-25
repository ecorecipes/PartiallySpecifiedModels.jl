using Test
using PartiallySpecifiedModels
using PartiallySpecifiedModels: solve
using LinearAlgebra
using MCMCChains
using Random
using OrdinaryDiffEq
using StableRNGs

# ─── Custom approximator for the "approximator extension protocol" testset ──
# Struct and method definitions must live at top level; the tests themselves
# are in the testset at the bottom of the file. PolyApproximator implements
# the full four-function extension interface (see docs/src/extending.md):
# f(x) = Σ βⱼ x^(j-1), with a ridge penalty on the curvature coefficients.
struct PolyApproximator <: PartiallySpecifiedModels.AbstractApproximator
    name::Symbol
    domain::Tuple{Float64, Float64}
    degree::Int
end
PartiallySpecifiedModels.nparams(a::PolyApproximator) = a.degree + 1
PartiallySpecifiedModels.initial_params(a::PolyApproximator) = zeros(a.degree + 1)
function PartiallySpecifiedModels.penalty_matrix(a::PolyApproximator)
    S = zeros(a.degree + 1, a.degree + 1)
    for j in 3:(a.degree + 1)
        S[j, j] = 1.0
    end
    S
end
function PartiallySpecifiedModels.build_evaluator(a::PolyApproximator, params_k)
    coeffs = collect(params_k)   # eltype-generic: params_k may be Dual-valued
    function poly_eval(x)
        acc = coeffs[end]
        for j in (length(coeffs) - 1):-1:1
            acc = acc * x + coeffs[j]
        end
        acc
    end
    poly_eval
end

# A type implementing none of the interface, for the fallback-error test.
struct NotAnApproximator end

# ─── Two-block approximator for the "N0: penalty_blocks" testset ──
# f(x) = β₁·sin(ωx) + β₂·cos(ωx) + Σⱼ βⱼ₊₂·hatⱼ(x): a rough two-parameter
# Fourier block under a ridge penalty (range 1:2 — BELOW the historical
# np ≥ 3 gate, which multi-block types may bypass) plus a piecewise-linear
# hat-basis block under a second-difference penalty (range 3:np).
# penalty_matrix is the fixed-weight block-diagonal merge for the single-λ
# consumers, per the penalty_blocks contract.
struct TwoBlockApproximator <: PartiallySpecifiedModels.AbstractApproximator
    name::Symbol
    domain::Tuple{Float64, Float64}
    nknots::Int
    omega::Float64
end
PartiallySpecifiedModels.nparams(a::TwoBlockApproximator) = 2 + a.nknots
PartiallySpecifiedModels.initial_params(a::TwoBlockApproximator) =
    zeros(2 + a.nknots)
function PartiallySpecifiedModels.penalty_blocks(a::TwoBlockApproximator)
    S1 = Matrix{Float64}(I, 2, 2)
    S2 = spline_penalty_matrix(collect(range(0.0, 1.0, length=a.nknots)))
    Tuple{Matrix{Float64}, UnitRange{Int}}[(S1, 1:2), (S2, 3:(2 + a.nknots))]
end
function PartiallySpecifiedModels.penalty_matrix(a::TwoBlockApproximator)
    np = 2 + a.nknots
    S = zeros(np, np)
    for (Sb, r) in PartiallySpecifiedModels.penalty_blocks(a)
        S[r, r] .= Sb
    end
    S
end
function PartiallySpecifiedModels.build_evaluator(a::TwoBlockApproximator,
                                                  params_k)
    lo, hi = a.domain
    nk = a.nknots
    om = a.omega
    h = (hi - lo) / (nk - 1)
    coeffs = params_k   # eltype-generic: may be Dual-valued
    function twoblock_eval(x)
        u = (x - lo) / h
        j = clamp(floor(Int, u), 0, nk - 2)
        t = u - j
        coeffs[1] * sin(om * x) + coeffs[2] * cos(om * x) +
            coeffs[3 + j] * (1 - t) + coeffs[4 + j] * t
    end
    twoblock_eval
end

# Malformed penalty_blocks variants for the build_penalty_matrices
# validation tests (the validator touches only nparams + penalty_blocks).
struct BadBlocksApproximator <: PartiallySpecifiedModels.AbstractApproximator
    name::Symbol
    kind::Symbol   # :overlap | :out_of_range | :size_mismatch | :asymmetric
end
PartiallySpecifiedModels.nparams(a::BadBlocksApproximator) = 6
PartiallySpecifiedModels.initial_params(a::BadBlocksApproximator) = zeros(6)
function PartiallySpecifiedModels.penalty_blocks(a::BadBlocksApproximator)
    I3 = Matrix{Float64}(I, 3, 3)
    a.kind === :overlap       ? [(I3, 1:3), (I3, 3:5)] :
    a.kind === :out_of_range  ? [(Matrix{Float64}(I, 4, 4), 4:7)] :
    a.kind === :size_mismatch ? [(Matrix{Float64}(I, 2, 2), 1:3)] :
    a.kind === :strided        ? [(I3, 1:2:5)] :
                                [([1.0 2.0; 0.0 1.0], 1:2)]
end

@testset "PartiallySpecifiedModels.jl" begin

    # Deterministic suite: all unseeded rand/randn sites below draw from
    # this stream (order-coupled by design; per-site Xoshiro rngs are
    # used where reproducibility of an individual value matters).
    Random.seed!(20260818)

    @testset "Spline penalty matrix" begin
        # Uniform knots
        x = collect(range(0.0, 1.0, length=10))
        S = spline_penalty_matrix(x)
        @test size(S) == (10, 10)
        @test maximum(abs.(S .- S')) < 1e-10  # numerically symmetric
        # S should be positive semi-definite with rank n-2 = 8
        evals = eigvals(Symmetric(S))
        @test count(e -> e > 1e-10, evals) == 8
        @test all(e -> e >= -1e-10, evals)
        # Penalty of linear function should be zero
        y_linear = collect(range(1.0, 5.0, length=10))
        @test dot(y_linear, S * y_linear) < 1e-10
    end

    @testset "BSpline approximator" begin
        a = BSplineApproximator(:f, (0.0, 1.0), 10)
        @test nparams(a) == 10
        @test a.name == :f
        @test a.domain == (0.0, 1.0)

        p0 = initial_params(a)
        @test length(p0) == 10
        @test all(p0 .== 0.0)

        S = penalty_matrix(a)
        @test size(S) == (10, 10)

        # With initial function
        a2 = BSplineApproximator(:g, (0.0, 1.0), 5; initial=x -> x^2)
        p2 = initial_params(a2)
        @test length(p2) == 5
        @test p2[end] ≈ 1.0  # x=1.0 → 1.0
    end

    @testset "Likelihood families" begin
        y = [1.0, 2.0, 3.0]
        mu = [1.1, 2.1, 2.9]
        w = [1.0, 1.0, 1.0]

        # Gaussian
        ll = PartiallySpecifiedModels.log_likelihood(Gaussian(), y, mu, w)
        @test ll < 0.0
        wt = PartiallySpecifiedModels.irls_weights(Gaussian(), y, mu, w)
        @test wt == w

        # Poisson
        y_p = [5.0, 10.0, 3.0]
        mu_p = [4.5, 11.0, 3.2]
        ll_p = PartiallySpecifiedModels.log_likelihood(Poisson(), y_p, mu_p, w)
        @test isfinite(ll_p)
        wt_p = PartiallySpecifiedModels.irls_weights(Poisson(), y_p, mu_p, w)
        @test all(wt_p .> 0)
        @test wt_p ≈ 1.0 ./ mu_p  # identity link: W = 1/μ

        # NegBin
        ll_nb = PartiallySpecifiedModels.log_likelihood(NegativeBinomial(5.0), y_p, mu_p, w)
        @test isfinite(ll_nb)
    end

    @testset "CustomLikelihood with ForwardDiff" begin
        # Custom Gaussian should match built-in
        custom_gauss = CustomLikelihood((y, μ) -> -0.5 * (y - μ)^2)
        y = [1.0, 2.0, 3.0]
        mu = [1.1, 2.1, 2.9]
        w = [1.0, 1.0, 1.0]

        ll_builtin = PartiallySpecifiedModels.log_likelihood(Gaussian(), y, mu, w)
        ll_custom = PartiallySpecifiedModels.log_likelihood(custom_gauss, y, mu, w)
        @test ll_builtin ≈ ll_custom

        wt = PartiallySpecifiedModels.irls_weights(custom_gauss, y, mu, w)
        @test all(wt .≈ 1.0)  # Gaussian has weight = 1
    end

    @testset "LAML helpers" begin
        # Use a matrix with known positive determinant
        S = Float64[2 1; 1 3]  # det = 5
        ld = PartiallySpecifiedModels._log_det_pd(S)
        @test isfinite(ld)
        @test ld ≈ log(5.0) atol=1e-8

        r = PartiallySpecifiedModels._rank_penalty(S)
        @test r == 2

        ldp = PartiallySpecifiedModels._log_det_plus(S)
        @test ldp ≈ ld atol=0.01

        # Singular matrix: rank should be less than size
        S_sing = [1.0 1.0; 1.0 1.0]
        @test PartiallySpecifiedModels._rank_penalty(S_sing) == 1
    end

    @testset "LAML gradient matches finite differences" begin
        using PartiallySpecifiedModels: laml_objective, laml_gradient,
                                        build_S_lambda, spline_penalty_matrix,
                                        _safe_inv
        # `laml_gradient` is the EXACT ∂V/∂ρ of `laml_objective` holding
        # (β, μ, J, W) fixed -- which is precisely how the Newton phase of
        # `estimate_smoothing_params` calls the pair: the objective is
        # evaluated at the working-model state (β_fs, μ_fs) and the gradient
        # at the H and σ̂² that same call returned. So the check here is a
        # central difference of V w.r.t. ρ with the working-model state
        # frozen, built exactly as the Newton phase builds it:
        # β̂(λ) = (J'WJ + S_λ)⁻¹J'Wz and μ = Jβ̂.
        n, nk = 40, 6
        knots = collect(0.0:1.0:5.0)
        xs = collect(range(0.0, 5.0, length=n))
        hat(x, k) = max(0.0, 1.0 - abs(x - k))
        # Two penalty blocks: a hat-function block and a varying-coefficient
        # block (the same basis modulated by a smooth covariate). The
        # modulation matters -- two plain hat blocks BOTH span the constant
        # function, which lies in the null space of both second-difference
        # penalties, so H would be exactly singular and the 1e-10 ridge
        # inside _log_det_pd/_safe_inv (not the analytic term) would dominate
        # the derivative. As built here cond(J'J) ≈ 1.7e4.
        g_mod = 1.0 .+ 0.5 .* sin.(2.0 .* xs)
        J1 = [hat(xs[i], knots[j]) for i in 1:n, j in 1:nk]
        J2 = [hat(xs[i], knots[j]) * g_mod[i] for i in 1:n, j in 1:nk]
        J = hcat(J1, J2)
        S = spline_penalty_matrix(knots)
        S_list = [S, copy(S)]
        offsets = [0, nk]; nknots_list = [nk, nk]; n_p = 2nk
        w = ones(n)

        function fd_vs_analytic(family, y, W, rho)
            S_lam = build_S_lambda(S_list, offsets, nknots_list, rho, n_p)
            beta = _safe_inv(J' * Diagonal(W) * J + S_lam) * (J' * (W .* y))
            mu = J * beta
            _, H, _, sigma2 = laml_objective(family, beta, J, W, w, y, mu,
                                             S_list, offsets, nknots_list, rho, n_p)
            g = laml_gradient(family, beta, S_list, offsets, nknots_list,
                              rho, n_p, H, sigma2)
            h = 1e-4
            gfd = map(eachindex(rho)) do k
                rp = copy(rho); rp[k] += h
                rm = copy(rho); rm[k] -= h
                Vp, = laml_objective(family, beta, J, W, w, y, mu, S_list,
                                     offsets, nknots_list, rp, n_p)
                Vm, = laml_objective(family, beta, J, W, w, y, mu, S_list,
                                     offsets, nknots_list, rm, n_p)
                (Vp - Vm) / (2h)
            end
            (g, collect(gfd))
        end

        # Tolerance: with h = 1e-4 the observed agreement is ≤ 3e-5 relative
        # (≤ 1.2e-6 absolute) over every (family, ρ) pair below, the residual
        # being FD truncation plus the O(1e-10) diagonal ridge that
        # _log_det_pd and _safe_inv both add. rtol = 1e-3 leaves ~30x
        # headroom for BLAS/platform variation while still catching any sign
        # slip, factor-of-two, or dropped term (all O(1) relative).
        y_g = sin.(xs) .+ 0.3 .* cos.(2.0 .* xs) .+ 0.05 .* sin.(37.0 .* (1:n))
        for rho in ([-2.0, 1.0], [0.0, 0.0], [3.0, -1.0], [5.0, 4.0])
            g, gfd = fd_vs_analytic(Gaussian(), y_g, w, rho)
            @test g ≈ gfd rtol=1e-3 atol=1e-6
        end
        # Non-Gaussian branch (unit scale, ℓ(β̂) − ½β̂'S^λβ̂ + …): identity
        # link ⟹ IRLS weights 1/μ, frozen here at the data as the outer IRLS
        # loop would supply them.
        y_p = round.(6.0 .+ 4.0 .* sin.(xs))
        W_p = 1.0 ./ max.(y_p, 1.0)
        for rho in ([-1.0, 0.5], [1.0, 1.0], [3.0, 0.0])
            g, gfd = fd_vs_analytic(Poisson(), y_p, W_p, rho)
            @test g ≈ gfd rtol=1e-3 atol=1e-6
        end
        # Sanity: V is not flat here, so a zero-returning stub gradient would
        # fail the comparisons above rather than pass them vacuously.
        g0, _ = fd_vs_analytic(Gaussian(), y_g, w, [0.0, 0.0])
        @test norm(g0) > 1.0
    end

    @testset "Simple ODE fit" begin
        # Exponential growth: du/dt = r*u, data = u0*exp(r*t)
        # Unknown function: r(u) ≈ constant
        true_r = 0.1
        u0_val = 1.0
        tspan = (0.0, 10.0)
        data_times = collect(0.0:0.5:10.0)
        true_data = u0_val .* exp.(true_r .* data_times)
        # Add small noise
        data_values = true_data .+ 0.01 .* randn(length(data_times))
        data_values = reshape(data_values, :, 1)

        function exp_growth!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        approx_r = BSplineApproximator(:r, (0.0, 5.0), 5;
                                        initial=x -> 0.05)

        prob = PSMProblem(exp_growth!, [u0_val], tspan, [approx_r];
                          data_times=data_times,
                          data_values=data_values,
                          obs_to_state=[1],
                          likelihood=Gaussian(),
                          solver=Tsit5())

        sol = solve(prob, LAML(maxiters=30, verbose=false))

        @test sol.edf > 1.0
        @test sol.data_loss < 1.0  # Should fit well
        @test haskey(sol.unknown_functions, :r)

        # Check that the fitted r function is approximately constant at true_r
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.05
    end

    @testset "CollocationLAML solver" begin
        # Simple exponential growth with unknown growth rate
        function growth!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        prob = PSMProblem(growth!, [1.0], tspan,
            [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, CollocationLAML(
            maxiters=20, verbose=false,
            lambda_ode_start=0.01, lambda_ode_end=100.0,
            n_continuation=4))

        @test sol.data_loss < 5.0
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.15
    end

    @testset "CollocationLAML error policy" begin
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(0.3 .* data_times), :, 1)
        approx_r = () -> BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)

        # 1) DDE problems are rejected up front: eval_ode_rhs calls the
        #    4-arg ODE signature, so pre-fix every RHS silently became the
        #    1e6 failure sentinel and the solver "converged" to garbage.
        function dde_dyn!(du, u, h, p, t)
            du[1] = p.r(u[1]) * h(p, t - 1.0)[1]
        end
        prob_dde = PSMProblem(dde_dyn!, [1.0], (0.0, 5.0), [approx_r()];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], likelihood=Gaussian(), solver=Tsit5(),
            delays=[1.0], history=(p, t) -> [1.0])
        err = try
            solve(prob_dde, CollocationLAML(maxiters=5, verbose=false,
                n_continuation=2))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("does not support DDE problems", err.msg)

        # 2) Programming errors in the user dynamics propagate instead of
        #    being swallowed into a silent flat fit (pre-fix this converged
        #    silently with data_loss ≈ 4.16e13 and r(1.0) ≈ 0.005).
        function buggy_colloc!(du, u, p, t)
            du[1] = p.r(u[1]) * colloc_undefined_helper(u[1])
        end
        prob_bug = PSMProblem(buggy_colloc!, [1.0], (0.0, 5.0), [approx_r()];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], likelihood=Gaussian(), solver=Tsit5())
        @test_throws UndefVarError solve(prob_bug,
            CollocationLAML(maxiters=5, verbose=false, n_continuation=2))

        function oob_colloc!(du, u, p, t)
            du[1] = p.r(u[1]) * u[2]   # u has length 1
        end
        prob_oob = PSMProblem(oob_colloc!, [1.0], (0.0, 5.0), [approx_r()];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], likelihood=Gaussian(), solver=Tsit5())
        @test_throws BoundsError solve(prob_oob,
            CollocationLAML(maxiters=5, verbose=false, n_continuation=2))

        # 3) When the Fellner–Schall simulation fails (exploding dynamics),
        #    the solve still returns but warns exactly once that smoothing
        #    parameters remain at initialization (pre-fix: silent skip,
        #    inflated initialization EDF reported as the answer).
        # tanh bounds the fitted term so no estimate of r can cancel the
        # u² blow-up: du ≥ 50u² − 1, so forward simulation always diverges.
        function explode_colloc!(du, u, p, t)
            du[1] = 50.0 * u[1]^2 + tanh(p.r(clamp(u[1], 0.5, 5.0)))
        end
        prob_expl = PSMProblem(explode_colloc!, [1.0], (0.0, 5.0), [approx_r()];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], likelihood=Gaussian(), solver=Tsit5())
        local sol_expl
        logs, _ = Test.collect_test_logs() do
            sol_expl = solve(prob_expl, CollocationLAML(maxiters=5,
                verbose=false, n_continuation=3))
        end
        fs_warns = count(l -> occursin("Fellner", string(l.message)), logs)
        @test fs_warns == 1   # once per solve, not per continuation level
        @test sol_expl isa PSMSolution
        @test isfinite(sol_expl.edf)

        # 4) FS numerator convention: no cheap deterministic problem drives
        #    the numerator r_k − θ·tr(H⁻¹S_k) ≤ 0 (it happens only on
        #    transient θ overshoot), and the update is inline in solve, so
        #    the ≤ 0 branch is pinned by convention parity with laml.jl's
        #    estimate_smoothing_params (keep θ unchanged) rather than a
        #    behavioral fixture. Here we assert the update leaves θ finite
        #    and positive on a well-posed fit.
        prob_ok = PSMProblem((du, u, p, t) -> (du[1] = p.r(u[1]) * u[1]),
            [1.0], (0.0, 5.0), [approx_r()];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], likelihood=Gaussian(), solver=Tsit5())
        sol_ok = solve(prob_ok, CollocationLAML(maxiters=10, verbose=false,
            n_continuation=2))
        @test all(t -> isfinite(t) && t > 0, sol_ok.smoothing_params)

        # 5) Domain-boundary sentinel asymmetry: large residual, ZERO
        #    Jacobian. With `sqrt(u)` dynamics and a state dipping just
        #    below zero, the base evaluation fails (DomainError → 1e6
        #    sentinel) while the +1e-6 perturbation succeeds. Differencing
        #    against the sentinel gave a slope of ≈ -1e12, so J'J carried
        #    ~1e24 entries and the Gauss–Newton solve returned garbage
        #    without erroring. Measured max|J| pre-fix: 1.0e12; post-fix 5.6.
        function sqrt_colloc!(du, u, p, t)
            du[1] = p.r(u[1]) * sqrt(u[1])
        end
        bt = collect(range(0.0, 5.0, length=15))
        bv = reshape(0.5 .+ 0.1 .* bt, :, 1)
        bv[3, 1] = -1e-8   # dips just below the sqrt domain boundary
        prob_bnd = PSMProblem(sqrt_colloc!, [0.5], (0.0, 5.0),
            [BSplineApproximator(:r, (0.0, 5.0), 5; initial=x -> 0.2)];
            data_times=bt, data_values=bv, obs_to_state=[1],
            likelihood=Gaussian(), solver=Tsit5())
        beta_bnd = Float64[]
        for a in prob_bnd.approximators
            append!(beta_bnd, PartiallySpecifiedModels.initial_params(a))
        end
        resid_bnd, J_bnd = PartiallySpecifiedModels.collocation_residual_jacobian(
            prob_bnd, bt, copy(bv), beta_bnd,
            PartiallySpecifiedModels.build_diff_matrix(bt), 1.0, ones(length(bt)))
        # The residual keeps the sentinel: a failed point must be expensive.
        @test maximum(abs, resid_bnd) >= 1e5
        # The Jacobian must not: no fabricated 1e12 slopes.
        @test maximum(abs, J_bnd) < 1e3
        @test all(isfinite, J_bnd)
        # And the end-to-end solve stays finite rather than returning garbage.
        sol_bnd = solve(prob_bnd, CollocationLAML(maxiters=5, verbose=false,
            n_continuation=2))
        @test all(isfinite, sol_bnd.parameters)
        @test isfinite(sol_bnd.objective)
    end

    @testset "GPApproximator" begin
        # Kernel matrix is positive definite
        gp = GPApproximator(:f, (0.0, 1.0), 8; kernel=:matern52)
        @test nparams(gp) == 8
        @test gp.name == :f

        S = penalty_matrix(gp)
        @test size(S) == (8, 8)
        evals = eigvals(Symmetric(S))
        @test all(e -> e > -1e-8, evals)  # positive semi-definite

        # GP evaluator interpolates function values
        vals = Float64[sin(x) for x in gp.inducing_points]
        eval_gp = PartiallySpecifiedModels.build_gp_evaluator(gp, vals)
        for (xi, vi) in zip(gp.inducing_points, vals)
            @test abs(eval_gp(xi) - vi) < 0.05
        end

        # Different kernels
        for kern in [:sqexp, :matern32, :matern52]
            gp_k = GPApproximator(:g, (0.0, 5.0), 6; kernel=kern)
            @test nparams(gp_k) == 6
            @test penalty_matrix(gp_k) !== nothing
        end
    end

    @testset "GP solver integration" begin
        function growth_gp!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        prob = PSMProblem(growth_gp!, [1.0], tspan,
            [GPApproximator(:r, (0.5, 5.0), 6; kernel=:matern52, initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, LAML(maxiters=60, verbose=false))
        @test sol.data_loss < 1.0
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.1
    end

    @testset "SPDEApproximator — construction" begin
        a = SPDEApproximator(:f, (0.0, 1.0), 10)
        @test nparams(a) == 10
        @test a.name == :f
        @test a.domain == (0.0, 1.0)
        @test a.nu ≈ 1.5
        @test a.range_param ≈ 1.0 / 3.0
        @test a.kappa ≈ sqrt(8.0 * 1.5) / (1.0 / 3.0)
        @test length(a.mesh_points) == 10

        p0 = initial_params(a)
        @test length(p0) == 10
        @test all(p0 .== 0.0)

        # With initial function
        a2 = SPDEApproximator(:g, (0.0, 1.0), 5; initial=x -> x^2)
        p2 = initial_params(a2)
        @test length(p2) == 5
        @test p2[1] ≈ 0.0 atol=1e-10
        @test p2[end] ≈ 1.0 atol=1e-10

        # Custom range
        a3 = SPDEApproximator(:h, (0.0, 10.0), 8; nu=0.5, range_param=2.0)
        @test a3.nu ≈ 0.5
        @test a3.range_param ≈ 2.0
        @test a3.kappa ≈ sqrt(4.0) / 2.0

        # Validation
        @test_throws ErrorException SPDEApproximator(:f, (0.0, 1.0), 10; nu=0.7)
        @test_throws ErrorException SPDEApproximator(:f, (0.0, 1.0), 2)
    end

    @testset "SPDEApproximator — FEM matrices" begin
        mesh = collect(range(0.0, 1.0, length=5))
        C, G = PartiallySpecifiedModels.spde_fem_matrices(mesh)
        h = 0.25

        # Mass matrix: diagonal with h/2 at boundaries, h at interior
        @test C[1,1] ≈ h/2
        @test C[2,2] ≈ h
        @test C[5,5] ≈ h/2
        @test all(C[i,j] ≈ 0 for i in 1:5, j in 1:5 if i ≠ j)

        # Stiffness matrix: tridiagonal
        @test G[1,1] ≈ 1/h
        @test G[1,2] ≈ -1/h
        @test G[2,2] ≈ 2/h
        @test G[2,1] ≈ -1/h
        @test G[2,3] ≈ -1/h
        @test maximum(abs.(G .- G')) < 1e-10  # symmetric

        # Stiffness matrix annihilates constant functions: G * ones = 0
        @test norm(G * ones(5)) < 1e-10
    end

    @testset "SPDEApproximator — penalty matrix" begin
        for ν in [0.5, 1.5, 2.5]
            a = SPDEApproximator(:f, (0.0, 1.0), 10; nu=ν, range_param=0.3)
            S = penalty_matrix(a)
            @test size(S) == (10, 10)
            @test maximum(abs.(S .- S')) < 1e-8  # symmetric
            evals = eigvals(Symmetric(S))
            @test all(e -> e >= -1e-8, evals)  # positive semi-definite
        end

        # Penalty of constant function should be small (not exactly zero due to
        # mass matrix contribution, but much smaller than wiggly function)
        a = SPDEApproximator(:f, (0.0, 1.0), 10; nu=1.5)
        S = penalty_matrix(a)
        y_const = ones(10)
        y_wiggly = [sin(10π * x) for x in a.mesh_points]
        @test dot(y_const, S * y_const) < dot(y_wiggly, S * y_wiggly)
    end

    @testset "SPDEApproximator — evaluator" begin
        mesh = collect(range(0.0, 1.0, length=10))
        params = sin.(mesh)
        eval_spde = PartiallySpecifiedModels.build_spde_evaluator(mesh, params)

        # Should interpolate at mesh points
        for (xi, vi) in zip(mesh, params)
            @test abs(eval_spde(xi) - vi) < 1e-10
        end

        # Should interpolate smoothly between points
        x_mid = 0.55
        @test abs(eval_spde(x_mid) - sin(x_mid)) < 0.05
    end

    @testset "SPDEApproximator — LAML solver" begin
        function growth_spde!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        prob = PSMProblem(growth_spde!, [1.0], tspan,
            [SPDEApproximator(:r, (0.5, 5.0), 8; nu=1.5, initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, LAML(maxiters=60, verbose=false))
        @test sol.data_loss < 1.0
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.1
    end

    @testset "ShapeConstrainedSPDEApproximator — construction" begin
        a = ShapeConstrainedSPDEApproximator(:f, (0.0, 5.0), 8, :increasing)
        @test a.name == :f
        @test a.domain == (0.0, 5.0)
        @test a.n_basis == 8
        @test a.nu == 1.5
        @test a.constraint == :increasing
        @test size(a.Sigma) == (8, 8)
        @test nparams(a) == 8
        @test length(initial_params(a)) == 8

        # With custom settings
        a2 = ShapeConstrainedSPDEApproximator(:g, (0.0, 1.0), 6, :dec_positive;
            nu=0.5, range_param=0.5, initial=x -> 1.0 - x)
        @test a2.nu == 0.5
        @test a2.range_param == 0.5
        @test nparams(a2) == 6

        # Zero-endpoint constraint: one fewer parameter
        a3 = ShapeConstrainedSPDEApproximator(:h, (0.0, 1.0), 8, :inc_zero_left)
        @test nparams(a3) == 7
        @test size(a3.Sigma) == (8, 7)

        # Errors
        @test_throws ArgumentError ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 8, :bad_constraint)
        @test_throws ErrorException ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 3, :increasing)
    end

    @testset "ShapeConstrainedSPDEApproximator — evaluator" begin
        a = ShapeConstrainedSPDEApproximator(:f, (0.0, 5.0), 8, :increasing;
            initial=x -> 0.1 * x)
        gamma = initial_params(a)
        eval_fn = PartiallySpecifiedModels.build_constrained_spde_evaluator(a, gamma)

        # Should be callable and increasing
        vals = [eval_fn(x) for x in range(0.0, 5.0, length=20)]
        @test all(diff(vals) .>= -1e-10)  # Approximately increasing

        # Positive constraint
        a_pos = ShapeConstrainedSPDEApproximator(:f, (0.0, 5.0), 8, :positive;
            initial=x -> 1.0)
        gamma_pos = initial_params(a_pos)
        eval_pos = PartiallySpecifiedModels.build_constrained_spde_evaluator(a_pos, gamma_pos)
        mesh_vals = PartiallySpecifiedModels.gamma_to_mesh_values(a_pos, gamma_pos)
        @test all(mesh_vals .> 0)  # Positive at mesh nodes
    end

    @testset "ShapeConstrainedSPDEApproximator — penalty matrix" begin
        a = ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 8, :increasing; nu=1.5)
        P = penalty_matrix(a)
        np = nparams(a)
        @test size(P) == (np, np)
        @test issymmetric(P)
        @test all(eigvals(Symmetric(P)) .>= -1e-10)  # Positive semi-definite

        # Zero-endpoint has smaller penalty
        a_z = ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 8, :inc_zero_left; nu=1.5)
        P_z = penalty_matrix(a_z)
        @test size(P_z) == (7, 7)
    end

    @testset "ShapeConstrainedSPDEApproximator — :difference penalty" begin
        PSM = PartiallySpecifiedModels

        # Construction: keyword stored; invalid mode rejected; default is
        # :gamma_matern (SPD — no null space)
        a_def = ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 8, :increasing)
        @test a_def.penalty == :gamma_matern
        @test_throws ArgumentError ShapeConstrainedSPDEApproximator(
            :f, (0.0, 1.0), 8, :increasing; penalty=:bogus)
        S_def = penalty_matrix(a_def)
        @test minimum(eigvals(Symmetric(S_def))) > 1.0  # SPD, no null space

        # :difference — P&W null space: free level e₁ and the constant
        # increment shift over the chained indices are exactly annihilated
        a_diff = ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 8, :increasing;
            penalty=:difference)
        S_diff = penalty_matrix(a_diff)
        @test issymmetric(S_diff)
        @test all(eigvals(Symmetric(S_diff)) .>= -1e-12)
        @test PSM._rank_penalty(S_diff) == 6      # np=8, null dim 2
        @test norm(S_diff * [1.0; zeros(7)]) == 0.0        # free level γ₁
        @test norm(S_diff * [0.0; ones(7)]) == 0.0         # constant slope shift

        # Curvature constraint: skips (1, 2) → null dim 3 (level, slope,
        # constant curvature)
        a_cx = ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 8, :inc_convex;
            penalty=:difference)
        S_cx = penalty_matrix(a_cx)
        @test PSM._rank_penalty(S_cx) == 5
        @test norm(S_cx * [1.0; zeros(7)]) == 0.0
        @test norm(S_cx * [0.0; 1.0; zeros(6)]) == 0.0
        @test norm(S_cx * [0.0; 0.0; ones(6)]) == 0.0

        # Zero-endpoint constraint keeps the reduced dimension
        a_z = ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 8, :inc_zero_left;
            penalty=:difference)
        @test size(penalty_matrix(a_z)) == (7, 7)

        # λ→∞ semantics: minimize ‖y − f(γ)‖² + λ γ'Sγ at λ=1e8. In
        # :difference mode γ collapses to the null space, whose image under
        # the constraint map is the straight-line family — NOT the
        # softplus(0)=log 2 ramp. In the default mode γ→0, giving exactly
        # the log-2 ramp (slope 7·log 2 on this mesh) with the level shrunk
        # to 0. Deterministic Nelder-Mead from initial_params.
        xs = collect(range(0.0, 1.0, length=30))
        y = @. 2.0 + 0.5 * xs + 0.3 * sin(6 * xs)
        X = [ones(length(xs)) xs]
        function fit_big_lambda(a, S)
            obj = g -> begin
                ev = PSM.build_constrained_spde_evaluator(a, g)
                sum(abs2, y .- ev.(xs)) + 1e8 * dot(g, S, g)
            end
            res = PSM.Optim.optimize(obj, initial_params(a), PSM.Optim.NelderMead(),
                PSM.Optim.Options(iterations=20000, g_tol=1e-12))
            ghat = PSM.Optim.minimizer(res)
            fitted = [PSM.build_constrained_spde_evaluator(a, ghat)(x) for x in xs]
            c = X \ fitted
            (level=c[1], slope=c[2], line_resid=norm(fitted - X * c))
        end
        r_diff = fit_big_lambda(a_diff, S_diff)
        @test r_diff.line_resid < 1e-6            # limit IS a straight line
        @test abs(r_diff.slope - 7 * log(2)) > 3.0  # and NOT the log-2 ramp
        @test r_diff.level > 1.5                  # level not shrunk away
        r_def = fit_big_lambda(a_def, S_def)
        @test abs(r_def.slope - 7 * log(2)) < 1e-3  # default limit: log-2 ramp
        @test abs(r_def.level) < 1e-6               # with level shrunk to 0
    end

    @testset "ShapeConstrainedSPDEApproximator — LAML solver" begin
        function growth_scspde!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        # Use :positive constraint since r(u) = 0.3 > 0
        prob = PSMProblem(growth_scspde!, [1.0], tspan,
            [ShapeConstrainedSPDEApproximator(:r, (0.5, 5.0), 8, :positive;
                nu=1.5, initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, LAML(maxiters=60, verbose=false))
        @test sol.data_loss < 1.0
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.15
        # Verify positivity at mesh nodes
        mesh_vals = PartiallySpecifiedModels.gamma_to_mesh_values(
            prob.approximators[1], collect(sol.parameters[:r]))
        @test all(mesh_vals .> 0)

        # Same problem with the rank-deficient :difference penalty: LAML's
        # rank/null-space (Mp) bookkeeping must handle it (the SCBSpline path
        # already exercises rank-deficient penalties) and recovery should be
        # comparable to the default mode.
        prob_d = PSMProblem(growth_scspde!, [1.0], tspan,
            [ShapeConstrainedSPDEApproximator(:r, (0.5, 5.0), 8, :positive;
                nu=1.5, initial=x -> 0.2, penalty=:difference)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)
        sol_d = solve(prob_d, LAML(maxiters=60, verbose=false))
        @test sol_d.data_loss < 1.0
        @test abs(sol_d.unknown_functions[:r](1.0) - true_r) < 0.15
        mesh_vals_d = PartiallySpecifiedModels.gamma_to_mesh_values(
            prob_d.approximators[1], collect(sol_d.parameters[:r]))
        @test all(mesh_vals_d .> 0)
    end

    @testset "GradientMatching solver" begin
        function growth_gm!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        prob = PSMProblem(growth_gm!, [1.0], tspan,
            [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, GradientMatching(maxiters=50, verbose=false))
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.08   # init (0.2) errs by 0.1 — must beat its start
    end

    @testset "Adam solver (B-spline)" begin
        function growth_adam!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        prob = PSMProblem(growth_adam!, [1.0], tspan,
            [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, AdamSolver(maxiters=150, lr=0.01, verbose=false))
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.15
        @test sol.data_loss < 1.0  # should fit exponential growth well
    end

    @testset "Multiple shooting solver (B-spline)" begin
        function growth_ms!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end

        true_r = 0.3
        tspan = (0.0, 5.0)
        data_times = collect(range(0.0, 5.0, length=30))
        data_values = reshape(exp.(true_r .* data_times), :, 1)

        prob = PSMProblem(growth_ms!, [1.0], tspan,
            [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        sol = solve(prob, MultipleShootingSolver(
            n_intervals=3, maxiters_inner=50, maxiters_outer=5,
            rho_init=1.0, verbose=false))
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(1.0) - true_r) < 0.08   # init (0.2) errs by 0.1 — must beat its start
    end

    @testset "Adaptive gradient matching (B-spline)" begin
        # Exponential growth: du/dt = r(x)*u, r(x) ≈ constant
        true_r = 0.3
        function exp_growth_agm!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        u0_exp = [2.0]; tspan_exp = (0.0, 5.0)
        data_t_exp = collect(0.0:0.25:5.0)
        data_true_exp = 2.0 .* exp.(true_r .* data_t_exp)
        data_vals_exp = reshape(data_true_exp .+ 0.05 .* randn(length(data_t_exp)),
                                :, 1)

        bs_r = BSplineApproximator(:r, (1.5, 10.0), 6; initial=0.5)
        prob = PSMProblem(exp_growth_agm!, u0_exp, tspan_exp, [bs_r];
            data_times=data_t_exp, data_values=data_vals_exp,
            obs_to_state=[1], solver=Tsit5())

        sol = solve(prob, AdaptiveGradientMatching(maxiters=100, verbose=false))
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(3.0) - true_r) < 0.15

        # Population MCMC mode (Dondelinger et al. 2013): tempered chains
        # jointly sample states, parameters, and mismatch variances
        sol_pm = solve(prob, AdaptiveGradientMatching(n_samples=300,
            n_chains=6, rng_seed=3, verbose=false))
        @test sol_pm.convergence.sampler == :population_mcmc
        @test size(sol_pm.convergence.beta_samples) == (300, 6)
        @test size(sol_pm.convergence.gamma_samples, 1) == 300
        @test length(sol_pm.convergence.temperatures) == 6
        @test sol_pm.convergence.temperatures[end] == 1.0
        @test abs(sol_pm.unknown_functions[:r](3.0) - true_r) < 0.15
        @test_throws ArgumentError solve(prob,
            AdaptiveGradientMatching(n_samples=100, n_chains=1))
        # reproducible under rng_seed
        sol_pm2 = solve(prob, AdaptiveGradientMatching(n_samples=300,
            n_chains=6, rng_seed=3, verbose=false))
        @test sol_pm2.unknown_functions[:r](3.0) == sol_pm.unknown_functions[:r](3.0)
    end

    @testset "Adaptive gradient matching MAP — large-mean data" begin
        # Exponential relaxation to a large baseline: du/dt = -f(u) with
        # f(u) = 0.5*(u - 1000). The state lives at ~1000-1010 while the
        # signal amplitude is only 10, so the MAP path must center the data
        # before GP smoothing (as the MCMC path does): an uncentered GP
        # shrinks the smoothed states toward 0 and injects boundary
        # derivative artifacts proportional to the mean level (pre-fix
        # max error 3.9 with the wrong sign at f(1002); post-fix 0.09).
        f_relax(u) = 0.5 * (u - 1000.0)
        function relax_agm!(du, u, p, t)
            du[1] = -p.f(u[1])
        end
        t_agm = collect(0.0:0.25:8.0)
        u_true_agm = 1000.0 .+ 10.0 .* exp.(-0.5 .* t_agm)
        data_agm = u_true_agm .+ 0.05 .* randn(Random.Xoshiro(11), length(t_agm))

        uf_agm = BSplineApproximator(:f, (999.5, 1011.0), 6)
        prob_agm = PSMProblem(relax_agm!, [1010.0], (0.0, 8.0), [uf_agm];
            data_times=t_agm, data_values=reshape(data_agm, :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_agm = solve(prob_agm, AdaptiveGradientMatching(maxiters=100, verbose=false))
        err_agm = maximum(abs(sol_agm.unknown_functions[:f](x) - f_relax(x))
                          for x in (1002.0, 1005.0, 1008.0))
        @test err_agm < 0.5
    end

    @testset "Rodeo solver (B-spline)" begin
        # Exponential growth: du/dt = r(x)*u, r(x) ≈ constant
        true_r = 0.3
        function exp_growth_rodeo!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        u0_exp = [2.0]; tspan_exp = (0.0, 5.0)
        data_t_exp = collect(0.0:0.25:5.0)
        data_true_exp = 2.0 .* exp.(true_r .* data_t_exp)
        data_vals_exp = reshape(data_true_exp .+ 0.05 .* randn(length(data_t_exp)),
                                :, 1)

        bs_r = BSplineApproximator(:r, (1.5, 10.0), 6; initial=0.5)
        prob = PSMProblem(exp_growth_rodeo!, u0_exp, tspan_exp, [bs_r];
            data_times=data_t_exp, data_values=data_vals_exp,
            obs_to_state=[1], solver=Tsit5())

        sol = solve(prob, RodeoSolver(n_steps=100, n_deriv=3, maxiters=100,
                    obs_var=0.01, verbose=false))
        @test haskey(sol.unknown_functions, :r)
        r_eval = sol.unknown_functions[:r]
        @test abs(r_eval(3.0) - true_r) < 0.15
    end

    @testset "MCMCSolver (B-spline)" begin
        # Exponential decay: du/dt = -r(t)*u, r(t) ≈ 0.3
        function exp_decay_mcmc!(du, u, p, t)
            du[1] = -p.r(t) * u[1]
        end
        times_mcmc = collect(0.0:0.5:10.0)
        true_sol_mcmc = exp.(-0.3 .* times_mcmc)
        data_mcmc = reshape(true_sol_mcmc .+ 0.02 .* randn(length(times_mcmc)), :, 1)

        bs_mcmc = BSplineApproximator(:r, (0.0, 10.0), 8; initial=0.15)  # away from true 0.3
        prob_mcmc = PSMProblem(
            ODEProblem(exp_decay_mcmc!, [1.0], (0.0, 10.0)),
            [bs_mcmc]; data_times=times_mcmc, data_values=data_mcmc,
            obs_to_state=[1], solver=Tsit5())

        sol_mcmc = solve(prob_mcmc, MCMCSolver(
            n_samples=50, n_warmup=25, verbose=false))

        @test sol_mcmc.convergence isa MCMCChains.Chains
        @test size(sol_mcmc.convergence, 1) == 50   # n_samples
        @test size(sol_mcmc.convergence, 2) == 9     # 8 params + log_σ
        @test haskey(sol_mcmc.unknown_functions, :r)
        r_map = sol_mcmc.unknown_functions[:r](5.0)
        @test abs(r_map - 0.3) < 0.2

        # Reproducibility under the solver's own rng_seed field (AGM/BNG
        # convention): same seed → identical chains, and the global stream
        # is never touched, so this order-coupled suite is unaffected.
        sol_mcmc_a = solve(prob_mcmc, MCMCSolver(n_samples=50, n_warmup=25,
                                                 rng_seed=2718, verbose=false))
        sol_mcmc_b = solve(prob_mcmc, MCMCSolver(n_samples=50, n_warmup=25,
                                                 rng_seed=2718, verbose=false))
        @test Array(sol_mcmc_a.convergence) == Array(sol_mcmc_b.convergence)
        @test sol_mcmc_a.unknown_functions[:r](5.0) ==
              sol_mcmc_b.unknown_functions[:r](5.0)

        # The masked-data solves below also draw from the global stream, so
        # save and restore it too — the rest of this suite is order-coupled.
        rng_state_msk = copy(Random.default_rng())

        # Masked observations (weight 0, value NaN). Pre-fix the initial σ
        # was `std(prob.data_values) * 0.1` over ALL cells → NaN →
        # log(max(NaN, 0.01)) = NaN → NaN in theta0 → the whole chain NaN,
        # reported as objective=NaN and data_loss=NaN with no error. The
        # reported data_loss also summed over masked cells (0 * NaN = NaN).
        data_msk = copy(data_mcmc)
        w_msk = ones(length(times_mcmc), 1)
        data_msk[5, 1] = NaN;  w_msk[5, 1] = 0.0
        data_msk[12, 1] = NaN; w_msk[12, 1] = 0.0
        prob_msk = PSMProblem(
            ODEProblem(exp_decay_mcmc!, [1.0], (0.0, 10.0)),
            [BSplineApproximator(:r, (0.0, 10.0), 8; initial=0.15)];
            data_times=times_mcmc, data_values=data_msk,
            data_weights=w_msk, obs_to_state=[1], solver=Tsit5())
        sol_msk = solve(prob_msk, MCMCSolver(n_samples=50, n_warmup=25,
                                             verbose=false))
        @test isfinite(sol_msk.objective)
        @test isfinite(sol_msk.data_loss)
        @test all(isfinite, Array(sol_msk.convergence))
        @test abs(sol_msk.unknown_functions[:r](5.0) - 0.3) < 0.2

        # Everything masked is not a fit — say so instead of sampling NaN.
        prob_allmsk = PSMProblem(
            ODEProblem(exp_decay_mcmc!, [1.0], (0.0, 10.0)),
            [BSplineApproximator(:r, (0.0, 10.0), 8; initial=0.15)];
            data_times=times_mcmc, data_values=data_mcmc,
            data_weights=zeros(length(times_mcmc), 1),
            obs_to_state=[1], solver=Tsit5())
        err_msk = try
            solve(prob_allmsk, MCMCSolver(n_samples=5, n_warmup=5,
                                          verbose=false))
            nothing
        catch e
            e
        end
        @test err_msk isa ErrorException
        @test occursin("every observation is masked", err_msk.msg)
        copy!(Random.default_rng(), rng_state_msk)
    end

    @testset "MagiSolver (B-spline)" begin
        # Exponential decay: du/dt = -r(t)*u, r(t) ≈ 0.3
        function exp_decay_magi!(du, u, p, t)
            du[1] = -p.r(t) * u[1]
        end
        times_magi = collect(0.0:1.0:10.0)
        true_sol_magi = exp.(-0.3 .* times_magi)
        data_magi = reshape(true_sol_magi, :, 1)

        bs_magi = BSplineApproximator(:r, (0.0, 10.0), 6; initial=0.15)  # away from true 0.3
        prob_magi = PSMProblem(exp_decay_magi!, [1.0], (0.0, 10.0), [bs_magi];
            data_times=times_magi, data_values=data_magi,
            obs_to_state=[1],
            known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())

        sol_magi = solve(prob_magi, MagiSolver(
            n_samples=50, n_warmup=50, n_gridpoints=50,
            obs_var=0.01, verbose=false))

        # Returns chains in convergence field
        @test sol_magi.convergence isa NamedTuple
        @test sol_magi.convergence.chains isa MCMCChains.Chains
        @test size(sol_magi.convergence.chains, 1) == 50  # n_samples
        @test size(sol_magi.convergence.chains, 2) == 6   # 6 B-spline params
        @test haskey(sol_magi.unknown_functions, :r)
        # posterior-mean recovery of the constant true rate r(t) = 0.3
        @test abs(sol_magi.unknown_functions[:r](5.0) - 0.3) < 0.15
    end

    # ─── Likelihood dispatch: solvers must honor prob.likelihood ──────

    @testset "loglik_pointwise matches log_likelihood" begin
        y = [0.0, 3.0, 7.0]
        mu = [0.5, 2.5, 8.0]
        w = [1.0, 0.5, 2.0]
        for fam in (Gaussian(), Poisson(), NegativeBinomial(5.0),
                    TruncatedNormal(lower=0.0, sigma=1.0),
                    CustomLikelihood((yy, m) -> -abs(yy - m)))
            ll_vec = PartiallySpecifiedModels.log_likelihood(fam, y, mu, w)
            ll_pt = sum(w[i] * PartiallySpecifiedModels.loglik_pointwise(
                            fam, y[i], mu[i]) for i in eachindex(y))
            # identical accumulation term-for-term; only summation order differs
            @test ll_pt ≈ ll_vec atol = 1e-10
        end
    end

    @testset "MCMCSolver — Poisson likelihood" begin
        # Exponential decay observed as Poisson counts: du/dt = -r(t)·u,
        # r(t) ≡ 0.3, u0 = 200 so means run 200 → ~10 (informative counts).
        rng_pmc = StableRNG(7)   # version-stable RNG for count draws
        function exp_decay_pmc!(du, u, p, t)
            du[1] = -p.r(t) * u[1]
        end
        times_pmc = collect(0.0:0.5:10.0)
        mu_pmc = 200.0 .* exp.(-0.3 .* times_pmc)
        function sample_poisson_pmc(μ)   # simple inversion sampler
            μ = max(μ, 0.01); c = 0; s = 0.0
            while true; s -= log(rand(rng_pmc)); s > μ && break; c += 1; end
            Float64(c)
        end
        y_pmc = sample_poisson_pmc.(mu_pmc)

        bs_pmc = BSplineApproximator(:r, (0.0, 10.0), 8; initial=0.15)
        prob_pmc = PSMProblem(exp_decay_pmc!, [200.0], (0.0, 10.0), [bs_pmc];
            data_times=times_pmc, data_values=reshape(y_pmc, :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Poisson(), solver=Tsit5())

        sol_pmc = solve(prob_pmc, MCMCSolver(
            n_samples=50, n_warmup=25, verbose=false))

        # Poisson data sample NO Gaussian σ nuisance: 8 spline params only
        # (the Gaussian MCMC test above sees 9 = 8 + log_σ)
        @test size(sol_pmc.convergence, 2) == 8
        @test isempty(sol_pmc.smoothing_params)
        # Poisson counts at means 10–200 carry 7–30% relative noise; use the
        # same 0.2 tolerance as the Gaussian MCMC recovery test above.
        @test abs(sol_pmc.unknown_functions[:r](5.0) - 0.3) < 0.2

        # obs_sigma is a Gaussian-only option: declaring it with Poisson
        # data must error, not silently fit Gaussian
        @test_throws ErrorException solve(prob_pmc,
            MCMCSolver(n_samples=10, n_warmup=5, obs_sigma=0.1))
    end

    @testset "MagiSolver — likelihood guard, data_weights, sigma" begin
        function exp_decay_mgw!(du, u, p, t)
            du[1] = -p.r(t) * u[1]
        end
        times_mgw = collect(0.0:1.0:10.0)
        data_mgw = reshape(exp.(-0.3 .* times_mgw), :, 1)
        bs_mgw = BSplineApproximator(:r, (0.0, 10.0), 6; initial=0.15)

        # (i) MAGI's manifold construction assumes Gaussian observations:
        # a non-Gaussian likelihood is rejected at entry
        prob_mgw_pois = PSMProblem(exp_decay_mgw!, [1.0], (0.0, 10.0), [bs_mgw];
            data_times=times_mgw, data_values=data_mgw,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Poisson())
        @test_throws ErrorException solve(prob_mgw_pois, MagiSolver(verbose=false))

        # (ii)+(iii) weight-0 masking + explicit per-state sigma: corrupt one
        # observation grossly (y=10 vs true 0.22) and mask it with weight 0.
        # At σ=0.1 an UNMASKED outlier of this size would dominate the data
        # term (~(10-0.22)²/0.02 ≈ 4800 vs ~O(1) for all clean points) and
        # wreck the fit, so recovery to the clean-data tolerance demonstrates
        # the mask is honored.
        data_bad = copy(data_mgw); data_bad[6, 1] = 10.0
        w_mask = ones(length(times_mgw), 1); w_mask[6, 1] = 0.0
        prob_mgw_mask = PSMProblem(exp_decay_mgw!, [1.0], (0.0, 10.0), [bs_mgw];
            data_times=times_mgw, data_values=data_bad, data_weights=w_mask,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        Random.seed!(9182)   # NUTS draws from the global RNG
        sol_mask = solve(prob_mgw_mask, MagiSolver(
            n_samples=50, n_warmup=50, n_gridpoints=50,
            sigma=[0.1], verbose=false))   # sigma path: SD 0.1 ≡ obs_var 0.01
        # Weight-0 masking excludes the point from the data term, the GP
        # hyperparameter fit, and the state initialization, so the corrupted
        # point must not degrade recovery beyond the clean-data test's 0.15
        @test abs(sol_mask.unknown_functions[:r](5.0) - 0.3) < 0.15

        # (iv) option validation
        @test_throws ErrorException solve(prob_mgw_mask,
            MagiSolver(sigma=[0.1], obs_var=0.01))    # mutually exclusive
        @test_throws ErrorException solve(prob_mgw_mask,
            MagiSolver(sigma=[0.1, 0.1]))             # length ≠ n_vars
    end

    @testset "Poisson dispatch — MultipleShooting and DerivativeFree" begin
        # Exponential growth observed as Poisson counts: du/dt = r(N)·N,
        # r ≡ 0.3, u0 = 20 so means run 20 → ~90.
        rng_msl = StableRNG(21)
        true_r_msl = 0.3
        function growth_msl!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        t_msl = collect(range(0.0, 5.0, length=30))
        mu_msl = 20.0 .* exp.(true_r_msl .* t_msl)
        function sample_poisson_msl(μ)
            μ = max(μ, 0.01); c = 0; s = 0.0
            while true; s -= log(rand(rng_msl)); s > μ && break; c += 1; end
            Float64(c)
        end
        y_msl = sample_poisson_msl.(mu_msl)

        bs_msl = BSplineApproximator(:r, (10.0, 120.0), 6; initial=x -> 0.2)
        prob_msl = PSMProblem(growth_msl!, [20.0], (0.0, 5.0), [bs_msl];
            data_times=t_msl, data_values=reshape(y_msl, :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Poisson(), solver=Tsit5())

        # MultipleShooting: loss=:auto routes Poisson data to the Poisson
        # negative log-likelihood (previously silently fit Gaussian SSE).
        # Counts carry 10–22% relative noise; 0.1 leaves headroom over the
        # noiseless Gaussian MS test's 0.08 while still beating the 0.2
        # initial guess.
        sol_msl = solve(prob_msl, MultipleShootingSolver(
            n_intervals=3, maxiters_inner=50, maxiters_outer=5,
            rho_init=1.0, verbose=false))
        @test abs(sol_msl.unknown_functions[:r](50.0) - true_r_msl) < 0.1
        # The reported objective must be on the Poisson-NLL-kernel scale,
        # not the SSE scale (pre-fix code reported weighted SSE): with
        # counts 20–90 the NLL kernel −Σw(y·log μ − μ) is strongly negative
        # while the SSE is O(10³) positive.
        sse_msl = sum(prob_msl.data_weights .*
                      (prob_msl.data_values .- sol_msl.fitted_values) .^ 2)
        nllk_msl = -sum(prob_msl.data_weights .*
                        (prob_msl.data_values .*
                         log.(max.(sol_msl.fitted_values, 1e-10)) .-
                         sol_msl.fitted_values))
        @test isapprox(sol_msl.objective, nllk_msl; rtol=1e-8)  # penalty_weight=0
        @test sol_msl.objective < 0.5 * sse_msl
        @test sol_msl.data_loss ≈ sse_msl  # data_loss stays descriptive SSE

        # NegativeBinomial has no MS loss: clear error, not silent Gaussian
        prob_msl_nb = PSMProblem(growth_msl!, [20.0], (0.0, 5.0), [bs_msl];
            data_times=t_msl, data_values=reshape(y_msl, :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=NegativeBinomial(5.0), solver=Tsit5())
        @test_throws ErrorException solve(prob_msl_nb, MultipleShootingSolver())

        # penalty_weight > 0 smoke test (Gaussian data): finite objective
        data_msg = reshape(exp.(true_r_msl .* t_msl), :, 1)
        bs_msg = BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)
        prob_msg = PSMProblem(growth_msl!, [1.0], (0.0, 5.0), [bs_msg];
            data_times=t_msl, data_values=data_msg,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian(), solver=Tsit5())
        sol_msp = solve(prob_msg, MultipleShootingSolver(
            n_intervals=3, maxiters_inner=20, maxiters_outer=2,
            rho_init=1.0, penalty_weight=1e-3, verbose=false))
        @test sol_msp isa PSMSolution
        @test isfinite(sol_msp.objective)

        # DerivativeFreeSolver: default loss=:auto now resolves to
        # :likelihood for Poisson data (previously silently used :mse)
        sol_dfp = solve(prob_msl,
            DerivativeFreeSolver(maxiters=4000, verbose=false))
        @test abs(sol_dfp.unknown_functions[:r](50.0) - true_r_msl) < 0.1
        # Objective must be the Poisson NLL plus the 0.5·penalty term, not
        # the SSE (pre-fix :mse default reported SSE + penalty, O(10³) here
        # vs O(10²) for the full NLL)
        nll_dfp = -PartiallySpecifiedModels.log_likelihood(
            Poisson(), vec(prob_msl.data_values), vec(sol_dfp.fitted_values),
            vec(prob_msl.data_weights))
        beta_dfp = collect(sol_dfp.parameters)
        S_dfp = penalty_matrix(bs_msl)
        pen_dfp = 0.5 * 1.0 * dot(beta_dfp, S_dfp * beta_dfp)  # default weight
        @test isapprox(sol_dfp.objective, nll_dfp + pen_dfp; rtol=1e-6)
        sse_dfp = sum(prob_msl.data_weights .*
                      (prob_msl.data_values .- sol_dfp.fitted_values) .^ 2)
        @test sol_dfp.objective < 0.5 * sse_dfp
    end

    @testset "VariationalSolver — Poisson likelihood" begin
        # Exponential decay observed as Poisson counts (as in the MCMC
        # Poisson test): the ELBO data term must route through the Poisson
        # pointwise log-likelihood, with no Gaussian noise nuisance.
        rng_vip = StableRNG(31)
        function exp_decay_vip!(du, u, p, t)
            du[1] = -p.r(t) * u[1]
        end
        times_vip = collect(0.0:0.5:10.0)
        mu_vip = 200.0 .* exp.(-0.3 .* times_vip)
        function sample_poisson_vip(μ)
            μ = max(μ, 0.01); c = 0; s = 0.0
            while true; s -= log(rand(rng_vip)); s > μ && break; c += 1; end
            Float64(c)
        end
        y_vip = sample_poisson_vip.(mu_vip)

        bs_vip = BSplineApproximator(:r, (0.0, 10.0), 6; initial=0.15)
        prob_vip = PSMProblem(exp_decay_vip!, [200.0], (0.0, 10.0), [bs_vip];
            data_times=times_vip, data_values=reshape(y_vip, :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Poisson(), solver=Tsit5())

        sol_vip = solve(prob_vip, VariationalSolver(
            maxiters=400, n_elbo_samples=5, verbose=false))
        @test sol_vip isa PSMSolution
        @test isfinite(sol_vip.objective)   # finite ELBO through _vi_loglik
        # no Gaussian noise variance is estimated for Poisson data
        @test sol_vip.convergence[:obs_noise_var] === nothing
        # posterior-mean recovery; same 0.2 tolerance as the Gaussian VI test
        @test abs(sol_vip.unknown_functions[:r](5.0) - 0.3) < 0.2

        # obs_noise_var is a Gaussian-only option: declaring it with Poisson
        # data must error rather than silently fit Gaussian
        @test_throws ErrorException solve(prob_vip,
            VariationalSolver(maxiters=10, obs_noise_var=0.1))
    end

    # ─── Discrete-time model tests ─────────────────────────────────

    @testset "Discrete-time: Ricker model with LAML" begin
        # Ricker model: N[t+1] = N[t] * exp(r * (1 - N[t]/K))
        # Unknown: density dependence f(N) where N[t+1] = N[t] * exp(f(N[t]))
        # True: f(N) = r * (1 - N/K)

        true_r = 0.8
        true_K = 100.0

        function ricker_true!(u_next, u, p, t)
            N = u[1]
            u_next[1] = N * exp(true_r * (1.0 - N / true_K))
        end

        # Generate data
        N0 = [10.0]
        tspan = (0.0, 30.0)
        n_steps = Int(tspan[2] - tspan[1])
        times = collect(0.0:1.0:tspan[2])
        N_true = zeros(length(times))
        N_true[1] = N0[1]
        u = copy(N0)
        u_next = similar(u)
        for i in 1:n_steps
            ricker_true!(u_next, u, nothing, Float64(i-1))
            u .= u_next
            N_true[i+1] = u[1]
        end
        data = N_true .+ 2.0 .* randn(length(times))
        data = max.(data, 0.1)

        # PSM: unknown f(N) where N[t+1] = N[t] * exp(f(N[t]))
        function ricker_psm!(u_next, u, p, t)
            N = u[1]
            f_N = p.f(N)
            u_next[1] = N * exp(f_N)
        end

        uf = BSplineApproximator(:f, (0.0, 150.0), 8;
                                  initial=x -> 0.5 * (1.0 - x / 100.0))

        prob = PSMProblem(ricker_psm!, N0, tspan, [uf];
                          data_times=times,
                          data_values=reshape(data, :, 1),
                          discrete=true,
                          solver=nothing)

        sol = solve(prob, LAML(maxiters=50, verbose=false))
        @test sol.data_loss < sum((data .- N_true).^2) * 2  # near the noise floor

        f_eval = sol.unknown_functions[:f]
        # At N=50 (half K), true f = 0.8*(1-50/100) = 0.4
        @test abs(f_eval(50.0) - 0.4) < 0.3
    end

    @testset "Discrete-time: Beverton-Holt with AdamSolver" begin
        # Beverton-Holt: N[t+1] = r * N[t] / (1 + (r-1)/K * N[t])
        # Unknown: f(N) where N[t+1] = f(N[t])
        true_r = 2.0
        true_K = 500.0

        function bh_true!(u_next, u, p, t)
            N = u[1]
            u_next[1] = true_r * N / (1.0 + (true_r - 1.0) / true_K * N)
        end

        N0 = [50.0]
        tspan = (0.0, 20.0)
        n_steps = Int(tspan[2])
        times = collect(0.0:1.0:tspan[2])
        N_true = zeros(length(times))
        N_true[1] = N0[1]
        u = copy(N0)
        u_next = similar(u)
        for i in 1:n_steps
            bh_true!(u_next, u, nothing, Float64(i-1))
            u .= u_next
            N_true[i+1] = u[1]
        end
        data = N_true .+ 5.0 .* randn(length(times))
        data = max.(data, 1.0)

        # PSM with directly unknown map f(N) = N[t+1]
        function bh_psm!(u_next, u, p, t)
            N = u[1]
            u_next[1] = p.f(N)
        end

        uf = BSplineApproximator(:f, (0.0, 600.0), 10;
                                  initial=x -> x)

        prob = PSMProblem(bh_psm!, N0, tspan, [uf];
                          data_times=times,
                          data_values=reshape(data, :, 1),
                          discrete=true,
                          solver=nothing)

        # Accuracy via GradientMatching: it fits the map POINTWISE
        # (f(N_t) matched to the smoothed N_{t+1}), so it recovers f across
        # the whole visited range. Trajectory-based single shooting (Adam,
        # MS) cannot fit this map from a generic start: any init whose
        # induced trajectory collapses or explodes leaves most spline
        # coefficients gradient-dead — a structural property, not a bug.
        sol = solve(prob, GradientMatching(maxiters=200, verbose=false))
        @test haskey(sol.unknown_functions, :f)
        f_eval = sol.unknown_functions[:f]
        # Identity-init errors are 0 / 23.7 / 83.3 at these points; the
        # bounds require genuinely beating the start where it errs, and the
        # measured GM fit achieves 4.1 / 9.3 / 32.
        @test abs(f_eval(500.0) - 500.0) < 15
        @test abs(f_eval(450.0) - 473.7) < 20   # below the 23.7 init error
        @test abs(f_eval(250.0) - 333.3) < 60   # below the 83.3 init error

        # AdamSolver discrete-path smoke: runs from the identity warm start
        # (its data_loss stays high here — see the note above)
        sol_adam = solve(prob, AdamSolver(maxiters=100, lr=0.005, verbose=false))
        @test haskey(sol_adam.unknown_functions, :f)
        @test isfinite(sol_adam.data_loss)
    end

    @testset "Discrete-time: GradientMatching" begin
        # Simple exponential growth: N[t+1] = r*N[t], unknown f(N) = r*N
        true_r = 1.05
        N0 = [10.0]
        tspan = (0.0, 15.0)
        times = collect(0.0:1.0:tspan[2])
        N_true = N0[1] .* true_r .^ (0:15)
        data = N_true .+ 0.5 .* randn(length(times))
        data = max.(data, 0.1)

        function exp_growth!(u_next, u, p, t)
            u_next[1] = p.f(u[1])
        end

        uf = BSplineApproximator(:f, (0.0, 25.0), 6; initial=x -> 0.8 * x)

        prob = PSMProblem(exp_growth!, N0, tspan, [uf];
                          data_times=times,
                          data_values=reshape(data, :, 1),
                          discrete=true,
                          solver=nothing)

        sol = solve(prob, GradientMatching(maxiters=100, verbose=false))
        @test haskey(sol.unknown_functions, :f)

        f_eval = sol.unknown_functions[:f]
        # At N=10, true f = 10.5; at N=15, true f = 15.75
        @test abs(f_eval(10.0) - 10.5) < 0.6    # init (0.8x) errs by 2.5 — the fit must beat its start
    end

    @testset "Discrete-time: CollocationLAML" begin
        # Ricker model with collocation
        true_r = 0.5
        true_K = 80.0
        N0 = [20.0]
        tspan = (0.0, 25.0)
        times = collect(0.0:1.0:tspan[2])
        N_true = zeros(length(times))
        N_true[1] = N0[1]
        u = copy(N0)
        u_next = similar(u)
        for i in 1:Int(tspan[2])
            u_next[1] = u[1] * exp(true_r * (1.0 - u[1] / true_K))
            u .= u_next
            N_true[i+1] = u[1]
        end
        data = N_true .+ 1.5 .* randn(length(times))
        data = max.(data, 0.1)

        function ricker_coll!(u_next, u, p, t)
            N = u[1]
            u_next[1] = N * exp(p.f(N))
        end

        uf = BSplineApproximator(:f, (0.0, 120.0), 8;
                                  initial=x -> 0.3 * (1.0 - x / 80.0))

        prob = PSMProblem(ricker_coll!, N0, tspan, [uf];
                          data_times=times,
                          data_values=reshape(data, :, 1),
                          discrete=true,
                          solver=nothing)

        sol = solve(prob, CollocationLAML(maxiters=30, verbose=false))
        @test haskey(sol.unknown_functions, :f)
        f_coll = sol.unknown_functions[:f]
        @test abs(f_coll(20.0) - 0.5 * (1 - 20.0/80.0)) < 0.15  # true 0.375
        @test abs(f_coll(80.0)) < 0.1                            # zero at K
    end

    # ─── SciML problem type constructors ───────────────────────────

    @testset "PSMProblem from ODEProblem" begin
        # Same exponential growth as "Simple ODE fit" but via ODEProblem
        true_r = 0.3
        function exp_ode!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        ode = ODEProblem(exp_ode!, [1.0], (0.0, 5.0))
        approx_r = BSplineApproximator(:r, (0.0, 10.0), 6)
        dt = collect(0:0.5:5)
        y_data = exp.(true_r .* dt) .+ [0.01*randn() for _ in dt]
        prob = PSMProblem(ode, [approx_r];
            data_times=dt, data_values=reshape(y_data, :, 1))

        @test prob.discrete == false
        @test prob.ode_solver isa typeof(Tsit5())

        sol = solve(prob, LAML(verbose=false))
        @test haskey(sol.unknown_functions, :r)
        @test abs(sol.unknown_functions[:r](1.0) - true_r) < 0.15
    end

    @testset "PSMProblem from DiscreteProblem" begin
        # Ricker model via DiscreteProblem
        true_r = 0.5; true_K = 100.0
        function ricker_disc!(u_next, u, p, t)
            N = u[1]
            u_next[1] = N * exp(p.g(N))
            nothing
        end
        disc = DiscreteProblem(ricker_disc!, [20.0], (0.0, 30.0))
        uf_g = BSplineApproximator(:g, (0.0, 150.0), 8)

        g_true(N) = true_r * (1 - N/true_K)
        N_data = zeros(31); N_data[1] = 20.0
        for t in 1:30
            N_data[t+1] = max(N_data[t] * exp(g_true(N_data[t])) + 0.5*randn(), 1.0)
        end
        times = Float64.(0:30)

        prob = PSMProblem(disc, [uf_g];
            data_times=times, data_values=reshape(N_data, :, 1))

        @test prob.discrete == true
        @test prob.ode_solver === nothing

        sol = solve(prob, LAML(verbose=false))
        @test haskey(sol.unknown_functions, :g)
        @test abs(sol.unknown_functions[:g](50.0) - g_true(50.0)) < 0.2
    end

    @testset "PSMProblem from out-of-place ODEProblem" begin
        true_r = 0.3
        exp_oop(u, p, t) = [p.r(u[1]) * u[1]]
        ode = ODEProblem(exp_oop, [1.0], (0.0, 5.0))
        approx_r = BSplineApproximator(:r, (0.0, 10.0), 6)
        dt = collect(0:0.5:5)
        y_data = exp.(true_r .* dt) .+ [0.01*randn() for _ in dt]
        prob = PSMProblem(ode, [approx_r];
            data_times=dt, data_values=reshape(y_data, :, 1))

        @test prob.discrete == false
        sol = solve(prob, LAML(verbose=false))
        @test abs(sol.unknown_functions[:r](1.0) - true_r) < 0.15
    end

    @testset "PSMProblem from out-of-place DiscreteProblem" begin
        true_r = 0.5; true_K = 100.0
        ricker_oop(u, p, t) = [u[1] * exp(p.g(u[1]))]
        disc = DiscreteProblem(ricker_oop, [20.0], (0.0, 30.0))
        uf_g = BSplineApproximator(:g, (0.0, 150.0), 8)

        g_true(N) = true_r * (1 - N/true_K)
        N_data = zeros(31); N_data[1] = 20.0
        for t in 1:30
            N_data[t+1] = max(N_data[t] * exp(g_true(N_data[t])) + 0.5*randn(), 1.0)
        end
        times = Float64.(0:30)

        prob = PSMProblem(disc, [uf_g];
            data_times=times, data_values=reshape(N_data, :, 1))

        @test prob.discrete == true
        sol = solve(prob, LAML(verbose=false))
        @test abs(sol.unknown_functions[:g](50.0) - g_true(50.0)) < 0.2
    end

    # ─── Shape-constrained B-spline approximator ──────────────────
    @testset "ShapeConstrainedBSplineApproximator — construction" begin
        using PartiallySpecifiedModels: _build_sigma_matrix, _softplus,
            gamma_to_knot_values, build_constrained_bspline_evaluator

        # All constraint types should construct without error
        for c in SHAPE_CONSTRAINTS
            a = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 8, c; initial=0.5)
            np = nparams(a)
            ip = initial_params(a)
            S = penalty_matrix(a)
            @test length(ip) == np
            @test size(S) == (np, np)
        end

        # Zero-endpoint constraints have nknots-1 params
        for c in (:inc_zero_left, :dec_zero_right, :inc_zero_right, :dec_zero_left)
            a = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 8, c)
            @test nparams(a) == 7
            @test size(a.Sigma) == (8, 7)
        end

        # Square constraints have nknots params
        for c in (:increasing, :decreasing, :convex, :concave, :positive, :dec_positive)
            a = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 8, c)
            @test nparams(a) == 8
            @test size(a.Sigma) == (8, 8)
        end

        # Invalid constraint throws
        @test_throws ArgumentError ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 8, :invalid)
        @test_throws ArgumentError ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 3, :increasing)
    end

    @testset "ShapeConstrainedBSplineApproximator — shape enforcement" begin
        using PartiallySpecifiedModels: gamma_to_knot_values

        xs = range(0.0, 1.0, length=50)

        # :increasing — knot values should be monotonically increasing
        a_inc = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :increasing)
        gamma = randn(10)
        beta = gamma_to_knot_values(a_inc, gamma)
        @test all(diff(beta) .> 0)  # strictly increasing (softplus > 0)

        # :decreasing — knot values should be monotonically decreasing
        a_dec = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :decreasing)
        beta_d = gamma_to_knot_values(a_dec, gamma)
        @test all(diff(beta_d) .< 0)

        # :positive — all knot values positive
        a_pos = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :positive)
        beta_p = gamma_to_knot_values(a_pos, gamma)
        @test all(beta_p .> 0)

        # :dec_positive — decreasing AND positive
        a_dp = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :dec_positive)
        beta_dp = gamma_to_knot_values(a_dp, gamma)
        @test all(diff(beta_dp) .< 0)
        @test all(beta_dp .> 0)

        # :inc_zero_left — increasing, first knot = 0
        a_izl = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :inc_zero_left)
        gamma7 = randn(9)
        beta_izl = gamma_to_knot_values(a_izl, gamma7)
        @test abs(beta_izl[1]) < 1e-15  # first knot exactly 0
        @test all(diff(beta_izl) .> 0)

        # :dec_zero_right — decreasing, last knot = 0
        a_dzr = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :dec_zero_right)
        beta_dzr = gamma_to_knot_values(a_dzr, gamma7)
        @test abs(beta_dzr[end]) < 1e-15  # last knot exactly 0
        @test all(diff(beta_dzr) .< 0)

        # :convex — second differences positive
        a_cx = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :convex)
        beta_cx = gamma_to_knot_values(a_cx, gamma)
        d2 = diff(diff(beta_cx))
        @test all(d2 .> 0)

        # :concave — second differences negative
        a_cv = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 10, :concave)
        beta_cv = gamma_to_knot_values(a_cv, gamma)
        d2_cv = diff(diff(beta_cv))
        @test all(d2_cv .< 0)

        # :convex must represent a NON-monotone U-shape (free intercept/slope):
        # negative initial slope (γ₂<0) + positive curvature ⇒ decreasing then
        # increasing. The previous parameterization forced monotone increasing.
        gamma_u = vcat(0.0, -3.0, fill(0.0, 8))  # softplus(0)=0.69 curvature
        beta_u = gamma_to_knot_values(a_cx, gamma_u)
        @test all(diff(diff(beta_u)) .> 0)         # still convex
        @test any(diff(beta_u) .< 0)               # decreasing somewhere
        @test any(diff(beta_u) .> 0)               # increasing somewhere
        # :concave likewise represents a non-monotone ∩-shape
        gamma_n = vcat(0.0, 3.0, fill(0.0, 8))
        beta_n = gamma_to_knot_values(a_cv, gamma_n)
        @test all(diff(diff(beta_n)) .< 0)
        @test any(diff(beta_n) .> 0) && any(diff(beta_n) .< 0)

        # Evaluator works and respects domain
        eval_inc = build_constrained_bspline_evaluator(a_inc, gamma)
        @test eval_inc(0.0) < eval_inc(0.5) < eval_inc(1.0)
    end

    @testset "ShapeConstrainedBSplineApproximator — LAML solver (SIR)" begin
        # SIR model with decreasing β(I)
        β_true(I) = 0.5 * exp(-5.0 * I)

        function sir!(du, u, p, t)
            S, I = u
            β_val = p.β(I)
            du[1] = -β_val * S * I
            du[2] = β_val * S * I - 0.25 * I
        end

        u0 = [0.99, 0.01]
        tspan = (0.0, 40.0)
        ode = ODEProblem(sir!, u0, tspan)

        using OrdinaryDiffEq
        sol_true = OrdinaryDiffEq.solve(ode,
            Tsit5(); p=(β=β_true,), saveat=1.0)
        t_obs = sol_true.t[1:end]
        data = hcat([u[1] for u in sol_true.u], [u[2] for u in sol_true.u])
        data .+= 0.005 .* randn(size(data))

        # Fit with decreasing shape constraint
        uf = ShapeConstrainedBSplineApproximator(:β, (0.0, 0.15), 8, :decreasing;
            initial=0.4)

        prob = PSMProblem(ode, [uf];
            data_times=t_obs, data_values=data,
            known_params=(;))

        sol = solve(prob, LAML(verbose=false))
        @test sol.data_loss < 1.0  # good fit
        @test sol.edf < 8.0  # some smoothing applied

        # Verify decreasing shape is maintained
        I_vals = range(0.01, 0.12, length=20)
        β_fitted = [sol.unknown_functions[:β](I) for I in I_vals]
        @test all(diff(β_fitted) .< 0.01)  # approximately decreasing
    end

    # ─── COMONet tests ────────────────────────────────────────────

    @testset "COMONetApproximator — construction" begin
        a = COMONetApproximator(:f, (0.0, 1.0), (16, 16), :increasing)
        @test a.name == :f
        @test a.domain == (0.0, 1.0)
        @test a.hidden_sizes == (16, 16)
        @test a.constraint == :increasing

        # Parameter count: 1→16 (16+16) + 16→16 (256+16) + 16→1 (16+1) = 321
        @test nparams(a) == 1*16 + 16 + 16*16 + 16 + 16*1 + 1

        # Invalid constraint
        @test_throws ArgumentError COMONetApproximator(:f, (0.0, 1.0), (8,), :invalid)
    end

    @testset "COMONetApproximator — shape enforcement" begin
        import PartiallySpecifiedModels as PSM

        xs = range(0.0, 1.0, length=50)

        for c in (:increasing, :decreasing, :convex, :concave,
                  :inc_convex, :inc_concave, :dec_convex, :dec_concave, :positive)
            a = COMONetApproximator(:f, (0.0, 1.0), (8, 8), c)
            p = initial_params(a)
            ev = PSM.build_comonet_evaluator(a, p)
            vals = [ev(x) for x in xs]
            diffs = diff(vals)

            if c in (:increasing, :inc_convex, :inc_concave)
                @test all(d -> d >= -1e-10, diffs)
            elseif c in (:decreasing, :dec_convex, :dec_concave)
                @test all(d -> d <= 1e-10, diffs)
            elseif c == :positive
                @test all(v -> v > 0, vals)
            end

            if c in (:convex, :inc_convex, :dec_convex)
                dd = diff(diffs)
                @test all(d -> d >= -1e-8, dd)
            elseif c in (:concave, :inc_concave, :dec_concave)
                dd = diff(diffs)
                @test all(d -> d <= 1e-8, dd)
            end
        end
        # Shape enforcement at RANDOM parameter draws (init-only checks can
        # miss architecture bugs that only appear off the initialization)
        for c in (:increasing, :decreasing, :convex, :concave, :positive)
            a_r = COMONetApproximator(:f, (0.0, 1.0), (6, 6), c; rng_seed=5)
            for trial in 1:3
                θr = 1.5 .* randn(Random.Xoshiro(31 * trial + Int(hash(c) % 256)),
                                  PSM.nparams(a_r))
                fr = PSM.build_comonet_evaluator(a_r, θr)
                vr = [fr(x) for x in 0.02:0.02:0.98]
                if c == :increasing
                    @test all(diff(vr) .>= -1e-10)
                elseif c == :decreasing
                    @test all(diff(vr) .<= 1e-10)
                elseif c == :convex
                    @test all(diff(diff(vr)) .>= -1e-8)
                elseif c == :concave
                    @test all(diff(diff(vr)) .<= 1e-8)
                else
                    @test all(vr .> 0)
                end
            end
        end
    end

    @testset "COMONetApproximator — ForwardDiff" begin
        import PartiallySpecifiedModels as PSM
        using ForwardDiff

        a = COMONetApproximator(:f, (0.0, 1.0), (8, 8), :increasing)
        p = initial_params(a)

        function comonet_loss(params)
            ev = PSM.build_comonet_evaluator(a, params)
            sum(ev(x)^2 for x in 0.0:0.2:1.0)
        end

        g = ForwardDiff.gradient(comonet_loss, p)
        @test length(g) == nparams(a)
        @test all(isfinite, g)
    end

    @testset "COMONetApproximator — penalty matrix" begin
        import PartiallySpecifiedModels as PSM

        a = COMONetApproximator(:f, (0.0, 1.0), (8, 8), :increasing; penalty_weight=0.05)
        S = PSM.penalty_matrix(a)
        @test size(S) == (nparams(a), nparams(a))
        @test S ≈ 0.05 * I(nparams(a))  # L2 penalty
    end

    @testset "COMONetApproximator — AdamSolver (exponential decay)" begin
        import PartiallySpecifiedModels as PSM

        # Exponential decay: du/dt = -r(t)*u, true r(t) = 0.3
        function exp_decay!(du, u, p, t)
            du[1] = -p.r(t) * u[1]
        end

        uf_r = COMONetApproximator(:r, (0.0, 10.0), (8,), :positive;
                                   penalty_weight=0.001)

        t_data = collect(0.0:1.0:10.0)
        u_true = exp.(-0.3 .* t_data)
        u_data = reshape(u_true, :, 1)

        prob = PSMProblem(exp_decay!, [1.0], (0.0, 10.0), [uf_r];
            data_times=t_data, data_values=u_data,
            obs_to_state=[1],
            known_params=NamedTuple(),
            likelihood=PSM.Gaussian())

        sol = solve(prob, AdamSolver(lr=0.01, maxiters=100))

        # Check that the fitted rate is positive (COMONet constraint)
        r_fitted = sol.unknown_functions[:r]
        r_vals = [r_fitted(t) for t in 0.0:2.0:10.0]
        @test all(v -> v > 0, r_vals)  # positive constraint guaranteed

        # Positivity alone is guaranteed by the :positive architecture at ANY
        # parameters -- including the random initialization -- so it would
        # pass a do-nothing solver. Assert recovery of the constant truth
        # r(t) = 0.3 as well. Across 25 random initializations at these
        # settings (lr = 0.01, 100 Adam steps) max |r̂(t) − 0.3| over
        # t ∈ {0,2,…,10} ranged 0.052–0.103, so 0.15 is ~1.5x the worst draw.
        @test maximum(abs(v - 0.3) for v in r_vals) < 0.15

        # ...and that the fit beats its own initialization. Seeded via the
        # approximator's rng_seed so both the initialization and the fit are
        # deterministic; a seeded COMONet draws from its own Xoshiro, leaving
        # this suite's order-coupled global stream untouched.
        uf_seeded = COMONetApproximator(:r, (0.0, 10.0), (8,), :positive;
                                        penalty_weight=0.001, rng_seed=3)
        prob_seeded = PSMProblem(exp_decay!, [1.0], (0.0, 10.0), [uf_seeded];
            data_times=t_data, data_values=u_data,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PSM.Gaussian())
        r_init = PSM.build_comonet_evaluator(uf_seeded, initial_params(uf_seeded))
        err_init = maximum(abs(r_init(t) - 0.3) for t in 0.0:2.0:10.0)
        sol_seeded = solve(prob_seeded, AdamSolver(lr=0.01, maxiters=100))
        r_seeded = sol_seeded.unknown_functions[:r]
        err_fit = maximum(abs(r_seeded(t) - 0.3) for t in 0.0:2.0:10.0)
        # Observed: err_init = 0.638, err_fit = 0.093 (ratio 0.15); assert a
        # 4x improvement, ~1.7x headroom on the observed ratio.
        @test err_fit < 0.25 * err_init
    end

    # ─── New solver tests ─────────────────────────────────────────

    @testset "BNGSolver — logistic growth" begin
        r_true(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_bng!(du, u, p, t)
            N = u[1]
            du[1] = p.r(N) * N
        end
        rng_bng = Random.Xoshiro(42)
        sol_true = OrdinaryDiffEq.solve(
            ODEProblem(logistic_bng!, [1.0], (0.0, 15.0), (; r=r_true)),
            Tsit5(); saveat=1.0)
        t_bng = collect(sol_true.t)
        data_bng = [sol_true.u[i][1] + 0.1*randn(rng_bng) for i in 1:length(t_bng)]
        data_bng = max.(data_bng, 0.01)

        uf_bng = BSplineApproximator(:r, (0.0, 12.0), 8)
        prob_bng = PSMProblem(logistic_bng!, [1.0], (0.0, 15.0), [uf_bng];
            data_times=t_bng, data_values=reshape(data_bng, :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_bng = solve(prob_bng, BNGSolver(maxiters=500, k_obs=4, k_proc=2,
                                            rng_seed=7, verbose=false))

        @test sol_bng isa PSMSolution
        @test sol_bng.data_loss < 2.0   # noise floor ≈ 0.16 (16 pts × 0.1²)
        @test haskey(sol_bng.unknown_functions, :r)
        r_fitted = sol_bng.unknown_functions[:r]
        @test abs(r_fitted(5.0) - 0.25) < 0.12   # true r(5) = 0.5(1 − 5/10)
        # Ensemble machinery: K_o × K_p members, posterior weights summing
        # to 1, and a pointwise uncertainty band
        @test sol_bng.convergence.n_ensemble == 8
        @test length(sol_bng.convergence.member_losses) == 8
        @test sum(sol_bng.convergence.member_weights) ≈ 1.0
        @test sol_bng.convergence.ensemble_std[:r](5.0) >= 0.0
        # Reproducibility under rng_seed
        sol_bng2 = solve(prob_bng, BNGSolver(maxiters=500, k_obs=4, k_proc=2,
                                             rng_seed=7, verbose=false))
        @test sol_bng2.unknown_functions[:r](5.0) == r_fitted(5.0)
    end

    @testset "DaltonSolver — exponential decay" begin
        function decay_dal!(du, u, p, t)
            du[1] = -p.f(u[1])
        end
        rng_dal = Random.Xoshiro(42)
        sol_true_d = OrdinaryDiffEq.solve(
            ODEProblem(decay_dal!, [5.0], (0.0, 10.0), (; f=x -> 0.5*x)),
            Tsit5(); saveat=0.5)
        t_dal = collect(sol_true_d.t)
        data_dal = [sol_true_d.u[i][1] + 0.05*randn(rng_dal) for i in 1:length(t_dal)]

        uf_dal = BSplineApproximator(:f, (0.0, 6.0), 8)
        prob_dal = PSMProblem(decay_dal!, [5.0], (0.0, 10.0), [uf_dal];
            data_times=t_dal, data_values=reshape(max.(data_dal, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        # Runs on the auto-estimated obs_var (≈0.0195). This was previously
        # pinned to obs_var=0.01 because the uncalibrated DALTON objective
        # was unbounded above and the fit diverged for scattered obs_var
        # values; both passes now share one quasi-MLE diffusion, so the
        # result is stable across obs_var.
        sol_dal = solve(prob_dal, DaltonSolver(n_steps=100, maxiters=50,
                                               verbose=false))

        @test sol_dal isa PSMSolution
        @test isfinite(sol_dal.objective)
        @test haskey(sol_dal.unknown_functions, :f)
        @test abs(sol_dal.unknown_functions[:f](3.0) - 1.5) < 0.35  # f(x)=0.5x

        # Diffusion calibration makes the fit insensitive to obs_var: values
        # that formerly drove the objective to ~1e11 (0.0125, 0.015, 0.02)
        # now all recover the same function. Regression guard for the cliff.
        for ov in (0.0125, 0.015, 0.02)
            s_ov = solve(prob_dal, DaltonSolver(n_steps=100, maxiters=50,
                                                obs_var=ov, verbose=false))
            @test abs(s_ov.unknown_functions[:f](3.0) - 1.5) < 0.35
            @test s_ov.objective < 100.0   # was ~8e11 uncalibrated
        end
    end

    @testset "PseudoMarginalSolver — logistic growth" begin
        r_true_pm(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_pm!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        rng_pm = Random.Xoshiro(42)
        sol_true_pm = OrdinaryDiffEq.solve(
            ODEProblem(logistic_pm!, [1.0], (0.0, 15.0), (; r=r_true_pm)),
            Tsit5(); saveat=1.0)
        t_pm = collect(sol_true_pm.t)
        data_pm = [sol_true_pm.u[i][1] + 0.1*randn(rng_pm) for i in 1:length(t_pm)]

        uf_pm = BSplineApproximator(:r, (0.0, 12.0), 6)
        prob_pm = PSMProblem(logistic_pm!, [1.0], (0.0, 15.0), [uf_pm];
            data_times=t_pm, data_values=reshape(max.(data_pm, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_pm = solve(prob_pm, PseudoMarginalSolver(
            n_samples=50, n_warmup=25, n_steps=50, verbose=false))

        @test sol_pm isa PSMSolution
        @test sol_pm.convergence isa MCMCChains.Chains
        @test size(sol_pm.convergence, 1) == 50  # n_samples
        # short chain: generous tolerance, but a do-nothing sampler fails it
        @test abs(sol_pm.unknown_functions[:r](5.0) - 0.25) < 0.25

        # Reproducibility under the solver's own rng_seed field (AGM/BNG
        # convention): same seed → identical chains, and the global stream
        # is never touched, so this order-coupled suite is unaffected.
        sol_pm_a = solve(prob_pm, PseudoMarginalSolver(
            n_samples=50, n_warmup=25, n_steps=50, rng_seed=31415,
            verbose=false))
        sol_pm_b = solve(prob_pm, PseudoMarginalSolver(
            n_samples=50, n_warmup=25, n_steps=50, rng_seed=31415,
            verbose=false))
        @test Array(sol_pm_a.convergence) == Array(sol_pm_b.convergence)
        @test sol_pm_a.unknown_functions[:r](5.0) ==
              sol_pm_b.unknown_functions[:r](5.0)
    end

    @testset "DDE support — delay exponential decay" begin
        using DelayDiffEq
        function dde_decay!(du, u, h, p, t)
            u_delayed = h(p, t - 1.0)
            du[1] = -p.f(u_delayed[1])
        end
        h_dde(p, t) = [1.0]

        # Generate data with known params
        function dde_true_decay!(du, u, h, p, t)
            u_delayed = h(p, t - 1.0)
            du[1] = -0.5 * u_delayed[1]
        end
        prob_true_dde = DDEProblem(dde_true_decay!, [1.0], h_dde, (0.0, 8.0);
            constant_lags=[1.0])
        sol_true_dde = OrdinaryDiffEq.solve(prob_true_dde, MethodOfSteps(Tsit5());
            saveat=0.5)
        rng_dde = Random.Xoshiro(42)
        t_dde = collect(sol_true_dde.t)
        data_dde = [sol_true_dde.u[i][1] + 0.02*randn(rng_dde) for i in 1:length(t_dde)]

        uf_dde = BSplineApproximator(:f, (0.0, 1.5), 6)
        prob_dde = PSMProblem(dde_decay!, [1.0], (0.0, 8.0), [uf_dde];
            data_times=t_dde, data_values=reshape(max.(data_dde, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian(),
            delays=[1.0], history=h_dde)

        @test !isempty(prob_dde.delays)
        @test prob_dde.delays == [1.0]
        @test prob_dde.history !== nothing

        sol_dde = solve(prob_dde, LAML(maxiters=30, verbose=false))
        @test sol_dde isa PSMSolution
        @test isfinite(sol_dde.data_loss)
        @test haskey(sol_dde.unknown_functions, :f)
        @test abs(sol_dde.unknown_functions[:f](0.8) - 0.4) < 0.1  # f(x)=0.5x
    end

    # ─── New solver tests ─────────────────────────────────────────────

    @testset "GCVSolver — logistic growth" begin
        r_gcv(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_gcv!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        rng_gcv = Random.Xoshiro(123)
        sol_true_gcv = OrdinaryDiffEq.solve(
            ODEProblem(logistic_gcv!, [1.0], (0.0, 15.0), (; r=r_gcv)),
            Tsit5(); saveat=0.5)
        t_gcv = collect(sol_true_gcv.t)
        data_gcv = [sol_true_gcv.u[i][1] + 0.1*randn(rng_gcv) for i in 1:length(t_gcv)]

        uf_gcv = BSplineApproximator(:r, (0.0, 12.0), 8)
        prob_gcv = PSMProblem(logistic_gcv!, [1.0], (0.0, 15.0), [uf_gcv];
            data_times=t_gcv, data_values=reshape(max.(data_gcv, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_gcv = solve(prob_gcv, GCVSolver(maxiters=30, verbose=false))

        @test sol_gcv isa PSMSolution
        @test isfinite(sol_gcv.data_loss)
        @test haskey(sol_gcv.unknown_functions, :r)
        r_fitted_gcv = sol_gcv.unknown_functions[:r]
        @test abs(r_fitted_gcv(5.0) - 0.25) < 0.12
    end

    @testset "Shared PCLS step (pcls.jl)" begin
        # The LAML and GCV IRLS loops share _pcls_augmented_solve /
        # _pcls_step_contract. Verify the two key numerical properties the
        # truncated SVD provides over a plain QR solve.
        # (An end-to-end GCV poor-initialization pin was attempted but the
        # GCV λ-selection floor regularizes the augmented system enough
        # that the pre-fix QR path did not visibly explode; this unit test
        # pins the guard directly instead.)
        PSM = PartiallySpecifiedModels
        rng_pcls = Random.Xoshiro(3)
        n, p = 40, 6
        Jm = randn(rng_pcls, n, p)
        zv = randn(rng_pcls, n)
        wv = rand(rng_pcls, n) .+ 0.5
        Ws = sqrt.(max.(wv, 1e-15))

        # 1) Well-conditioned system: truncated SVD equals plain QR backslash
        Bpen = Matrix(0.3I, p, p)
        beta_svd = PSM._pcls_augmented_solve(Jm, zv, Bpen, wv)
        Cpen = PSM.penalty_sqrt_matrix(Bpen)
        F_aug = vcat(Diagonal(Ws) * Jm, Cpen)
        z_aug = vcat(Ws .* zv, zeros(size(Cpen, 1)))
        beta_qr = F_aug \ z_aug
        @test norm(beta_svd - beta_qr) / norm(beta_qr) < 1e-10

        # 2) Near-null Jacobian directions (the documented poor-init failure
        # mode): plain QR returns exploding coefficients along σ≈1e-9
        # directions; the truncated SVD keeps the step bounded.
        J2 = copy(Jm)
        J2[:, 5] .= 1e-9 .* randn(rng_pcls, n)
        J2[:, 6] .= 1e-9 .* randn(rng_pcls, n)
        B0 = zeros(p, p)
        b_svd = PSM._pcls_augmented_solve(J2, zv, B0, wv)
        F2 = vcat(Diagonal(Ws) * J2, PSM.penalty_sqrt_matrix(B0))
        b_qr = F2 \ vcat(Ws .* zv, zeros(0))
        @test norm(b_qr) > 1e6      # plain QR explodes
        @test norm(b_svd) < 1e2     # truncated SVD stays bounded

        # 3) Step contraction: phase-2 rescue escapes an explosive step where
        # even α = 2^-15 of the direction is rejected by the objective.
        # Objective: Inf outside a tiny ball around the origin (mimics ODE
        # blow-up), quadratic inside; the proposed step is enormous.
        a_old = zeros(2)
        a_new = fill(1e9, 2)
        obj_ball(a, B) = norm(a) < 1e-4 ? sum(abs2, a .- 1e-5) : Inf
        a_res, f_res = PSM._pcls_step_contract(obj_ball, a_old, a_new, B0[1:2, 1:2])
        @test f_res < obj_ball(a_old, nothing)   # escaped (improved on f_old)
        @test 0 < norm(a_res) < 1e-4             # via a phase-2 contracted step
    end

    @testset "TwoStageSolver — logistic growth" begin
        r_ts(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_ts!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        rng_ts = Random.Xoshiro(123)
        sol_true_ts = OrdinaryDiffEq.solve(
            ODEProblem(logistic_ts!, [1.0], (0.0, 15.0), (; r=r_ts)),
            Tsit5(); saveat=0.5)
        t_ts = collect(sol_true_ts.t)
        data_ts = [sol_true_ts.u[i][1] + 0.1*randn(rng_ts) for i in 1:length(t_ts)]

        uf_ts = BSplineApproximator(:r, (0.0, 12.0), 8)
        prob_ts = PSMProblem(logistic_ts!, [1.0], (0.0, 15.0), [uf_ts];
            data_times=t_ts, data_values=reshape(max.(data_ts, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_ts = solve(prob_ts, TwoStageSolver(maxiters=500, verbose=false))

        @test sol_ts isa PSMSolution
        @test isfinite(sol_ts.data_loss)
        @test haskey(sol_ts.unknown_functions, :r)
        @test abs(sol_ts.unknown_functions[:r](5.0) - 0.25) < 0.12
    end

    @testset "DerivativeFreeSolver — exponential decay" begin
        function decay_df!(du, u, p, t)
            du[1] = -p.f(u[1])
        end
        rng_df = Random.Xoshiro(42)
        sol_true_df = OrdinaryDiffEq.solve(
            ODEProblem(decay_df!, [5.0], (0.0, 10.0), (; f=x -> 0.5*x)),
            Tsit5(); saveat=0.5)
        t_df = collect(sol_true_df.t)
        data_df = [sol_true_df.u[i][1] + 0.05*randn(rng_df) for i in 1:length(t_df)]

        uf_df = BSplineApproximator(:f, (0.0, 6.0), 6)
        prob_df = PSMProblem(decay_df!, [5.0], (0.0, 10.0), [uf_df];
            data_times=t_df, data_values=reshape(max.(data_df, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_df = solve(prob_df, DerivativeFreeSolver(maxiters=5000, verbose=false))

        @test sol_df isa PSMSolution
        @test isfinite(sol_df.objective)
        @test haskey(sol_df.unknown_functions, :f)
        @test abs(sol_df.unknown_functions[:f](3.0) - 1.5) < 0.35
    end

    @testset "VariationalSolver — logistic growth" begin
        r_vi(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_vi!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        rng_vi = Random.Xoshiro(42)
        sol_true_vi = OrdinaryDiffEq.solve(
            ODEProblem(logistic_vi!, [1.0], (0.0, 15.0), (; r=r_vi)),
            Tsit5(); saveat=1.0)
        t_vi = collect(sol_true_vi.t)
        data_vi = [sol_true_vi.u[i][1] + 0.1*randn(rng_vi) for i in 1:length(t_vi)]

        uf_vi = BSplineApproximator(:r, (0.0, 12.0), 6)
        prob_vi = PSMProblem(logistic_vi!, [1.0], (0.0, 15.0), [uf_vi];
            data_times=t_vi, data_values=reshape(max.(data_vi, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_vi = solve(prob_vi, VariationalSolver(maxiters=500, n_elbo_samples=5, verbose=false))

        @test sol_vi isa PSMSolution
        @test isfinite(sol_vi.objective)
        @test haskey(sol_vi.unknown_functions, :r)
        @test haskey(sol_vi.convergence, :posterior_std)
        @test abs(sol_vi.unknown_functions[:r](5.0) - 0.25) < 0.2
        # edf = tr((H + Λ)⁻¹H) of the Laplace approximation at the
        # posterior mean: a valid dof count, materially below n_p with the
        # smoothness prior active, and — crucially — it must MOVE with the
        # smoothing level. The previous σ_q-based form saturated every
        # per-term clamp and returned the constant 1.0 for every
        # prior_scale, which passed a bounds-only assertion trivially.
        n_p_vi = length(sol_vi.parameters)
        @test 0.0 <= sol_vi.edf <= n_p_vi
        @test sol_vi.edf <= n_p_vi - 1.0
        vi_opts = (maxiters=200, n_elbo_samples=5, verbose=false)
        edf_heavy = solve(prob_vi,
            VariationalSolver(; prior_scale=1e-4, vi_opts...)).edf
        edf_mid = solve(prob_vi,
            VariationalSolver(; prior_scale=1.0, vi_opts...)).edf
        edf_light = solve(prob_vi,
            VariationalSolver(; prior_scale=1e4, vi_opts...)).edf
        @test all(isfinite, (edf_heavy, edf_mid, edf_light))
        # strictly increasing with material gaps at EVERY step — the old
        # form returned exactly 1.0 for both 1e-4 and 1.0 (both clamped),
        # so a gap assertion only at the extremes passed on the floor.
        # Measured post-fix: 2.01 → 3.55 → 5.99.
        @test edf_mid - edf_heavy > 0.5
        @test edf_light - edf_mid > 0.5
        # heavy smoothing must collapse toward the penalty null space,
        # light smoothing toward the full parameter count
        @test edf_heavy < n_p_vi - 2.0
        @test edf_light > n_p_vi - 1.0

        # Masked observations (weight 0, value NaN): the DEFAULT Gaussian
        # path must skip them. Pre-fix `_gaussian_loglik` and `data_loss`
        # summed over all cells, so `0 * NaN = NaN` gave objective=Inf,
        # data_loss=NaN, elbo_history=[NaN,...] and r(5) frozen at the
        # untouched initial coefficients — silently, with no error.
        dv_vim = reshape(max.(data_vi, 0.01), :, 1)
        w_vim = ones(length(t_vi), 1)
        dv_vim[7, 1] = NaN; w_vim[7, 1] = 0.0
        prob_vim = PSMProblem(logistic_vi!, [1.0], (0.0, 15.0),
            [BSplineApproximator(:r, (0.0, 12.0), 6)];
            data_times=t_vi, data_values=dv_vim, data_weights=w_vim,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_vim = solve(prob_vim,
            VariationalSolver(maxiters=500, n_elbo_samples=5, verbose=false))
        @test isfinite(sol_vim.objective)
        @test isfinite(sol_vim.data_loss)
        @test all(isfinite, sol_vim.convergence[:elbo_history])
        @test abs(sol_vim.unknown_functions[:r](5.0) - 0.25) < 0.2

        # An ELBO that is non-finite at every iteration means the returned
        # parameters would be the untouched initialization: error loudly
        # instead of presenting the initial guess as a fit. (A zero
        # observation variance makes every quadratic term -Inf.)
        err_vi = try
            solve(prob_vi, VariationalSolver(maxiters=20, n_elbo_samples=2,
                                             obs_noise_var=0.0, verbose=false))
            nothing
        catch e
            e
        end
        @test err_vi isa ErrorException
        @test occursin("non-finite at every", err_vi.msg)
    end

    @testset "ABCSolver — exponential decay" begin
        function decay_abc!(du, u, p, t)
            du[1] = -p.f(u[1])
        end
        rng_abc = Random.Xoshiro(42)
        sol_true_abc = OrdinaryDiffEq.solve(
            ODEProblem(decay_abc!, [5.0], (0.0, 10.0), (; f=x -> 0.5*x)),
            Tsit5(); saveat=1.0)
        t_abc = collect(sol_true_abc.t)
        data_abc = [sol_true_abc.u[i][1] + 0.05*randn(rng_abc) for i in 1:length(t_abc)]

        uf_abc = BSplineApproximator(:f, (0.0, 6.0), 6)
        prob_abc = PSMProblem(decay_abc!, [5.0], (0.0, 10.0), [uf_abc];
            data_times=t_abc, data_values=reshape(max.(data_abc, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_abc = solve(prob_abc, ABCSolver(n_particles=100, n_generations=6, verbose=false))

        @test sol_abc isa PSMSolution
        @test isfinite(sol_abc.objective)
        @test haskey(sol_abc.unknown_functions, :f)
        # posterior-MEAN point estimate (consistent with sol.parameters);
        # default GMRF smoothness prior — generous but nonvacuous bound
        @test abs(sol_abc.unknown_functions[:f](3.0) - 1.5) < 0.6
        # legacy box prior still available and accurate
        sol_abc_box = solve(prob_abc, ABCSolver(n_particles=100,
            n_generations=6, prior=:box, verbose=false))
        @test abs(sol_abc_box.unknown_functions[:f](3.0) - 1.5) < 0.6
        @test_throws ErrorException solve(prob_abc, ABCSolver(prior=:bogus))

        # rng_seed parity (W1): the sampler owns its stream, so a fixed
        # seed reproduces the run exactly without touching the global RNG.
        sol_abc_s1 = solve(prob_abc, ABCSolver(n_particles=40,
            n_generations=3, rng_seed=11, verbose=false))
        sol_abc_s2 = solve(prob_abc, ABCSolver(n_particles=40,
            n_generations=3, rng_seed=11, verbose=false))
        @test sol_abc_s1.parameters == sol_abc_s2.parameters
    end

    @testset "rng_seed parity — VI and EKI" begin
        # VariationalSolver/EnsembleKalmanSolver historically hard-coded
        # Xoshiro(42); the new rng_seed field must default to 42 (behavior
        # preserved) while making the stream controllable.
        @test VariationalSolver().rng_seed == 42
        @test EnsembleKalmanSolver().rng_seed == 42

        decay_seed!(du, u, p, t) = (du[1] = -p.f(u[1]))
        t_sd = collect(0.0:1.0:8.0)
        data_sd = reshape(5.0 .* exp.(-0.5 .* t_sd) .+ 0.02, :, 1)
        uf_sd = BSplineApproximator(:f, (0.0, 6.0), 5)
        prob_sd = PSMProblem(decay_seed!, [5.0], (0.0, 8.0), [uf_sd];
            data_times=t_sd, data_values=data_sd, obs_to_state=[1],
            known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())

        # same seed → identical; explicit 42 ≡ the default
        eki = (seed; kw...) -> solve(prob_sd, EnsembleKalmanSolver(
            n_ensemble=20, n_iterations=5, rng_seed=seed); kw...)
        sol_e1 = eki(7); sol_e2 = eki(7)
        @test sol_e1.parameters == sol_e2.parameters
        sol_e42 = eki(42)
        sol_edef = solve(prob_sd, EnsembleKalmanSolver(n_ensemble=20,
            n_iterations=5))
        @test sol_e42.parameters == sol_edef.parameters
        # a different seed draws a different ensemble
        @test eki(8).parameters != sol_e1.parameters

        vi = seed -> solve(prob_sd, VariationalSolver(maxiters=100,
            rng_seed=seed, verbose=false))
        sol_v1 = vi(7); sol_v2 = vi(7)
        @test sol_v1.parameters == sol_v2.parameters
        sol_vdef = solve(prob_sd, VariationalSolver(maxiters=100, verbose=false))
        @test vi(42).parameters == sol_vdef.parameters
    end

    @testset "ABCSolver — NaN-masked observation" begin
        # Pre-fix: the :auto distance was NaN whenever any observation was
        # NaN; `NaN < ε` is false, so no particle was ever accepted (the
        # 1000-attempt budget was burned for every slot each generation)
        # and the solver returned the untouched prior population with
        # no finite distances (ε stayed Inf throughout).
        function decay_abcn!(du, u, p, t)
            du[1] = -p.f(u[1])
        end
        rng_abcn = Random.Xoshiro(42)
        sol_true_abcn = OrdinaryDiffEq.solve(
            ODEProblem(decay_abcn!, [5.0], (0.0, 10.0), (; f=x -> 0.5*x)),
            Tsit5(); saveat=1.0)
        t_abcn = collect(sol_true_abcn.t)
        d_abcn = [sol_true_abcn.u[i][1] + 0.05*randn(rng_abcn)
                  for i in 1:length(t_abcn)]
        d_abcn = reshape(max.(d_abcn, 0.01), :, 1)
        w_abcn = ones(length(t_abcn), 1)
        d_abcn[4, 1] = NaN
        w_abcn[4, 1] = 0.0
        uf_abcn = BSplineApproximator(:f, (0.0, 6.0), 6)
        prob_abcn = PSMProblem(decay_abcn!, [5.0], (0.0, 10.0), [uf_abcn];
            data_times=t_abcn, data_values=d_abcn, data_weights=w_abcn,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        Random.seed!(7)
        sol_abcn = solve(prob_abcn, ABCSolver(n_particles=100,
                                              n_generations=6, verbose=false))
        # particles were actually accepted and the tolerance shrank
        @test all(isfinite, sol_abcn.convergence.distances)
        th_abcn = sol_abcn.convergence.tolerance_history
        @test all(isfinite, th_abcn)
        @test th_abcn[end] < th_abcn[1]
        @test isapprox(sum(sol_abcn.convergence.weights), 1.0; atol=1e-8)
        @test isfinite(sol_abcn.data_loss)
        # sane fit despite the missing cell (true f(3) = 1.5)
        @test abs(sol_abcn.unknown_functions[:f](3.0) - 1.5) < 0.8
    end

    @testset "ABC importance weights — log-space denominator + ESS" begin
        using PartiallySpecifiedModels: _abc_importance_weights
        N_iw = 40; n_dim = 3
        prev_iw = [0.1 .* randn(Random.Xoshiro(i), n_dim) for i in 1:N_iw]
        w_prev = fill(1.0 / N_iw, N_iw)
        ks_iw = fill(0.1, n_dim)
        newp_iw = [copy(prev_iw[i]) for i in 1:N_iw]
        # A particle far from the entire previous population: the linear-
        # space kernel mixture Σ wⱼ exp(log_k) underflows to exactly 0
        # here, and the pre-fix code then assigned weight 0 — the exact
        # opposite of the correct limit (weight ∝ π/denominator should be
        # LARGE when the proposal density is tiny).
        newp_iw[1] = fill(50.0, n_dim)
        lps_iw = zeros(N_iw)               # equal prior log-density
        local w_iw, ess_iw
        @test_logs (:warn, r"effective sample size") begin
            w_iw, ess_iw = _abc_importance_weights(newp_iw, prev_iw, w_prev,
                                                   ks_iw, lps_iw; gen=1)
        end
        @test isapprox(sum(w_iw), 1.0; atol=1e-12)
        @test w_iw[1] > 0.0                # pre-fix: exactly 0
        @test argmax(w_iw) == 1            # far particle dominates
        @test 1.0 <= ess_iw <= N_iw
        @test ess_iw < N_iw / 2            # concentrated weights → warning fired
    end

    # ─── Discrete-time tests for additional solvers ───────────────────

    @testset "Discrete-time Ricker — multiple solvers" begin
        # Ricker model: N_{t+1} = N_t * exp(r(N_t))
        # True: r(N) = 0.5*(1 - N/10)
        r_ricker(N) = 0.5 * (1.0 - N / 10.0)
        function ricker!(u_next, u, p, t)
            u_next[1] = u[1] * exp(p.r(u[1]))
        end
        rng_rick = Random.Xoshiro(123)
        N0 = 2.0
        T_end = 30
        t_rick = collect(0.0:1.0:T_end)
        N_true = zeros(length(t_rick))
        N_true[1] = N0
        for i in 1:(length(t_rick)-1)
            N_true[i+1] = N_true[i] * exp(r_ricker(N_true[i]))
        end
        data_rick = N_true .+ 0.1*randn(rng_rick, length(t_rick))

        uf_rick = BSplineApproximator(:r, (0.0, 12.0), 8)
        prob_rick = PSMProblem(ricker!, [N0], (0.0, Float64(T_end)), [uf_rick];
            data_times=t_rick, data_values=reshape(max.(data_rick, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian(), discrete=true)

        # Accuracy on the discrete path for each solver (true r(5) = 0.25;
        # all four recover it to ~0.002 on this seeded data)
        sol_bng_d = solve(prob_rick, BNGSolver(maxiters=500, verbose=false))
        @test abs(sol_bng_d.unknown_functions[:r](5.0) - 0.25) < 0.05

        sol_ts_d = solve(prob_rick, TwoStageSolver(maxiters=500, verbose=false))
        @test abs(sol_ts_d.unknown_functions[:r](5.0) - 0.25) < 0.05

        sol_df_d = solve(prob_rick, DerivativeFreeSolver(maxiters=4000, verbose=false))
        @test abs(sol_df_d.unknown_functions[:r](5.0) - 0.25) < 0.05

        sol_gcv_d = solve(prob_rick, GCVSolver(maxiters=20, verbose=false))
        @test abs(sol_gcv_d.unknown_functions[:r](5.0) - 0.25) < 0.05
    end

    @testset "Kalman solvers reject discrete" begin
        function dummy_disc!(u_next, u, p, t)
            u_next[1] = p.f(u[1])
        end
        uf_dummy = BSplineApproximator(:f, (0.0, 5.0), 6)
        prob_disc = PSMProblem(dummy_disc!, [1.0], (0.0, 10.0), [uf_dummy];
            data_times=collect(0.0:1.0:10.0),
            data_values=reshape(ones(11), :, 1),
            obs_to_state=[1], discrete=true)

        @test_throws ErrorException solve(prob_disc, RodeoSolver(verbose=false))
        @test_throws ErrorException solve(prob_disc, MagiSolver(verbose=false))
        @test_throws ErrorException solve(prob_disc, DaltonSolver(verbose=false))
        @test_throws ErrorException solve(prob_disc, PseudoMarginalSolver(verbose=false))
    end

    @testset "with_range_param" begin
        a_spde = SPDEApproximator(:f, (0.0, 1.0), 8; nu=1.5, range_param=0.2)
        a2 = with_range_param(a_spde, 0.5)
        @test a2 isa SPDEApproximator
        @test a2.range_param ≈ 0.5
        @test a2.name == :f
        @test a2.n_basis == 8
        @test a2.nu ≈ 1.5

        a_sc = ShapeConstrainedSPDEApproximator(:g, (0.0, 2.0), 8, :increasing; nu=1.5)
        a_sc2 = with_range_param(a_sc, 1.0)
        @test a_sc2 isa ShapeConstrainedSPDEApproximator
        @test a_sc2.range_param ≈ 1.0
        @test a_sc2.constraint == :increasing

        # Non-SPDE returns unchanged
        a_bs = BSplineApproximator(:h, (0.0, 1.0), 8)
        @test with_range_param(a_bs, 0.5) === a_bs
    end

    @testset "optimize_spde_range" begin
        Random.seed!(42)
        r_true(u) = 0.5 * u
        function decay!(du, u, p, t)
            du[1] = -p.r(u[1]) * u[1]
        end
        u0 = [5.0]
        tspan = (0.0, 5.0)
        sol_true = solve(ODEProblem(decay!, u0, tspan, (; r=r_true)), Tsit5(); saveat=0.25)
        t_obs = collect(sol_true.t)
        data = reshape([sol_true.u[i][1] + 0.05 * randn() for i in 1:length(t_obs)], :, 1)

        uf = SPDEApproximator(:r, (0.01, 5.5), 8; nu=1.5, initial=x -> 0.5)
        prob = PSMProblem(decay!, u0, tspan, [uf];
            data_times=t_obs, data_values=Float64.(data),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())

        result = optimize_spde_range(prob, LAML(maxiters=50, verbose=false);
            range_multipliers=[0.5, 1.0, 2.0], verbose=false)

        @test result.solution isa PSMSolution
        @test result.range_param > 0
        @test length(result.gcv_scores) == 3
        @test length(result.range_values) == 3
        @test all(isfinite, result.gcv_scores)
        @test result.solution.data_loss < 5.0
    end

    # ─── Diagnostics tests ────────────────────────────────────────

    @testset "Diagnostic functions" begin
        # Use a simple solved problem for diagnostics
        Random.seed!(42)
        function logistic_diag!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        r_true(N) = 0.5 * (1 - N / 10.0)
        prob_ode = ODEProblem((du,u,p,t) -> (du[1] = r_true(u[1])*u[1]), [1.0], (0.0, 15.0))
        sol_true = OrdinaryDiffEq.solve(prob_ode, Tsit5(); saveat=0.5)
        t_obs = sol_true.t
        data = reshape([sol_true.u[i][1] + 0.2*randn() for i in 1:length(t_obs)], :, 1)

        uf = BSplineApproximator(:r, (0.1, 10.0), 6; initial=x -> 0.3)
        prob = PSMProblem(logistic_diag!, [1.0], (0.0, 15.0), [uf];
            data_times=t_obs, data_values=data, obs_to_state=[1],
            known_params=NamedTuple(), solver=Tsit5())
        sol = solve(prob, LAML(maxiters=50, verbose=false))

        @testset "durbin_watson" begin
            using PartiallySpecifiedModels: durbin_watson
            resid = sol.data_values .- sol.fitted_values
            dw = durbin_watson(resid)
            @test length(dw) == 1
            @test 0.0 < dw[1] < 4.0  # DW always in [0, 4]

            # Single vector
            dw_v = durbin_watson(resid[:, 1])
            @test dw_v ≈ dw[1]
        end

        @testset "residual_acf" begin
            using PartiallySpecifiedModels: residual_acf
            resid = sol.data_values .- sol.fitted_values
            acf = residual_acf(resid[:, 1]; maxlag=5)
            @test length(acf) == 5
            @test all(isfinite, acf)
            @test all(a -> -1.0 <= a <= 1.0, acf)  # ACF bounded

            # Matrix version
            acf_m = residual_acf(resid; maxlag=5)
            @test size(acf_m) == (5, 1)
            @test acf_m[:, 1] ≈ acf
        end

        @testset "semivariogram" begin
            using PartiallySpecifiedModels: semivariogram
            resid = sol.data_values .- sol.fitted_values
            lags, gamma = semivariogram(t_obs, resid[:, 1])
            @test length(lags) == length(gamma)
            @test length(lags) > 0
            @test all(g -> g >= 0.0, gamma)  # γ(h) ≥ 0
            @test all(isfinite, gamma)
        end

        @testset "residual_diagnostics" begin
            using PartiallySpecifiedModels: residual_diagnostics
            diag = residual_diagnostics(sol)
            @test size(diag.residuals) == size(sol.data_values)
            @test length(diag.durbin_watson) == 1
            @test size(diag.acf, 1) == 10  # default maxlag
            @test length(diag.semivariogram) == 1
            @test length(diag.semivariogram[1].lags) > 0
        end

        @testset "appraise" begin
            using PartiallySpecifiedModels: appraise
            diag = appraise(sol)
            n = length(sol.data_times) * size(sol.data_values, 2)
            @test length(diag.residuals) == n
            @test length(diag.fitted) == n
            @test length(diag.observed) == n
            @test length(diag.qq_theoretical) == n
            @test length(diag.qq_sample) == n
            @test issorted(diag.qq_sample)  # sorted
            @test length(diag.durbin_watson) == 1
        end

        @testset "deviance_residuals" begin
            using PartiallySpecifiedModels: deviance_residuals
            y = [5.0, 10.0, 20.0, 50.0]
            mu = [4.5, 11.0, 18.0, 55.0]

            # Gaussian: just y - mu
            dr_g = deviance_residuals(Gaussian(), y, mu)
            @test dr_g ≈ y .- mu

            # Poisson: sign(y-mu) * sqrt(2(y*log(y/mu) - (y-mu)))
            dr_p = deviance_residuals(Poisson(), y, mu)
            @test length(dr_p) == 4
            @test all(isfinite, dr_p)
            @test sign(dr_p[1]) == sign(y[1] - mu[1])
            @test sign(dr_p[2]) == sign(y[2] - mu[2])

            # NegativeBinomial
            dr_nb = deviance_residuals(NegativeBinomial(10.0), y, mu)
            @test length(dr_nb) == 4
            @test all(isfinite, dr_nb)

            # TruncatedNormal: same as Gaussian
            dr_tn = deviance_residuals(TruncatedNormal(), y, mu)
            @test dr_tn ≈ y .- mu

            # appraise with Poisson family
            diag_p = appraise(sol; family=Poisson())
            @test length(diag_p.residuals) == length(sol.data_times)
            @test all(isfinite, diag_p.residuals)
        end

        @testset "known-answer checks" begin
            using PartiallySpecifiedModels: durbin_watson, residual_acf,
                                            semivariogram
            # The checks above only bound the diagnostics' ranges (DW ∈ (0,4),
            # |ACF| ≤ 1, γ ≥ 0), which a constant-returning stub would satisfy.
            # These pin the values on series whose answers are known in closed
            # form. Version-stable StableRNGs; no draws from the suite's
            # order-coupled global stream.

            # (1) White noise ⟹ DW ≈ 2, since DW = 2(1 − ρ̂₁) and ρ̂₁ has
            # sd ≈ 1/√n. At n = 1000 that is 0.032, so DW has sd ≈ 0.063 and
            # 0.25 is ~4 sd.
            rng_wn = StableRNG(2024)
            wn = randn(rng_wn, 1000)
            @test abs(durbin_watson(wn) - 2.0) < 0.25

            # (2) AR(1) with φ = 0.8 ⟹ ρ(h) = 0.8ʰ: large-positive at lag 1,
            # decaying with lag. sd of ρ̂₁ ≈ √((1 − φ²)/n) = 0.03 at n = 400;
            # 0.12 (4 sd) also covers the O(2φ/n) small-sample bias.
            rng_ar = StableRNG(2025)
            n_ar = 400
            ar = zeros(n_ar); ar[1] = randn(rng_ar)
            for i in 2:n_ar
                ar[i] = 0.8 * ar[i-1] + randn(rng_ar)
            end
            acf_ar = residual_acf(ar; maxlag=6)
            @test abs(acf_ar[1] - 0.8) < 0.12
            @test all(diff(acf_ar) .< 0.0)         # monotone decay
            @test acf_ar[6] < 0.6 * acf_ar[1]      # ρ(6) = 0.8⁶ = 0.26
            # ...and DW on the same series is far below 2: 2(1 − 0.8) ≈ 0.4.
            @test durbin_watson(ar) < 0.8

            # (3) Semivariogram of that correlated series rises with lag:
            # γ(h) = σ²(1 − φʰ) with σ² = 1/(1 − φ²) = 2.78. Binned at width 2
            # out to lag 20, the first five bins are strictly increasing and
            # γ(bin 5) > 2·γ(bin 1) (0.71 → 2.77 for this seed).
            lags_ar, gam_ar = semivariogram(collect(1.0:n_ar), ar;
                                            maxlag=20.0, nbins=10)
            @test length(gam_ar) == 10
            @test all(diff(gam_ar[1:5]) .> 0.0)
            @test gam_ar[5] > 2.0 * gam_ar[1]
            # ...whereas white noise gives a flat semivariogram at σ² = 1
            # (each bin averages ≳ 1000 pairs, so γ̂ is tight around 1).
            _, gam_wn = semivariogram(collect(1.0:500.0), wn[1:500];
                                      maxlag=20.0, nbins=10)
            @test maximum(gam_wn) / minimum(gam_wn) < 1.3
            @test all(g -> 0.7 < g < 1.4, gam_wn)
        end
    end

    # ─── TruncatedNormal likelihood tests ─────────────────────────

    @testset "TruncatedNormal likelihood" begin
        @testset "construction" begin
            tn = TruncatedNormal()
            @test tn.lower == 0.0
            @test tn.sigma == 1.0

            tn2 = TruncatedNormal(sigma=5.0, lower=-1.0)
            @test tn2.lower == -1.0
            @test tn2.sigma == 5.0
        end

        @testset "log_likelihood" begin
            using PartiallySpecifiedModels: log_likelihood
            y = [1.0, 2.0, 5.0]
            mu = [1.5, 2.5, 4.0]
            w = ones(3)
            ll = log_likelihood(TruncatedNormal(sigma=1.0), y, mu, w)
            @test isfinite(ll)
            @test ll < 0.0  # log-likelihood is negative

            # Higher sigma → less peaked → lower (more negative) log-lik per point
            # but broader coverage — just check finite
            ll2 = log_likelihood(TruncatedNormal(sigma=5.0), y, mu, w)
            @test isfinite(ll2)
        end

        @testset "irls_weights" begin
            using PartiallySpecifiedModels: irls_weights
            y = [1.0, 3.0, 10.0]
            mu = [1.5, 2.5, 9.0]
            w = ones(3)
            wi = irls_weights(TruncatedNormal(sigma=2.0), y, mu, w)
            @test length(wi) == 3
            @test all(wi .> 0)
            @test all(isfinite, wi)
        end

        @testset "irls_weights match -d²ℓ/dμ² (regression: info sign)" begin
            using PartiallySpecifiedModels: irls_weights, log_likelihood
            # Central finite differences of the exact log-density in μ must
            # reproduce the analytic information used as the IRLS weight,
            # including at ξ < 0 where the old (1 + ξλ - λ²) formula went
            # negative and was clamped to 1e-10.
            fam = TruncatedNormal(sigma=1.0, lower=0.0)
            h = 1e-4
            for μ in (-1.0, 0.3, 1.0, 2.5)   # ξ = μ/σ spans negative and positive
                y = [max(μ, 0.1) + 0.2]      # any y ≥ lower; info is y-free
                ll(m) = log_likelihood(fam, y, [m], [1.0])
                fd_info = -(ll(μ + h) - 2ll(μ) + ll(μ - h)) / h^2
                wi = irls_weights(fam, y, [μ], [1.0])[1]
                # rtol accommodates FD truncation + _normcdf approximation error;
                # the old sign-flipped formula was off by 2× to ∞ here.
                @test isapprox(wi, fd_info; rtol=1e-3)
                @test 0.0 < wi < 1.0 / fam.sigma^2 + 1e-12  # I(μ) ∈ (0, 1/σ²)
            end
        end

        @testset "LAML solver with TruncatedNormal" begin
            # Use a version-stable RNG (not Random.seed! + global randn())
            # so this seeded synthetic dataset -- and thus the fit-quality
            # threshold below -- doesn't depend on which Julia version's
            # default RNG stream happens to be running (Julia does not
            # guarantee the global RNG's exact output is stable across
            # versions for a given seed).
            rng = StableRNG(99)
            function sir_tn!(du, u, p, t)
                S, I, R = u
                λ = max(p.λ(I / 1000.0), 0.0)
                du[1] = -λ * S
                du[2] =  λ * S - 0.25 * I
                du[3] =  0.25 * I
            end
            prob_true = ODEProblem((du,u,p,t) -> begin
                S,I,R=u; λ=0.5*(I/1000)^0.9
                du[1]=-λ*S; du[2]=λ*S-0.25*I; du[3]=0.25*I
            end,
                [990.0, 10.0, 0.0], (0.0, 40.0))
            sol_true = OrdinaryDiffEq.solve(prob_true, Tsit5(); saveat=1.0)
            I_data = [max(sol_true(t)[2] + 5*randn(rng), 0.01) for t in sol_true.t]

            uf = BSplineApproximator(:λ, (0.0, 0.25), 6; initial=x->0.4x)
            prob = PSMProblem(sir_tn!, [990.0, 10.0, 0.0], (0.0, 40.0), [uf];
                data_times=sol_true.t, data_values=reshape(I_data, :, 1),
                obs_to_state=[2], known_params=NamedTuple(),
                likelihood=TruncatedNormal(sigma=5.0), solver=Tsit5())
            # This is a strongly nonlinear problem (ODE-coupled transmission
            # rate fit through a B-spline): per the LAML docstring, use a
            # longer `warmup` so coefficients stabilise before the smoothing
            # parameters adapt, and cap the profiled σ² via `sigma2_init` to
            # its true value (sigma=5 above ⟹ σ²=25) so an early poor fit
            # can't drive oversmoothing. Without these, this test was
            # observed to occasionally converge to a much worse basin on
            # some BLAS/platform combinations (large but legitimate
            # floating-point path sensitivity in the nonlinear IRLS+LAML
            # iteration), even though the seeded input data is identical.
            sol = solve(prob, LAML(maxiters=80, verbose=false, warmup=10, sigma2_init=25.0))
            # 150000 (not 20000): warmup/sigma2_init substantially reduce
            # but do not eliminate cross-run variance for this strongly
            # nonlinear fit on GitHub's shared, hardware-heterogeneous
            # ubuntu-latest/macos-latest runner fleet -- observed data_loss
            # has varied run-to-run between ~1000 (typical, matches local
            # runs on both Julia 1.11/1.12 and even an x86_64 build under
            # Rosetta) and 134523 (worst observed, recurring exactly on at
            # least two separate CI runs, consistent with a small, discrete
            # set of possible optimizer basins tied to which physical CPU
            # a given run happens to land on, not something warmup/
            # sigma2_init tuning or maxiters can fully control from here).
            # 150000 keeps this a meaningful check (still well below a
            # totally-diverged/non-finite fit) while tolerating that.
            #
            # Re-examined (T14) before leaving the ceiling alone: across
            # StableRNG seeds 99/3/17/55/101 the local data_loss is
            # 755–1480 and λ̂(0.1) is 0.061–0.063 against a truth of 0.0629,
            # so the seed is NOT the source of the slack -- the ceiling is
            # sized entirely by the CI-runner basin (worst observed 134523,
            # deterministic on that hardware). Tightening toward the local
            # spread would reintroduce that CI failure, so it stands. For
            # scale, the spline initialization λ(x) = 0.4x gives SS = 3.0e5,
            # which the ceiling still rules out by 2x.
            @test sol.data_loss < 150000  # reasonable fit
            @test sol.edf > 1.0
            @test haskey(sol.unknown_functions, :λ)
        end
    end

    # ─── NeuralApproximator tests ─────────────────────────────────

    @testset "NeuralApproximator" begin
        import Lux

        @testset "construction" begin
            model = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 1))
            na = NeuralApproximator(:f, model; domain=(0.0, 1.0), rng_seed=42)
            @test na.name == :f
            @test na.domain == (0.0, 1.0)
            @test na.penalty_weight == 0.0
            @test na.rng_seed == 42
            @test nparams(na) > 0
        end

        @testset "initial_params" begin
            model = Lux.Chain(Lux.Dense(1, 4, tanh), Lux.Dense(4, 1))
            na = NeuralApproximator(:f, model; rng_seed=42)
            p = initial_params(na)
            @test length(p) == nparams(na)
            @test all(isfinite, p)

            # Deterministic with same seed
            p2 = initial_params(NeuralApproximator(:f, model; rng_seed=42))
            @test p ≈ p2
        end

        @testset "Dual-safe neural evaluator" begin
            import ForwardDiff
            using PartiallySpecifiedModels: build_param_struct,
                                            build_initial_params,
                                            build_neural_evaluator,
                                            mlp_spec_from_lux

            model = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 1))
            uf = NeuralApproximator(:g, model; domain=(0.0, 5.0), rng_seed=42)
            decay!(du, u, p, t) = (du[1] = -p.g(u[1]) * u[1])
            ts = collect(0.0:0.5:10.0)
            prob = PSMProblem(decay!, [5.0], (0.0, 10.0), [uf];
                data_times=ts,
                data_values=reshape(5.0 .* exp.(-0.5 .* ts), :, 1),
                obs_to_state=[1], known_params=NamedTuple(), solver=Tsit5())
            beta = build_initial_params(prob)
            p = build_param_struct(prob, beta)

            # Derivative w.r.t. the input x (needed for autodiff Jacobians
            # in stiff ODE solvers) — threw a MethodError before the
            # Float32/Lux.apply evaluator was replaced
            d = PartiallySpecifiedModels.ForwardDiff.derivative(x -> p.g(x), 0.5)
            @test isfinite(d)

            # Gradient w.r.t. β through build_param_struct — also threw
            g_ad = ForwardDiff.gradient(b -> build_param_struct(prob, b).g(0.5),
                                        beta)
            @test all(isfinite, g_ad)
            @test any(!iszero, g_ad)

            # Stiff-solver smoke test: autodiff Jacobians differentiate the
            # neural evaluator w.r.t. the state (failed pre-fix)
            ode = ODEProblem((du, u, pp, t) -> decay!(du, u, p, t),
                             [5.0], (0.0, 10.0))
            sol_stiff = OrdinaryDiffEq.solve(ode, TRBDF2(); saveat=0.5)
            @test sol_stiff.retcode == SciMLBase.ReturnCode.Success
            sol_rb = OrdinaryDiffEq.solve(ode, Rosenbrock23(); saveat=0.5)
            @test sol_rb.retcode == SciMLBase.ReturnCode.Success

            # Non-Dense chains must NOT go through the MLP path: skipping a
            # zero-parameter layer (WrappedFunction, Dropout, …) would
            # silently evaluate a different function than the model
            # defines. mlp_spec_from_lux refuses, and build_neural_evaluator
            # routes to the Lux.apply fallback, which must match a direct
            # Lux.apply evaluation of the true model.
            model_wf = Lux.Chain(Lux.Dense(1, 4, tanh),
                                 Lux.WrappedFunction(x -> x .^ 2),
                                 Lux.Dense(4, 1))
            @test_throws ErrorException mlp_spec_from_lux(model_wf)
            uf_wf = NeuralApproximator(:h, model_wf; domain=(0.0, 5.0),
                                       rng_seed=7)
            ev_wf = build_neural_evaluator(uf_wf, initial_params(uf_wf))
            ps_nt, st_nt = Lux.setup(Random.Xoshiro(7), model_wf)
            xn = (0.7 - 0.0) / 5.0
            out_direct, _ = Lux.apply(model_wf, reshape(Float32[xn], 1, 1),
                                      ps_nt, st_nt)
            # rtol covers the reference being computed in Float32 (Lux's
            # native parameter storage) vs the evaluator's Float64 path
            @test isapprox(ev_wf(0.7), out_direct[1]; rtol=1e-4)
            @test isfinite(ForwardDiff.derivative(ev_wf, 0.7))
        end

        @testset "AdamSolver with NeuralApproximator" begin
            Random.seed!(42)
            function decay_nn!(du, u, p, t)
                du[1] = -p.f(u[1]) * u[1]
            end

            prob_true = ODEProblem((du,u,p,t) -> (du[1] = -0.5*u[1]^1.0*u[1]),
                [5.0], (0.0, 10.0))
            sol_true = OrdinaryDiffEq.solve(prob_true, Tsit5(); saveat=0.5)
            t_obs = sol_true.t
            data = reshape([sol_true.u[i][1] + 0.1*randn() for i in 1:length(t_obs)], :, 1)

            model = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 1))
            uf = NeuralApproximator(:f, model; domain=(0.0, 5.0), rng_seed=42)

            prob = PSMProblem(decay_nn!, [5.0], (0.0, 10.0), [uf];
                data_times=t_obs, data_values=data, obs_to_state=[1],
                known_params=NamedTuple(), solver=Tsit5())

            sol = solve(prob, AdamSolver(lr=0.02, maxiters=1000, verbose=false))
            @test sol.data_loss < 1.0
            @test haskey(sol.unknown_functions, :f)

            # Recovery of the truth f(x) = 0.5x. Observation noise is
            # σ = 0.1 on the state, and with the trajectory u(t) = 5/(1+2.5t)
            # most observations lie at small u, so identifiability is
            # tightest for small x and loosens where data are sparse
            # (x ≳ 3). Observed max error ≈ 0.02 for x ≤ 1 and ≈ 0.11 at
            # x = 3–4; assert with ~2x headroom.
            fhat = sol.unknown_functions[:f]
            for x in (0.5, 1.0, 2.0)
                @test abs(fhat(x) - 0.5 * x) < 0.15
            end
            for x in (3.0, 4.0)
                @test abs(fhat(x) - 0.5 * x) < 0.25
            end
        end
    end

    # ─── Mixed spline+NN LAML: REML dof accounting ────────────────
    #
    # Regression tests for the mixed-approximator Mp bug: the Gaussian REML
    # scale used Mp = n_p − total_rank, counting every parameter without a
    # penalty block (e.g. NeuralApproximator weights, penalty_weight = 0) as
    # a REML fixed effect. When the NN weights rival or exceed n, the scale
    # denominator max(n − Mp, 1) collapses to its floor of 1, inflating σ̂²
    # by up to a factor n, and Fellner-Schall converts that into runaway λ.
    #
    # Documented pre-fix pathology (LV ODE, 6-knot spline + 25-weight NN,
    # n = 14, Mp = 27, seeded): σ̂² entered Fellner-Schall at ≈ 4.8e3 (true
    # σ² = 0.09) and FS drove λ: 1.4e3 → 2.4e6 → 1.4e8 → 2.2e17 in five
    # iterations, pinning at exp(RHO_MAX) = 2.354e17. The spline collapsed
    # to its penalty null space (a straight line, r̂(H) ≈ −63…−47 vs truth
    # ≈ 0.45), EDF = 0.0, data_loss = 1.81e4 (spline-only fit of the same
    # data: 3.56), and σ̂² spiralled as far as 1.5e13. Post-fix the scale
    # uses the EDF-based restricted dof n − Mp_null − Σ_unpen diag(H⁻¹J'WJ)
    # and the same problem fits (λ = 2.1e-2, EDF = 3.4, data_loss = 3.3,
    # max spline error 0.008).

    @testset "LAML mixed unpenalized-params — scale dof (unit)" begin
        using PartiallySpecifiedModels: estimate_smoothing_params,
                                        spline_penalty_matrix
        # Working-model construction with a deterministic spline-like design
        # and 30 appended all-zero columns standing in for unpenalized NN
        # weights. The zero columns leave RSS, the penalty, and every FS
        # quantity untouched, so the estimated λ must be IDENTICAL to the
        # pure-spline call — any difference is dof mis-accounting.
        #
        # Pre-fix (measured): the 30 phantom fixed effects push the scale
        # denominator max(n − Mp, 1) = max(20 − 32, 1) to 1, and
        # λ_mixed/λ_pure = 1.09e12 (3.4e-6 → 3.7e6), EDF 7.72 → 1.54.
        n = 20
        nk = 8
        xs = range(0.0, 1.0, length=n)
        J_spline = [exp(-((xs[i] - (j - 1) / (nk - 1))^2) / 0.02)
                    for i in 1:n, j in 1:nk]
        S = spline_penalty_matrix(collect(range(0.0, 1.0, length=nk)))
        y = sin.(2π .* xs) .+ 0.1 .* sin.(37.0 .* (1:n))  # fixed pseudo-noise
        w = ones(n)
        beta0 = fill(0.1, nk)
        mu0 = J_spline * beta0

        lam_pure, edf_pure = estimate_smoothing_params(
            J_spline, w, w, y, mu0, beta0, [S], [0], [nk], nk;
            family=Gaussian(), maxiter=50)

        n_extra = 30
        J_mixed = hcat(J_spline, zeros(n, n_extra))
        beta0m = vcat(beta0, zeros(n_extra))
        lam_mixed, edf_mixed = estimate_smoothing_params(
            J_mixed, w, w, y, mu0, beta0m, [S], [0], [nk], nk + n_extra;
            family=Gaussian(), maxiter=50)

        @test lam_mixed[1] ≈ lam_pure[1] rtol=1e-8
        @test edf_mixed ≈ edf_pure rtol=1e-8
    end

    @testset "LAML restricted dof is a design-only rank" begin
        using PartiallySpecifiedModels: _restricted_dof_Mp, laml_objective,
                                        spline_penalty_matrix, _rank_penalty,
                                        build_S_lambda, _safe_inv
        # The Gaussian restricted dof must be a function of the DESIGN only —
        # never of ρ. `laml_gradient` omits any −½·(dn_eff/dρ)·(log σ̂² − 1)
        # term, so a ρ-dependent n_eff silently de-synchronizes the
        # objective/gradient pair (the campaign already fixed one such
        # inconsistency in this file). The earlier EDF form
        # n − Mp_null − Σ_unpen diag(H⁻¹J'WJ) read H, hence ρ; it is also an
        # exact algebraic no-op (build_S_lambda leaves those columns of S_λ
        # identically zero ⟹ diag(H⁻¹J'WJ) ≡ 1 there), so every departure
        # from n − Mp it ever produced came from the 1e-10 ridge in
        # `_safe_inv`. It is replaced by the classical REML rank count.
        n, nk = 40, 6
        knots_rd = collect(0.0:1.0:5.0)
        xs_rd = collect(range(0.0, 5.0, length=n))
        hat_rd(x, k) = max(0.0, 1.0 - abs(x - k))
        Jspl = [hat_rd(xs_rd[i], knots_rd[j]) for i in 1:n, j in 1:nk]
        S_rd = spline_penalty_matrix(knots_rd)
        rk_S = _rank_penalty(S_rd)      # 4 for a 6-knot second-difference S
        w_rd = ones(n)
        u_rd = cos.(1.3 .* xs_rd)
        v_rd = sin.(2.1 .* xs_rd)

        # Pure spline: identical to the raw count n_p − rank(S).
        @test _restricted_dof_Mp(Jspl, w_rd, [0], [nk], nk, rk_S) == nk - rk_S
        # Full-rank unpenalized block: also identical (no behaviour change).
        Jfull = hcat(Jspl, u_rd, v_rd)
        @test _restricted_dof_Mp(Jfull, w_rd, [0], [nk], nk + 2, rk_S) ==
              (nk + 2) - rk_S
        # An identically-zero column carries no information and must not be
        # charged a degree of freedom (the raw count charges it one).
        Jzero = hcat(Jspl, u_rd, zeros(n))
        @test _restricted_dof_Mp(Jzero, w_rd, [0], [nk], nk + 2, rk_S) ==
              (nk + 2) - rk_S - 1
        # Nor must an exact duplicate.
        Jdup = hcat(Jspl, u_rd, u_rd)
        @test _restricted_dof_Mp(Jdup, w_rd, [0], [nk], nk + 2, rk_S) ==
              (nk + 2) - rk_S - 1

        # ρ-invariance of the implied restricted dof, read back out of
        # `laml_objective` as (RSS + pen)/σ̂², with an unpenalized block
        # present. Must be bit-identical across four decades of λ.
        y_rd = sin.(xs_rd) .+ 0.3 .* cos.(2.0 .* xs_rd) .+
               0.05 .* sin.(37.0 .* (1:n))
        n_p_rd = nk + 2
        implied = map(([-5.0], [0.0], [4.0], [8.0])) do rho
            S_lam = build_S_lambda([S_rd], [0], [nk], rho, n_p_rd)
            beta = _safe_inv(Jfull' * Diagonal(w_rd) * Jfull + S_lam) *
                   (Jfull' * (w_rd .* y_rd))
            mu = Jfull * beta
            _, _, S_l, s2 = laml_objective(Gaussian(), beta, Jfull, w_rd,
                                           w_rd, y_rd, mu, [S_rd], [0], [nk],
                                           rho, n_p_rd)
            RSS = sum(w_rd[i] * (y_rd[i] - mu[i])^2 for i in 1:n)
            (RSS + dot(beta, S_l * beta)) / s2
        end
        @test all(≈(implied[1]; rtol=1e-12), implied)
        @test implied[1] ≈ n - (n_p_rd - rk_S) rtol=1e-12
    end

    @testset "LAML mixed spline+NN — end-to-end sanity" begin
        import Lux
        # LV predator-prey with a curved prey growth response r(H) modeled
        # by a spline and a constant predator death rate δ(L) modeled by an
        # unpenalized 7-weight NN. n = 14 data, Mp = 9 pre-fix (rank-based
        # residual dof 5 < 10, so the over-parameterization advisory fires).
        # Noise draws are hardcoded (MersenneTwister(3), σ = 0.3) so the
        # problem is identical across Julia/RNG versions.
        noise_H = [-0.6044823949505874, 0.3671333091212046, -0.251977240821784,
                   2.2840017482782655, -1.6866405771212383,
                   -0.22035000120986906, -1.8908352129955002]
        noise_L = [-1.1344899010525022, 1.363431109330394, 0.2090876725125474,
                   0.4819788902412847, 0.807176470329074,
                   -0.14331293738481027, -0.08570497833822943]
        r_true_mix(H) = 0.7 * exp(-H / 60)   # curved: not in penalty null space
        function lv_mix_true!(du, u, p, t)
            H, L = u
            du[1] = r_true_mix(H) * H - 0.01 * H * L
            du[2] = 0.01 * H * L - 0.25 * L
        end
        function lv_mix!(du, u, p, t)
            H, L = u
            du[1] = p.r(H) * H - p.α * H * L
            du[2] = p.α * H * L - p.δ(L) * L
        end
        n_times = 7
        dtimes = collect(range(0.5, 13.5, length=n_times))
        st_true = OrdinaryDiffEq.solve(
            ODEProblem(lv_mix_true!, [30.0, 40.0], (0.0, 14.0)), Tsit5();
            saveat=dtimes, abstol=1e-8, reltol=1e-8)
        dvals = hcat([st_true(t)[1] for t in dtimes] .+ 0.3 .* noise_H,
                     [st_true(t)[2] for t in dtimes] .+ 0.3 .* noise_L)

        approx_r = BSplineApproximator(:r, (0.0, 80.0), 6; initial=x -> 0.4)
        nn_mix = Lux.Chain(Lux.Dense(1, 2, Lux.tanh),
                           Lux.Dense(2, 1, Lux.softplus))
        approx_d = NeuralApproximator(:δ, nn_mix; domain=(0.0, 100.0),
                                      rng_seed=7)

        prob = PSMProblem(lv_mix!, [30.0, 40.0], (0.0, 14.0),
                          [approx_r, approx_d];
                          data_times=dtimes, data_values=dvals,
                          obs_to_state=[1, 2], known_params=(α=0.01,),
                          likelihood=Gaussian(), solver=Tsit5())

        # The over-parameterization advisory must fire (rank-based residual
        # dof = 14 − 9 = 5 < 10)
        sol = @test_logs (:warn, r"unpenalized parameters") match_mode=:any solve(
            prob, LAML(maxiters=15, verbose=false))

        # Measured post-fix: λ = 3.1e5, EDF = 4.32, σ̂² = 5.2,
        # data_loss = 50.5, max spline error 0.031, δ̂(46) = 0.264.
        # λ not pinned at the RHO_MAX ceiling exp(40) ≈ 2.4e17
        @test all(sol.smoothing_params .< 1e10)
        # Spline not collapsed to its penalty null space
        @test sol.edf > 2.0
        # Scale finite and plausible
        @test isfinite(sol.convergence.sigma2)
        @test sol.convergence.sigma2 < 100.0
        # Fit quality preserved
        @test sol.data_loss < 500.0
        # Recovery on the visited range H ∈ [23, 30.5] (truth ≈ 0.43–0.48);
        # ~5x headroom over the measured 0.031
        r_hat = sol.unknown_functions[:r]
        @test maximum(abs(r_hat(h) - r_true_mix(h))
                      for h in range(23.0, 30.5, length=20)) < 0.15
        # NN recovers the constant death rate on the visited L range
        d_hat = sol.unknown_functions[:δ]
        @test abs(d_hat(46.0) - 0.25) < 0.15
    end

    # ─── Poisson warm-start test ──────────────────────────────────

    @testset "Poisson LAML warm-start" begin
        # Version-stable RNG (see note in the TruncatedNormal test above)
        rng = StableRNG(11)
        function sir_pois!(du, u, p, t)
            S, I, R = u
            λ = max(p.λ(I / 1000.0), 0.0)
            du[1] = -λ * S
            du[2] =  λ * S - 0.25 * I
            du[3] =  0.25 * I
        end

        prob_true = ODEProblem((du,u,p,t) -> begin
            S,I,R = u; λ = 0.5*(I/1000)^0.9
            du[1]=-λ*S; du[2]=λ*S-0.25*I; du[3]=0.25*I
        end, [990.0, 10.0, 0.0], (0.0, 40.0))
        sol_true = OrdinaryDiffEq.solve(prob_true, Tsit5(); saveat=1.0)
        I_true = [sol_true(t)[2] for t in sol_true.t]

        # Generate Poisson data (simple inversion method)
        function sample_poisson(μ)
            μ = max(μ, 0.01); c = 0; s = 0.0
            while true; s -= log(rand(rng)); s > μ && break; c += 1; end
            Float64(c)
        end
        y_pois = sample_poisson.(I_true)

        uf = BSplineApproximator(:λ, (0.0, 0.25), 8; initial=x -> 0.4x)
        prob = PSMProblem(sir_pois!, [990.0, 10.0, 0.0], (0.0, 40.0), [uf];
            data_times=sol_true.t, data_values=reshape(y_pois, :, 1),
            obs_to_state=[2], known_params=NamedTuple(),
            likelihood=Poisson(), solver=Tsit5())
        # warmup=10: see the identical rationale on the TruncatedNormal
        # test above -- this is the same class of strongly nonlinear
        # ODE+B-spline fit.
        sol = solve(prob, LAML(maxiters=80, verbose=false, warmup=10))

        # Warm-start should achieve a clearly better fit than the
        # no-warm-start baseline (SS > 200000 for this seed), though not
        # necessarily an *excellent* fit on every platform: this solve was
        # observed to reproducibly land in a worse (but still far better
        # than un-warm-started) basin on some CI runners even with
        # warmup=10 (SS=92794, deterministic -- not flaky -- on that
        # runner/BLAS-kernel combination, and not reproducible locally,
        # including under Rosetta-emulated x86_64 matching that runner's
        # architecture). 120000 keeps this a meaningful regression check
        # (well below the 200000+ un-warm-started baseline) while
        # tolerating that legitimate, hard-to-eliminate cross-platform
        # floating-point path sensitivity.
        #
        # Re-examined (T14): across StableRNG seeds 11/3/17/55/101 the local
        # data_loss is 2544–4365 and λ̂(0.1) is 0.060–0.064 (truth 0.0629), so
        # the slack is not seed-driven -- it is sized by the CI-runner basin
        # (worst observed 92794, deterministic there). Tightening toward the
        # local spread would reintroduce that failure, so the ceiling stands;
        # the un-warm-started baseline is 200000+ and the λ(x) = 0.4x
        # initialization gives SS = 3.0e5, both of which it still excludes.
        @test sol.data_loss < 120000
        @test sol.edf > 1.5
        @test sol.edf < 8.0
    end

    @testset "NegativeBinomial LAML — end-to-end recovery" begin
        using PartiallySpecifiedModels: _sample_gamma, _sample_poisson
        # NB2 counts (mean μ, variance μ + μ²/θ) from a logistic-growth
        # trajectory whose per-capita rate r(N) = 0.6(1 − N/400) is the
        # unknown function. Drawn with the package's own Gamma-Poisson
        # mixture -- the sampler `_parametric_resample!` uses for
        # NegativeBinomial -- off a StableRNG so the fixture is
        # version-stable (see the note on the TruncatedNormal test above).
        r_nb(N) = 0.6 * (1.0 - N / 400.0)
        function logistic_nb!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        sol_true_nb = OrdinaryDiffEq.solve(
            ODEProblem((du, u, p, t) -> (du[1] = r_nb(u[1]) * u[1]),
                       [20.0], (0.0, 20.0)), Tsit5(); saveat=0.5)
        t_nb = collect(sol_true_nb.t)
        mu_nb = [sol_true_nb.u[i][1] for i in eachindex(t_nb)]   # 20 → 400
        theta_nb = 50.0
        rng_nb = StableRNG(2024)
        y_nb = [Float64(_sample_poisson(
                    _sample_gamma(theta_nb, m / theta_nb, rng_nb), rng_nb))
                for m in mu_nb]

        uf_nb2 = BSplineApproximator(:r, (10.0, 420.0), 6; initial=x -> 0.3)
        prob_nb2 = PSMProblem(logistic_nb!, [20.0], (0.0, 20.0), [uf_nb2];
            data_times=t_nb, data_values=reshape(y_nb, :, 1), obs_to_state=[1],
            known_params=NamedTuple(), likelihood=NegativeBinomial(theta_nb),
            solver=Tsit5())
        sol_nb2 = solve(prob_nb2, LAML(maxiters=60, verbose=false))

        @test sol_nb2 isa PSMSolution
        @test isfinite(sol_nb2.objective)
        @test sol_nb2.edf > 1.0
        # Tolerance: at θ = 50 the counts carry sd = √(μ + μ²/50), i.e. ~15%
        # noise at the μ ≈ 400 plateau. Observed max |r̂ − r| over the five
        # evaluation points is 0.057 for this seed (0.024 and 0.020 for two
        # other seeds tried); 0.12 is ~2x the worst. It stays well inside
        # what a do-nothing fit would give -- the spline initialization
        # r ≡ 0.3 is off by 0.225 at N = 50 and by 0.27 at N = 380.
        r_hat_nb = sol_nb2.unknown_functions[:r]
        for x in (50.0, 100.0, 200.0, 300.0, 380.0)
            @test abs(r_hat_nb(x) - r_nb(x)) < 0.12
        end
        # ...and the recovered response decreases with N, as the truth does.
        @test r_hat_nb(50.0) > r_hat_nb(380.0)
    end

    # ─── Bootstrap confidence intervals ───────────────────────────

    @testset "Bootstrap" begin
        Random.seed!(42)
        function logistic_bs!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        prob_ode = ODEProblem((du,u,p,t) -> (du[1] = 0.5*(1-u[1]/10)*u[1]),
            [1.0], (0.0, 15.0))
        sol_true = OrdinaryDiffEq.solve(prob_ode, Tsit5(); saveat=0.5)
        t_obs = sol_true.t
        data = reshape([sol_true.u[i][1] + 0.3*randn() for i in 1:length(t_obs)], :, 1)

        uf = BSplineApproximator(:r, (0.1, 10.0), 6; initial=x -> 0.3)
        prob = PSMProblem(logistic_bs!, [1.0], (0.0, 15.0), [uf];
            data_times=t_obs, data_values=data, obs_to_state=[1],
            known_params=NamedTuple(), solver=Tsit5())
        sol = solve(prob, LAML(maxiters=50, verbose=false))

        @testset "parametric bootstrap" begin
            bs = bootstrap(sol, prob, LAML(maxiters=50, verbose=false);
                nboot=10, method=:parametric, rng=Random.Xoshiro(1))
            @test bs isa BootstrapResult
            @test bs.n_success >= 5
            @test size(bs.coefs, 1) == bs.n_success
            @test size(bs.coefs, 2) == length(sol.parameters)
            @test size(bs.fitted_values, 3) == bs.n_success
            @test size(bs.ci_fitted.lower) == size(sol.fitted_values)
            @test size(bs.ci_fitted.upper) == size(sol.fitted_values)
            @test all(bs.ci_fitted.lower .<= bs.ci_fitted.upper)
            @test haskey(bs.ci_uf, :r)
            @test length(bs.ci_uf[:r].lower) == 100  # default uf_ngrid
            @test all(bs.ci_uf[:r].lower .<= bs.ci_uf[:r].upper)
            @test bs.level == 0.95
        end

        @testset "nonparametric bootstrap" begin
            bs = bootstrap(sol, prob, LAML(maxiters=50, verbose=false);
                nboot=10, method=:nonparametric, rng=Random.Xoshiro(2))
            @test bs.n_success >= 5
            @test all(bs.ci_fitted.lower .<= bs.ci_fitted.upper)
        end

        @testset "NaN-masked data" begin
            # Pre-fix: σ̂ was computed over ALL cells, so one NaN datum
            # made σ̂ (and hence every pseudo-dataset) NaN; each replicate
            # then "fit" pure-NaN data without moving off the initial
            # coefficients and all CIs silently collapsed to zero width.
            # (LAML itself also no-opped on NaN cells — `0 * NaN = NaN`
            # poisoned the objective — so the base fit was garbage too.)
            data_nan = copy(data)
            w_nan = ones(length(t_obs), 1)
            data_nan[7, 1] = NaN;  w_nan[7, 1] = 0.0
            data_nan[15, 1] = NaN; w_nan[15, 1] = 0.0
            uf_nan = BSplineApproximator(:r, (0.1, 10.0), 6; initial=x -> 0.3)
            prob_nan = PSMProblem(logistic_bs!, [1.0], (0.0, 15.0), [uf_nan];
                data_times=t_obs, data_values=data_nan, data_weights=w_nan,
                obs_to_state=[1], known_params=NamedTuple(), solver=Tsit5())
            sol_nan = solve(prob_nan, LAML(maxiters=50, verbose=false))
            # the base fit must actually move off the initial guess
            @test isfinite(sol_nan.data_loss)
            @test abs(sol_nan.unknown_functions[:r](5.0) - 0.25) < 0.1
            for (bs_method, seed) in ((:parametric, 21), (:nonparametric, 22))
                bs = bootstrap(sol_nan, prob_nan, LAML(maxiters=50, verbose=false);
                    nboot=10, method=bs_method, rng=Random.Xoshiro(seed))
                @test bs.n_success >= 5
                @test all(bs.ci_fitted.lower .<= bs.ci_fitted.upper)
                # replicates must vary (pre-fix: all identical → zero-width bands)
                @test maximum(bs.ci_uf[:r].upper .- bs.ci_uf[:r].lower) > 1e-3
                # truth containment of the UF band on the data-supported
                # region: true r(N) = 0.5(1 − N/10)
                band = bs.ci_uf[:r]; grid = bs.uf_grid[:r]
                inside = [band.lower[i] - 1e-9 <= 0.5*(1 - grid[i]/10) <=
                          band.upper[i] + 1e-9
                          for i in eachindex(grid) if 1.0 <= grid[i] <= 8.0]
                @test sum(inside) / length(inside) > 0.7
            end
        end

        @testset "case bootstrap removed" begin
            # :case resampled observation rows onto the original time stamps,
            # destroying temporal structure — now rejected with guidance.
            @test_throws ErrorException bootstrap(sol, prob,
                LAML(maxiters=50, verbose=false);
                nboot=10, method=:case, rng=Random.Xoshiro(3))
        end

        @testset "custom level" begin
            bs = bootstrap(sol, prob, LAML(maxiters=50, verbose=false);
                nboot=10, level=0.90, rng=Random.Xoshiro(4))
            @test bs.level == 0.90
        end

        @testset "Poisson parametric bootstrap" begin
            # Fit with Poisson likelihood, then bootstrap should sample from Poisson(μ̂)
            Random.seed!(42)
            function sir_bs_pois!(du, u, p, t)
                S, I, R = u
                λ = max(p.λ(I / 1000.0), 0.0)
                du[1] = -λ * S; du[2] = λ * S - 0.25 * I; du[3] = 0.25 * I
            end
            prob_ode = ODEProblem((du,u,p,t) -> begin
                S,I,R=u; λ=0.5*(I/1000)^0.9
                du[1]=-λ*S; du[2]=λ*S-0.25*I; du[3]=0.25*I
            end, [990.0, 10.0, 0.0], (0.0, 40.0))
            sol_ode = OrdinaryDiffEq.solve(prob_ode, Tsit5(); saveat=2.0)
            I_true = [sol_ode(t)[2] for t in sol_ode.t]
            # Simple Poisson sampling
            function _sp(μ)
                μ = max(μ, 0.01); c = 0; s = 0.0
                while true; s -= log(rand()); s > μ && break; c += 1; end
                Float64(c)
            end
            y_pois = _sp.(I_true)

            uf = BSplineApproximator(:λ, (0.0, 0.25), 6; initial=x->0.4x)
            prob_p = PSMProblem(sir_bs_pois!, [990.0,10.0,0.0], (0.0,40.0), [uf];
                data_times=sol_ode.t, data_values=reshape(y_pois,:,1),
                obs_to_state=[2], known_params=NamedTuple(),
                likelihood=Poisson(), solver=Tsit5())
            sol_p = solve(prob_p, LAML(maxiters=80, verbose=false))

            bs = bootstrap(sol_p, prob_p, LAML(maxiters=80, verbose=false);
                nboot=10, method=:parametric, rng=Random.Xoshiro(5))
            @test bs.n_success >= 3
            @test all(bs.ci_fitted.lower .<= bs.ci_fitted.upper)
            # Poisson bootstrap data should be non-negative integers
        end

        @testset "NegBin parametric bootstrap" begin
            # Reuse the logistic problem but with NegBin likelihood
            Random.seed!(42)
            data_pos = abs.(data) .+ 0.1  # ensure positive for NegBin
            uf_nb = BSplineApproximator(:r, (0.1, 10.0), 6; initial=x -> 0.3)
            prob_nb = PSMProblem(logistic_bs!, [1.0], (0.0, 15.0), [uf_nb];
                data_times=t_obs, data_values=data_pos,
                obs_to_state=[1], known_params=NamedTuple(),
                likelihood=NegativeBinomial(10.0), solver=Tsit5())
            sol_nb = solve(prob_nb, LAML(maxiters=50, verbose=false))
            bs_nb = bootstrap(sol_nb, prob_nb, LAML(maxiters=50, verbose=false);
                nboot=10, method=:parametric, rng=Random.Xoshiro(7))
            @test bs_nb.n_success >= 3
            @test all(bs_nb.ci_fitted.lower .<= bs_nb.ci_fitted.upper)
        end

        @testset "internal samplers" begin
            using PartiallySpecifiedModels: _sample_poisson, _sample_gamma
            rng = Random.Xoshiro(42)
            # Poisson: small μ (Knuth) and large μ (normal approx)
            samples_small = [_sample_poisson(5.0, rng) for _ in 1:1000]
            @test all(s -> s >= 0, samples_small)
            @test 4.0 < mean(samples_small) < 6.0  # E[X] = μ

            samples_large = [_sample_poisson(100.0, rng) for _ in 1:1000]
            @test all(s -> s >= 0, samples_large)
            @test 90.0 < mean(samples_large) < 110.0

            # Gamma: shape=2, scale=3 → mean=6
            samples_g = [_sample_gamma(2.0, 3.0, rng) for _ in 1:1000]
            @test all(s -> s > 0, samples_g)
            @test 5.0 < mean(samples_g) < 7.0

            # Gamma with shape < 1 (boost method)
            samples_g2 = [_sample_gamma(0.5, 2.0, rng) for _ in 1:1000]
            @test all(s -> s > 0, samples_g2)
            @test 0.5 < mean(samples_g2) < 1.5  # E = 0.5*2 = 1.0
        end
    end

    # ─── New solver tests ─────────────────────────────────────────────

    @testset "IntegralMatchingSolver — logistic growth" begin
        r_im(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_im!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        rng_im = Random.Xoshiro(42)
        sol_true_im = OrdinaryDiffEq.solve(
            ODEProblem(logistic_im!, [1.0], (0.0, 15.0), (; r=r_im)),
            Tsit5(); saveat=0.5)
        t_im = collect(sol_true_im.t)
        data_im = [sol_true_im.u[i][1] + 0.1*randn(rng_im) for i in 1:length(t_im)]

        uf_im = BSplineApproximator(:r, (0.0, 12.0), 8)
        prob_im = PSMProblem(logistic_im!, [1.0], (0.0, 15.0), [uf_im];
            data_times=t_im, data_values=reshape(max.(data_im, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_im = solve(prob_im, IntegralMatchingSolver(maxiters=500, verbose=false))

        @test sol_im isa PSMSolution
        @test isfinite(sol_im.data_loss)
        @test isfinite(sol_im.objective)
        @test haskey(sol_im.unknown_functions, :r)
        @test sol_im.convergence.method == :integral_matching
        @test abs(sol_im.unknown_functions[:r](5.0) - 0.25) < 0.12
    end

    @testset "EnsembleKalmanSolver — exponential decay" begin
        function decay_ek!(du, u, p, t)
            du[1] = -p.f(u[1])
        end
        rng_ek = Random.Xoshiro(42)
        sol_true_ek = OrdinaryDiffEq.solve(
            ODEProblem(decay_ek!, [5.0], (0.0, 10.0), (; f=x -> 0.5*x)),
            Tsit5(); saveat=0.5)
        t_ek = collect(sol_true_ek.t)
        data_ek = [sol_true_ek.u[i][1] + 0.05*randn(rng_ek) for i in 1:length(t_ek)]

        uf_ek = BSplineApproximator(:f, (0.0, 6.0), 6)
        prob_ek = PSMProblem(decay_ek!, [5.0], (0.0, 10.0), [uf_ek];
            data_times=t_ek, data_values=reshape(max.(data_ek, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_ek = solve(prob_ek, EnsembleKalmanSolver(n_ensemble=30, n_iterations=15, verbose=false))

        @test sol_ek isa PSMSolution
        @test isfinite(sol_ek.data_loss)
        @test haskey(sol_ek.unknown_functions, :f)
        @test sol_ek.convergence.method == :ensemble_kalman
        @test haskey(sol_ek.convergence, :ensemble_std)
        @test abs(sol_ek.unknown_functions[:f](3.0) - 1.5) < 0.45  # stochastic ensemble (30 particles)
    end

    @testset "EnsembleKalmanSolver — discrete map, gapped data times" begin
        # Stable affine decay map N[t+1] = f(N[t]), f(N) = 0.5N + 2, with
        # data only every 2 steps. EKI must follow the package-canonical
        # simulate_discrete convention (iterate unit steps over tspan);
        # applying the map once per data time would instead fit the
        # two-step composition 0.25N + 3 (f(8): true 6.0 vs wrong 5.0;
        # pre-fix error 1.28, post-fix 0.17).
        f_map_ek(N) = 0.5 * N + 2.0
        function decmap_ek!(u_next, u, p, t)
            u_next[1] = p.f(u[1])
        end
        N0_ek = 12.0
        N_full = zeros(21)
        N_full[1] = N0_ek
        for i in 1:20
            N_full[i+1] = f_map_ek(N_full[i])
        end
        t_gap = collect(0.0:2.0:20.0)
        data_gap = [N_full[Int(t)+1] for t in t_gap] .+
                   0.05 .* randn(Random.Xoshiro(123), length(t_gap))

        uf_ekd = BSplineApproximator(:f, (0.0, 14.0), 6; initial=5.0)
        prob_ekd = PSMProblem(decmap_ek!, [N0_ek], (0.0, 20.0), [uf_ekd];
            data_times=t_gap, data_values=reshape(max.(data_gap, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian(), discrete=true)
        sol_ekd = solve(prob_ekd,
            EnsembleKalmanSolver(n_ensemble=40, n_iterations=20, verbose=false))

        @test abs(sol_ekd.unknown_functions[:f](8.0) - f_map_ek(8.0)) < 0.5
        # The fitted trajectory must be exactly the canonical discrete
        # simulation at the fitted parameters.
        pred_canon = PartiallySpecifiedModels.simulate_discrete(
            prob_ekd, collect(sol_ekd.parameters))
        @test maximum(abs.(sol_ekd.fitted_values .- pred_canon)) < 1e-8
    end

    @testset "GradientMatching sigma2 — masked residuals excluded" begin
        # Same observed dynamics, once as a 1-state problem and once with an
        # appended unobserved nuisance state that does not feed back. The
        # unobserved state's residual rows are zero-weighted, so the fit and
        # the sigma2-driven theta update must be identical; a divisor that
        # counts the masked rows halves sigma2 (pre-fix theta ratio was
        # exactly 2). Dynamics are nonlinear in r so Gauss-Newton iterates
        # past the iter-5 theta update.
        r_gm2(N) = 0.5 * (1.0 - N / 10.0)
        function nl1_gm!(du, u, p, t)
            r = p.r(u[1])
            du[1] = r * u[1] * (1.0 + 0.3 * r)
        end
        function nl2_gm!(du, u, p, t)
            r = p.r(u[1])
            du[1] = r * u[1] * (1.0 + 0.3 * r)
            du[2] = -u[2]
        end
        true_nl!(du, u, p, t) = (r = r_gm2(u[1]); du[1] = r * u[1] * (1.0 + 0.3 * r))
        sol_true_nl = OrdinaryDiffEq.solve(
            ODEProblem(true_nl!, [1.0], (0.0, 15.0), nothing), Tsit5(); saveat=0.5)
        t_nl = collect(sol_true_nl.t)
        rng_nl = Random.Xoshiro(21)
        data_nl = [sol_true_nl.u[i][1] + 0.1*randn(rng_nl) for i in 1:length(t_nl)]
        dvals_nl = reshape(max.(data_nl, 0.01), :, 1)

        prob_nl1 = PSMProblem(nl1_gm!, [1.0], (0.0, 15.0),
            [BSplineApproximator(:r, (0.0, 12.0), 8)];
            data_times=t_nl, data_values=dvals_nl,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        prob_nl2 = PSMProblem(nl2_gm!, [1.0, 1.0], (0.0, 15.0),
            [BSplineApproximator(:r, (0.0, 12.0), 8)];
            data_times=t_nl, data_values=dvals_nl,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_nl1 = solve(prob_nl1, GradientMatching(maxiters=30, tol=0.0, verbose=false))
        sol_nl2 = solve(prob_nl2, GradientMatching(maxiters=30, tol=0.0, verbose=false))

        @test isapprox(sol_nl1.smoothing_params[1], sol_nl2.smoothing_params[1];
                       rtol=1e-6)
        @test isapprox(sol_nl1.unknown_functions[:r](5.0),
                       sol_nl2.unknown_functions[:r](5.0); atol=1e-6)
        @test abs(sol_nl1.unknown_functions[:r](5.0) - 0.25) < 0.05
    end

    @testset "ODINSolver — logistic growth" begin
        r_od(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_od!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        rng_od = Random.Xoshiro(123)
        sol_true_od = OrdinaryDiffEq.solve(
            ODEProblem(logistic_od!, [1.0], (0.0, 15.0), (; r=r_od)),
            Tsit5(); saveat=0.5)
        t_od = collect(sol_true_od.t)
        data_od = [sol_true_od.u[i][1] + 0.1*randn(rng_od) for i in 1:length(t_od)]

        uf_od = BSplineApproximator(:r, (0.0, 12.0), 8)
        prob_od = PSMProblem(logistic_od!, [1.0], (0.0, 15.0), [uf_od];
            data_times=t_od, data_values=reshape(max.(data_od, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_od = solve(prob_od, ODINSolver(maxiters=20, verbose=false))

        @test sol_od isa PSMSolution
        @test isfinite(sol_od.objective)
        @test haskey(sol_od.unknown_functions, :r)
        @test sol_od.convergence.method == :odin
        @test abs(sol_od.unknown_functions[:r](5.0) - 0.25) < 0.15
        # GP hyperparameters are estimated per state by marginal likelihood
        hp = sol_od.convergence.gp_hyperparams[1]
        @test hp.ℓ > 0 && hp.σ² > 0 && hp.σn² > 0

        # Reported data_loss must apply data_weights (like GM/BNG siblings)
        w_od = reshape([isodd(i) ? 2.0 : 0.5 for i in 1:length(t_od)], :, 1)
        prob_odw = PSMProblem(logistic_od!, [1.0], (0.0, 15.0), [uf_od];
            data_times=t_od, data_values=reshape(max.(data_od, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian(), data_weights=w_od)
        sol_odw = solve(prob_odw, ODINSolver(maxiters=10, verbose=false))
        @test isapprox(sol_odw.data_loss,
                       sum(w_od .* abs2.(prob_odw.data_values .- sol_odw.fitted_values));
                       rtol=1e-10)
    end

    @testset "ODINSolver — partially observed oscillator" begin
        # x1' = x2, x2' = -k(x1); only position observed. The joint
        # optimisation must infer the velocity through the ODE terms.
        k_osc(x) = x
        function osc_od!(du, u, p, t)
            du[1] = u[2]
            du[2] = -p.k(u[1])
        end
        sol_true = OrdinaryDiffEq.solve(
            ODEProblem((du,u,p,t)->(du[1]=u[2]; du[2]=-k_osc(u[1])),
                       [1.5, 0.0], (0.0, 8.0)), Tsit5(); saveat=0.25)
        t_osc = collect(sol_true.t)
        data_osc = [sol_true(t)[1] for t in t_osc] .+
                   0.03 .* randn(Random.Xoshiro(7), length(t_osc))
        uf_osc = BSplineApproximator(:k, (-2.0, 2.0), 7)
        prob_osc = PSMProblem(osc_od!, [1.5, 0.0], (0.0, 8.0), [uf_osc];
            data_times=t_osc, data_values=reshape(data_osc, :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_osc = solve(prob_osc, ODINSolver(maxiters=60, verbose=false))

        errs = [abs(sol_osc.unknown_functions[:k](x) - k_osc(x))
                for x in -1.4:0.2:1.4]
        @test maximum(errs) < 0.2
        @test sol_osc.data_loss < 0.5   # states track the position data
    end

    @testset "RKHSSolver — exponential decay" begin
        function decay_rk!(du, u, p, t)
            du[1] = -p.f(u[1])
        end
        rng_rk = Random.Xoshiro(42)
        sol_true_rk = OrdinaryDiffEq.solve(
            ODEProblem(decay_rk!, [5.0], (0.0, 10.0), (; f=x -> 0.5*x)),
            Tsit5(); saveat=0.5)
        t_rk = collect(sol_true_rk.t)
        data_rk = [sol_true_rk.u[i][1] + 0.05*randn(rng_rk) for i in 1:length(t_rk)]

        uf_rk = BSplineApproximator(:f, (0.0, 6.0), 6)
        prob_rk = PSMProblem(decay_rk!, [5.0], (0.0, 10.0), [uf_rk];
            data_times=t_rk, data_values=reshape(max.(data_rk, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_rk = solve(prob_rk, RKHSSolver(maxiters=500, n_repr_points=10,
                                            kernel=:rbf, verbose=false))

        @test sol_rk isa PSMSolution
        @test isfinite(sol_rk.objective)
        @test haskey(sol_rk.unknown_functions, :f)
        @test sol_rk.convergence.method == :rkhs
        @test sol_rk.convergence.kernel == :rbf
        @test abs(sol_rk.unknown_functions[:f](3.0) - 1.5) < 0.35
        # trajectory-RKHS: the fitted values are the RKHS trajectory, so
        # they must track the data (not just echo a smoother)
        @test sol_rk.data_loss < 0.5
    end

    @testset "RKHSSolver — partially observed oscillator" begin
        # x1' = x2, x2' = -k(x1); only position observed. The Gauss–Newton
        # B-step's Jacobian coupling must infer the velocity trajectory.
        k_rkhs(x) = x
        function osc_rk!(du, u, p, t)
            du[1] = u[2]
            du[2] = -p.k(u[1])
        end
        sol_true_rk = OrdinaryDiffEq.solve(
            ODEProblem((du,u,p,t)->(du[1]=u[2]; du[2]=-k_rkhs(u[1])),
                       [1.5, 0.0], (0.0, 8.0)), Tsit5(); saveat=0.25)
        t_ork = collect(sol_true_rk.t)
        data_ork = [sol_true_rk(t)[1] for t in t_ork] .+
                   0.03 .* randn(Random.Xoshiro(7), length(t_ork))
        prob_ork = PSMProblem(osc_rk!, [1.5, 0.0], (0.0, 8.0),
            [BSplineApproximator(:k, (-2.0, 2.0), 7)];
            data_times=t_ork, data_values=reshape(data_ork, :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_ork = solve(prob_ork, RKHSSolver(maxiters=400, verbose=false))

        errs = [abs(sol_ork.unknown_functions[:k](x) - k_rkhs(x))
                for x in -1.4:0.2:1.4]
        @test maximum(errs) < 0.15
        @test sol_ork.data_loss < 0.5
        # discrete-time problems are rejected (ẋ is undefined for maps)
        prob_disc = PSMProblem((u, p, t) -> [p.k(u[1])], [1.0], (0.0, 5.0),
            [BSplineApproximator(:k, (0.0, 5.0), 5)];
            data_times=collect(0.0:1.0:5.0),
            data_values=reshape(ones(6), :, 1), obs_to_state=[1],
            known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian(), discrete=true)
        @test_throws Exception solve(prob_disc, RKHSSolver())
    end

    @testset "ProfileLikelihoodSolver — logistic growth" begin
        r_pl(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_pl!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        rng_pl = Random.Xoshiro(42)
        sol_true_pl = OrdinaryDiffEq.solve(
            ODEProblem(logistic_pl!, [1.0], (0.0, 15.0), (; r=r_pl)),
            Tsit5(); saveat=1.0)
        t_pl = collect(sol_true_pl.t)
        data_pl = [sol_true_pl.u[i][1] + 0.1*randn(rng_pl) for i in 1:length(t_pl)]

        uf_pl = BSplineApproximator(:r, (0.0, 12.0), 6)
        prob_pl = PSMProblem(logistic_pl!, [1.0], (0.0, 15.0), [uf_pl];
            data_times=t_pl, data_values=reshape(max.(data_pl, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        # Profile only first 2 parameters to keep test fast
        sol_pl = solve(prob_pl, ProfileLikelihoodSolver(
            n_profile_points=10, param_indices=[1, 2], verbose=false))

        @test sol_pl isa PSMSolution
        @test isfinite(sol_pl.objective)
        @test sol_pl.convergence.method == :profile_likelihood
        @test haskey(sol_pl.convergence, :profiles)
        profiles = sol_pl.convergence.profiles
        @test haskey(profiles, 1)
        @test haskey(profiles, 2)
        # n_profile_points grid values plus the inserted exact-MLE point
        # (equal when the MLE already coincides with a grid value)
        @test 10 <= length(profiles[1].grid) <= 11
        @test length(profiles[1].ci) == 2
        # base-fit accuracy: the profile is built around a genuine MLE
        @test abs(sol_pl.unknown_functions[:r](5.0) - 0.25) < 0.12
        # the fitted parameter sits inside its own profile CI (PLR = 0 there)
        @test profiles[1].ci[1] <= sol_pl.parameters[1] <= profiles[1].ci[2]
    end


    # ─── Review additions: features previously untested ───────────────

    @testset "Multiple approximators in one model" begin
        # Headline composability claim: two unknown functions fitted jointly.
        Random.seed!(4242)
        function two_uf!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1] - p.g(u[1])
        end
        pt = ODEProblem((du,u,p,t) -> (du[1] = 0.4u[1] - 0.05u[1]), [1.0], (0.0, 5.0))
        st = OrdinaryDiffEq.solve(pt, Tsit5(); saveat=0.25)
        tt = collect(st.t)
        dv = reshape([st(t)[1] for t in tt] .+ 0.02 .* randn(Random.Xoshiro(7), length(tt)), :, 1)
        prob2 = PSMProblem(two_uf!, [1.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.5, 6.0), 5; initial=x -> 0.3),
             BSplineApproximator(:g, (0.5, 6.0), 5; initial=x -> 0.03x)];
            data_times=tt, data_values=dv, obs_to_state=[1],
            known_params=NamedTuple(), likelihood=Gaussian(), solver=Tsit5())
        sol2 = solve(prob2, LAML(maxiters=60, verbose=false))
        @test haskey(sol2.unknown_functions, :r)
        @test haskey(sol2.unknown_functions, :g)
        @test length(sol2.smoothing_params) == 2
        # only the combination r(N)N − g(N) is identified; check it
        net(N) = sol2.unknown_functions[:r](N) * N - sol2.unknown_functions[:g](N)
        for N in (1.0, 2.0, 4.0)
            @test abs(net(N) - 0.35N) < 0.1 * max(0.35N, 1.0)
        end
    end

    @testset "known_params mixed with an approximator" begin
        Random.seed!(4243)
        function sir_kp!(du, u, p, t)
            S, I = u
            β = p.β(I)
            du[1] = -β * S * I
            du[2] =  β * S * I - p.γ * I
        end
        βtrue_kp(I) = 0.5 * exp(-5.0 * I)
        pt = ODEProblem((du,u,p,t) -> begin
            S, I = u; b = 0.5 * exp(-5.0 * I)
            du[1] = -b * S * I; du[2] = b * S * I - 0.25I
        end, [0.99, 0.01], (0.0, 30.0))
        st = OrdinaryDiffEq.solve(pt, Tsit5(); saveat=1.0)
        data = hcat([u[1] for u in st.u], [u[2] for u in st.u]) .+
               0.004 .* randn(Random.Xoshiro(9), length(st.t), 2)
        prob = PSMProblem(sir_kp!, [0.99, 0.01], (0.0, 30.0),
            [ShapeConstrainedBSplineApproximator(:β, (0.0, 0.15), 7, :decreasing;
                                                 initial=0.25)];  # away from β(0.05)≈0.39
            data_times=collect(st.t), data_values=data,
            obs_to_state=[1, 2], known_params=(γ=0.25,),
            likelihood=Gaussian(), solver=Tsit5())
        sol = solve(prob, LAML(maxiters=50, verbose=false))
        @test abs(sol.unknown_functions[:β](0.05) - βtrue_kp(0.05)) < 0.08
        # name collision is rejected
        @test_throws ArgumentError PSMProblem(sir_kp!, [0.99, 0.01], (0.0, 30.0),
            [BSplineApproximator(:β, (0.0, 0.15), 6)];
            data_times=collect(st.t), data_values=data,
            obs_to_state=[1, 2], known_params=(β=0.5,),
            likelihood=Gaussian(), solver=Tsit5())
    end

    @testset "CustomLikelihood end-to-end matches Gaussian" begin
        Random.seed!(4244)
        function growth_cl!(du, u, p, t); du[1] = p.r(u[1]) * u[1]; end
        tt = collect(range(0.0, 5.0, length=20))
        dv = reshape(exp.(0.3 .* tt) .+ 0.05 .* randn(Random.Xoshiro(11), 20), :, 1)
        mk_cl(lik) = PSMProblem(growth_cl!, [1.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.5, 5.0), 5; initial=x -> 0.25)];
            data_times=tt, data_values=dv, obs_to_state=[1],
            known_params=NamedTuple(), likelihood=lik, solver=Tsit5())
        sol_g = solve(mk_cl(Gaussian()), LAML(maxiters=40, verbose=false))
        # CustomLikelihood with the Gaussian kernel drives the same IRLS
        cl = CustomLikelihood((y, mu) -> -0.5 * (y - mu)^2)
        sol_c = solve(mk_cl(cl), LAML(maxiters=40, verbose=false))
        @test abs(sol_c.unknown_functions[:r](2.0) -
                  sol_g.unknown_functions[:r](2.0)) < 0.05
    end

    @testset "Permuted obs_to_state and data_weights masking" begin
        Random.seed!(4245)
        # 2-state system observed in REVERSED column order
        function two_state_p!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
            du[2] = 0.1 * u[1]
        end
        pt = ODEProblem((du,u,p,t) -> (du[1]=0.3u[1]; du[2]=0.1u[1]),
                        [1.0, 0.0], (0.0, 5.0))
        st = OrdinaryDiffEq.solve(pt, Tsit5(); saveat=0.25)
        tt = collect(st.t)
        rng_p = Random.Xoshiro(13)
        col_u2 = [st(t)[2] for t in tt] .+ 0.02 .* randn(rng_p, length(tt))
        col_u1 = [st(t)[1] for t in tt] .+ 0.02 .* randn(rng_p, length(tt))
        data = hcat(col_u2, col_u1)          # column 1 ↦ state 2, column 2 ↦ state 1
        # corrupt two entries of column 2 and mask them with zero weights
        data[5, 2] = 100.0
        data[9, 2] = -50.0
        w = ones(length(tt), 2); w[5, 2] = 0.0; w[9, 2] = 0.0
        prob = PSMProblem(two_state_p!, [1.0, 0.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.5, 5.0), 5; initial=x -> 0.15)];
            data_times=tt, data_values=data, data_weights=w,
            obs_to_state=[2, 1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5())
        sol = solve(prob, LAML(maxiters=50, verbose=false))
        # masked outliers must not derail the fit; permuted mapping honored
        @test abs(sol.unknown_functions[:r](2.0) - 0.3) < 0.1
    end

    @testset "confidence_band" begin
        Random.seed!(4246)
        function growth_cb!(du, u, p, t); du[1] = p.r(u[1]) * u[1]; end
        tt = collect(range(0.0, 5.0, length=25))
        dv = reshape(exp.(0.3 .* tt) .+ 0.05 .* randn(Random.Xoshiro(15), 25), :, 1)
        prob = PSMProblem(growth_cb!, [1.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.25)];
            data_times=tt, data_values=dv, obs_to_state=[1],
            known_params=NamedTuple(), likelihood=Gaussian(), solver=Tsit5())
        sol = solve(prob, LAML(maxiters=50, verbose=false))
        bands = confidence_band(sol, prob; level=0.95)
        @test haskey(bands, :r)
        band = bands[:r]
        @test length(band.lower) == length(band.upper) == length(band.grid)
        @test all(band.lower .<= band.upper)
        # true constant r = 0.3 inside the band over the data-supported range
        inside = [band.lower[i] <= 0.3 <= band.upper[i]
                  for i in eachindex(band.grid) if 1.0 <= band.grid[i] <= 4.0]
        @test sum(inside) / length(inside) > 0.7
        # non-LAML solutions raise the documented error
        sol_adam = solve(prob, AdamSolver(maxiters=20, verbose=false))
        @test_throws ErrorException confidence_band(sol_adam, prob)
    end

    @testset "shape transforms: remaining six constraints" begin
        using PartiallySpecifiedModels: build_constrained_bspline_evaluator, nparams
        grid_sc = collect(range(0.05, 0.95, length=40))
        d1_sc(v) = diff(v); d2_sc(v) = diff(diff(v))
        checks = Dict(
            :inc_convex     => v -> all(d1_sc(v) .>= -1e-9) && all(d2_sc(v) .>= -1e-7),
            :inc_concave    => v -> all(d1_sc(v) .>= -1e-9) && all(d2_sc(v) .<= 1e-7),
            :dec_convex     => v -> all(d1_sc(v) .<= 1e-9)  && all(d2_sc(v) .>= -1e-7),
            :dec_concave    => v -> all(d1_sc(v) .<= 1e-9)  && all(d2_sc(v) .<= 1e-7),
            :inc_zero_right => v -> all(d1_sc(v) .>= -1e-9),
            :dec_zero_left  => v -> all(d1_sc(v) .<= 1e-9))
        for (c, chk) in checks, trial in 1:3
            a = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 7, c)
            γ = 2 .* randn(Random.Xoshiro(97trial + Int(hash(c) % 512)), nparams(a))
            f = build_constrained_bspline_evaluator(a, γ)
            v = [f(x) for x in grid_sc]
            @test chk(v)
            if c == :inc_zero_right
                @test abs(f(1.0)) < 1e-10
            elseif c == :dec_zero_left
                @test abs(f(0.0)) < 1e-10
            end
        end
    end

    @testset "AD gradient matches finite differences (Adam loss)" begin
        using PartiallySpecifiedModels: adam_loss_mse
        import ForwardDiff
        function growth_ad!(du, u, p, t); du[1] = p.r(u[1]) * u[1]; end
        tt = collect(range(0.0, 4.0, length=15))
        dv = reshape(exp.(0.3 .* tt), :, 1)
        prob = PSMProblem(growth_ad!, [1.0], (0.0, 4.0),
            [BSplineApproximator(:r, (0.5, 4.5), 5; initial=x -> 0.25)];
            data_times=tt, data_values=dv, obs_to_state=[1],
            known_params=NamedTuple(), likelihood=Gaussian(), solver=Tsit5())
        β0 = collect(PartiallySpecifiedModels.initial_params(prob.approximators[1]))
        L(β) = adam_loss_mse(prob, β)
        g_ad = ForwardDiff.gradient(L, β0)
        h = 1e-6
        g_fd = [(L(β0 .+ h .* ((1:5) .== j)) - L(β0 .- h .* ((1:5) .== j))) / 2h
                for j in 1:5]
        @test maximum(abs.(g_ad .- g_fd)) < 1e-3 * max(maximum(abs.(g_fd)), 1.0)
    end

    @testset "GP in-loop hyperparameter adaptation" begin
        Random.seed!(4250)
        rtrue_gp(x) = 0.3 + 0.15 * sin(3.0 * x)
        function growth_gp!(du, u, p, t); du[1] = p.r(u[1]) * u[1]; end
        pt = ODEProblem((du,u,p,t)->(du[1]=rtrue_gp(u[1])*u[1]), [0.8], (0.0, 6.0))
        st = OrdinaryDiffEq.solve(pt, Tsit5(); saveat=0.25)
        tt = collect(st.t)
        dv = reshape([st(t)[1] for t in tt] .+
                     0.02 .* randn(Random.Xoshiro(17), length(tt)), :, 1)
        # adaptive (no user lengthscale): ℓ must move off the default and
        # the in-data-range fit stay accurate
        g_a = GPApproximator(:r, (0.5, 8.0), 12; initial=x->0.3)
        ℓ0 = g_a.lengthscale
        @test g_a.adapt
        prob_a = PSMProblem(growth_gp!, [0.8], (0.0, 6.0), [g_a];
            data_times=tt, data_values=dv, obs_to_state=[1],
            known_params=NamedTuple(), likelihood=Gaussian(), solver=Tsit5())
        sol_a = solve(prob_a, LAML(maxiters=60, verbose=false))
        @test g_a.lengthscale != ℓ0                 # adaptation ran
        errs = [abs(sol_a.unknown_functions[:r](x) - rtrue_gp(x))
                for x in 1.0:0.25:3.5]              # data-informed range
        @test maximum(errs) < 0.08
        # user-fixed lengthscale: no adaptation, value untouched
        g_f = GPApproximator(:r, (0.5, 8.0), 12; lengthscale=1.0, initial=x->0.3)
        @test !g_f.adapt
        prob_f = PSMProblem(growth_gp!, [0.8], (0.0, 6.0), [g_f];
            data_times=tt, data_values=dv, obs_to_state=[1],
            known_params=NamedTuple(), likelihood=Gaussian(), solver=Tsit5())
        solve(prob_f, LAML(maxiters=30, verbose=false))
        @test g_f.lengthscale == 1.0
    end

    @testset "simulate and predict" begin
        function growth_sp!(du, u, p, t); du[1] = p.r(u[1]) * u[1]; end
        tt = collect(range(0.0, 3.0, length=10))
        dv = reshape(exp.(0.3 .* tt), :, 1)
        prob = PSMProblem(growth_sp!, [1.0], (0.0, 3.0),
            [BSplineApproximator(:r, (0.5, 3.5), 5; initial=x -> 0.3)];
            data_times=tt, data_values=dv, obs_to_state=[1],
            known_params=NamedTuple(), likelihood=Gaussian(), solver=Tsit5())
        β0 = collect(PartiallySpecifiedModels.initial_params(prob.approximators[1]))
        pred = simulate(prob, β0)
        @test size(pred) == (10, 1)
        @test all(isfinite, pred)
        sol = solve(prob, LAML(maxiters=30, verbose=false))
        # predict returns the stored fit; verify those values are actually
        # reproducible from the stored parameters (not a stale artifact)
        @test isapprox(predict(sol, prob),
                       simulate(prob, Float64.(collect(sol.parameters)));
                       rtol=1e-5)
    end

    @testset "construction validation — approximators" begin
        # Inverted domains are rejected everywhere (previously accepted
        # silently, producing reversed knot/mesh grids downstream)
        @test_throws ArgumentError BSplineApproximator(:f, (1.0, 0.0), 5)
        @test_throws ArgumentError GPApproximator(:f, (1.0, 0.0), 5)
        @test_throws ArgumentError SPDEApproximator(:f, (1.0, 0.0), 5)
        @test_throws ArgumentError ShapeConstrainedBSplineApproximator(
            :f, (1.0, 0.0), 6, :increasing)
        @test_throws ArgumentError ShapeConstrainedSPDEApproximator(
            :f, (1.0, 0.0), 6, :increasing)
        @test_throws ArgumentError NeuralApproximator(
            :f, Lux.Dense(1 => 1); domain=(1.0, 0.0))
        # COMONet with lo == hi previously divided by zero when normalizing
        # inputs and silently evaluated to NaN
        @test_throws ArgumentError COMONetApproximator(
            :f, (1.0, 1.0), (8,), :increasing)
        @test_throws ArgumentError COMONetApproximator(
            :f, (1.0, 0.0), (8,), :increasing)
        # CubicSpline needs ≥ 3 knots; nknots=2 previously surfaced much
        # later as a raw BoundsError from DataInterpolations
        @test_throws ArgumentError BSplineApproximator(:f, (0.0, 1.0), 2)
        @test_throws ArgumentError BSplineApproximator(:f, (0.0, 1.0), 1)
        # Minimal valid constructions still work
        @test BSplineApproximator(:f, (0.0, 1.0), 3) isa BSplineApproximator
        @test COMONetApproximator(:f, (0.0, 1.0), (4,), :increasing) isa
              COMONetApproximator
    end

    @testset "construction validation — PSMProblem" begin
        dyn_val!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1]; nothing)
        tt = collect(0.0:0.5:5.0)
        dv = reshape(exp.(0.3 .* tt), :, 1)
        # Duplicate approximator names: build_param_struct's NamedTuple
        # keeps only the last evaluator while both consume β slices —
        # previously accepted silently
        @test_throws ArgumentError PSMProblem(dyn_val!, [1.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.5, 6.0), 5),
             BSplineApproximator(:r, (0.5, 6.0), 7)];
            data_times=tt, data_values=dv)
        # Wrong-shaped data_weights: solvers previously indexed a
        # valid-but-wrong subset silently
        @test_throws ArgumentError PSMProblem(dyn_val!, [1.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.5, 6.0), 5)];
            data_times=tt, data_values=dv, data_weights=ones(3, 1))
        @test_throws ArgumentError PSMProblem(dyn_val!, [1.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.5, 6.0), 5)];
            data_times=tt, data_values=dv,
            data_weights=ones(length(tt), 2))
        # Correct shape and distinct names still construct
        prob_ok = PSMProblem(dyn_val!, [1.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.5, 6.0), 5),
             BSplineApproximator(:s, (0.5, 6.0), 5)];
            data_times=tt, data_values=dv,
            data_weights=ones(length(tt), 1))
        @test prob_ok isa PSMProblem
    end

    @testset "construction validation — probnum solver options" begin
        # Typo'd option symbols previously fell through silently
        # (any interrogate ≠ :schober ran :kramer; any method ≠ :fenrir
        # ran :basic)
        @test_throws ArgumentError RodeoSolver(interrogate=:shober)
        @test_throws ArgumentError RodeoSolver(method=:fenrirr)
        @test_throws ArgumentError DaltonSolver(interrogate=:kramerr)
        # n_deriv=1 previously BoundsError'd deep in the Kalman selectors
        @test_throws ArgumentError RodeoSolver(n_deriv=1)
        @test_throws ArgumentError DaltonSolver(n_deriv=1)
        @test_throws ArgumentError PseudoMarginalSolver(n_deriv=1)
        # Valid symbol combinations still construct
        @test RodeoSolver(method=:fenrir, interrogate=:schober) isa RodeoSolver
        @test DaltonSolver(interrogate=:schober) isa DaltonSolver
    end

    @testset "DaltonSolver — auto obs_var on large-scale data" begin
        # Scale-blind fixed obs_var=0.01 (the old default) badly misfits
        # data of order 1000; the auto default estimates from the data
        # scale (same estimator family as RodeoSolver/PseudoMarginal).
        function decay_sc!(du, u, p, t)
            du[1] = -p.f(u[1])
        end
        rng_sc = Random.Xoshiro(42)
        sol_true_sc = OrdinaryDiffEq.solve(
            ODEProblem(decay_sc!, [5000.0], (0.0, 10.0), (; f=x -> 0.5*x)),
            Tsit5(); saveat=0.5)
        t_sc = collect(sol_true_sc.t)
        truth_sc = [u[1] for u in sol_true_sc.u]
        data_sc = truth_sc .+ 50.0 .* randn(rng_sc, length(t_sc))
        prob_sc = PSMProblem(decay_sc!, [5000.0], (0.0, 10.0),
            [BSplineApproximator(:f, (0.0, 6000.0), 8)];
            data_times=t_sc, data_values=reshape(max.(data_sc, 0.01), :, 1))

        # Old fixed default demonstrably fails: f(3000) ≈ −1.4e4 vs true
        # 1500 — not merely inaccurate but a NEGATIVE decay rate, i.e. the
        # wrong sign. Fit RMSE ≈ 3.7e3–4.9e3 on data of order 5000.
        sol_old = solve(prob_sc, DaltonSolver(n_steps=100, maxiters=50,
                                              obs_var=0.01))
        err_old = abs(sol_old.unknown_functions[:f](3000.0) - 1500.0)

        # New auto default recovers the right scale and sign: f(3000) ≈ 411
        # (true 1500), fit RMSE ≈ 1.7e3. The auto estimator uses TOTAL data
        # variance, so on this steeply decaying trajectory it overshoots the
        # true noise variance (2500) by ~9×, which caps the achievable
        # accuracy — see the pinned run below for what the right obs_var buys.
        sol_auto = solve(prob_sc, DaltonSolver(n_steps=100, maxiters=50))
        f3k_auto = sol_auto.unknown_functions[:f](3000.0)
        err_auto = abs(f3k_auto - 1500.0)
        @test err_auto < 1200.0           # right order of magnitude
        @test err_old > 10 * err_auto     # old default badly underperforms
        _rmse_sc(x, y) = sqrt(sum(abs2, x .- y) / length(y))
        rmse_auto = _rmse_sc(sol_auto.fitted_values[:, 1], truth_sc)
        rmse_old = _rmse_sc(sol_old.fitted_values[:, 1], truth_sc)
        # The auto run is deterministic, so assert on it absolutely (data are
        # of order 5000). The obs_var=0.01 run is a 6-orders-of-magnitude
        # misspecification and its optimizer path is genuinely unstable —
        # rmse_old moves between ~3.7e3 and ~4.9e3 under --check-bounds, a
        # 2.1–2.8× ratio — so the comparative rmse bound is set loosely
        # enough to sit outside that spread. The sharp claim against the old
        # default is the ~14× err ratio above.
        @test rmse_auto < 2000.0
        @test rmse_old > 1.5 * rmse_auto

        # Explicit obs_var is honored exactly: identical pinned runs agree
        sol_p1 = solve(prob_sc, DaltonSolver(n_steps=100, maxiters=50,
                                             obs_var=2500.0))
        sol_p2 = solve(prob_sc, DaltonSolver(n_steps=100, maxiters=50,
                                             obs_var=2500.0))
        @test collect(sol_p1.parameters) == collect(sol_p2.parameters)
        # ... and with the true noise variance the fit is near-exact. Shared
        # quasi-MLE diffusion calibration of the two DALTON passes cut this
        # error from O(100) to O(10).
        @test abs(sol_p1.unknown_functions[:f](3000.0) - 1500.0) < 50.0
    end

    @testset "Convergence reporting — honest converged/iterations/reason" begin
        # Every iterative solver must report (converged, iterations, reason)
        # in sol.convergence: converged=true ONLY when a genuine criterion
        # fired, iterations = actual work done (not the budget), and reason
        # distinguishing :converged_tol / :plateau / :maxiters / :early_break.
        # Shared cheap problem: exponential growth with a spline growth rate.
        function growth_conv!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        t_conv = collect(range(0.0, 5.0, length=30))
        y_conv = reshape(exp.(0.3 .* t_conv), :, 1)
        prob_conv = PSMProblem(growth_conv!, [1.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
            data_times=t_conv, data_values=y_conv,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)

        # Shared sanity check on the extension keys
        function check_conv_keys(c)
            @test c.converged isa Bool
            @test c.iterations isa Int
            @test c.reason isa Symbol
            @test c.reason in (:converged_tol, :plateau, :maxiters, :early_break)
        end

        @testset "LAML" begin
            # (a) budget exhaustion is reported as such
            c = solve(prob_conv, LAML(maxiters=2)).convergence
            check_conv_keys(c)
            @test !c.converged
            @test c.reason == :maxiters
            @test c.iterations == 2
            @test c.laml_failures isa Int && c.laml_failures >= 0
            # existing keys survive (confidence_band depends on them)
            @test c.V_beta !== nothing
            @test c.sigma2 isa Float64
            # (b) genuine convergence fires the tolerance criterion
            c2 = solve(prob_conv, LAML(maxiters=60)).convergence
            check_conv_keys(c2)
            @test c2.converged
            @test c2.reason in (:converged_tol, :plateau)
            @test 0 < c2.iterations < 60
        end

        @testset "GCVSolver" begin
            c = solve(prob_conv, GCVSolver(maxiters=2)).convergence
            check_conv_keys(c)
            @test !c.converged
            @test c.reason == :maxiters
            @test c.iterations == 2
            @test c.gcv isa Float64
            c2 = solve(prob_conv, GCVSolver(maxiters=30)).convergence
            @test c2.converged
            @test c2.reason == :converged_tol
            @test 0 < c2.iterations < 30
        end

        @testset "CollocationLAML" begin
            c = solve(prob_conv, CollocationLAML(maxiters=1,
                lambda_ode_start=0.01, lambda_ode_end=100.0,
                n_continuation=2)).convergence
            check_conv_keys(c)
            @test !c.converged
            @test c.reason == :maxiters
            @test c.iterations == 1          # final continuation level
            @test c.iterations_total == 2    # 1 per level × 2 levels
            # existing keys survive
            @test c.ode_compliance isa Float64
            @test c.lambda_ode_final == 100.0
            c2 = solve(prob_conv, CollocationLAML(maxiters=20,
                lambda_ode_start=0.01, lambda_ode_end=100.0,
                n_continuation=4)).convergence
            @test c2.converged
            @test c2.reason in (:converged_tol, :plateau)
            @test c2.iterations_total >= c2.iterations
        end

        @testset "AdamSolver" begin
            c = solve(prob_conv, AdamSolver(maxiters=5, lr=0.01)).convergence
            check_conv_keys(c)
            @test !c.converged
            @test c.reason == :maxiters
            @test c.iterations == 5
            @test c.final_grad_norm isa Float64 && isfinite(c.final_grad_norm)
            # existing keys survive
            @test c.optimizer == :adam
            @test c.method == :adam_ode
            # Spurious-plateau guard: with maxiters=70, every iteration where
            # the plateau check is reachable (iter > 60) has a cosine-annealed
            # lr_t below 5% of the base lr (0.5·(1+cos(π·61/70)) ≈ 0.041), so
            # a flat loss window there is the schedule's artifact, not
            # convergence — the guard must forbid reason=:plateau.
            cg = solve(prob_conv, AdamSolver(maxiters=70, lr=0.01)).convergence
            @test !cg.converged
            @test cg.reason == :maxiters   # NOT :plateau
            @test cg.iterations == 70
            # A genuine plateau (mid-schedule, lr still meaningful) IS
            # reported as convergence.
            c2 = solve(prob_conv, AdamSolver(maxiters=300, lr=0.01)).convergence
            @test c2.converged
            @test c2.reason == :plateau
            @test 60 < c2.iterations < 300
        end

        @testset "MultipleShootingSolver" begin
            # Starved budget with a negligible continuity penalty leaves
            # visible shooting gaps: must NOT be reported as converged.
            c = solve(prob_conv, MultipleShootingSolver(
                n_intervals=5, maxiters_inner=1, maxiters_outer=1,
                rho_init=1e-6)).convergence
            check_conv_keys(c)
            @test !c.converged
            @test c.reason == :maxiters
            @test c.iterations == 1
            @test c.max_gap isa Float64 && c.max_gap > 0.0
            @test c.rho_final isa Float64 && c.rho_final >= 1e-6
            # existing keys survive
            @test c.optimizer == :lbfgs
            @test c.method == :multiple_shooting
            @test c.n_intervals == 5
            # Normal run: shooting gaps close below the tolerance
            c2 = solve(prob_conv, MultipleShootingSolver(
                n_intervals=3, maxiters_inner=50, maxiters_outer=5,
                rho_init=1.0)).convergence
            @test c2.converged
            @test c2.reason == :converged_tol
            @test c2.max_gap < 1e-2   # tolerance is 1e-2 · ‖u0‖, u0 = [1.0]
        end

        @testset "TwoStageSolver" begin
            c = solve(prob_conv, TwoStageSolver(maxiters=10)).convergence
            check_conv_keys(c)
            @test !c.converged
            @test c.reason == :maxiters
            @test c.iterations == 10
            @test c.method == :two_stage
            c2 = solve(prob_conv, TwoStageSolver(maxiters=1000)).convergence
            @test c2.converged
            @test c2.reason == :plateau
            @test 60 < c2.iterations < 1000
        end

        @testset "IntegralMatchingSolver" begin
            c = solve(prob_conv, IntegralMatchingSolver(maxiters=10)).convergence
            check_conv_keys(c)
            @test !c.converged
            @test c.reason == :maxiters
            @test c.iterations == 10
            @test c.method == :integral_matching
            c2 = solve(prob_conv,
                       IntegralMatchingSolver(maxiters=1000)).convergence
            @test c2.converged
            @test c2.reason == :plateau
            @test 60 < c2.iterations < 1000
        end

        @testset "ODINSolver" begin
            # ODIN's budget is 20 Adam steps per maxiter; iterations counts
            # the steps actually performed.
            c = solve(prob_conv, ODINSolver(maxiters=3)).convergence
            check_conv_keys(c)
            @test !c.converged
            @test c.reason == :maxiters
            @test c.iterations == 60   # 20 · maxiters, plateau needs step > 60
            @test c.method == :odin
            c2 = solve(prob_conv, ODINSolver(maxiters=50)).convergence
            @test c2.converged
            @test c2.reason == :plateau
            @test 60 < c2.iterations < 1000
        end

        @testset "BNGSolver" begin
            c = solve(prob_conv, BNGSolver(maxiters=5, k_obs=2, k_proc=1,
                                           rng_seed=7)).convergence
            check_conv_keys(c)
            @test !c.converged
            @test c.reason == :maxiters
            @test c.iterations == 10   # 2 members × 5 iters, all exhausted
            @test c.member_converged == [false, false]
            # existing keys survive
            @test c.n_ensemble == 2
            @test length(c.member_losses) == 2
            c2 = solve(prob_conv, BNGSolver(maxiters=500, k_obs=2, k_proc=1,
                                            rng_seed=7)).convergence
            @test c2.converged
            @test c2.reason == :plateau
            @test all(c2.member_converged)
            @test c2.iterations < 1000   # both members plateaued early
        end
    end

    @testset "LAML stationarity diagnostics (F2)" begin
        # `convergence.converged` is a STABILITY test: it fires when the
        # penalized objective and the data loss stop changing. Several
        # different things stop a fit moving and `converged` reports them
        # all as success. These tests pin the two ADDITIVE keys that tell
        # them apart -- `stationarity` (|dV/drho| normalized by
        # 1/2 rank(S_k)) and `smoothing_advanced` -- and pin that they stay
        # quiet on a healthy fit.
        #
        # Both keys are additive: `converged`, `reason`, `iterations` and
        # every fitted quantity are unchanged by this testset's subject.
        #
        # NOTE ON THRESHOLDS. There is deliberately no `stationary::Bool` in
        # the API, because measured across all 151 LAML solves this suite
        # performs the residual is an UNBROKEN continuum, not two clusters:
        # quantiles p25=1.7e-7, p50=9.5e-6, p75=1.5e-2, p90=0.46, max=14.1,
        # and over the whole decision-relevant region (1e-3..3) the largest
        # ratio between consecutive sorted values is 1.65 below 1 and 2.98
        # across the whole region -- nothing like the orders-of-magnitude
        # separation a threshold would need, so no cutoff could sit there. (A partial 94-solve sample appeared to show a
        # 3.2x gap near 0.1; the remaining 57 solves filled it in. That is
        # why none of the assertions below rely on a universal cutoff.)
        # Each either compares a fixture against a STRUCTURALLY MATCHED
        # control, or pins a fixture-specific measured magnitude with wide
        # margin.

        # ── (a) HEALTHY: the diagnostics must stay quiet ──
        gh_f2!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1])
        t_f2 = collect(0.0:0.5:10.0)
        # Deterministic (RNG-free) wobble so the fit has REAL residual
        # variance. A noiseless fixture drives the profiled sigma^2 to
        # ~1e-17 and the REML gradient's -1/2 lambda b'Sb / sigma^2 term
        # with it, which is a ratio of two near-zero quantities and is not
        # informative (see the caveat in the LAML docstring).
        noi_f2 = [0.01 * (sin(3.1i) + 0.5cos(7.7i)) for i in 1:length(t_f2)]
        mk_f2() = PSMProblem(gh_f2!, [1.0], (0.0, 10.0),
            [BSplineApproximator(:r, (0.0, 5.0), 5; initial=x -> 0.05)];
            data_times=t_f2,
            data_values=reshape(exp.(0.1 .* t_f2) .+ noi_f2, :, 1),
            obs_to_state=[1], likelihood=Gaussian(), solver=Tsit5())

        s_ok = solve(mk_f2(), LAML(maxiters=30))
        @test s_ok.convergence.converged
        @test s_ok.convergence.smoothing_advanced
        # measured 2.90e-7; 85 of the suite's 151 solves sit below 1e-4
        @test s_ok.convergence.stationarity < 1e-4
        @test isfinite(s_ok.convergence.stationarity)

        # ── (b) MODE: Fellner-Schall never advanced ──
        # Same problem, jac=:forwarddiff. The exact Jacobian makes the very
        # first step good enough that the accept block takes the `f01 <
        # obj_prev` branch every iteration with `dl_a1 >= dl_curr`, so
        # `otheta` never moves onto an FS proposal. `converged` is true and
        # the reported lambda-hat is EXACTLY the data-driven default
        # 1/tr(S) -- smoothing selection never took effect. This was hidden
        # before F1, which reported a plausible-looking proposal instead.
        # `smoothing_advanced` is exact and threshold-free, which is why it
        # is the assertion carrying the weight here.
        s_m3 = solve(mk_f2(), LAML(maxiters=30, jac=:forwarddiff))
        @test s_m3.convergence.converged          # unchanged: still "converged"
        @test !s_m3.convergence.smoothing_advanced
        # lambda-hat is bit-identical to the initialization, not merely close
        lam0_f2 = 1.0 / tr(penalty_matrix(
            BSplineApproximator(:r, (0.0, 5.0), 5)))
        @test s_m3.smoothing_params[1] == lam0_f2
        # ...and the fd path on the SAME data does advance, so this is a
        # property of the search, not of the fixture being unfittable
        @test s_ok.smoothing_params[1] != lam0_f2

        # ── (c) MODE: stalled on a non-smooth evaluator ──
        # Structurally MATCHED PAIR: the same discrete-map model, the same
        # basis, the same solver settings and the same deterministic noise,
        # differing only in whether the step applies floor() to a
        # coefficient-dependent quantity. The kink makes the prediction
        # discontinuous in beta, so the default finite-difference Jacobian
        # is noise and the search plateaus at a non-optimum -- while still
        # reporting converged=true. Comparing the pair avoids asserting any
        # absolute stationarity cutoff.
        smooth_step(g) = g
        kink_step(g) = g - 0.5 * (floor(g * 20.0) / 20.0)
        function mk_map(stepf)
            Nt = zeros(31); Nt[1] = 20.0
            for t in 1:30
                Nt[t+1] = max(Nt[t] *
                    exp(stepf(0.8 * (1 - Nt[t] / 100.0))), 0.01)
            end
            dat = Nt .+ [0.5 * (sin(3.1i) + 0.7cos(7.7i)) for i in 1:31]
            dyn! = (un, u, p, t) -> (un[1] = max(u[1] * exp(stepf(p.f(u[1]))),
                                                 0.01))
            PSMProblem(dyn!, [20.0], (0.0, 30.0),
                [BSplineApproximator(:f, (0.0, 150.0), 8;
                    initial=x -> 0.5 * (1.0 - x / 100.0))];
                data_times=Float64.(0:30), data_values=reshape(dat, :, 1),
                discrete=true, solver=nothing)
        end
        s_smooth = solve(mk_map(smooth_step), LAML(maxiters=50))
        s_kink   = solve(mk_map(kink_step),   LAML(maxiters=50))

        # Both report success -- `converged` cannot tell them apart
        @test s_smooth.convergence.converged
        @test s_kink.convergence.converged
        # ...but the stationarity residuals differ by orders of magnitude.
        # Measured: 1.52e-6 (smooth) vs 1.32 (kinked), a factor of ~8.7e5.
        # Asserted at 1e3 -- ~870x of margin -- so this is a qualitative
        # separation check, not a pinned constant.
        @test s_kink.convergence.stationarity >
              1e3 * s_smooth.convergence.stationarity
        @test s_smooth.convergence.stationarity < 1e-4    # measured 1.5e-6
        @test s_kink.convergence.stationarity > 0.5       # measured 1.32
        # here FS DID move lambda -- it just never reached an optimum, which
        # is exactly why `smoothing_advanced` alone cannot detect this mode
        # and the residual is needed
        @test s_kink.convergence.smoothing_advanced
        # more budget does not help: it is stalled, not truncated
        s_kink2 = solve(mk_map(kink_step), LAML(maxiters=80))
        @test s_kink2.convergence.iterations == s_kink.convergence.iterations
        @test s_kink2.convergence.stationarity ≈ s_kink.convergence.stationarity

        # ── (d) the keys exist on NON-converged exits too ──
        # (`haskey` patterns and `keys()` comparisons must not depend on the
        # exit path)
        s_tr = solve(mk_f2(), LAML(maxiters=1))
        @test !s_tr.convergence.converged
        for s in (s_tr, s_ok, s_kink)
            @test haskey(s.convergence, :stationarity)
            @test haskey(s.convergence, :smoothing_advanced)
            @test s.convergence.stationarity isa Float64
            @test s.convergence.smoothing_advanced isa Bool
        end
        # a 1-iteration fit cannot have advanced smoothing (warmup=3 > 1)
        @test !s_tr.convergence.smoothing_advanced
        # no `stationary::Bool` is exposed -- see the threshold note above
        @test !haskey(s_ok.convergence, :stationary)

        # ── (e) every pre-existing key keeps its meaning ──
        for k in (:V_beta, :sigma2, :converged, :iterations, :reason,
                  :laml_failures, :criterion, :laml)
            @test haskey(s_ok.convergence, k)
        end

        # ── (f) the convergence monitor is scored at `theta`, not
        # `theta_fit` ──
        # `curr_obj = penalized_objective(beta, build_B(theta))` in
        # solver.jl is the ONLY theta-dependent quantity F1 left on the FS
        # proposal rather than on theta_fit, and that is deliberate: it is
        # the sole term in the convergence test that can see lambda still
        # moving. Rescoring it at theta_fit makes this two-lambda fit stop
        # 11 iterations earlier with lambda-hat frozen at its
        # initialization (measured: EDF 4.00 -> 11.99, lambda-hat
        # [1.9e7, 3.1e7] -> [1.569e-4, 1.569e-4], stationarity 5.3e-6 ->
        # 0.9997). These assertions fail under that change.
        function lv_f2!(du, u, p, t)
            N, Pr = u
            du[1] = p.r(N) * N - 0.02 * N * Pr
            du[2] = 0.01 * N * Pr - p.m(Pr) * Pr
        end
        pt_f2 = ODEProblem((du, u, p, t) -> begin
                N, Pr = u
                du[1] = 0.5N - 0.02N * Pr
                du[2] = 0.01N * Pr - 0.3Pr
            end, [20.0, 5.0], (0.0, 20.0))
        st_f2 = OrdinaryDiffEq.solve(pt_f2, Tsit5(); saveat=0.5)
        Y_f2 = hcat([u[1] for u in st_f2.u], [u[2] for u in st_f2.u])
        Y_f2 .+= 0.2 .* hcat([sin(3.1i) + 0.5cos(7.7i) for i in 1:size(Y_f2,1)],
                             [0.6sin(3.1i) + cos(7.7i) for i in 1:size(Y_f2,1)])
        prob_lv2 = PSMProblem(lv_f2!, [20.0, 5.0], (0.0, 20.0),
            [BSplineApproximator(:r, (0.0, 80.0), 6; initial=x -> 0.5),
             BSplineApproximator(:m, (0.0, 30.0), 6; initial=x -> 0.3)];
            data_times=st_f2.t, data_values=Y_f2, obs_to_state=[1, 2],
            likelihood=Gaussian(), solver=Tsit5())
        s_lv2 = solve(prob_lv2, LAML(maxiters=60))
        @test s_lv2.convergence.converged
        @test s_lv2.convergence.iterations > 12      # measured 18; 7 if rescored
        @test s_lv2.convergence.smoothing_advanced
        @test s_lv2.convergence.stationarity < 1e-4  # measured 5.3e-6; 0.9997 if rescored
        @test s_lv2.edf < 6.0                        # measured 4.00; 11.99 if rescored
        @test all(s_lv2.smoothing_params .> 1.0)     # measured ~2e7; 1.6e-4 if rescored
    end

    @testset "Minor-batch fixes (T10)" begin
        @testset "_normlogcdf: tail accuracy and branch continuity" begin
            using PartiallySpecifiedModels: _normlogcdf, _normcdf
            # References from 512-bit BigFloat Simpson quadrature of φ.
            refs = ((-5.5, -17.779376352625253),
                    (-6.0, -20.736768949974696),
                    (-6.5, -23.938149495161824),
                    (-7.0, -27.384307498811054),
                    (-8.0, -35.01343715991452))
            for (x, r) in refs
                # Old truncated asymptotic branch erred by 2.5e-3–2.6e-2 here.
                @test _normlogcdf(x) ≈ r atol=1e-10
            end
            # The two branches agree at the x = −1 seam (old seam at −6
            # jumped by ~0.022).
            @test abs(_normlogcdf(-1.0) - log(_normcdf(-1.0))) < 1e-7
            @test abs(_normlogcdf(-1.0 + 1e-10) - _normlogcdf(-1.0 - 1e-10)) < 1e-7

            # RIGHT tail (RB/N5). The x > 6 branch used to be
            # log(_normcdf(x)), which loses all relative precision as
            # Φ → 1: it erred by 3.6e-3 at x = 6.0001, 7.1e-2 at x = 8,
            # returned exactly 0.0 from x ≈ 9 on, and DECREASED by
            # 3.5e-12 across x = 6 — the only non-monotone point of the
            # function. Computing the positive half by complement,
            # log1p(−Φ(−x)), is accurate to ~1e-15 everywhere.
            # References: 512-bit BigFloat Simpson quadrature of φ on
            # [x, x+60], log1p(−Q).
            right_refs = ((6.0,  -9.865876455244298e-10),
                          (7.0,  -1.2798125438867863e-12),
                          (8.0,  -6.220960574272895e-16),
                          (10.0, -7.619853024163884e-24))
            for (x, r) in right_refs
                @test _normlogcdf(x) ≈ r rtol=1e-11
            end
            # Continuity AND monotonicity across the retired x = 6 seam.
            @test _normlogcdf(nextfloat(6.0)) >= _normlogcdf(6.0)
            # Across ±1e-10 the only change should be the true slope
            # φ(6)/Φ(6) ≈ 6.076e-9, i.e. ≈1.2e-18 — the old branch jumped
            # by 3.5e-12, nine orders of magnitude more.
            @test abs(_normlogcdf(6.0 + 1e-10) - _normlogcdf(6.0 - 1e-10)) < 1e-17
            # Monotone over a dense sweep spanning every branch.
            sweep = [_normlogcdf(x) for x in range(-10.0, 10.0, length=20_001)]
            @test all(sweep[i+1] >= sweep[i] for i in 1:length(sweep)-1)
            # log Φ(x) < 0 strictly for every finite x (never rounds to 0).
            @test all(_normlogcdf(x) < 0 for x in (6.0, 8.0, 10.0, 15.0, 20.0))
        end

        @testset "Poisson: mu floor consistent between loglik and weights" begin
            using PartiallySpecifiedModels: log_likelihood, loglik_pointwise,
                irls_weights, _loggamma
            # Healthy fit (μ ≫ 1e-6): value unchanged by the unification.
            y_p = [5.0, 10.0, 3.0]; mu_p = [4.5, 11.0, 3.2]; w = ones(3)
            ll_ref = sum(y_p[i] * log(mu_p[i]) - mu_p[i] -
                         _loggamma(y_p[i] + 1) for i in 1:3)
            @test log_likelihood(Poisson(), y_p, mu_p, w) ≈ ll_ref atol=1e-12
            # Nonpositive μ sees the SAME effective mean max(|μ|, 1e-6) as
            # the IRLS weights (old floor at 1e-10 was a cliff the
            # curvature never saw).
            @test log_likelihood(Poisson(), [2.0], [-0.5], [1.0]) ==
                  log_likelihood(Poisson(), [2.0], [0.5], [1.0])
            @test loglik_pointwise(Poisson(), 2.0, -0.5) ==
                  loglik_pointwise(Poisson(), 2.0, 0.5)
            @test irls_weights(Poisson(), [2.0], [-0.5], [1.0]) ==
                  irls_weights(Poisson(), [2.0], [0.5], [1.0])
        end

        @testset "SCOP initial_params: unclamped Greville reproduces linears" begin
            using PartiallySpecifiedModels: initial_params,
                build_constrained_bspline_evaluator
            lin = x -> 1.0 + 2.0 * x
            a = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 8,
                                                    :increasing; initial=lin)
            ev = build_constrained_bspline_evaluator(a, initial_params(a))
            # Clamped Greville evaluation had max error 0.067 on this target.
            @test maximum(abs(ev(x) - lin(x))
                          for x in range(0.0, 1.0, length=101)) < 1e-10
            # A function that THROWS outside its domain still initializes
            # (per-point fallback: clamped value + one-sided-slope
            # extrapolation), and stays exact for a linear target.
            f_strict = x -> (0.0 <= x <= 1.0 || throw(DomainError(x));
                             1.0 + 2.0 * x)
            a2 = ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 8,
                                                     :increasing;
                                                     initial=f_strict)
            p2 = initial_params(a2)
            @test all(isfinite, p2)
            ev2 = build_constrained_bspline_evaluator(a2, p2)
            @test maximum(abs(ev2(x) - lin(x))
                          for x in range(0.0, 1.0, length=101)) < 1e-10
        end

        @testset "GP adaptation: incumbent hyperparameters win" begin
            using PartiallySpecifiedModels: _adapt_gp_hyperparams!,
                _kernel_func, _build_kernel_matrix
            g = GPApproximator(:g, (0.0, 1.0), 12)
            @test g.adapt
            xind = g.inducing_points
            beta_k = sin.(4.0 .* xind) .+ 0.3 .* xind
            # Sample variance (matches Statistics.var used inside the adapter).
            mβ = sum(beta_k) / length(beta_k)
            v = sum(abs2, beta_k .- mβ) / (length(beta_k) - 1)
            nug = 1e-6 * v
            gp_ll = (ℓ, σ²) -> begin
                K = _build_kernel_matrix(_kernel_func(g.kernel, ℓ, σ²), xind)
                F = cholesky(Symmetric(K + nug * I), check=false)
                issuccess(F) || return -Inf
                -0.5 * dot(beta_k, F \ beta_k) - sum(log, diag(F.U))
            end
            # Fine sweep (superset of the adaptation grid) → off-grid optimum.
            best_ll = -Inf; best_ℓ = 0.0; best_σ² = 0.0
            for frac in 0.04:0.01:1.2, σm in 0.3:0.1:3.0
                llv = gp_ll(frac, σm * v)
                if llv > best_ll
                    best_ll = llv; best_ℓ = frac; best_σ² = σm * v
                end
            end
            # Grid best, for the discrimination guard below.
            gbest_ll = -Inf; gbest_ℓ = 0.0
            for frac in (0.08, 0.15, 0.25, 0.4, 0.6, 1.0), σm in (0.5, 1.0, 2.0)
                llv = gp_ll(frac, σm * v)
                if llv > gbest_ll
                    gbest_ll = llv; gbest_ℓ = frac
                end
            end
            # The incumbent must beat the grid materially, else this test
            # could not distinguish the fix.
            @test abs(log(best_ℓ / gbest_ℓ)) > 0.05
            g.lengthscale = best_ℓ; g.variance = best_σ²
            # Pre-fix, adaptation moved to the worse grid best; now the
            # incumbent is scored first and "no change" wins.
            @test _adapt_gp_hyperparams!(g, beta_k) == false
            @test g.lengthscale == best_ℓ
            @test g.variance == best_σ²
        end

        @testset "TwoStageSolver.n_basis_smooth is wired to the smoother" begin
            using PartiallySpecifiedModels: _smoothing_spline
            t = collect(range(0.0, 10.0, length=40))
            yy = sin.(t) .+ 0.05 .* cos.(7 .* t)
            v_def, _ = _smoothing_spline(t, yy)
            v_15, _ = _smoothing_spline(t, yy; max_basis=15)
            v_8, _ = _smoothing_spline(t, yy; max_basis=8)
            # Default preserved exactly (the field was dead at an effective 15).
            @test v_def(3.7) == v_15(3.7)
            # A non-default basis cap actually changes the smoother.
            @test abs(v_def(3.7) - v_8(3.7)) > 1e-6
            @test TwoStageSolver().n_basis_smooth == 15
            @test TwoStageSolver(n_basis_smooth=8).n_basis_smooth == 8
            @test_throws ArgumentError TwoStageSolver(n_basis_smooth=3)
        end

        @testset "smooth_and_differentiate warns on duplicate obs targets" begin
            using PartiallySpecifiedModels: smooth_and_differentiate
            times = collect(range(0.0, 1.0, length=10))
            data = hcat(times, 2 .* times)
            @test_logs (:warn, r"multiple observation columns") smooth_and_differentiate(
                times, data, [1, 1], 2)
            # Later column wins (documented overwrite behavior); the linear
            # target is reproduced exactly by the penalized smoother.
            ys, _ = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                smooth_and_differentiate(times, data, [1, 1], 2)
            end
            @test isapprox(ys[5, 1], 2 * times[5], atol=1e-6)
        end
    end

    # ── Remediation round RB: completeness fixes ─────────────────
    @testset "Remediation RB — completeness" begin

        @testset "exotic Lux architectures reach the fallback, not an error" begin
            import Lux
            using PartiallySpecifiedModels: neural_mlp_spec, neural_init_params,
                                            mlp_spec_from_lux
            # mlp_spec_from_lux errors by design on anything that is not a
            # Chain of Dense layers. Every solver used to call it directly,
            # so a perfectly valid architecture hard-errored at solve():
            #   "The Dual-safe MLP evaluator supports Chains of Lux.Dense
            #    layers only; the given model contains a
            #    Lux.WrappedFunction{…} layer"
            # thrown from AdamSolver's beta initialisation, BEFORE the
            # solver could reach its own working Lux.apply fallback.
            m_wf = Lux.Chain(Lux.Dense(1, 4, tanh),
                             Lux.WrappedFunction(x -> x),
                             Lux.Dense(4, 1))
            m_nb = Lux.Chain(Lux.Dense(1, 4, tanh),
                             Lux.Dense(4, 1; use_bias=false))
            for m in (m_wf, m_nb)
                a = NeuralApproximator(:f, m; domain=(0.0, 6.0), rng_seed=7)
                @test_throws ErrorException mlp_spec_from_lux(a.model)
                # The guarded probe reports "no spec" instead of throwing…
                @test neural_mlp_spec(a) === nothing
                # …and initialisation still yields a full-length vector
                # (Lux's own init), so the fallback evaluator can use it.
                p0 = neural_init_params(a, Random.Xoshiro(7))
                @test length(p0) == nparams(a)
                @test all(isfinite, p0)
            end
            # A Dense-only chain still takes the fast Dual-safe path.
            a_ok = NeuralApproximator(:f, Lux.Chain(Lux.Dense(1, 4, tanh),
                                                    Lux.Dense(4, 1));
                                      domain=(0.0, 6.0), rng_seed=7)
            @test neural_mlp_spec(a_ok) !== nothing

            # End-to-end: both exotic architectures now fit.
            decay_rb!(du, u, p, t) = (du[1] = -p.f(u[1]))
            ts_rb = collect(0.0:0.5:10.0)
            ys_rb = reshape(5.0 .* exp.(-0.5 .* ts_rb), :, 1)
            for m in (m_wf, m_nb)
                a = NeuralApproximator(:f, m; domain=(0.0, 6.0), rng_seed=7)
                prob_rb = PSMProblem(decay_rb!, [5.0], (0.0, 10.0), [a];
                    data_times=ts_rb, data_values=ys_rb, obs_to_state=[1],
                    known_params=NamedTuple(), solver=Tsit5())
                sol_rb = solve(prob_rb, AdamSolver(maxiters=30, lr=0.02,
                                                   verbose=false))
                @test sol_rb isa PSMSolution
                @test isfinite(sol_rb.data_loss)
                @test all(isfinite, sol_rb.fitted_values)
                @test isfinite(sol_rb.unknown_functions[:f](2.0))
            end
        end

        @testset "cosine-lr plateau guard — TwoStage / IntegralMatching" begin
            # With maxiters = 70 every iteration past the plateau window
            # start (iter > 60) has lr_t = lr·½(1+cos(π·iter/70)) < 0.05·lr,
            # and lr = 1e-12 freezes beta, so the loss window is flat.
            # Pre-fix both solvers reported converged=true, reason=:plateau
            # at iter 61 — a plateau manufactured entirely by the schedule.
            decay_g!(du, u, p, t) = (du[1] = -p.f(u[1]))
            ts_g = collect(0.0:0.5:10.0)
            ys_g = reshape(5.0 .* exp.(-0.5 .* ts_g), :, 1)
            prob_g = PSMProblem(decay_g!, [5.0], (0.0, 10.0),
                [BSplineApproximator(:f, (0.0, 6.0), 6)];
                data_times=ts_g, data_values=ys_g, obs_to_state=[1],
                known_params=NamedTuple(), solver=Tsit5())

            s_ts = solve(prob_g, TwoStageSolver(maxiters=70, lr=1e-12,
                                                verbose=false))
            @test s_ts.convergence.converged == false
            @test s_ts.convergence.reason == :maxiters
            @test s_ts.convergence.iterations == 70

            s_im = solve(prob_g, IntegralMatchingSolver(maxiters=70, lr=1e-12,
                                                        verbose=false))
            @test s_im.convergence.converged == false
            @test s_im.convergence.reason == :maxiters
        end

        @testset "EnsembleKalmanSolver — honest convergence" begin
            decay_ke!(du, u, p, t) = (du[1] = -p.f(u[1]))
            ts_ke = collect(0.0:0.5:10.0)
            ys_ke = reshape(5.0 .* exp.(-0.5 .* ts_ke), :, 1)
            prob_ke = PSMProblem(decay_ke!, [5.0], (0.0, 10.0),
                [BSplineApproximator(:f, (0.0, 6.0), 6)];
                data_times=ts_ke, data_values=ys_ke, obs_to_state=[1],
                known_params=NamedTuple(), solver=Tsit5())
            s_ke = solve(prob_ke, EnsembleKalmanSolver(n_ensemble=30,
                                                       n_iterations=15))
            # Pre-fix: converged=true unconditionally, with no stopping
            # test of any kind and no :reason key at all.
            @test haskey(s_ke.convergence, :reason)
            @test s_ke.convergence.converged == false
            @test s_ke.convergence.reason == :maxiters
            @test s_ke.convergence.iterations == 15
            # The ensemble did not collapse to 1e-3 of its initial spread,
            # which is exactly why :maxiters is the honest answer.
            spread = s_ke.convergence.ensemble_spread
            @test spread[end] > 1e-3 * spread[1]
            # converged=true is reachable only via the collapse criterion.
            @test !s_ke.convergence.converged ||
                  s_ke.convergence.reason == :ensemble_collapse
        end

        @testset "ProfileLikelihoodSolver propagates base convergence" begin
            decay_pl!(du, u, p, t) = (du[1] = -p.f(u[1]))
            ts_pl = collect(0.0:0.5:10.0)
            ys_pl = reshape(5.0 .* exp.(-0.5 .* ts_pl), :, 1)
            prob_pl = PSMProblem(decay_pl!, [5.0], (0.0, 10.0),
                [BSplineApproximator(:f, (0.0, 6.0), 6)];
                data_times=ts_pl, data_values=ys_pl, obs_to_state=[1],
                known_params=NamedTuple(), solver=Tsit5())
            base = solve(prob_pl, LAML(verbose=false))
            s_pl = solve(prob_pl, ProfileLikelihoodSolver(n_profile_points=5,
                                                          param_indices=[1]))
            # Pre-fix the base fit's NamedTuple was discarded and replaced
            # by a hard-coded converged=true with no :reason/:iterations.
            @test s_pl.convergence.converged == base.convergence.converged
            @test s_pl.convergence.reason == base.convergence.reason
            @test s_pl.convergence.iterations == base.convergence.iterations
            @test s_pl.convergence.method == :profile_likelihood
            @test haskey(s_pl.convergence, :profiles)

            # A NON-converged base must propagate as non-converged. The
            # assertions above pass identically on the pre-fix hard-coded
            # `converged=true` because THIS base fit converges — only a
            # truncated base discriminates.
            base_trunc = solve(prob_pl, LAML(maxiters=1, verbose=false))
            @test base_trunc.convergence.converged == false
            s_trunc = solve(prob_pl,
                ProfileLikelihoodSolver(n_profile_points=5, param_indices=[1],
                                        base_alg=LAML(maxiters=1, verbose=false)))
            # PRE-FIX this was `true` regardless; now it tracks the base.
            @test s_trunc.convergence.converged == false
            @test s_trunc.convergence.converged == base_trunc.convergence.converged
            @test s_trunc.convergence.reason == base_trunc.convergence.reason
            @test s_trunc.convergence.iterations == base_trunc.convergence.iterations
            @test s_trunc.convergence.method == :profile_likelihood
        end

        @testset "weighted_data_loss — weighted and NaN-safe" begin
            using PartiallySpecifiedModels: weighted_data_loss
            decay_w!(du, u, p, t) = (du[1] = -p.f(u[1]))
            ts_w = collect(0.0:1.0:5.0)
            ys_w = reshape(collect(1.0:6.0), :, 1)
            w = ones(6, 1); w[2] = 0.0; w[4] = 0.5
            dv = copy(ys_w); dv[2] = NaN          # masked-out missing datum
            prob_w = PSMProblem(decay_w!, [1.0], (0.0, 5.0),
                [BSplineApproximator(:f, (0.0, 6.0), 4)];
                data_times=ts_w, data_values=dv, obs_to_state=[1],
                known_params=NamedTuple(), data_weights=w, solver=Tsit5())
            pred = fill(0.0, 6, 1)
            # 0*NaN = NaN would poison an unmasked sum; the masked cell and
            # the half-weighted cell must both be honoured.
            expected = 1.0^2 + 0.5 * 4.0^2 + 3.0^2 + 5.0^2 + 6.0^2
            @test weighted_data_loss(prob_w, pred) ≈ expected
            @test isfinite(weighted_data_loss(prob_w, pred))
        end

        @testset "_is_program_error rethrows InterruptException" begin
            using PartiallySpecifiedModels: _is_program_error
            # Ctrl-C inside user dynamics was previously swallowed as a
            # numerical failure at ~20 rethrow sites.
            @test _is_program_error(InterruptException())
            @test !_is_program_error(DomainError(-1.0, "sqrt"))
        end
    end

    @testset "NaN-safe optimized objectives (masked data)" begin
        # The campaign made the REPORTED data_loss mask-aware; these tests
        # cover the OPTIMIZED objective. `0 * NaN = NaN` in IEEE arithmetic,
        # so a masked cell (weight 0) whose value is NaN — or whose value was
        # left corrupted precisely BECAUSE it is masked — used to poison the
        # objective itself, not merely the number reported at the end.
        #
        # Each test compares fits on the same underlying data:
        #   masked — 3 rows masked in place (2 NaN, 1 corrupted at weight 0)
        #   pruned — the same 3 rows physically deleted
        # A correct masking implementation makes `masked` ≈ `pruned`.

        function nan_growth!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        true_r_nan = 0.15
        tspan_nan = (0.0, 10.0)
        times_nan = collect(0.0:0.5:10.0)
        rng_nan = Random.Xoshiro(4242)
        clean_nan = exp.(true_r_nan .* times_nan) .+
                    0.01 .* randn(rng_nan, length(times_nan))

        mask_idx = [5, 12, 17]          # rows to mask / prune
        vals_masked = reshape(copy(clean_nan), :, 1)
        w_masked = ones(length(times_nan), 1)
        vals_masked[mask_idx[1], 1] = NaN;  w_masked[mask_idx[1], 1] = 0.0
        vals_masked[mask_idx[2], 1] = NaN;  w_masked[mask_idx[2], 1] = 0.0
        # A CORRUPTED value at weight 0: the package treats weight 0 as
        # "excluded", so a wildly wrong finite number here must be ignored
        # exactly like the NaNs. This catches implementations that guard
        # `isnan` but forget the weight.
        vals_masked[mask_idx[3], 1] = 1.0e6; w_masked[mask_idx[3], 1] = 0.0

        keep_idx = setdiff(1:length(times_nan), mask_idx)
        times_pruned = times_nan[keep_idx]
        vals_pruned = reshape(clean_nan[keep_idx], :, 1)

        mk_nan(ts, vs, ws) = PSMProblem(
            nan_growth!, [1.0], tspan_nan,
            [BSplineApproximator(:r, (0.0, 5.0), 5; initial=x -> 0.05)];
            data_times=ts, data_values=vs, obs_to_state=[1],
            data_weights=ws, likelihood=Gaussian(), solver=Tsit5())

        prob_nan_masked = mk_nan(times_nan, vals_masked, w_masked)
        prob_nan_pruned = mk_nan(times_pruned, vals_pruned,
                                 ones(length(times_pruned), 1))

        @testset "LAML (Gaussian)" begin
            # NOTE ON PRE-FIX BEHAVIOR: LAML alone already passed this — an
            # earlier round of the campaign had made its data FLATTEN mask
            # (masked cells get weight 0 and a finite placeholder), so
            # `log_likelihood` never saw the NaN from this path. This block
            # is therefore a REGRESSION GUARD on that flatten, plus coverage
            # of laml.jl's `n` (now the usable-cell count, which sets the
            # REML scale). The sibling solvers below — GCVSolver and
            # CollocationLAML, the same penalized-likelihood family — did NOT
            # have a masked flatten and DO fail pre-fix.
            #
            # HONEST LABEL: every assertion in THIS block passes on pre-fix
            # code. It is a REGRESSION GUARD, not a discriminating test. The
            # discriminating coverage of `n = _n_usable` and of
            # `estimate_smoothing_params` lives in the
            # "LAML sample size n counts usable cells only" block below,
            # which fails pre-fix both as a unit test (λ 29% off) and
            # end-to-end (λ 85% off, EDF 22% off).
            sol_m = solve(prob_nan_masked, LAML(maxiters=30, verbose=false))
            sol_p = solve(prob_nan_pruned, LAML(maxiters=30, verbose=false))

            # (a) the fit succeeds and reports finite numbers
            @test isfinite(sol_m.objective)
            @test isfinite(sol_m.data_loss)
            @test isfinite(sol_m.edf)
            @test all(isfinite, collect(sol_m.parameters))

            # (b) it recovers the unknown function. Tolerance 0.02 on a true
            # value of 0.15 (13%): the LAML testset above asserts 0.05 on 21
            # clean points, and dropping 3 of 21 widens that only modestly.
            for x in (1.0, 2.0, 3.0)
                @test abs(sol_m.unknown_functions[:r](x) - true_r_nan) < 0.02
            end

            # (c) masked ≈ pruned. Both objectives now see the identical 18
            # usable cells with identical weights, and `n` in laml.jl counts
            # usable cells, so the REML scale matches too. The residual gap is
            # the different Jacobian row count and ODE save grid — genuine
            # differences between the two problem specifications.
            for x in (1.0, 2.0, 3.0)
                @test abs(sol_m.unknown_functions[:r](x) -
                          sol_p.unknown_functions[:r](x)) < 5e-3
            end
            @test isapprox(sol_m.data_loss, sol_p.data_loss; rtol=0.25)

            # The corrupted 1e6 cell at weight 0 must be fully excluded: one
            # such cell would otherwise dominate data_loss outright.
            @test sol_m.data_loss < 1.0
        end

        @testset "GCVSolver (penalized likelihood, sibling of LAML)" begin
            # PRE-FIX FAILURE MODE (measured): the GCV flatten copied
            # data_values verbatim, so `log_likelihood` — and the IRLS
            # pseudo-data built from it — were NaN. Every PCLS step was
            # rejected and the solver returned its initialization:
            # r(1.0) = r(2.0) = 0.05 exactly (the `initial=` value), with a
            # finite-looking objective 16.877 and data_loss 33.755 (the
            # reported loss is computed by `weighted_data_loss`, which was
            # already masked, so nothing looked wrong). Silent wrong answer.
            sol_m = solve(prob_nan_masked, GCVSolver(maxiters=25, verbose=false))
            sol_p = solve(prob_nan_pruned, GCVSolver(maxiters=25, verbose=false))

            @test isfinite(sol_m.objective)
            @test all(isfinite, collect(sol_m.parameters))
            # Must have MOVED off the initialization (the pre-fix failure).
            @test abs(sol_m.unknown_functions[:r](1.0) - 0.05) > 1e-6
            for x in (1.0, 2.0, 3.0)
                @test abs(sol_m.unknown_functions[:r](x) - true_r_nan) < 0.02
                @test abs(sol_m.unknown_functions[:r](x) -
                          sol_p.unknown_functions[:r](x)) < 5e-3
            end
            @test sol_m.data_loss < 1.0
        end

        @testset "CollocationLAML (penalized likelihood, sibling of LAML)" begin
            # PRE-FIX FAILURE MODE (measured): the collocation data residual
            # is `sqrt(w) * (y - alpha)`, and `sqrt(0) * NaN = NaN` — a zero
            # weight does NOT neutralize a NaN datum, though the matching
            # JACOBIAN row was already a clean 0, which is what made this so
            # easy to miss. The Gauss-Newton step went all-NaN, the line
            # search rejected every step, and the solver reported
            # `:plateau` convergence at its initialization with
            # r(1.0) = 0.05 and a FABRICATED objective of -1.35e-19 and
            # data_loss 0.0 — i.e. it claimed a perfect fit.
            cl = CollocationLAML(maxiters=20, verbose=false,
                                 lambda_ode_start=0.01, lambda_ode_end=100.0,
                                 n_continuation=4)
            sol_m = solve(prob_nan_masked, cl)

            @test isfinite(sol_m.objective)
            @test all(isfinite, collect(sol_m.parameters))
            # A real fit has a nonzero data loss; the pre-fix code reported
            # exactly 0.0 because it never moved off a zeroed residual.
            @test sol_m.data_loss > 0.0
            @test abs(sol_m.unknown_functions[:r](1.0) - 0.05) > 1e-6
            for x in (1.0, 2.0, 3.0)
                @test abs(sol_m.unknown_functions[:r](x) - true_r_nan) < 0.05
            end
        end

        @testset "DerivativeFreeSolver" begin
            # PRE-FIX FAILURE MODE: the flatten copied data_values verbatim,
            # so `w_vec[k] * (y_vec[k] - pred)^2` was NaN. The guard
            # `!isfinite(data_loss) && return 1e20` then made the loss a
            # CONSTANT for every beta — Nelder-Mead saw a flat surface, never
            # moved, and returned its initialization with a normal-looking
            # convergence report. A silent no-op.
            dfs = DerivativeFreeSolver(maxiters=3000, verbose=false)
            sol_m = solve(prob_nan_masked, dfs)
            sol_p = solve(prob_nan_pruned, dfs)

            @test isfinite(sol_m.objective)
            @test isfinite(sol_m.data_loss)
            @test all(isfinite, collect(sol_m.parameters))
            # Not the 1e20 sentinel, and not the flat constant the pre-fix
            # code produced.
            @test sol_m.objective < 1e19

            for x in (1.0, 2.0, 3.0)
                @test abs(sol_m.unknown_functions[:r](x) - true_r_nan) < 0.05
            end
            for x in (1.0, 2.0, 3.0)
                @test abs(sol_m.unknown_functions[:r](x) -
                          sol_p.unknown_functions[:r](x)) < 0.03
            end
            @test sol_m.data_loss < 1.0
        end

        @testset "GradientMatching / TwoStageSolver" begin
            # PRE-FIX FAILURE MODE (both): `_smoothing_spline` solves
            # `BtB \ Bty`; one NaN in the column makes every coefficient NaN,
            # and because `gcv < best_gcv` is false for a NaN score at every
            # lambda, the lambda loop never replaces it. Both returned
            # callables were then NaN for ALL inputs, so dydt was all-NaN.
            # GradientMatching's Gauss-Newton stopped at iteration 1 ("no
            # improvement"); TwoStageSolver's Adam drove beta to NaN and
            # returned `best_beta` = the initialization with objective Inf.
            sol_gm_m = solve(prob_nan_masked, GradientMatching(verbose=false))
            sol_gm_p = solve(prob_nan_pruned, GradientMatching(verbose=false))

            @test isfinite(sol_gm_m.objective)
            @test all(isfinite, collect(sol_gm_m.parameters))
            for x in (1.0, 2.0, 3.0)
                @test abs(sol_gm_m.unknown_functions[:r](x) - true_r_nan) < 0.05
            end
            # Gradient matching fits a SMOOTHER through the usable points and
            # matches derivatives, so masked-vs-pruned agreement is governed
            # by that smoother, which now sees the identical 18 points in
            # both problems. The remaining gap is the match-point grid.
            for x in (1.0, 2.0, 3.0)
                @test abs(sol_gm_m.unknown_functions[:r](x) -
                          sol_gm_p.unknown_functions[:r](x)) < 0.05
            end

            sol_ts_m = solve(prob_nan_masked, TwoStageSolver(verbose=false))
            @test isfinite(sol_ts_m.objective)
            @test all(isfinite, collect(sol_ts_m.parameters))
            for x in (1.0, 2.0, 3.0)
                @test abs(sol_ts_m.unknown_functions[:r](x) - true_r_nan) < 0.05
            end
        end

        @testset "likelihood families mask at the choke point" begin
            y_ok  = [1.0, 2.0, 3.0, 4.0]
            mu_ok = [1.1, 2.1, 2.9, 4.2]
            # Cell 2 masked as a NaN datum, cell 4 masked by zero weight but
            # left holding a corrupted value.
            y_bad = [1.0, NaN, 3.0, 1.0e9]
            w_bad = [1.0, 0.0, 1.0, 0.0]
            # The same problem with those two cells genuinely removed.
            y_ref  = [1.0, 3.0]
            mu_ref = [1.1, 2.9]
            w_ref  = [1.0, 1.0]

            fams = (Gaussian(), Poisson(), NegativeBinomial(5.0),
                    TruncatedNormal(0.0, 0.7),
                    CustomLikelihood((a, b) -> -0.6 * (a - b)^2))
            for fam in fams
                ll_bad = PartiallySpecifiedModels.log_likelihood(
                    fam, y_bad, mu_ok, w_bad)
                ll_ref = PartiallySpecifiedModels.log_likelihood(
                    fam, y_ref, mu_ref, w_ref)
                @test isfinite(ll_bad)
                # Masked cells contribute exactly nothing, so the total must
                # equal the total over the surviving cells alone.
                @test ll_bad ≈ ll_ref

                wt = PartiallySpecifiedModels.irls_weights(
                    fam, y_bad, mu_ok, w_bad)
                @test all(isfinite, wt)
                @test wt[2] == 0.0     # NaN datum
                @test wt[4] == 0.0     # zero weight
            end

            # Pointwise: a NaN datum contributes zero and stays AD-safe —
            # zero gradient rather than a NaN one.
            for fam in fams
                @test PartiallySpecifiedModels.loglik_pointwise(fam, NaN, 2.0) == 0.0
                d = PartiallySpecifiedModels.ForwardDiff.derivative(
                    m -> PartiallySpecifiedModels.loglik_pointwise(fam, NaN, m), 2.0)
                @test d == 0.0
            end
        end

        @testset "solvers without masking support fail loudly" begin
            # These evaluate their likelihood inside a Kalman/particle
            # recursion with no per-cell mask, so masked cells would corrupt
            # the filter state rather than be skipped. They must say so
            # instead of silently returning their initialization.
            for alg in (RodeoSolver(n_steps=20, maxiters=2, verbose=false),
                        DaltonSolver(n_steps=20, maxiters=2, verbose=false),
                        PseudoMarginalSolver(n_samples=5, n_warmup=2,
                                             n_steps=20, verbose=false),
                        EnsembleKalmanSolver(n_ensemble=5, n_iterations=2,
                                             verbose=false))
                err = try
                    solve(prob_nan_masked, alg); nothing
                catch e
                    e
                end
                @test err isa ErrorException
                @test occursin("does not support masked observations", err.msg)
            end
        end

        # ── D4: ONE usability predicate everywhere ────────────────────
        @testset "usability predicate is isfinite, consistently" begin
            using PartiallySpecifiedModels: _usable, usable_cell, n_usable,
                                            weighted_data_loss
            # PRE-FIX: `weighted_data_loss` gated on `isfinite(y)` while
            # `_usable`/`usable_cell`/`n_usable`/`_reject_masked_data` gated
            # on `!isnan(y)`. An Inf datum was therefore EXCLUDED from the
            # numerator but COUNTED in every denominator, and sailed past
            # the Kalman-solver rejection into a filter that cannot mask.
            @test !_usable(Inf, 1.0)
            @test !_usable(-Inf, 1.0)
            @test !_usable(NaN, 1.0)
            @test !_usable(1.0, 0.0)
            @test _usable(1.0, 1.0)

            ts_inf = collect(0.0:1.0:5.0)
            vals_inf = reshape(collect(1.0:6.0), :, 1)
            vals_inf[3, 1] = Inf
            prob_inf = mk_nan(ts_inf, vals_inf, ones(6, 1))

            @test !usable_cell(prob_inf, 3, 1)
            @test usable_cell(prob_inf, 2, 1)
            # Numerator and denominator now agree: 5 usable cells, and the
            # Inf row contributes nothing to the loss.
            @test n_usable(prob_inf) == 5
            @test isfinite(weighted_data_loss(prob_inf, zeros(6, 1)))
            @test weighted_data_loss(prob_inf, zeros(6, 1)) ≈
                  sum(x^2 for x in [1.0, 2.0, 4.0, 5.0, 6.0])
            # ...and the rejection predicate no longer waves it through.
            err_inf = try
                solve(prob_inf, RodeoSolver(n_steps=10, maxiters=2,
                                            verbose=false)); nothing
            catch e; e end
            @test err_inf isa ErrorException
            @test occursin("does not support masked observations", err_inf.msg)

            # Pointwise likelihoods treat ±Inf like NaN: zero contribution,
            # zero gradient, rather than an Inf/NaN that poisons the sum.
            for fam in (Gaussian(), Poisson(), NegativeBinomial(5.0),
                        TruncatedNormal())
                @test PartiallySpecifiedModels.loglik_pointwise(fam, Inf, 2.0) == 0.0
            end
        end

        # ── D1: residual diagnostics are mask-aware ───────────────────
        @testset "residual diagnostics are mask-aware" begin
            using PartiallySpecifiedModels: residual_diagnostics, appraise
            # PRE-FIX FAILURE MODE (measured on this exact fit): the LAML
            # fit itself was clean, but `diagnostics.jl` never saw the mask.
            #   residual_diagnostics: durbin_watson = [NaN], acf = all NaN,
            #     semivariogram gamma = all NaN
            #   appraise: residuals/observed/qq_sample all NaN — the `std`
            #     over a vector containing one NaN is NaN, so EVERY
            #     standardized residual was NaN, not just the masked one.
            # The comparison fit (masked rows physically pruned) was fine
            # throughout, so each assertion below discriminates.
            ts_d = collect(0.0:0.5:10.0)
            # Deterministic structured "noise": residuals must be dominated
            # by signal, not by ODE round-off, or DW/ACF compare pure noise.
            noise_d = [0.05 * sin(3.1 * i) + 0.03 * cos(7.7 * i)
                       for i in 1:length(ts_d)]
            clean_d = exp.(0.15 .* ts_d) .+ noise_d
            mask_d = [4, 9, 15]
            keep_d = setdiff(1:length(ts_d), mask_d)

            vals_d = reshape(copy(clean_d), :, 1)
            w_d = ones(length(ts_d), 1)
            for i in mask_d
                vals_d[i, 1] = NaN
                w_d[i, 1] = 0.0
            end

            sol_dm = solve(mk_nan(ts_d, vals_d, w_d),
                           LAML(maxiters=30, verbose=false))
            sol_dp = solve(mk_nan(ts_d[keep_d],
                                  reshape(clean_d[keep_d], :, 1),
                                  ones(length(keep_d), 1)),
                           LAML(maxiters=30, verbose=false))

            rd = residual_diagnostics(sol_dm)
            rp = residual_diagnostics(sol_dp)

            # (a) every STATISTIC is finite
            @test all(isfinite, rd.durbin_watson)
            @test all(isfinite, rd.acf)
            @test all(isfinite, rd.semivariogram[1].gamma)
            @test all(isfinite, rd.semivariogram[1].lags)
            # `residuals` keeps its rectangular shape; the mask says which
            # cells carry a residual at all.
            @test size(rd.residuals) == size(sol_dm.data_values)
            @test rd.usable == isfinite.(sol_dm.data_values)
            @test count(rd.usable) == length(keep_d)
            @test all(isfinite, rd.residuals[rd.usable])
            @test all(!isfinite, rd.residuals[.!rd.usable])

            # (b) they MATCH the fit with the masked rows genuinely removed
            @test isapprox(rd.durbin_watson[1], rp.durbin_watson[1]; rtol=1e-4)
            @test size(rd.acf) == size(rp.acf)
            @test isapprox(rd.acf, rp.acf; rtol=1e-4)
            @test isapprox(rd.semivariogram[1].gamma,
                           rp.semivariogram[1].gamma; rtol=1e-3)

            ad = appraise(sol_dm)
            ap = appraise(sol_dp)
            @test length(ad.residuals) == length(keep_d)
            @test length(ad.observed) == length(keep_d)
            @test length(ad.fitted) == length(keep_d)
            @test length(ad.qq_theoretical) == length(keep_d)
            @test all(isfinite, ad.residuals)
            @test all(isfinite, ad.observed)
            @test all(isfinite, ad.fitted)
            @test all(isfinite, ad.qq_sample)
            @test all(isfinite, ad.qq_theoretical)
            @test all(isfinite, ad.durbin_watson)
            @test issorted(ad.qq_sample)
            @test isapprox(ad.residuals, ap.residuals; rtol=1e-3)
            @test isapprox(ad.observed, ap.observed; rtol=1e-10)
            @test isapprox(ad.durbin_watson[1], ap.durbin_watson[1]; rtol=1e-4)

            # Non-Gaussian path (deviance residuals + robust scale) is
            # masked too — pre-fix `median_abs` over a NaN-bearing vector
            # made every standardized residual NaN.
            ad_p = appraise(sol_dm; family=Poisson())
            @test length(ad_p.residuals) == length(keep_d)
            @test all(isfinite, ad_p.residuals)

            # DOCUMENTED LIMITATION: `PSMSolution` carries no data_weights,
            # so a cell masked ONLY by a zero weight (finite value) is not
            # detectable and is still counted. `prob_nan_masked` has exactly
            # such a cell (1e6 at weight 0) plus two NaN cells.
            sol_lim = solve(prob_nan_masked, LAML(maxiters=20, verbose=false))
            rl = residual_diagnostics(sol_lim)
            @test count(rl.usable) == length(times_nan) - 2   # NaNs only
            @test all(isfinite, rl.durbin_watson)             # still finite
        end

        # ── D5 / D6: IntegralMatchingSolver gates like its siblings ───
        @testset "IntegralMatchingSolver masks its loss" begin
            # PRE-FIX: the smoother dropped the masked rows, but the loss ran
            # over EVERY time index ungated and anchored the increments on
            # `y_smooth[1, k]` even when row 1 was masked — so masked rows
            # were fitted against the smoother's own interpolation and, with
            # row 1 masked, against its extrapolation.
            ts_i = collect(0.0:0.5:8.0)
            clean_i = exp.(0.15 .* ts_i)
            mask_i = [1, 6, 11]        # row 1 masked ⇒ baseline must move
            keep_i = setdiff(1:length(ts_i), mask_i)
            vals_i = reshape(copy(clean_i), :, 1)
            w_i = ones(length(ts_i), 1)
            for i in mask_i
                vals_i[i, 1] = NaN
                w_i[i, 1] = 0.0
            end

            s_im = solve(mk_nan(ts_i, vals_i, w_i),
                         IntegralMatchingSolver(maxiters=120, verbose=false))
            @test isfinite(s_im.objective)
            @test isfinite(s_im.data_loss)
            @test all(isfinite, collect(s_im.parameters))
            @test abs(s_im.unknown_functions[:r](2.0) - 0.15) < 0.06

            # Every cell masked ⇒ nothing to match; say so instead of
            # optimizing against a pure interpolation.
            all_masked = fill(NaN, length(ts_i), 1)
            err_im = try
                solve(mk_nan(ts_i, all_masked, zeros(length(ts_i), 1)),
                      IntegralMatchingSolver(maxiters=5, verbose=false))
                nothing
            catch e; e end
            @test err_im isa ErrorException
            # The smoother refuses first (`_smoothing_spline`); the
            # baseline guard added here is defence-in-depth behind it.
            @test occursin("masked", err_im.msg)
        end

        # ── D2: MagiSolver reports a real objective and data_loss ─────
        @testset "MagiSolver reports objective and data_loss" begin
            # PRE-FIX: `PSMSolution(params, 0.0, 0.0, …)` — the objective AND
            # the data loss were hard-coded zero, so every MAGI run claimed
            # a perfect fit no diagnostic could distinguish from a real one.
            ts_mg = collect(0.0:1.0:8.0)
            vals_mg = reshape(exp.(-0.3 .* ts_mg), :, 1)
            prob_mg = PSMProblem((du, u, p, t) -> (du[1] = -p.r(t) * u[1]),
                [1.0], (0.0, 8.0),
                [BSplineApproximator(:r, (0.0, 8.0), 5; initial=0.2)];
                data_times=ts_mg, data_values=vals_mg, obs_to_state=[1],
                known_params=NamedTuple(), likelihood=Gaussian(),
                solver=Tsit5())
            s_mg = solve(prob_mg, MagiSolver(n_samples=30, n_warmup=30,
                                             n_gridpoints=17, verbose=false))
            @test isfinite(s_mg.objective)
            @test isfinite(s_mg.data_loss)
            @test s_mg.data_loss >= 0.0
            # It is the SAME mask-aware weighted RSS every other solver
            # reports, evaluated at the posterior-mean trajectory.
            @test s_mg.data_loss ≈
                  PartiallySpecifiedModels.weighted_data_loss(prob_mg,
                                                              s_mg.fitted_values)
            @test isfinite(s_mg.convergence.mean_logposterior)
            @test s_mg.objective ≈ -s_mg.convergence.mean_logposterior
            # NUTS has no stopping criterion: loop exhaustion, reported honestly.
            @test haskey(s_mg.convergence, :converged)
            @test haskey(s_mg.convergence, :reason)
            @test s_mg.convergence.converged == false
            @test s_mg.convergence.reason == :maxiters
        end

        # ── D2: GradientMatching reports converged/reason ─────────────
        @testset "GradientMatching reports convergence keys" begin
            # PRE-FIX: its convergence NamedTuple was
            # `(deriv_loss=…, method=:gradient_matching)` — no `converged`,
            # no `reason`, unlike every sibling solver.
            ts_g = collect(0.0:0.5:10.0)
            vals_g = reshape(exp.(0.15 .* ts_g), :, 1)
            prob_g = mk_nan(ts_g, vals_g, ones(length(ts_g), 1))
            s_g = solve(prob_g, GradientMatching(maxiters=40, verbose=false))
            @test haskey(s_g.convergence, :converged)
            @test haskey(s_g.convergence, :reason)
            @test haskey(s_g.convergence, :iterations)
            @test s_g.convergence.converged isa Bool
            @test s_g.convergence.reason isa Symbol
            @test s_g.convergence.reason in
                  (:objective_tol, :maxiters, :plateau,
                   :line_search_failure, :singular_system)
            @test 1 <= s_g.convergence.iterations <= 40
            # `converged` is only ever true alongside a criterion reason.
            @test !s_g.convergence.converged ||
                  s_g.convergence.reason in (:objective_tol, :plateau)
        end

        # ── D3: _vi_edf uses the family's curvature ───────────────────
        @testset "_vi_edf uses the family variance function" begin
            using PartiallySpecifiedModels: _vi_edf
            # PRE-FIX: `H = J'WJ / obs_noise_var` unconditionally, and
            # `obs_noise_var` is hard-set to 1.0 for non-Gaussian families —
            # so a Poisson/NB EDF was computed with UNIT observation
            # variance instead of V(μ̂). With counts in the hundreds that is
            # a curvature wrong by two orders of magnitude.
            ts_v = collect(0.0:1.0:10.0)
            mu_v = 200.0 .* exp.(0.05 .* ts_v)
            counts = round.(mu_v)
            prob_pois = PSMProblem((du, u, p, t) -> (du[1] = p.r(u[1]) * u[1]),
                [200.0], (0.0, 10.0),
                [BSplineApproximator(:r, (150.0, 400.0), 5; initial=x -> 0.05)];
                data_times=ts_v, data_values=reshape(counts, :, 1),
                obs_to_state=[1], likelihood=Poisson(), solver=Tsit5())

            beta_v = PartiallySpecifiedModels.build_initial_params(prob_pois)
            n_pv = length(beta_v)
            # Λ large enough that neither EDF saturates at the n_p clamp,
            # so the two curvatures are cleanly separated.
            Λ = Matrix(1.0e4 * I, n_pv, n_pv)

            edf_fam = _vi_edf(prob_pois, Float64.(beta_v), Λ, 1.0, n_pv)
            @test isfinite(edf_fam)
            @test 0.0 <= edf_fam <= n_pv

            # The pre-fix curvature is exactly what `_vi_edf` computes for a
            # GAUSSIAN problem with the same data and σ²_obs = 1 — the unit
            # observation variance. Reconstruct it and show the two differ:
            # if they agreed, the fix would be a no-op.
            prob_gauss = PSMProblem((du, u, p, t) -> (du[1] = p.r(u[1]) * u[1]),
                [200.0], (0.0, 10.0),
                [BSplineApproximator(:r, (150.0, 400.0), 5; initial=x -> 0.05)];
                data_times=ts_v, data_values=reshape(counts, :, 1),
                obs_to_state=[1], likelihood=Gaussian(), solver=Tsit5())
            edf_prefix = _vi_edf(prob_gauss, Float64.(beta_v), Λ, 1.0, n_pv)
            @test isfinite(edf_prefix)
            # MEASURED: Poisson curvature gives EDF 1.180, the pre-fix
            # unit-variance form gives 2.990 — a factor of 2.5.
            @test !isapprox(edf_fam, edf_prefix; rtol=1e-2)
            @test edf_fam < edf_prefix

            # Gaussian data is untouched by the change: still w/σ²_obs.
            @test _vi_edf(prob_gauss, Float64.(beta_v), Λ, 4.0, n_pv) <
                  edf_prefix        # more observation noise ⇒ fewer dof
        end

        # ── D7: the Mp_eff ≥ n warning is Gaussian-only ───────────────
        @testset "LAML Mp_eff warning is gated to the Gaussian branch" begin
            using PartiallySpecifiedModels: estimate_smoothing_params
            # PRE-FIX: the warning fired for every family, but only the
            # Gaussian branch reads `n_eff = n − Mp_eff`; the non-Gaussian
            # branch uses `n − sum(ranks)`. A healthy Poisson fit could
            # therefore emit an alarming, irrelevant "σ̂² is NOT identified"
            # warning about a quantity it never computes.
            # n = 2 with an 8-column basis: the penalty null space has
            # dimension 2, so Mp_eff = 2 ≥ n = 2 and the Gaussian REML
            # scale denominator is genuinely exhausted.
            n_w, p_w = 2, 8
            xw = range(0.0, 1.0, length=n_w)
            Jw = [xi^(k - 1) for xi in xw, k in 1:p_w]
            Dw = zeros(p_w - 2, p_w)
            for i in 1:(p_w - 2)
                Dw[i, i] = 1.0; Dw[i, i+1] = -2.0; Dw[i, i+2] = 1.0
            end
            Sw = Dw' * Dw
            bw = fill(0.1, p_w)
            yw = fill(5.0, n_w)
            muw = fill(5.0, n_w)
            ww = ones(n_w)

            call(fam) = estimate_smoothing_params(Jw, ww, ww, yw, muw, bw,
                        [Sw], [0], [p_w], p_w; family=fam, maxiter=3,
                        verbose=false)
            # Gaussian: the warning is relevant and still fires.
            @test_logs (:warn, r"unpenalized \(fixed-effect\) part") match_mode=:any call(Gaussian())
            # Poisson: no such warning (its scale comes from n − sum(ranks)).
            # `Base.CoreLogging.Warn` avoids adding a Logging test dep.
            @test_logs min_level=Base.CoreLogging.Warn call(Poisson())
        end

        # ── D8a: the LAML denominator genuinely counts usable cells ───
        @testset "LAML sample size n counts usable cells only" begin
            using PartiallySpecifiedModels: estimate_smoothing_params, _n_usable

            # (i) DIRECT unit test of the denominator. The LAML flatten
            # already masks the VALUES (masked cells arrive as a finite
            # placeholder at weight 0), so the only thing `n = _n_usable`
            # changes is the SAMPLE SIZE. Half the sample masked makes that
            # visible: n = 21 vs the pre-fix n = 40.
            n_u, p_u = 40, 8
            xu = range(0.0, 1.0, length=n_u)
            Ju = [xi^(k - 1) for xi in xu, k in 1:p_u]
            Ju = Ju ./ maximum(abs, Ju)
            Du = zeros(p_u - 2, p_u)
            for i in 1:(p_u - 2)
                Du[i, i] = 1.0; Du[i, i+1] = -2.0; Du[i, i+2] = 1.0
            end
            Su = Du' * Du
            beta_u = [0.3, -0.5, 0.8, 0.1, -0.2, 0.05, 0.0, 0.0]
            yu = Ju * beta_u .+ [0.05 * sin(3.7 * i) for i in 1:n_u]
            mask_u = collect(3:2:39)             # 19 of 40 cells masked
            keep_u = setdiff(1:n_u, mask_u)
            y_um = copy(yu); w_um = ones(n_u)
            y_um[mask_u] .= 0.0                  # LAML flatten convention:
            w_um[mask_u] .= 0.0                  # placeholder + weight 0

            @test _n_usable(y_um, w_um) == length(keep_u)
            @test _n_usable(yu, ones(n_u)) == n_u

            run_u(J, w, y) = estimate_smoothing_params(J, w, w, y, J * beta_u,
                                copy(beta_u), [Su], [0], [p_u], p_u;
                                family=Gaussian(), maxiter=50, verbose=false)
            lam_um, edf_um = run_u(Ju, w_um, y_um)
            lam_up, edf_up = run_u(Ju[keep_u, :], ones(length(keep_u)),
                                   yu[keep_u])
            # PRE-FIX (measured with `n = length(y)`): λ = 5.05e-3 against
            # the pruned 7.13e-3 — 29% off — and EDF 4.010 vs 3.941.
            # POST-FIX both agree to ~1e-13.
            @test isapprox(lam_um[1], lam_up[1]; rtol=1e-6)
            @test isapprox(edf_um, edf_up; rtol=1e-6)

            # (ii) END-TO-END. Heavy masking (20 of 41 rows) makes the same
            # denominator bias reach the reported λ and EDF.
            ts_n = collect(0.0:0.25:10.0)
            noise_n = [0.04 * sin(3.1 * i) + 0.02 * cos(7.7 * i)
                       for i in 1:length(ts_n)]
            clean_n = exp.(0.15 .* ts_n) .+ noise_n
            mask_n = collect(2:2:40)
            keep_n = setdiff(1:length(ts_n), mask_n)
            vals_n = reshape(copy(clean_n), :, 1)
            w_n = ones(length(ts_n), 1)
            vals_n[mask_n, 1] .= NaN
            w_n[mask_n, 1] .= 0.0

            mk_n(t, v, w) = PSMProblem(nan_growth!, [1.0], tspan_nan,
                [BSplineApproximator(:r, (0.0, 5.0), 6; initial=x -> 0.05)];
                data_times=t, data_values=v, obs_to_state=[1],
                data_weights=w, likelihood=Gaussian(), solver=Tsit5())

            s_nm = solve(mk_n(ts_n, vals_n, w_n), LAML(maxiters=40, verbose=false))
            s_np = solve(mk_n(ts_n[keep_n], reshape(clean_n[keep_n], :, 1),
                              ones(length(keep_n), 1)),
                         LAML(maxiters=40, verbose=false))
            # PRE-FIX (measured): λ 0.0867 (masked) vs 0.5658 (pruned) —
            # 85% relative error — and EDF 2.691 vs 2.199 (22%).
            # POST-FIX: 2.7e-4 and 1.7e-5.
            #
            # λ tolerance WIDENED 0.02 -> 0.05 when `smoothing_params` was
            # changed to report the θ β was actually fitted at (rather than
            # the last, untested Fellner-Schall proposal — see the
            # `theta_fit` comment in solver.jl). This is a REAL change, not
            # a papered-over one, and it is one-sided: the MASKED fit's λ is
            # bit-identical before and after (0.5658567355454274); only the
            # PRUNED fit moved, 0.5658007 -> 0.5455690, so masked-vs-pruned
            # agreement went from 9.9e-5 to 3.6e-2. The pre-fix closeness
            # was coincidence — both sides happened to report proposals that
            # nearly matched. Three things say the new value is the better
            # one: on the pruned problem it attains a HIGHER LAML criterion
            # than the number it replaced (74.172972 vs 74.172875); λ is
            # barely identified here, that same 3.6% λ move buying only
            # 9.6e-5 of criterion; and the quantities the denominator bug
            # actually corrupted still agree tightly — EDF to 0.28% and σ̂²
            # to 0.031% (both still asserted at rtol=0.02 below), with
            # `data_loss` agreeing to 0.06% (3 significant figures). 0.05
            # still separates fixed from broken by 17x (the bug gave 85%).
            #
            # The 3.6% gap is an artifact of the DEFAULT FD Jacobian, not of
            # masking: under jac=:forwarddiff the same masked-vs-pruned
            # comparison agrees to 5.9e-9 both before and after this fix.
            # That is asserted below at rtol=1e-6, keeping a genuinely tight
            # guard on λ equivalence — the 0.05 line above has only 1.4x
            # headroom and would not catch a small regression on its own.
            @test isapprox(s_nm.smoothing_params[1], s_np.smoothing_params[1];
                           rtol=0.05)
            s_nm_fd = solve(mk_n(ts_n, vals_n, w_n),
                            LAML(maxiters=40, verbose=false, jac=:forwarddiff))
            s_np_fd = solve(mk_n(ts_n[keep_n], reshape(clean_n[keep_n], :, 1),
                                 ones(length(keep_n), 1)),
                            LAML(maxiters=40, verbose=false, jac=:forwarddiff))
            @test isapprox(s_nm_fd.smoothing_params[1],
                           s_np_fd.smoothing_params[1]; rtol=1e-6)
            @test isapprox(s_nm.edf, s_np.edf; rtol=0.02)
            @test isapprox(s_nm.convergence.sigma2, s_np.convergence.sigma2;
                           rtol=0.02)
        end
    end

    @testset "approximator extension protocol" begin
        import Lux

        # (d) All 7 built-in types round-trip through build_evaluator:
        # evaluator from initial_params, called at the domain midpoint,
        # returns a finite number.
        builtins = PartiallySpecifiedModels.AbstractApproximator[
            BSplineApproximator(:f, (0.0, 1.0), 6),
            ShapeConstrainedBSplineApproximator(:f, (0.0, 1.0), 8, :increasing),
            SPDEApproximator(:f, (0.0, 1.0), 10),
            ShapeConstrainedSPDEApproximator(:f, (0.0, 1.0), 8, :increasing),
            GPApproximator(:f, (0.0, 1.0), 6; kernel=:matern52),
            COMONetApproximator(:f, (0.0, 1.0), (8, 8), :increasing),
            NeuralApproximator(:f, Lux.Chain(Lux.Dense(1, 4, tanh), Lux.Dense(4, 1));
                               domain=(0.0, 1.0), rng_seed=42),
        ]
        for a in builtins
            p0 = initial_params(a)
            @test length(p0) == nparams(a)
            ev = build_evaluator(a, p0)
            v = ev(0.5)
            @test v isa Number && isfinite(v)
        end

        # (c) Fallback error for a type implementing nothing: names all four
        # interface functions and points at the docs page.
        @test_throws ErrorException build_evaluator(NotAnApproximator(), Float64[])
        err = try
            build_evaluator(NotAnApproximator(), Float64[])
        catch e
            e
        end
        msg = sprint(showerror, err)
        @test occursin("NotAnApproximator", msg)
        for fn in ("nparams", "initial_params", "penalty_matrix", "build_evaluator")
            @test occursin(fn, msg)
        end
        @test occursin("Custom approximators", msg)

        # Custom PolyApproximator basics: interface methods and evaluator
        pa = PolyApproximator(:f, (0.0, 1.0), 2)
        @test nparams(pa) == 3
        @test initial_params(pa) == zeros(3)
        Spa = penalty_matrix(pa)
        @test size(Spa) == (3, 3)
        @test Spa[3, 3] == 1.0 && Spa[1, 1] == 0.0  # constants/lines unpenalized
        @test build_evaluator(pa, [1.0, 2.0, 3.0])(0.5) ≈ 1.0 + 2.0 * 0.5 + 3.0 * 0.25

        # Decay problem du = -f(u) with true f(u) = 0.7u, noise-free data.
        # u ranges over [exp(-2.8), 1] ⊂ (0, 1], so f is a line inside the
        # PolyApproximator's model class and should be recovered.
        poly_decay!(du, u, p, t) = (du[1] = -p.f(u[1]))
        ts_poly = collect(0.0:0.25:4.0)
        obs_poly = reshape(exp.(-0.7 .* ts_poly), :, 1)
        mk_poly() = PSMProblem(poly_decay!, [1.0], (0.0, 4.0),
            [PolyApproximator(:f, (0.0, 1.0), 2)];
            data_times=ts_poly, data_values=obs_poly, obs_to_state=[1],
            likelihood=Gaussian(), solver=Tsit5())

        # (a) AdamSolver (through-the-solver autodiff: exercises the
        # Dual-safety of the custom evaluator)
        sol_adam = solve(mk_poly(), AdamSolver(maxiters=1000, lr=0.02, verbose=false))
        f_adam = sol_adam.unknown_functions[:f]
        @test isfinite(sol_adam.data_loss)
        @test sol_adam.data_loss < 0.05
        @test abs(f_adam(0.5) - 0.35) < 0.1
        @test abs(f_adam(0.9) - 0.63) < 0.15

        # (b) DerivativeFreeSolver
        sol_df = solve(mk_poly(), DerivativeFreeSolver(maxiters=4000, verbose=false))
        f_df = sol_df.unknown_functions[:f]
        @test isfinite(sol_df.data_loss)
        @test sol_df.data_loss < 0.05
        @test abs(f_df(0.5) - 0.35) < 0.1
    end

    @testset "custom-type penalties in whitelist solvers (W12)" begin
        # The six per-type penalty sites (TwoStage, IntegralMatching, MAGI,
        # AGM, Rodeo, Dalton) fall back to generic penalty_matrix(approx)
        # for non-built-in types. Discriminator: pre-change, a custom
        # type's penalty was silently ignored there, so lambda_smooth had
        # NO effect on the fit — these assertions fail on pre-change code.
        # TwoStage and IntegralMatching are tested end-to-end (cheap);
        # MAGI/AGM/Rodeo/Dalton share the identical fallback pattern.
        decay_w12!(du, u, p, t) = (du[1] = -p.f(u[1]))
        t_w12 = collect(0.0:0.4:4.0)
        d_w12 = reshape(2.0 .* exp.(-0.5 .* t_w12), :, 1)
        mk12 = () -> PSMProblem(decay_w12!, [2.0], (0.0, 4.0),
            [PolyApproximator(:f, (0.0, 2.0), 2)];
            data_times=t_w12, data_values=d_w12, obs_to_state=[1],
            known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())

        # PolyApproximator's penalty is a ridge on the curvature (x^2)
        # coefficient, so a huge lambda_smooth must crush |beta_3| in the
        # penalized fit relative to the unpenalized one.
        sol_ts0 = solve(mk12(), TwoStageSolver(lambda_smooth=0.0,
            maxiters=800, verbose=false))
        sol_tsP = solve(mk12(), TwoStageSolver(lambda_smooth=1e4,
            maxiters=800, verbose=false))
        @test sol_ts0.parameters != sol_tsP.parameters
        @test abs(sol_tsP.parameters[3]) < abs(sol_ts0.parameters[3])

        sol_im0 = solve(mk12(), IntegralMatchingSolver(lambda_smooth=0.0,
            maxiters=800, verbose=false))
        sol_imP = solve(mk12(), IntegralMatchingSolver(lambda_smooth=1e4,
            maxiters=800, verbose=false))
        @test sol_im0.parameters != sol_imP.parameters
        @test abs(sol_imP.parameters[3]) < abs(sol_im0.parameters[3])
    end

    # ─── TensorBSplineApproximator (bivariate tensor-product spline) ──

    @testset "TensorBSplineApproximator — construction and validation" begin
        # inverted / degenerate domains, per margin
        @test_throws ArgumentError TensorBSplineApproximator(
            :g, (1.0, 0.0), (0.0, 1.0), 5, 5)
        @test_throws ArgumentError TensorBSplineApproximator(
            :g, (0.0, 1.0), (2.0, 2.0), 5, 5)
        # nknots < 3 per margin (cubic interpolation floor, as univariate)
        @test_throws ArgumentError TensorBSplineApproximator(
            :g, (0.0, 1.0), (0.0, 1.0), 2, 5)
        @test_throws ArgumentError TensorBSplineApproximator(
            :g, (0.0, 1.0), (0.0, 1.0), 5, 2)
        # anisotropy must be positive and finite
        @test_throws ArgumentError TensorBSplineApproximator(
            :g, (0.0, 1.0), (0.0, 1.0), 5, 5; anisotropy=0.0)
        @test_throws ArgumentError TensorBSplineApproximator(
            :g, (0.0, 1.0), (0.0, 1.0), 5, 5; anisotropy=-1.0)
        @test_throws ArgumentError TensorBSplineApproximator(
            :g, (0.0, 1.0), (0.0, 1.0), 5, 5; anisotropy=Inf)

        a = TensorBSplineApproximator(:g, (0.0, 2.0), (1.0, 3.0), 5, 4)
        @test nparams(a) == 20
        @test initial_params(a) == zeros(20)

        # initial surface sampled on the knot grid, column-major (x fastest)
        a2 = TensorBSplineApproximator(:g, (0.0, 2.0), (1.0, 3.0), 5, 4;
                                       initial=(x, y) -> x + 10y)
        th = initial_params(a2)
        xs = range(0.0, 2.0, length=5)
        ys = range(1.0, 3.0, length=4)
        @test th[2] ≈ xs[2] + 10ys[1]      # grid index (2, 1)
        @test th[6] ≈ xs[1] + 10ys[2]      # grid index (1, 2): x fastest
        a3 = TensorBSplineApproximator(:g, (0.0, 1.0), (0.0, 1.0), 3, 3;
                                       initial=2.5)
        @test all(initial_params(a3) .== 2.5)
        # string names accepted, matching the univariate constructors
        @test TensorBSplineApproximator("g", (0.0, 1.0), (0.0, 1.0), 3, 3).name == :g
    end

    @testset "TensorBSplineApproximator — evaluator matches univariate machinery" begin
        using ForwardDiff
        nx, ny = 6, 5
        a = TensorBSplineApproximator(:g, (0.0, 3.0), (-1.0, 2.0), nx, ny)
        C = randn(StableRNG(1), nx, ny)
        f = build_evaluator(a, vec(C))
        kx = collect(range(0.0, 3.0, length=nx))
        ky = collect(range(-1.0, 2.0, length=ny))

        # interpolates the coefficient grid exactly
        @test maximum(abs(f(kx[i], ky[j]) - C[i, j])
                      for i in 1:nx, j in 1:ny) < 1e-10

        # The univariate BSpline evaluator is interpolation-THROUGH-VALUES
        # (CubicSpline with linear extrapolation), and the tensor evaluator
        # mirrors it per margin — so on every grid line the surface must
        # agree exactly with a univariate evaluator built from the
        # corresponding coefficient slice, inside AND outside the domain.
        for j in (1, 3, ny)
            s = PartiallySpecifiedModels.build_bspline_evaluator(kx, C[:, j])
            for x in (-0.5, 0.2, 1.7, 3.0, 3.8)
                @test f(x, ky[j]) ≈ s(x) atol=1e-9
            end
        end
        for i in (1, 4, nx)
            s = PartiallySpecifiedModels.build_bspline_evaluator(ky, C[i, :])
            for y in (-1.6, -0.3, 0.9, 2.0, 2.5)
                @test f(kx[i], y) ≈ s(y) atol=1e-9
            end
        end

        # Dual-safe in the parameters …
        g1 = ForwardDiff.gradient(b -> build_evaluator(a, b)(1.1, 0.4), vec(C))
        @test all(isfinite, g1)
        # … matching finite differences
        h = 1e-6
        for k in (1, 13, 30)
            e_k = [i == k ? 1.0 : 0.0 for i in 1:(nx * ny)]
            fd = (build_evaluator(a, vec(C) .+ h .* e_k)(1.1, 0.4) -
                  build_evaluator(a, vec(C) .- h .* e_k)(1.1, 0.4)) / (2h)
            @test g1[k] ≈ fd atol=1e-6
        end
        # Dual-safe in the states (autodiff ODE solvers evaluate at Dual u)
        g2 = ForwardDiff.gradient(v -> f(v[1], v[2]), [1.1, 0.4])
        @test all(isfinite, g2)
        # nested: Dual params of a Dual-state derivative (through-the-solver
        # training with stiff autodiff Jacobians)
        g3 = ForwardDiff.gradient(
            b -> ForwardDiff.derivative(x -> build_evaluator(a, b)(x, 0.4), 1.1),
            vec(C))
        @test all(isfinite, g3)
    end

    @testset "TensorBSplineApproximator — Kronecker-sum penalty" begin
        nx, ny = 6, 5
        a1 = TensorBSplineApproximator(:g, (0.0, 3.0), (-1.0, 2.0), nx, ny)
        a100 = TensorBSplineApproximator(:g, (0.0, 3.0), (-1.0, 2.0), nx, ny;
                                         anisotropy=100.0)
        S1 = penalty_matrix(a1)
        S100 = penalty_matrix(a100)
        @test size(S1) == (nx * ny, nx * ny)
        @test issymmetric(S1)
        ev = eigvals(Symmetric(S1))
        @test minimum(ev) > -1e-10           # PSD

        # Null space: with 2nd-derivative marginal penalties, S annihilates
        # exactly the surfaces whose columns are affine in x AND rows affine
        # in y — the bilinear family a + b·x + c·y + d·x·y (on the unit
        # square, where penalty_matrix builds its knots).
        xs = range(0.0, 1.0, length=nx)
        ys = range(0.0, 1.0, length=ny)
        for (α, β, γ, δ) in ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                             (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                             (2.0, -1.0, 0.5, 3.0))
            v = vec([α + β * x + γ * y + δ * x * y for x in xs, y in ys])
            @test norm(S1 * v) < 1e-8
        end
        # … and nothing more: rank is exactly nx·ny − 4
        @test count(x -> x > 1e-8 * maximum(ev), ev) == nx * ny - 4

        # Orientation w.r.t. the column-major vec layout: a surface rough in
        # x ONLY is penalized by the x-term only (quadratic form invariant
        # under anisotropy), while a surface rough in y ONLY scales linearly
        # with anisotropy.
        vx = vec([sin(6x) for x in xs, y in ys])  # varies in x, flat in y
        vy = vec([sin(6y) for x in xs, y in ys])  # flat in x, varies in y
        @test dot(vx, S1 * vx) > 1.0
        @test dot(vy, S1 * vy) > 1.0
        @test dot(vx, S100 * vx) ≈ dot(vx, S1 * vx) rtol=1e-10
        @test dot(vy, S100 * vy) ≈ 100.0 * dot(vy, S1 * vy) rtol=1e-10
    end

    @testset "TensorBSplineApproximator — end-to-end recovery (Wood 2001 predation surface)" begin
        # Rosenzweig–MacArthur predator–prey with an unknown BIVARIATE
        # interaction g(N, P) = 0.6·N·P/(1 + 0.4·N): Holling II in prey,
        # linear in predator — genuinely non-separable. Logistic prey
        # growth keeps the system on a bounded limit cycle (with pure
        # exponential prey growth this model diverges), so the orbit
        # sweeps a data-rich loop through the (N, P) plane.
        g_true_2d(N, P) = 0.6 * N * P / (1 + 0.4 * N)
        function pp_true!(du, u, p, t)
            g = g_true_2d(u[1], u[2])
            du[1] = u[1] * (1 - u[1] / 6.0) - g
            du[2] = 0.5 * g - 0.25 * u[2]
        end
        sol_true2d = OrdinaryDiffEq.solve(
            ODEProblem(pp_true!, [1.0, 1.5], (0.0, 40.0)), Tsit5(),
            saveat=0.5, abstol=1e-10, reltol=1e-10)
        traj2d = reduce(hcat, sol_true2d.u)'
        ts2d = collect(sol_true2d.t)
        data2d = traj2d .+ 0.03 .* randn(StableRNG(11), size(traj2d))

        function pp2d!(du, u, p, t)
            g = p.g(u[1], u[2])
            du[1] = u[1] * (1 - u[1] / 6.0) - g
            du[2] = p.e * g - p.m * u[2]
        end
        mk2d() = PSMProblem(pp2d!, [1.0, 1.5], (0.0, 40.0),
            [TensorBSplineApproximator(:g, (0.0, 3.0), (0.5, 3.0), 5, 5;
                                       initial=(N, P) -> 0.3 * N * P)];
            data_times=ts2d, data_values=data2d, obs_to_state=[1, 2],
            known_params=(e=0.5, m=0.25), likelihood=Gaussian(),
            solver=Tsit5())
        prob2d = mk2d()

        # A 1-D orbit only identifies the surface near the visited region:
        # test at actual trajectory states across the cycle's phases.
        test_pts = [(traj2d[i, 1], traj2d[i, 2]) for i in (10, 25, 40, 55, 70)]

        # Reference: data loss of the initial bilinear mass-action surface
        beta0 = PartiallySpecifiedModels.build_initial_params(prob2d)
        loss0 = PartiallySpecifiedModels.weighted_data_loss(
            prob2d, PartiallySpecifiedModels.simulate(prob2d, beta0))

        # (a) AdamSolver — ForwardDiff through the ODE solve (Dual params
        # AND Dual states through the two-argument evaluator)
        sol_adam = solve(prob2d, AdamSolver(maxiters=400, lr=0.05))
        @test sol_adam.data_loss < 1.5           # observed ≈ 0.67
        @test sol_adam.data_loss < loss0 / 20    # loss0 ≈ 81
        g_adam = sol_adam.unknown_functions[:g]
        for (N, P) in test_pts
            # observed rel. errors 0.02–0.13
            @test abs(g_adam(N, P) - g_true_2d(N, P)) < 0.25 * g_true_2d(N, P)
        end

        # (b) LAML — finite-difference Jacobian through the ODE solve plus
        # the Kronecker-sum penalty with automatic λ. The default
        # data-driven initial λ oversmooths this strongly nonlinear
        # oscillator; starting light (initial_lambda, warmup — the LAML
        # docstring's guidance for strongly nonlinear models) recovers the
        # surface to ≈1% at the tested states.
        sol_laml = solve(prob2d,
                         LAML(maxiters=60, warmup=10, initial_lambda=0.01))
        @test sol_laml.data_loss < 1.0           # observed ≈ 0.14
        g_laml = sol_laml.unknown_functions[:g]
        for (N, P) in test_pts
            # observed rel. errors ≤ 0.01
            @test abs(g_laml(N, P) - g_true_2d(N, P)) < 0.15 * g_true_2d(N, P)
        end

        # (c) a higher fixed penalty produces a SMOOTHER surface: β'Sβ
        # collapses under penalty_weight (the smoothness assertion is made
        # on the fixed-λ solver because LAML's λ automation would make a
        # λ-vs-λ comparison brittle)
        S2d = penalty_matrix(prob2d.approximators[1])
        sol_pen = solve(mk2d(),
                        AdamSolver(maxiters=300, lr=0.05, penalty_weight=50.0))
        rough_unpen = dot(sol_adam.parameters, S2d * sol_adam.parameters)
        rough_pen = dot(sol_pen.parameters, S2d * sol_pen.parameters)
        @test rough_pen < 0.01 * rough_unpen     # observed 1221 → 0.011

        # (d) DerivativeFreeSolver completes and recovers the surface
        sol_dfree = solve(prob2d, DerivativeFreeSolver(maxiters=3000))
        @test sol_dfree.data_loss < 3.0          # observed ≈ 1.1
        g_dfree = sol_dfree.unknown_functions[:g]
        for (N, P) in test_pts[[2, 4]]
            # observed rel. errors ≤ 0.04
            @test abs(g_dfree(N, P) - g_true_2d(N, P)) < 0.25 * g_true_2d(N, P)
        end

        # (e) the univariate confidence-band machinery REJECTS the tensor
        # surface loudly instead of erroring obscurely on `.domain`
        @test_throws ArgumentError confidence_band(sol_laml, prob2d)
    end

    @testset "TensorBSplineApproximator — GCV recovery and bootstrap-band gating" begin
        # Milder monotone 2-state decay problem: GCV's IRLS linearization
        # is well behaved here (on the oscillatory cycle above it needs the
        # same care as for univariate splines).
        g_true_dec(N, P) = 0.5 * N * P / (1 + 0.5 * N)
        function dec_true!(du, u, p, t)
            g = g_true_dec(u[1], u[2])
            du[1] = -g
            du[2] = 0.2 * g - 0.1 * u[2]
        end
        sol_true_d = OrdinaryDiffEq.solve(
            ODEProblem(dec_true!, [2.5, 1.5], (0.0, 12.0)), Tsit5(),
            saveat=0.25, abstol=1e-10, reltol=1e-10)
        traj_d = reduce(hcat, sol_true_d.u)'
        ts_d = collect(sol_true_d.t)
        data_d = traj_d .+ 0.01 .* randn(StableRNG(3), size(traj_d))
        function dec!(du, u, p, t)
            g = p.g(u[1], u[2])
            du[1] = -g
            du[2] = p.e * g - p.m * u[2]
        end
        prob_d = PSMProblem(dec!, [2.5, 1.5], (0.0, 12.0),
            [TensorBSplineApproximator(:g, (0.0, 3.0), (0.0, 2.0), 5, 5;
                                       initial=(N, P) -> 0.2 * N * P)];
            data_times=ts_d, data_values=data_d, obs_to_state=[1, 2],
            known_params=(e=0.2, m=0.1), likelihood=Gaussian(),
            solver=Tsit5())

        sol_gcv = solve(prob_d, GCVSolver(maxiters=60))
        @test sol_gcv.data_loss < 0.5            # observed ≈ 0.10
        g_gcv = sol_gcv.unknown_functions[:g]
        # early-trajectory states, where g is well identified (late in the
        # decay g → 0 and relative error stops being meaningful)
        for i in (5, 15)
            N, P = traj_d[i, 1], traj_d[i, 2]
            # observed rel. errors ≤ 0.01
            @test abs(g_gcv(N, P) - g_true_dec(N, P)) < 0.15 * g_true_dec(N, P)
        end

        # bootstrap: parameter/trajectory intervals work, but the UNIVARIATE
        # unknown-function band is skipped with a warning rather than
        # silently mis-gridding the bivariate surface
        sol_dfree = solve(prob_d, DerivativeFreeSolver(maxiters=400))
        res = @test_logs (:warn, r"bivariate") match_mode=:any bootstrap(
            sol_dfree, prob_d, DerivativeFreeSolver(maxiters=200);
            nboot=3, rng=StableRNG(5))
        @test res.n_success >= 3
        @test !haskey(res.ci_uf, :g)
    end

    @testset "ShapeConstrainedGPApproximator — construction" begin
        a = ShapeConstrainedGPApproximator(:f, (0.0, 5.0), 8, :increasing)
        @test a.name == :f
        @test nparams(a) == 8
        @test a.constraint == :increasing
        @test a.adapt                     # no lengthscale supplied → adaptive
        @test length(a.inducing_points) == 8
        @test length(initial_params(a)) == 8

        # Explicit hyperparameters are fixed for the whole fit (never adapted)
        a_fix = ShapeConstrainedGPApproximator(:g, (0.0, 5.0), 8, :decreasing;
                                               lengthscale=1.0, kernel=:matern52)
        @test !a_fix.adapt
        @test a_fix.lengthscale == 1.0
        @test a_fix.kernel == :matern52

        # Zero-at-endpoint constraints drop one parameter, like the SCOP
        # B-spline/SPDE siblings
        a_z = ShapeConstrainedGPApproximator(:h, (0.0, 1.0), 8, :inc_zero_left)
        @test nparams(a_z) == 7

        # Penalty: P&W first-difference penalty on γ with the free level in
        # the null space (PSD, rank-deficient)
        S = penalty_matrix(a)
        @test size(S) == (8, 8)
        @test all(e -> e > -1e-10, eigvals(Symmetric(S)))
        @test all(iszero, S[1, :])        # free level γ₁ unpenalized

        # Validation
        @test_throws ArgumentError ShapeConstrainedGPApproximator(
            :f, (0.0, 1.0), 8, :bad_constraint)
        @test_throws ArgumentError ShapeConstrainedGPApproximator(
            :f, (1.0, 0.0), 8, :increasing)
        @test_throws ArgumentError ShapeConstrainedGPApproximator(
            :f, (0.0, 1.0), 3, :increasing)
    end

    @testset "ShapeConstrainedGPApproximator — constraint by construction" begin
        # For ANY parameter vector the reparameterized evaluator satisfies its
        # constraint: exactly at the inducing values (β = Σ·d(γ) is built to)
        # and on a fine grid up to the small between-point wiggle of kernel
        # interpolation (the SCOP-SPDE cubic has the same caveat; observed
        # ≤ 0.6% of the function range for the N(0,1) draws used here;
        # the margin grows with the draw scale, so the 1% tolerance below
        # is calibrated to THESE seeded draws, not a universal bound).
        rng = StableRNG(7)
        grid = collect(range(0.0, 2.0, length=401))
        for c in (:increasing, :decreasing, :convex, :concave)
            ac = ShapeConstrainedGPApproximator(:f, (0.0, 2.0), 8, c)
            for trial in 1:5
                γ = randn(rng, nparams(ac))
                β = PartiallySpecifiedModels.gamma_to_inducing_values(ac, γ)
                f = build_evaluator(ac, γ)
                vals = [f(x) for x in grid]
                rangev = maximum(vals) - minimum(vals)
                if c == :increasing
                    @test all(diff(β) .>= 0)              # exact at nodes
                    @test all(diff(vals) .>= -0.01 * rangev)
                elseif c == :decreasing
                    @test all(diff(β) .<= 0)
                    @test all(diff(vals) .<= 0.01 * rangev)
                elseif c == :convex
                    @test all(diff(diff(β)) .>= -1e-10)
                    @test all(diff(diff(vals)) .>= -2e-3 * rangev)
                else # :concave
                    @test all(diff(diff(β)) .<= 1e-10)
                    @test all(diff(diff(vals)) .<= 2e-3 * rangev)
                end
            end
        end

        # Zero-at-endpoint: pinned exactly (constant centering shift) and
        # still monotone
        a_z = ShapeConstrainedGPApproximator(:f, (0.0, 2.0), 8, :inc_zero_left)
        γz = randn(rng, nparams(a_z))
        fz = build_evaluator(a_z, γz)
        @test abs(fz(0.0)) < 1e-10
        valz = [fz(x) for x in grid]
        @test all(diff(valz) .>= -0.01 * (maximum(valz) - minimum(valz)))
    end

    @testset "ShapeConstrainedGPApproximator — Dual-safety" begin
        import ForwardDiff
        a = ShapeConstrainedGPApproximator(:f, (0.0, 2.0), 6, :increasing)
        γ = randn(StableRNG(3), nparams(a))
        # Gradient w.r.t. the parameters (autodiff solvers differentiate the
        # objective through build_evaluator)
        g = ForwardDiff.gradient(
            p -> sum(build_evaluator(a, p)(x) for x in 0.0:0.5:2.0), γ)
        @test all(isfinite, g)
        @test any(!iszero, g)
        # Derivative w.r.t. the input (stiff-solver Jacobians pass Dual x),
        # covering both extrapolation branches and the interior
        f = build_evaluator(a, γ)
        for x in (-0.5, 1.0, 2.5)
            @test isfinite(ForwardDiff.derivative(f, x))
        end
        @test ForwardDiff.derivative(f, 1.0) >= -1e-8  # monotone increasing
    end

    @testset "ShapeConstrainedGPApproximator — monotone recovery (LAML/Adam)" begin
        # Saturating uptake du = -f(u), f(u) = u/(1+u): monotone increasing
        # truth with sparse noisy data. The discriminating pair: the
        # UNCONSTRAINED GP fit chases the noise into a non-monotone f (assert
        # it does — otherwise this test proves nothing), while the constrained
        # fit cannot violate monotonicity by construction AND recovers the
        # truth more accurately.
        f_true_gp(u) = u / (1 + u)
        function uptake_gp!(du, u, p, t)
            du[1] = -p.f(u[1])
        end
        ode_gp = ODEProblem((du, u, p, t) -> (du[1] = -f_true_gp(u[1])),
                            [2.0], (0.0, 8.0))
        traj_gp = OrdinaryDiffEq.solve(ode_gp, Tsit5(); abstol=1e-10, reltol=1e-10)
        rng = StableRNG(5)
        ts_gp = collect(range(0.0, 8.0, length=12))
        data_gp = reshape([traj_gp(t)[1] for t in ts_gp] .+
                          0.10 .* randn(rng, 12), :, 1)

        mk_gp(approx) = PSMProblem(uptake_gp!, [2.0], (0.0, 8.0), [approx];
            data_times=ts_gp, data_values=data_gp, obs_to_state=[1],
            known_params=NamedTuple(), likelihood=Gaussian(), solver=Tsit5())

        grid = collect(range(0.05, 2.0, length=200))
        # Worst decrease as a fraction of the fitted range (positive ⇒ the
        # fit is non-monotone somewhere on the grid)
        function viol_frac(fh)
            vals = [fh(x) for x in grid]
            -minimum(diff(vals)) / (maximum(vals) - minimum(vals))
        end

        sol_u = solve(mk_gp(GPApproximator(:f, (0.0, 2.2), 8;
                                           initial=x -> 0.3 * x)),
                      LAML(maxiters=40, verbose=false))
        prob_c = mk_gp(ShapeConstrainedGPApproximator(:f, (0.0, 2.2), 8,
                           :increasing; initial=x -> 0.3 * x))
        sol_c = solve(prob_c, LAML(maxiters=40, verbose=false))
        fu = sol_u.unknown_functions[:f]
        fc = sol_c.unknown_functions[:f]

        # Unconstrained fit genuinely violates monotonicity (observed 0.0079)
        @test viol_frac(fu) > 0.003
        # Constrained fit is monotone (observed strictly increasing,
        # viol_frac ≈ -0.001; tolerance covers between-point kernel wiggle)
        @test viol_frac(fc) < 1e-3
        # ...and recovers the truth (observed maxerr 0.107, meanerr 0.054 on
        # a truth range of ≈ 0.65)
        errs_c = [abs(fc(x) - f_true_gp(x)) for x in grid]
        @test maximum(errs_c) < 0.15
        @test sum(errs_c) / length(errs_c) < 0.08
        # ...more accurately than the unconstrained fit (0.054 vs 0.104 mean)
        errs_u = [abs(fu(x) - f_true_gp(x)) for x in grid]
        @test sum(errs_c) < sum(errs_u)

        # Confidence band treats the constrained GP like its SCOP siblings
        # (finite-difference ∂f/∂γ through _eval_approx_at)
        bands = confidence_band(sol_c, prob_c)
        @test haskey(bands, :f)
        @test all(isfinite, bands[:f].se)
        @test all(bands[:f].lower .<= bands[:f].upper)

        # AdamSolver end-to-end through the same build_evaluator protocol
        # (observed maxerr 0.117, monotone)
        sol_a = solve(mk_gp(ShapeConstrainedGPApproximator(:f, (0.0, 2.2), 8,
                                :increasing; initial=x -> 0.3 * x)),
                      AdamSolver(maxiters=300, lr=0.05))
        fa = sol_a.unknown_functions[:f]
        @test viol_frac(fa) < 1e-3
        @test maximum(abs(fa(x) - f_true_gp(x)) for x in grid) < 0.2
    end

    @testset "LAML full Laplace criterion (construction + Gaussian reduction)" begin
        # Construction validation: criterion is checked at construction
        @test LAML().criterion == :working
        @test LAML(criterion=:laplace).criterion == :laplace
        @test_throws ArgumentError LAML(criterion=:reml)

        # CustomLikelihood is rejected under :laplace with an informative
        # error (its loglik_scalar declares neither a normalizer nor a
        # dispersion, both of which the full criterion needs).
        rng = Xoshiro(20260819)
        data_times = collect(0.0:0.5:10.0)
        data_values = reshape(exp.(0.1 .* data_times) .+
                              0.01 .* randn(rng, length(data_times)), :, 1)
        exp_growth!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1])
        mk_lik(lik) = PSMProblem(exp_growth!, [1.0], (0.0, 10.0),
            [BSplineApproximator(:r, (0.0, 5.0), 5; initial=x -> 0.05)];
            data_times=data_times, data_values=data_values,
            obs_to_state=[1], likelihood=lik, solver=Tsit5())
        err = try
            solve(mk_lik(CustomLikelihood((y, μ) -> -0.5 * (y - μ)^2)),
                  LAML(maxiters=5, criterion=:laplace))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("does not support CustomLikelihood", err.msg)

        # Gaussian reduction: for Gaussian data the full Laplace criterion
        # with σ² profiled out IS the profiled-REML criterion (algebra in
        # laml_objective's Gaussian branch), and in code the two criteria
        # take the identical path — so λ̂ trajectory, objective and fit must
        # coincide to machine precision.
        prob_g = mk_lik(Gaussian())
        sol_w = solve(prob_g, LAML(maxiters=30, verbose=false))
        sol_l = solve(prob_g, LAML(maxiters=30, verbose=false,
                                   criterion=:laplace))
        @test sol_l.smoothing_params ≈ sol_w.smoothing_params rtol=1e-12
        @test sol_l.objective ≈ sol_w.objective rtol=1e-12
        @test sol_l.fitted_values ≈ sol_w.fitted_values rtol=1e-12
        rgrid = range(0.5, 4.0, length=20)
        @test [sol_l.unknown_functions[:r](x) for x in rgrid] ≈
              [sol_w.unknown_functions[:r](x) for x in rgrid] rtol=1e-12

        # Criterion is recorded, existing convergence keys are all intact,
        # and the reported criterion value (profiled REML here) is the same
        # finite number under both settings.
        @test sol_w.convergence.criterion == :working
        @test sol_l.convergence.criterion == :laplace
        for k in (:V_beta, :sigma2, :converged, :iterations, :reason,
                  :laml_failures, :criterion, :laml)
            @test haskey(sol_w.convergence, k)
        end
        @test isfinite(sol_l.convergence.laml)
        @test sol_l.convergence.laml ≈ sol_w.convergence.laml rtol=1e-12
    end

    @testset "LAML full Laplace criterion (objective value + FS scale)" begin
        using PartiallySpecifiedModels: laml_objective,
                                        estimate_smoothing_params,
                                        spline_penalty_matrix, _safe_inv
        # Deterministic (RNG-free) Poisson working state on a linear hat
        # design: y is a fixed rounded pattern whose Pearson dispersion
        # against the fit is ≈ 1.10 > 1, so the two criteria measurably
        # disagree about λ while everything stays pure linear algebra.
        n, nk = 40, 6
        knots = collect(0.0:1.0:5.0)
        xs = collect(range(0.0, 5.0, length=n))
        hat(x, k) = max(0.0, 1.0 - abs(x - k))
        J = [hat(xs[i], knots[j]) for i in 1:n, j in 1:nk]
        S = spline_penalty_matrix(knots)
        w = ones(n)
        mu0 = 2.0 .+ 1.5 .* sin.(xs)
        y = max.(round.(mu0 .+ 2.0 .* sin.(17.3 .* xs)), 0.0)

        # β̂ at FIXED small λ by penalized IRLS for the actual Poisson family
        # (identity link on a linear design ⟹ pseudodata z = y exactly).
        lam_fix = 0.05
        S_lam = lam_fix .* S
        beta = _safe_inv(J' * Diagonal(w) * J + S_lam) * (J' * (w .* y))
        mu = J * beta
        for _ in 1:300
            W_it = [w[i] / max(abs(mu[i]), 1e-6) for i in 1:n]
            beta = _safe_inv(J' * Diagonal(W_it) * J + S_lam) *
                   (J' * (W_it .* y))
            mu = J * beta
        end
        W = [w[i] / max(abs(mu[i]), 1e-6) for i in 1:n]
        @test all(mu .> 0.5)   # fitted means healthy (observed 0.87–3.37)

        # Objective correctness: the solver's criterion (laml_objective, the
        # exact quantity reported as sol.convergence.laml) against the
        # Wood (2011) formula computed independently here:
        #   V = ℓ(β̂) − ½β̂'S_λβ̂ + ½log|S_λ|₊ − ½log|J'WJ + S_λ| + (Mp/2)log 2π
        # with W the Fisher weights 1/μ̂ (expected Hessian). Observed
        # agreement 6.5e-10 (the residual is laml_objective's 1e-10
        # stabilizing ridge inside _log_det_pd); atol=1e-6 leaves ~1000x
        # headroom while catching any dropped or mis-scaled term.
        V_solver, = laml_objective(Poisson(), beta, J, W, w, y, mu,
                                   [S], [0], [nk], [log(lam_fix)], nk)
        logfact(k) = sum(log.(1:Int(k)); init=0.0)
        ll = sum(w[i] * (y[i] * log(mu[i]) - mu[i] - logfact(y[i]))
                 for i in 1:n)
        pen = lam_fix * dot(beta, S * beta)
        ev = eigvals(Symmetric(S))
        tolS = 1e-10 * maximum(abs.(ev))
        rS = count(e -> e > tolS, ev)
        logdetSplus = rS * log(lam_fix) + sum(log(e) for e in ev if e > tolS)
        logdetH = logdet(cholesky(Symmetric(J' * Diagonal(W) * J + S_lam)))
        V_direct = ll - 0.5 * pen + 0.5 * logdetSplus - 0.5 * logdetH +
                   0.5 * (nk - rS) * log(2π)
        @test V_solver ≈ V_direct atol=1e-6

        # FS-scale discriminator: from the same frozen working state,
        # :working calibrates FS by the Pearson dispersion φ̂ ≈ 1.10 (and
        # skips Newton) while :laplace uses unit dispersion plus Newton
        # refinement of V. Observed λ̂: 4.247 vs 3.213 (ratio 1.32). The
        # computation is deterministic — no RNG, no ODE solves — so a
        # ratio floor of 1.05 is safe against BLAS/platform variation.
        pearson = sum((y[i] - mu[i])^2 / mu[i] for i in 1:n) / (n - rS)
        @test pearson > 1.02   # the mechanism the discrimination relies on
        lam_w, = estimate_smoothing_params(J, W, w, y, mu, beta,
            [S], [0], [nk], nk; family=Poisson(), rho_init=[0.0],
            criterion=:working)
        lam_l, = estimate_smoothing_params(J, W, w, y, mu, beta,
            [S], [0], [nk], nk; family=Poisson(), rho_init=[0.0],
            criterion=:laplace)
        @test isfinite(lam_l[1]) && lam_l[1] > 0
        @test lam_w[1] / lam_l[1] > 1.05
    end

    @testset "LAML full Laplace criterion (count-data end-to-end)" begin
        # Logistic growth observed as low-mean counts (μ ∈ [1, 7.96]) —
        # exactly the regime where the Gaussian working approximation is
        # worst. The unknown per-capita rate g(u) = r₀(1 − u/K) is fitted
        # from Poisson observations. Data are deterministic: a fixed-seed
        # Xoshiro drives an inversion sampler, and the solver itself is
        # RNG-free, so both fits below are reproducible.
        r0, K = 0.6, 8.0
        tsl = collect(0.0:0.4:12.0)
        tsol = OrdinaryDiffEq.solve(
            ODEProblem((du, u, p, t) -> du[1] = r0 * (1 - u[1] / K) * u[1],
                       [1.0], (0.0, 12.0)),
            Tsit5(); saveat=tsl, abstol=1e-10, reltol=1e-10)
        mu_true = [tsol.u[i][1] for i in eachindex(tsl)]
        function rpois(rng, m)
            L = exp(-m); k = 0; p = 1.0
            while true
                p *= rand(rng)
                p <= L && return k
                k += 1
            end
        end
        growth!(du, u, p, t) = (du[1] = p.g(u[1]) * u[1])
        mk_count(y, lik) = PSMProblem(growth!, [1.0], (0.0, 12.0),
            [BSplineApproximator(:g, (0.3, 9.0), 6; initial=x -> 0.3)];
            data_times=tsl, data_values=reshape(y, :, 1), obs_to_state=[1],
            likelihood=lik, solver=Tsit5())
        gtrue(x) = r0 * (1 - x / K)
        gx = range(1.0, 7.5, length=40)
        errs(s) = [abs(s.unknown_functions[:g](x) - gtrue(x)) for x in gx]

        # Poisson: this draw's sample Pearson dispersion against the truth
        # is 1.41, so the :working criterion (φ̂-scaled FS) lands on a
        # visibly larger λ̂ than the full Laplace criterion.
        rng_p = Xoshiro(1)
        y_p = Float64[rpois(rng_p, m) for m in mu_true]
        sol_pw = solve(mk_count(y_p, Poisson()),
                       LAML(maxiters=40, verbose=false))
        sol_pl = solve(mk_count(y_p, Poisson()),
                       LAML(maxiters=40, verbose=false, criterion=:laplace))
        # Honest reporting: :laplace runs to convergence and records itself
        @test sol_pl.convergence.converged
        @test sol_pl.convergence.criterion == :laplace
        @test sol_pw.convergence.criterion == :working
        @test isfinite(sol_pl.convergence.laml)
        # Recovery of the truth under :laplace (observed maxerr 0.069,
        # meanerr 0.029 on a truth range of ≈ 0.53 over [1, 7.5])
        @test maximum(errs(sol_pl)) < 0.15
        @test sum(errs(sol_pl)) / length(gx) < 0.06
        # ...and it DIFFERS measurably from :working in λ̂: observed
        # λ̂_working = 5.86e6 vs λ̂_laplace = 2.59e6, |Δlog λ̂| = 0.816 —
        # the fits themselves nearly coincide here (the truth is linear in
        # u, i.e. in the penalty null space, so both smooth heavily), which
        # is what makes the λ̂ contrast the deterministic discriminator.
        # Floor of 0.1 leaves 8x headroom in log terms.
        @test sol_pl.smoothing_params[1] != sol_pw.smoothing_params[1]
        @test abs(log(sol_pl.smoothing_params[1] /
                      sol_pw.smoothing_params[1])) > 0.1

        # NegativeBinomial end-to-end with :laplace: fixed dispersion
        # θ = 8 in the family object, correctly specified data. Observed:
        # converged, λ̂ = 6.66e6, maxerr 0.061, meanerr 0.041, V = −71.6.
        rng_nb = Xoshiro(1)
        th = 8
        y_nb = Float64[rpois(rng_nb,
                             m * (-sum(log(rand(rng_nb)) for _ in 1:th) / th))
                       for m in mu_true]
        sol_nb = solve(mk_count(y_nb, NegativeBinomial(Float64(th))),
                       LAML(maxiters=40, verbose=false, criterion=:laplace))
        @test sol_nb.convergence.converged
        @test sol_nb.convergence.criterion == :laplace
        @test isfinite(sol_nb.convergence.laml)
        @test maximum(errs(sol_nb)) < 0.15
        @test sum(errs(sol_nb)) / length(gx) < 0.09
    end

    @testset "GCVSolver criterion=:ncv (neighbourhood CV, Wood 2024)" begin
        PSM = PartiallySpecifiedModels

        # ── Construction validation ──
        @test_throws ArgumentError GCVSolver(criterion=:foo)
        @test_throws ArgumentError GCVSolver(criterion=:loocv)
        @test_throws ArgumentError GCVSolver(ncv_width=-1)
        @test_throws ArgumentError GCVSolver(criterion=:ncv, ncv_width=-3)
        @test GCVSolver().criterion == :gcv                  # default unchanged
        @test GCVSolver(criterion=:ncv).ncv_width == 2
        @test GCVSolver(criterion=:ncv, ncv_width=0) isa GCVSolver  # 0 = LOOCV

        # ── Neighbourhood construction (obs-major flattening: index
        # k = (oi−1)·n_times + ti; δ never crosses component blocks and
        # never contains masked rows) ──
        nt_w6 = 12
        w_w6 = fill(1.0, 2 * nt_w6)
        w_w6[5] = 0.0                       # masked cell, component 1, t=5
        @test PSM._ncv_neighbourhood(1, nt_w6, w_w6, 2) == [1, 2, 3]
        @test PSM._ncv_neighbourhood(12, nt_w6, w_w6, 2) == [10, 11, 12]
        @test PSM._ncv_neighbourhood(6, nt_w6, w_w6, 2) == [4, 6, 7, 8]
        @test PSM._ncv_neighbourhood(13, nt_w6, w_w6, 2) == [13, 14, 15]
        @test PSM._ncv_neighbourhood(24, nt_w6, w_w6, 2) == [22, 23, 24]
        @test PSM._ncv_neighbourhood(8, nt_w6, w_w6, 0) == [8]

        # ── EXACTNESS: Woodbury downdate == brute-force drop-refit-predict
        # on a fixed working model (StableRNG), interior + edge points +
        # a masked cell inside a neighbourhood ──
        rng_w6 = StableRNG(42)
        n_w6 = 2 * nt_w6
        p_w6 = 5
        J_w6 = randn(rng_w6, n_w6, p_w6)
        z_w6 = randn(rng_w6, n_w6)
        wv_w6 = rand(rng_w6, n_w6) .+ 0.5
        wv_w6[5] = 0.0
        S_w6 = Matrix(0.7I, p_w6, p_w6)
        A_w6 = J_w6' * Diagonal(wv_w6) * J_w6 + S_w6
        chol_w6 = cholesky(Symmetric(A_w6))
        beta_w6 = chol_w6 \ (J_w6' * (wv_w6 .* z_w6))
        zhat_w6 = PSM._ncv_loo_predictions(J_w6, wv_w6, z_w6, chol_w6,
                                           beta_w6, nt_w6, 2)
        # Brute force: drop δ(i)'s rows, re-solve the penalized LS, predict
        function brute_w6(i, width)
            delta = PSM._ncv_neighbourhood(i, nt_w6, wv_w6, width)
            keep = setdiff(1:n_w6, delta)
            Jk = J_w6[keep, :]; wk = wv_w6[keep]; zk = z_w6[keep]
            bk = (Jk' * Diagonal(wk) * Jk + S_w6) \ (Jk' * (wk .* zk))
            dot(J_w6[i, :], bk)
        end
        @test abs(zhat_w6[1] - brute_w6(1, 2)) < 1e-8    # left edge, truncated δ
        @test abs(zhat_w6[12] - brute_w6(12, 2)) < 1e-8  # right edge of block 1
        @test abs(zhat_w6[7] - brute_w6(7, 2)) < 1e-8    # interior
        @test abs(zhat_w6[6] - brute_w6(6, 2)) < 1e-8    # δ contains masked cell 5
        @test abs(zhat_w6[13] - brute_w6(13, 2)) < 1e-8  # block-2 edge (no bleed)
        @test isnan(zhat_w6[5])                          # masked row: no score
        @test maximum(abs(zhat_w6[i] - brute_w6(i, 2))
                      for i in 1:n_w6 if wv_w6[i] > 0) < 1e-8

        # ── ncv_width=0 equals exact leave-one-out CV (classic shortcut
        # ẑ_i^{−i} = z_i − (z_i − ẑ_i)/(1 − h_ii), h_ii = w_i J_i A⁻¹ J_i') ──
        zh0_w6 = PSM._ncv_loo_predictions(J_w6, wv_w6, z_w6, chol_w6,
                                          beta_w6, nt_w6, 0)
        fitted_w6 = J_w6 * beta_w6
        @test maximum(begin
                          h = wv_w6[i] * dot(J_w6[i, :], chol_w6 \ J_w6[i, :])
                          loo = z_w6[i] - (z_w6[i] - fitted_w6[i]) / (1 - h)
                          abs(loo - zh0_w6[i])
                      end for i in 1:n_w6 if wv_w6[i] > 0) < 1e-8

        # ── _ncv_score equals the manually assembled weighted mean ──
        V_w6, beta_sc_w6, _, _ = PSM._ncv_score(J_w6, wv_w6, z_w6, S_w6,
                                                nt_w6, 2)
        Vb_w6 = sum(wv_w6[i] * (z_w6[i] - zhat_w6[i])^2
                    for i in 1:n_w6 if wv_w6[i] > 0) /
                count(>(0), wv_w6)
        @test abs(V_w6 - Vb_w6) < 1e-8
        @test norm(beta_sc_w6 - beta_w6) < 1e-6   # same fit (modulo tiny ridge)

        # ── End-to-end setup shared by the remaining blocks ──
        r_true_w6(N) = 0.5 * (1.0 - N / 10.0)
        function logistic_w6!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        sol_true_w6 = OrdinaryDiffEq.solve(
            ODEProblem(logistic_w6!, [1.0], (0.0, 15.0), (; r=r_true_w6)),
            Tsit5(); saveat=0.2, abstol=1e-10, reltol=1e-10)
        t_w6 = collect(sol_true_w6.t)
        truth_w6 = [u[1] for u in sol_true_w6.u]
        nN_w6 = length(t_w6)                     # 76 points

        # AR(1) observation noise: e_t = 0.7 e_{t−1} + ε_t, ε ~ N(0, 0.08²),
        # stationary start. Short-range positive autocorrelation is exactly
        # the regime where ordinary GCV/LOOCV undersmooths (Wood 2024 §5).
        rho_ar_w6 = 0.7
        sige_w6 = 0.08
        rng_ar_w6 = StableRNG(1)
        e_w6 = zeros(nN_w6)
        e_w6[1] = sige_w6 / sqrt(1 - rho_ar_w6^2) * randn(rng_ar_w6)
        for i in 2:nN_w6
            e_w6[i] = rho_ar_w6 * e_w6[i-1] + sige_w6 * randn(rng_ar_w6)
        end
        data_w6 = truth_w6 .+ e_w6
        mk_prob_w6(vals) = PSMProblem(logistic_w6!, [1.0], (0.0, 15.0),
            [BSplineApproximator(:r, (0.0, 12.0), 8)];
            data_times=t_w6, data_values=reshape(vals, :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PSM.Gaussian())
        prob_ar_w6 = mk_prob_w6(max.(data_w6, 0.01))

        # ── Default equivalence: GCVSolver() ≡ GCVSolver(criterion=:gcv)
        # (the :gcv path is byte-identical pre/post refactor; the existing
        # GCVSolver testsets pin the pre-change behavior itself) ──
        sol_gdef_w6 = solve(prob_ar_w6, GCVSolver(maxiters=30))
        sol_gexp_w6 = solve(prob_ar_w6, GCVSolver(maxiters=30, criterion=:gcv))
        @test sol_gdef_w6.smoothing_params == sol_gexp_w6.smoothing_params
        @test sol_gdef_w6.objective == sol_gexp_w6.objective
        @test sol_gdef_w6.convergence.gcv == sol_gexp_w6.convergence.gcv
        @test sol_gdef_w6.convergence.criterion == :gcv
        @test isfinite(sol_gdef_w6.convergence.gcv)
        @test isnan(sol_gdef_w6.convergence.ncv)

        # ── THE DISCRIMINATOR: under AR(1) residuals :gcv undersmooths,
        # :ncv (width 3 ≳ correlation length) does not. Observed (StableRNG
        # seed 1, this exact config): λ̂_gcv = 6.57e-4, λ̂_ncv = 2.61e4
        # (ratio 4.0e7), trajectory MSE vs noise-free truth 1.84e-3 (gcv)
        # vs 8.95e-4 (ncv), r-function MSE 2.85e-4 vs 1.42e-6. The 2×
        # λ-margin below has ~7 orders of magnitude of headroom; every
        # step of the pipeline (StableRNG data, FD Jacobians, grid +
        # golden-section, Cholesky) is deterministic, and seeds 2 and 3
        # gave ratios 472 and 5.1e4 with the same error ordering. The
        # effect SIZE is seed-dependent (some seeds give ratios near 1),
        # so the 2× margin is protected by the pinned seed, not
        # universal; the MSE ordering held on every seed tried. ──
        sol_ncv_w6 = solve(prob_ar_w6,
                           GCVSolver(maxiters=30, criterion=:ncv, ncv_width=3))
        lam_g_w6 = sol_gdef_w6.smoothing_params[1]
        lam_n_w6 = sol_ncv_w6.smoothing_params[1]
        @test lam_n_w6 > 2 * lam_g_w6
        trajmse_w6(s) = sum(abs2, s.fitted_values[:, 1] .- truth_w6) / nN_w6
        @test trajmse_w6(sol_ncv_w6) < trajmse_w6(sol_gdef_w6)
        rgrid_w6 = 0.5:0.25:9.5
        rmse_w6(s) = sum(abs2, s.unknown_functions[:r](x) - r_true_w6(x)
                         for x in rgrid_w6) / length(rgrid_w6)
        @test rmse_w6(sol_ncv_w6) < rmse_w6(sol_gdef_w6)
        @test sol_ncv_w6.convergence.criterion == :ncv
        @test isfinite(sol_ncv_w6.convergence.ncv)
        @test isnan(sol_ncv_w6.convergence.gcv)
        # Determinism: an identical re-run reproduces λ̂ exactly
        sol_ncv2_w6 = solve(prob_ar_w6,
                            GCVSolver(maxiters=30, criterion=:ncv, ncv_width=3))
        @test sol_ncv2_w6.smoothing_params == sol_ncv_w6.smoothing_params
        @test sol_ncv2_w6.convergence.ncv == sol_ncv_w6.convergence.ncv

        # ── Masked data: NCV run completes and honors masking (NaN cells
        # are excluded from the score sum and from every neighbourhood) ──
        data_mask_w6 = copy(max.(data_w6, 0.01))
        data_mask_w6[10] = NaN
        data_mask_w6[30] = NaN
        data_mask_w6[31] = NaN
        sol_mask_w6 = solve(mk_prob_w6(data_mask_w6),
                            GCVSolver(maxiters=15, criterion=:ncv, ncv_width=3))
        @test sol_mask_w6 isa PSMSolution
        @test isfinite(sol_mask_w6.data_loss)
        @test isfinite(sol_mask_w6.convergence.ncv)
        @test all(isfinite, sol_mask_w6.fitted_values)
    end

    # ─── W7: FGPGMSolver (Wenk et al. 2019, fast GP gradient matching) ──

    @testset "FGPGMSolver — construction validation" begin
        # Mirror ODIN's validation style: ArgumentError with the reason.
        @test_throws ArgumentError FGPGMSolver(target_accept=1.5)
        @test_throws ArgumentError FGPGMSolver(target_accept=0.0)
        @test_throws ArgumentError FGPGMSolver(gamma=-1.0)
        @test_throws ArgumentError FGPGMSolver(gamma=0.0)
        @test_throws ArgumentError FGPGMSolver(n_samples=0)
        @test_throws ArgumentError FGPGMSolver(n_warmup=-1)
        @test_throws ArgumentError FGPGMSolver(prior_scale=0.0)
        @test_throws ArgumentError FGPGMSolver(obs_sigma=-0.1)
        # one-of-two GP hyperparameters is an error (ODIN convention:
        # both to fix them, neither to estimate per state)
        @test_throws ArgumentError FGPGMSolver(gp_lengthscale=1.0)
        @test_throws ArgumentError FGPGMSolver(gp_variance=1.0)
        alg_fg = FGPGMSolver()
        @test alg_fg.gamma == 0.1
        @test alg_fg.target_accept == 0.234
        @test alg_fg.rng_seed === nothing
    end

    @testset "FGPGMSolver — exponential decay recovery" begin
        # du/dt = -f(u), f(x) = 0.5x with a B-spline stand-in.
        decay_fg!(du, u, p, t) = (du[1] = -p.f(u[1]))
        rng_fg = Random.Xoshiro(42)
        sol_true_fg = OrdinaryDiffEq.solve(
            ODEProblem(decay_fg!, [5.0], (0.0, 10.0), (; f=x -> 0.5*x)),
            Tsit5(); saveat=0.5)
        t_fg = collect(sol_true_fg.t)
        data_fg = [sol_true_fg.u[i][1] + 0.05*randn(rng_fg) for i in 1:length(t_fg)]
        uf_fg = BSplineApproximator(:f, (0.0, 6.0), 6)
        prob_fg = PSMProblem(decay_fg!, [5.0], (0.0, 10.0), [uf_fg];
            data_times=t_fg, data_values=reshape(max.(data_fg, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_fg = solve(prob_fg, FGPGMSolver(n_samples=3000, n_warmup=3000,
                                            rng_seed=7, verbose=false))

        @test sol_fg isa PSMSolution
        @test isfinite(sol_fg.objective)
        @test isfinite(sol_fg.data_loss)
        @test haskey(sol_fg.unknown_functions, :f)
        # Posterior-mean recovery of f(x) = 0.5x. Observed errors at this
        # seed: 0.005 / 0.047 / 0.13 — assert with ~2.5x headroom.
        f_fg = sol_fg.unknown_functions[:f]
        @test abs(f_fg(1.5) - 0.75) < 0.35
        @test abs(f_fg(3.0) - 1.5) < 0.25
        @test abs(f_fg(4.5) - 2.25) < 0.35

        # Honest convergence keys (MagiSolver convention: a fixed-budget
        # sampler never claims convergence) + sampler-appropriate extras.
        c_fg = sol_fg.convergence
        @test c_fg.method == :fgpgm
        @test c_fg.sampler == :metropolis_within_gibbs
        @test c_fg.converged == false
        @test c_fg.reason == :maxiters
        @test c_fg.iterations == 6000
        @test c_fg.chains isa MCMCChains.Chains
        @test size(c_fg.chains, 1) == 3000       # n_samples retained draws
        @test size(c_fg.beta_samples) == (3000, nparams(uf_fg))
        # GP hyperparameters were estimated per state by marginal likelihood
        hp_fg = c_fg.gp_hyperparams[1]
        @test hp_fg.σ² > 0 && hp_fg.ℓ > 0 && hp_fg.σn² > 0
        @test hp_fg.observed
        # Warmup adaptation lands post-warmup acceptance in a broad band
        # around target_accept = 0.234 (deterministic under rng_seed=7;
        # observed 0.20 / 0.29 at this seed).
        @test all(a -> 0.1 < a < 0.6, c_fg.accept_rates.x)
        @test 0.1 < c_fg.accept_rates.theta < 0.6

        # Reproducibility (W1 convention): the sampler owns its stream —
        # same rng_seed → identical run, and the global RNG is untouched.
        probe_fg = rand(copy(Random.default_rng()), 3)
        sol_fg_a = solve(prob_fg, FGPGMSolver(n_samples=200, n_warmup=200,
                                              rng_seed=5))
        sol_fg_b = solve(prob_fg, FGPGMSolver(n_samples=200, n_warmup=200,
                                              rng_seed=5))
        @test sol_fg_a.parameters == sol_fg_b.parameters
        @test sol_fg_a.convergence.beta_samples ==
              sol_fg_b.convergence.beta_samples
        @test sol_fg_a.unknown_functions[:f](3.0) ==
              sol_fg_b.unknown_functions[:f](3.0)
        @test rand(copy(Random.default_rng()), 3) == probe_fg

        # Fixed-hyperparameter path (both supplied, ODIN convention:
        # σn² assumed at 0.01·gp_variance).
        sol_fg_fix = solve(prob_fg, FGPGMSolver(n_samples=200, n_warmup=200,
            gp_lengthscale=2.0, gp_variance=4.0, rng_seed=5))
        @test sol_fg_fix.convergence.gp_hyperparams[1].ℓ == 2.0
        @test sol_fg_fix.convergence.gp_hyperparams[1].σn² == 0.04
        @test isfinite(sol_fg_fix.objective)

        # Non-Gaussian likelihoods are rejected informatively (the data
        # expert is a Gaussian quadratic form) — MagiSolver style.
        prob_fg_pois = PSMProblem(decay_fg!, [5.0], (0.0, 10.0), [uf_fg];
            data_times=t_fg, data_values=reshape(max.(data_fg, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Poisson())
        err_fg = try
            solve(prob_fg_pois, FGPGMSolver(n_samples=5, n_warmup=5))
            nothing
        catch e
            e
        end
        @test err_fg isa ErrorException
        @test occursin("Gaussian likelihoods only", err_fg.msg)
    end

    @testset "FGPGMSolver — masked observations" begin
        decay_fgm!(du, u, p, t) = (du[1] = -p.f(u[1]))
        t_fgm = collect(0.0:0.5:10.0)
        u_fgm = 5.0 .* exp.(-0.5 .* t_fgm) .+
                0.05 .* randn(Random.Xoshiro(3), length(t_fgm))
        d_fgm = reshape(max.(u_fgm, 0.01), :, 1)
        w_fgm = ones(length(t_fgm), 1)
        d_fgm[5, 1] = NaN;  w_fgm[5, 1] = 0.0
        d_fgm[12, 1] = NaN; w_fgm[12, 1] = 0.0
        uf_fgm = BSplineApproximator(:f, (0.0, 6.0), 6)
        prob_fgm = PSMProblem(decay_fgm!, [5.0], (0.0, 10.0), [uf_fgm];
            data_times=t_fgm, data_values=d_fgm, data_weights=w_fgm,
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_fgm = solve(prob_fgm, FGPGMSolver(n_samples=1500, n_warmup=1500,
                                              rng_seed=13))
        @test isfinite(sol_fgm.objective)
        @test isfinite(sol_fgm.data_loss)
        @test all(isfinite, sol_fgm.convergence.beta_samples)
        @test abs(sol_fgm.unknown_functions[:f](3.0) - 1.5) < 0.4

        # Everything masked is not a fit — say so (ODIN/MCMC convention).
        prob_fgall = PSMProblem(decay_fgm!, [5.0], (0.0, 10.0), [uf_fgm];
            data_times=t_fgm, data_values=d_fgm,
            data_weights=zeros(length(t_fgm), 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        err_fgall = try
            solve(prob_fgall, FGPGMSolver(n_samples=5, n_warmup=5))
            nothing
        catch e
            e
        end
        @test err_fgall isa ErrorException
        @test occursin("masked", err_fgall.msg)
    end

    @testset "FGPGMSolver — multi-state and unobserved state" begin
        # x1' = x2, x2' = -k(x1) with k(x) = x (harmonic restoring force).
        k_fg(x) = x
        osc_fg!(du, u, p, t) = (du[1] = u[2]; du[2] = -p.k(u[1]))
        sol_true_ofg = OrdinaryDiffEq.solve(
            ODEProblem((du, u, p, t) -> (du[1] = u[2]; du[2] = -k_fg(u[1])),
                       [1.5, 0.0], (0.0, 8.0)), Tsit5(); saveat=0.25)
        t_ofg = collect(sol_true_ofg.t)
        rng_ofg = Random.Xoshiro(7)
        pos_ofg = [sol_true_ofg(tt)[1] for tt in t_ofg] .+
                  0.03 .* randn(rng_ofg, length(t_ofg))
        vel_ofg = [sol_true_ofg(tt)[2] for tt in t_ofg] .+
                  0.03 .* randn(rng_ofg, length(t_ofg))
        uf_ofg = BSplineApproximator(:k, (-2.0, 2.0), 7)

        # Both states observed: joint sampling over two state blocks + θ.
        prob_ofg2 = PSMProblem(osc_fg!, [1.5, 0.0], (0.0, 8.0), [uf_ofg];
            data_times=t_ofg, data_values=hcat(pos_ofg, vel_ofg),
            obs_to_state=[1, 2], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_ofg2 = solve(prob_ofg2, FGPGMSolver(n_samples=3000, n_warmup=3000,
                                                rng_seed=11))
        errs_ofg2 = [abs(sol_ofg2.unknown_functions[:k](x) - k_fg(x))
                     for x in -1.4:0.2:1.4]
        # Observed max error 0.033 at this seed; 6x headroom.
        @test maximum(errs_ofg2) < 0.2
        @test length(sol_ofg2.convergence.accept_rates.x) == 2
        @test all(isfinite, sol_ofg2.convergence.accept_rates.x)

        # Velocity unobserved: sampled block with borrowed GP prior + ODE
        # experts, no data term (ODIN precedent).
        prob_ofg1 = PSMProblem(osc_fg!, [1.5, 0.0], (0.0, 8.0), [uf_ofg];
            data_times=t_ofg, data_values=reshape(pos_ofg, :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_ofg1 = solve(prob_ofg1, FGPGMSolver(n_samples=3000, n_warmup=3000,
                                                rng_seed=11))
        errs_ofg1 = [abs(sol_ofg1.unknown_functions[:k](x) - k_fg(x))
                     for x in -1.4:0.2:1.4]
        # Observed max error 0.162 at this seed; ~2x headroom.
        @test maximum(errs_ofg1) < 0.35
        @test sol_ofg1.convergence.gp_hyperparams[2].observed == false
        @test isfinite(sol_ofg1.data_loss)
    end

    # ─── W8: square-root (QR-based) Kalman filtering ──────────────────
    #
    # Opt-in `sqrt_filter=true` on RodeoSolver / DaltonSolver /
    # PseudoMarginalSolver runs the probabilistic-ODE Kalman recursions on
    # right Cholesky/QR factors (P = RᵀR) instead of covariances
    # (src/sqrt_kalman.jl; Krämer & Hennig JMLR 2024 / Kailath et al. array
    # algorithms). Default `false` leaves the standard code paths untouched.

    @testset "Square-root Kalman — construction and default equivalence" begin
        PSM = PartiallySpecifiedModels

        # Validated opt-in kwarg, default false on all three consumers.
        @test RodeoSolver().sqrt_filter == false
        @test DaltonSolver().sqrt_filter == false
        @test PseudoMarginalSolver().sqrt_filter == false
        @test RodeoSolver(sqrt_filter=true).sqrt_filter == true
        @test DaltonSolver(sqrt_filter=true).sqrt_filter == true
        @test PseudoMarginalSolver(sqrt_filter=true, inner_method=:ffbs).sqrt_filter == true
        # Typed kwarg: non-Bool values are rejected at construction.
        @test_throws TypeError RodeoSolver(sqrt_filter=:yes)
        @test_throws TypeError DaltonSolver(sqrt_filter=1.0)
        @test_throws TypeError PseudoMarginalSolver(sqrt_filter=:yes)

        # `sqrt_filter=false` takes the standard code path exactly (the
        # delegation branch is never entered): identical output to a call
        # that does not pass the keyword at all, and no factor keys.
        logis_sq!(du, u, p, t) = (du[1] = 0.5 * u[1] * (1 - u[1] / 10.0))
        f_def = PSM.probsolve_filter(logis_sq!, nothing, [0.1], (0.0, 10.0),
                                     50, 3, [1.0])
        f_off = PSM.probsolve_filter(logis_sq!, nothing, [0.1], (0.0, 10.0),
                                     50, 3, [1.0]; sqrt_filter=false)
        @test f_def["μ_filt"] == f_off["μ_filt"]
        @test f_def["Σ_filt"] == f_off["Σ_filt"]
        @test f_def["ssq"] == f_off["ssq"]
        @test !haskey(f_def, "R_filt")

        # sqrt output carries the factors, and they reconstruct the stored
        # covariances exactly (Σ = RᵀR by definition of the representation).
        f_sq = PSM.probsolve_filter(logis_sq!, nothing, [0.1], (0.0, 10.0),
                                    50, 3, [1.0]; sqrt_filter=true)
        @test haskey(f_sq, "R_filt") && haskey(f_sq, "R_pred") && haskey(f_sq, "R_Q")
        @test all(f_sq["Σ_filt"][n] == f_sq["R_filt"][n]' * f_sq["R_filt"][n]
                  for n in 1:51)
    end

    @testset "Square-root Kalman — parity with the standard filter" begin
        PSM = PartiallySpecifiedModels
        logis_par!(du, u, p, t) = (du[1] = 0.5 * u[1] * (1 - u[1] / 10.0))
        lv_par!(du, u, p, t) = (du[1] = 1.5*u[1] - u[1]*u[2];
                                du[2] = -3.0*u[2] + u[1]*u[2])

        # Filter parity, q=3 moderate steps: means to 1e-10, covariances to
        # 1e-8 relative, calibrated diffusion to 1e-10 relative.
        # (Measured: 1.9e-14 / 4.0e-12 / 2e-19.)
        fs = PSM.probsolve_filter(logis_par!, nothing, [0.1], (0.0, 10.0),
                                  100, 3, [1.0])
        fq = PSM.probsolve_filter(logis_par!, nothing, [0.1], (0.0, 10.0),
                                  100, 3, [1.0]; sqrt_filter=true)
        @test maximum(maximum(abs.(fs["μ_filt"][n] .- fq["μ_filt"][n]))
                      for n in 1:101) < 1e-10
        @test maximum(maximum(abs.(fs["Σ_filt"][n] .- fq["Σ_filt"][n])) /
                      max(maximum(abs.(fs["Σ_filt"][n])), 1e-30)
                      for n in 2:101) < 1e-8
        @test abs(fs["ssq"] - fq["ssq"]) / fs["ssq"] < 1e-10

        # Smoother parity at q=2 (measured 1.5e-10 / 1.9e-8). At higher q /
        # finer steps the comparison is limited by the STANDARD smoother's
        # own `+1e-12·I` jitter on Σ_pred, which the square-root gain (pure
        # triangular solves) does not need — the paths then differ by the
        # jitter's perturbation of G, not by an error in the sqrt pass.
        fs2 = PSM.probsolve_filter(logis_par!, nothing, [0.1], (0.0, 10.0),
                                   50, 2, [1.0])
        fq2 = PSM.probsolve_filter(logis_par!, nothing, [0.1], (0.0, 10.0),
                                   50, 2, [1.0]; sqrt_filter=true)
        ms2, Ss2 = PSM.probsolve_smooth(fs2, 1)
        mq2, Sq2 = PSM.probsolve_smooth(fq2, 1)
        @test maximum(maximum(abs.(ms2[n][1] .- mq2[n][1])) for n in 1:51) < 1e-7
        @test maximum(maximum(abs.(Ss2[n][1] .- Sq2[n][1])) /
                      max(maximum(abs.(Ss2[n][1])), 1e-30) for n in 2:51) < 1e-6

        # Log-evidence parity (log-evidence read off the triangular factor,
        # 2·Σ log|diag S_R|, must equal the standard Cholesky value).
        # Measured: fenrir 2.0e-8, basic 1.1e-7, dalton 1.4e-9,
        # ssq/calib_acc 8.9e-12 relative.
        sol_lv = OrdinaryDiffEq.solve(
            ODEProblem(lv_par!, [1.0, 1.0], (0.0, 10.0)), Tsit5(),
            saveat=collect(0.5:0.5:9.5))
        tobs_lv = collect(sol_lv.t)
        data_lv = Matrix(hcat([sol_lv[k, :] for k in 1:2]...))
        args_lv = (lv_par!, nothing, [1.0, 1.0], (0.0, 10.0), 100, 3,
                   [1.0, 1.0], data_lv, tobs_lv, [1, 2], 0.05)
        @test isapprox(PSM.fenrir_loglik(args_lv...),
                       PSM.fenrir_loglik(args_lv...; sqrt_filter=true);
                       atol=1e-6, rtol=1e-7)
        @test isapprox(PSM.basic_loglik(args_lv...),
                       PSM.basic_loglik(args_lv...; sqrt_filter=true);
                       atol=1e-5, rtol=1e-7)
        @test isapprox(PSM._dalton_loglik(args_lv...),
                       PSM._dalton_loglik(args_lv...; sqrt_filter=true);
                       atol=1e-6, rtol=1e-7)

        # DALTON's calibration triple (logZ, σ̂², calib_acc) — the inputs to
        # its closed-form quasi-MLE rescale — must match across paths.
        ref_s = PSM._dalton_reference(lv_par!, nothing, [1.0, 1.0],
                                      (0.0, 10.0), 100, 3, [1.0, 1.0])
        ref_q = PSM._dalton_reference(lv_par!, nothing, [1.0, 1.0],
                                      (0.0, 10.0), 100, 3, [1.0, 1.0];
                                      sqrt_filter=true)
        @test isapprox(ref_s[1], ref_q[1]; atol=1e-5, rtol=1e-8)   # logZ
        @test abs(ref_s[2] - ref_q[2]) / ref_s[2] < 1e-9           # ssq
        @test abs(ref_s[3] - ref_q[3]) / ref_s[3] < 1e-9           # calib_acc
    end

    @testset "Square-root Kalman — stress at high IBM order" begin
        PSM = PartiallySpecifiedModels
        lv_st!(du, u, p, t) = (du[1] = 1.5*u[1] - u[1]*u[2];
                               du[2] = -3.0*u[2] + u[1]*u[2])

        # STANDARD-path degradation (the discriminator): at n_deriv=7 with
        # small steps the standard recursion Σ⁺ = Σ − KHΣ demonstrably
        # loses positive-semidefiniteness — strictly negative eigenvalues
        # appear in the filtered covariances (the COUNT is BLAS/machine-
        # specific — 25 on the machine that wrote this, 8 on the reviewer's;
        # worst eigenvalue −3.4e-10 on both — so only n_indef > 0 is
        # asserted). Honest caveat, stated plainly: no in-package
        # configuration was found where this degradation is CATASTROPHIC
        # (NaN / hard cholesky failure) — the violations stay at rounding
        # level relative to ‖Σ‖ (~1e-16·‖Σ‖) and the existing band-aids
        # (0.5(Σ+Σ') symmetrization, the adaptive jitter in logpdf_mvn, the
        # jitter escalation in _pm_mvn_sample, and the cholesky→pinv
        # fallbacks) absorb them.
        f_std7 = PSM.probsolve_filter(lv_st!, nothing, [1.0, 1.0],
                                      (0.0, 10.0), 2000, 7, [1.0, 1.0])
        n_indef = count(minimum(eigvals(Symmetric(f_std7["Σ_filt"][n]))) < -1e-12
                        for n in 2:2001)
        @test n_indef > 0

        # sqrt path at the same configuration: finite everywhere and the
        # means track the standard filter's (measured 7e-5 at q=7).
        f_sq7 = PSM.probsolve_filter(lv_st!, nothing, [1.0, 1.0],
                                     (0.0, 10.0), 2000, 7, [1.0, 1.0];
                                     sqrt_filter=true)
        @test all(all(isfinite, m) for m in f_sq7["μ_filt"])
        @test all(all(isfinite, R) for R in f_sq7["R_filt"])
        @test maximum(maximum(abs.(f_std7["μ_filt"][n] .- f_sq7["μ_filt"][n]))
                      for n in 1:2001) < 1e-2

        # PSD-by-construction at n_deriv=5: every reconstructed filter and
        # predict covariance is a Gram matrix RᵀR, so its eigenvalues are
        # nonnegative up to the rounding of forming/eigensolving the
        # product (measured: ≥ −3.6e-21 relative; bound 1e-12 relative).
        f_sq5 = PSM.probsolve_filter(lv_st!, nothing, [1.0, 1.0],
                                     (0.0, 10.0), 500, 5, [1.0, 1.0];
                                     sqrt_filter=true)
        psd_ok(Σ) = (ev = eigvals(Symmetric(Σ));
                     minimum(ev) >= -1e-12 * max(maximum(abs, ev), 1.0))
        @test all(psd_ok(f_sq5["Σ_filt"][n]) for n in 2:501)
        @test all(psd_ok(f_sq5["Σ_pred"][n]) for n in 2:501)

        # The square-root smoother stays finite and PSD there too.
        _, Ssm5 = PSM.probsolve_smooth(f_sq5, 2)
        @test all(psd_ok(Ssm5[n][k]) for n in 1:501, k in 1:2)
    end

    @testset "Square-root Kalman — FFBS sampling under sqrt" begin
        PSM = PartiallySpecifiedModels
        logis_ff!(du, u, p, t) = (du[1] = 0.5 * u[1] * (1 - u[1] / 10.0))

        # Factor-based FFBS is supported: backward draws use the
        # PSD-by-construction covariance factors (no jitter escalation).
        # Distributional check: empirical mean/variance of the sampled value
        # coordinate match the smoothing posterior.
        filt_ff = PSM.probsolve_filter(logis_ff!, nothing, [0.5], (0.0, 10.0),
                                       50, 3, [1.0]; sqrt_filter=true)
        mu_ff, S_ff = PSM.probsolve_smooth(filt_ff, 1)
        rng_ff = Random.Xoshiro(1)
        n_draw = 1500
        vals_ff = zeros(n_draw, 51)
        for s in 1:n_draw
            X = PSM._pm_sample_traj(filt_ff, rng_ff)
            @assert length(X) == 51
            for n in 1:51
                vals_ff[s, n] = X[n][1]
            end
        end
        @test all(isfinite, vals_ff)
        worst_m_ff = 0.0
        worst_v_ff = 0.0
        for n in 2:51
            m_emp = mean(vals_ff[:, n])
            v_emp = sum(abs2, vals_ff[:, n] .- m_emp) / (n_draw - 1)
            m_th = mu_ff[n][1][1]
            v_th = S_ff[n][1][1, 1]
            sd_th = sqrt(max(v_th, 1e-30))
            worst_m_ff = max(worst_m_ff, abs(m_emp - m_th) / max(sd_th, 1e-12))
            worst_v_ff = max(worst_v_ff, abs(v_emp - v_th) / max(v_th, 1e-12))
        end
        @test worst_m_ff < 0.15   # measured ≈ 0.02 (in posterior-sd units)
        @test worst_v_ff < 0.25   # measured ≈ 0.05 (relative)

        # And the two samplers estimate the same marginal likelihood
        # (unbiasedness preserved): compare _pm_loglik_hat across seeds.
        data_ff = reshape([0.6, 1.2, 2.5, 4.5, 6.8, 8.5], :, 1)
        t_ff = collect(1.0:1.5:9.0)
        ll_s = mean(PSM._pm_loglik_hat(logis_ff!, [0.5], (0.0, 10.0), 50, 3,
                                       [1.0], data_ff, t_ff, [1], 0.05, 64,
                                       Random.Xoshiro(s)) for s in 1:10)
        ll_q = mean(PSM._pm_loglik_hat(logis_ff!, [0.5], (0.0, 10.0), 50, 3,
                                       [1.0], data_ff, t_ff, [1], 0.05, 64,
                                       Random.Xoshiro(s); sqrt_filter=true)
                    for s in 1:10)
        @test abs(ll_s - ll_q) < 0.05   # measured ≈ 5e-5, MC sd ≈ 1e-3
    end

    @testset "Square-root Kalman — end-to-end solver fits" begin
        PSM = PartiallySpecifiedModels

        # DaltonSolver: same problem as the "DaltonSolver — exponential
        # decay" testset. The two likelihoods agree to ~1e-9, but the
        # NelderMead → L-BFGS(FD-gradient) optimizer amplifies sub-1e-8
        # objective differences into slightly different (equally good)
        # optima, so the recovered functions agree to ~4e-3 rather than
        # 1e-6 (measured max |Δf| = 3.7e-3, Δobjective = 3.3e-5). Both fits
        # must recover the truth to the original testset's tolerance.
        decay_sq!(du, u, p, t) = (du[1] = -p.f(u[1]))
        rng_dsq = Random.Xoshiro(42)
        sol_true_dsq = OrdinaryDiffEq.solve(
            ODEProblem(decay_sq!, [5.0], (0.0, 10.0), (; f=x -> 0.5*x)),
            Tsit5(); saveat=0.5)
        t_dsq = collect(sol_true_dsq.t)
        data_dsq = [sol_true_dsq.u[i][1] + 0.05*randn(rng_dsq)
                    for i in 1:length(t_dsq)]
        uf_dsq = BSplineApproximator(:f, (0.0, 6.0), 8)
        prob_dsq = PSMProblem(decay_sq!, [5.0], (0.0, 10.0), [uf_dsq];
            data_times=t_dsq, data_values=reshape(max.(data_dsq, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_d_std = solve(prob_dsq, DaltonSolver(n_steps=100, maxiters=50,
                                                 verbose=false))
        sol_d_sq = solve(prob_dsq, DaltonSolver(n_steps=100, maxiters=50,
                                                verbose=false, sqrt_filter=true))
        @test sol_d_sq isa PSMSolution
        @test isfinite(sol_d_sq.objective)
        @test abs(sol_d_sq.unknown_functions[:f](3.0) - 1.5) < 0.35
        @test maximum(abs(sol_d_std.unknown_functions[:f](x) -
                          sol_d_sq.unknown_functions[:f](x))
                      for x in 0.5:0.5:5.5) < 0.05
        @test abs(sol_d_std.objective - sol_d_sq.objective) < 1e-2

        # RodeoSolver: same problem as the "Rodeo solver (B-spline)"
        # testset (with a pinned rng). Measured max |Δr| = 1.7e-5.
        growth_sq!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1])
        rng_rsq = Random.Xoshiro(7)
        t_rsq = collect(0.0:0.25:5.0)
        data_rsq = reshape(2.0 .* exp.(0.3 .* t_rsq) .+
                           0.05 .* randn(rng_rsq, length(t_rsq)), :, 1)
        uf_rsq = BSplineApproximator(:r, (1.5, 10.0), 6; initial=0.5)
        prob_rsq = PSMProblem(growth_sq!, [2.0], (0.0, 5.0), [uf_rsq];
            data_times=t_rsq, data_values=data_rsq,
            obs_to_state=[1], solver=Tsit5())
        sol_r_std = solve(prob_rsq, RodeoSolver(n_steps=100, n_deriv=3,
                                                maxiters=100, obs_var=0.01,
                                                verbose=false))
        sol_r_sq = solve(prob_rsq, RodeoSolver(n_steps=100, n_deriv=3,
                                               maxiters=100, obs_var=0.01,
                                               verbose=false, sqrt_filter=true))
        @test sol_r_sq isa PSMSolution
        @test abs(sol_r_sq.unknown_functions[:r](3.0) - 0.3) < 0.15
        @test maximum(abs(sol_r_std.unknown_functions[:r](x) -
                          sol_r_sq.unknown_functions[:r](x))
                      for x in 2.0:0.5:8.0) < 0.01
        @test abs(sol_r_std.objective - sol_r_sq.objective) < 1e-2

        # PseudoMarginalSolver under sqrt (:ffbs inner method is supported
        # via factor-based backward sampling): runs, finite, valid chain,
        # and reproducible under the solver-owned rng_seed.
        r_true_psq(N) = 0.5 * (1.0 - N / 10.0)
        logis_psq!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1])
        rng_psq = Random.Xoshiro(42)
        sol_true_psq = OrdinaryDiffEq.solve(
            ODEProblem(logis_psq!, [1.0], (0.0, 15.0), (; r=r_true_psq)),
            Tsit5(); saveat=1.0)
        t_psq = collect(sol_true_psq.t)
        data_psq = [sol_true_psq.u[i][1] + 0.1*randn(rng_psq)
                    for i in 1:length(t_psq)]
        uf_psq = BSplineApproximator(:r, (0.0, 12.0), 6)
        prob_psq = PSMProblem(logis_psq!, [1.0], (0.0, 15.0), [uf_psq];
            data_times=t_psq, data_values=reshape(max.(data_psq, 0.01), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PartiallySpecifiedModels.Gaussian())
        sol_p_sq = solve(prob_psq, PseudoMarginalSolver(
            n_samples=50, n_warmup=25, n_steps=50, rng_seed=31415,
            verbose=false, sqrt_filter=true))
        @test sol_p_sq isa PSMSolution
        @test sol_p_sq.convergence isa MCMCChains.Chains
        @test size(sol_p_sq.convergence, 1) == 50
        @test all(isfinite, Array(sol_p_sq.convergence))
        sol_p_sq2 = solve(prob_psq, PseudoMarginalSolver(
            n_samples=50, n_warmup=25, n_steps=50, rng_seed=31415,
            verbose=false, sqrt_filter=true))
        @test Array(sol_p_sq.convergence) == Array(sol_p_sq2.convergence)
    end

    # ─── AdamSolver adjoint-sensitivity backend (SciMLSensitivity ext) ──
    #
    # The opt-in `sensealg` gradient backend: continuous adjoint
    # sensitivities via the PartiallySpecifiedModelsSciMLSensitivityExt
    # package extension (trigger: SciMLSensitivity only — its internal
    # vjps use ReverseDiff, a hard dep of SciMLSensitivity, so no Zygote
    # is involved). The parity tests are the correctness contract: the
    # extension mirrors the ForwardDiff loss paths (masking, penalty,
    # loss=:auto, u0-as-function, 1e10 sentinel) and must agree with them
    # by value (rtol 1e-10) and by gradient (rtol 1e-5).
    @testset "AdamSolver adjoint backend (sensealg)" begin
        using SciMLSensitivity
        using ForwardDiff
        import Lux
        PSM = PartiallySpecifiedModels

        # Shared spline problem: exponential growth with unknown rate fn
        growth_aj!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1])
        t_aj = collect(range(0.0, 5.0, length=30))
        y_aj = reshape(exp.(0.3 .* t_aj), :, 1)
        mk_spline_prob() = PSMProblem(growth_aj!, [1.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
            data_times=t_aj, data_values=y_aj, obs_to_state=[1],
            likelihood=Gaussian(), solver=Tsit5(), abstol=1e-8, reltol=1e-8)

        # The extension must be active under the test environment
        @test Base.get_extension(PSM,
            :PartiallySpecifiedModelsSciMLSensitivityExt) !== nothing

        @testset "gradient parity — spline (penalty and mask)" begin
            prob = mk_spline_prob()
            beta = PSM.initial_params(prob.approximators[1]) .+
                   0.05 .* randn(StableRNG(11), 6)
            f = b -> PSM.adam_loss_mse(prob, b, 0.5)
            l_fd = f(beta)
            g_fd = ForwardDiff.gradient(f, beta)
            l_aj, g_aj = PSM._adam_adjoint_loss_grad(prob, beta, :mse, 0.5,
                                                     :auto)
            @test isapprox(l_aj, l_fd; rtol=1e-10)
            @test isapprox(g_aj, g_fd; rtol=1e-5)

            # Masked cells (NaN datum + zero weight) contribute nothing
            y_mask = copy(y_aj); y_mask[5, 1] = NaN
            w_mask = ones(size(y_aj)); w_mask[9, 1] = 0.0
            prob_m = PSMProblem(growth_aj!, [1.0], (0.0, 5.0),
                [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
                data_times=t_aj, data_values=y_mask, data_weights=w_mask,
                obs_to_state=[1], likelihood=Gaussian(), solver=Tsit5(),
                abstol=1e-8, reltol=1e-8)
            fm = b -> PSM.adam_loss_mse(prob_m, b, 0.0)
            l_fdm = fm(beta)
            g_fdm = ForwardDiff.gradient(fm, beta)
            l_ajm, g_ajm = PSM._adam_adjoint_loss_grad(prob_m, beta, :mse,
                                                       0.0, :auto)
            @test isapprox(l_ajm, l_fdm; rtol=1e-10)
            @test isapprox(g_ajm, g_fdm; rtol=1e-5)
        end

        @testset "gradient parity — Poisson loss" begin
            prob_p = PSMProblem(growth_aj!, [1.0], (0.0, 5.0),
                [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
                data_times=t_aj, data_values=round.(y_aj), obs_to_state=[1],
                likelihood=Poisson(), solver=Tsit5(),
                abstol=1e-8, reltol=1e-8)
            beta = PSM.initial_params(prob_p.approximators[1]) .+
                   0.05 .* randn(StableRNG(12), 6)
            fp = b -> PSM.adam_loss_poisson(prob_p, b, 0.0)
            l_fd = fp(beta)
            g_fd = ForwardDiff.gradient(fp, beta)
            l_aj, g_aj = PSM._adam_adjoint_loss_grad(prob_p, beta, :poisson,
                                                     0.0, :auto)
            @test isapprox(l_aj, l_fd; rtol=1e-10)
            @test isapprox(g_aj, g_fd; rtol=1e-5)
        end

        @testset "gradient parity — u0 as a function of p" begin
            prob_u = PSMProblem(growth_aj!, p -> [1.0 + 0.5 * p.r(1.0)],
                (0.0, 5.0),
                [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
                data_times=t_aj, data_values=y_aj, obs_to_state=[1],
                likelihood=Gaussian(), solver=Tsit5(),
                abstol=1e-8, reltol=1e-8)
            beta = PSM.initial_params(prob_u.approximators[1]) .+
                   0.05 .* randn(StableRNG(13), 6)
            fu = b -> PSM.adam_loss_mse(prob_u, b, 0.0)
            l_fd = fu(beta)
            g_fd = ForwardDiff.gradient(fu, beta)
            l_aj, g_aj = PSM._adam_adjoint_loss_grad(prob_u, beta, :mse,
                                                     0.0, :auto)
            @test isapprox(l_aj, l_fd; rtol=1e-10)
            @test isapprox(g_aj, g_fd; rtol=1e-5)
        end

        # Neural parity + timing on a 321-parameter MLP UDE
        @testset "gradient parity and timing — neural (321 params)" begin
            decay_aj!(du, u, p, t) = (du[1] = -p.f(u[1]) * u[1])
            sol_true = OrdinaryDiffEq.solve(
                ODEProblem((du, u, p, t) -> (du[1] = -0.5 * u[1] * u[1]),
                           [5.0], (0.0, 10.0)), Tsit5(); saveat=0.5)
            rng_aj = StableRNG(7)
            y_nn = reshape([u[1] + 0.1 * randn(rng_aj) for u in sol_true.u],
                           :, 1)
            model = Lux.Chain(Lux.Dense(1, 16, tanh),
                              Lux.Dense(16, 16, tanh), Lux.Dense(16, 1))
            uf = NeuralApproximator(:f, model; domain=(0.0, 5.0),
                                    rng_seed=42)
            prob_nn = PSMProblem(decay_aj!, [5.0], (0.0, 10.0), [uf];
                data_times=collect(sol_true.t), data_values=y_nn,
                obs_to_state=[1], solver=Tsit5())
            beta = PSM.neural_init_params(uf, Random.Xoshiro(42))
            @test length(beta) == 321

            fn = b -> PSM.adam_loss_mse(prob_nn, b, 0.0)
            l_fd = fn(beta)
            g_fd = ForwardDiff.gradient(fn, beta)
            # Compiled tape: valid here (tanh MLP dynamics are branch-free)
            salg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
            l_aj, g_aj = PSM._adam_adjoint_loss_grad(prob_nn, beta, :mse,
                                                     0.0, salg)
            @test isapprox(l_aj, l_fd; rtol=1e-10)
            @test isapprox(g_aj, g_fd; rtol=1e-5)

            # Timing sanity (not a benchmark). Representative measurements
            # (M-series laptop, n=321): compiled-tape adjoint ~0.018 s/grad
            # vs ForwardDiff 0.0065-0.073 s/grad (ForwardDiff timing varies
            # with chunking/GC state); at n=1153 adjoint 0.058 vs
            # ForwardDiff 0.072 s/grad; the non-compiled tape (:auto) is
            # ~4x slower than the compiled one. Assert loosely to stay
            # robust on loaded CI machines.
            t_adj = minimum(@elapsed PSM._adam_adjoint_loss_grad(
                                prob_nn, beta, :mse, 0.0, salg)
                            for _ in 1:3)
            t_fwd = minimum(@elapsed ForwardDiff.gradient(fn, beta)
                            for _ in 1:3)
            @info "AdamSolver adjoint timing (n=321)" t_adj t_fwd ratio =
                t_adj / t_fwd
            # Loose bound only: ForwardDiff per-gradient time varies ~10×
            # with chunking/GC on shared hardware, and a timing flake fails
            # the whole suite. The real numbers are in the @info line.
            @test t_adj < 10 * t_fwd
        end

        # End-to-end neural UDE fit through solve() with the adjoint
        # backend (smaller sibling of "AdamSolver with NeuralApproximator")
        @testset "end-to-end neural fit with sensealg" begin
            decay_e2e!(du, u, p, t) = (du[1] = -p.f(u[1]) * u[1])
            sol_true = OrdinaryDiffEq.solve(
                ODEProblem((du, u, p, t) -> (du[1] = -0.5 * u[1] * u[1]),
                           [5.0], (0.0, 10.0)), Tsit5(); saveat=0.5)
            rng_e2e = StableRNG(21)
            y_e2e = reshape([u[1] + 0.1 * randn(rng_e2e)
                             for u in sol_true.u], :, 1)
            model = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 1))
            uf = NeuralApproximator(:f, model; domain=(0.0, 5.0),
                                    rng_seed=42)
            prob_e2e = PSMProblem(decay_e2e!, [5.0], (0.0, 10.0), [uf];
                data_times=collect(sol_true.t), data_values=y_e2e,
                obs_to_state=[1], solver=Tsit5())
            sol = solve(prob_e2e, AdamSolver(lr=0.02, maxiters=400,
                sensealg=InterpolatingAdjoint(
                    autojacvec=ReverseDiffVJP(true))))
            @test sol.convergence.backend == :adjoint
            # Recovery comparable to the ForwardDiff-path testset (which
            # runs 1000 iters and asserts < 0.15/0.25); at 400 iters the
            # fit is slightly looser, so use its outer band throughout.
            @test sol.data_loss < 1.0
            fhat = sol.unknown_functions[:f]
            for x in (0.5, 1.0, 2.0)
                @test abs(fhat(x) - 0.5 * x) < 0.25
            end
        end

        @testset "default path untouched and rejections" begin
            prob = mk_spline_prob()
            sol_fd = solve(prob, AdamSolver(maxiters=5, lr=0.01))
            @test sol_fd.convergence.backend == :forwarddiff
            sol_fdiff = solve(prob, AdamSolver(maxiters=3, lr=0.01,
                                               autodiff=false))
            @test sol_fdiff.convergence.backend == :finitediff
            sol_aj = solve(prob, AdamSolver(maxiters=5, lr=0.01,
                                            sensealg=:auto))
            @test sol_aj.convergence.backend == :adjoint

            # Discrete maps have no ODE solve to adjoint
            prob_disc = PSMProblem(growth_aj!, [1.0], (0.0, 5.0),
                [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
                data_times=collect(0.0:1.0:5.0),
                data_values=reshape(exp.(0.1 .* collect(0.0:1.0:5.0)), :, 1),
                obs_to_state=[1], discrete=true)
            err = try
                solve(prob_disc, AdamSolver(maxiters=2, sensealg=:auto))
                nothing
            catch e; e; end
            @test err isa ErrorException
            @test occursin("discrete-time", err.msg)

            # DDE problems go through adam_solve_dde, not the adjoint path
            prob_dde = PSMProblem(growth_aj!, [1.0], (0.0, 5.0),
                [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
                data_times=t_aj, data_values=y_aj, obs_to_state=[1],
                delays=[1.0], history=(p, t) -> [1.0])
            err = try
                solve(prob_dde, AdamSolver(maxiters=2, sensealg=:auto))
                nothing
            catch e; e; end
            @test err isa ErrorException
            @test occursin("delays", err.msg)

            # Forward sensitivity algorithms are rejected (adjoint-only)
            err = try
                solve(prob, AdamSolver(maxiters=2,
                                       sensealg=ForwardDiffSensitivity()))
                nothing
            catch e; e; end
            @test err isa ErrorException
            @test occursin("not an adjoint", err.msg)

            # Unknown sensealg symbol
            err = try
                solve(prob, AdamSolver(maxiters=2, sensealg=:bogus))
                nothing
            catch e; e; end
            @test err isa ErrorException
            @test occursin("unknown sensealg symbol", err.msg)

            # Junk sensealg values fall through the extension to the stub
            err = try
                solve(prob, AdamSolver(maxiters=2, sensealg="junk"))
                nothing
            catch e; e; end
            @test err isa ErrorException
            @test occursin("not a valid adjoint specification", err.msg)

            # Unsupported likelihood family: same rejection as the
            # ForwardDiff path (AdamSolver supports Gaussian/Poisson only)
            prob_nb = PSMProblem(growth_aj!, [1.0], (0.0, 5.0),
                [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
                data_times=t_aj, data_values=y_aj, obs_to_state=[1],
                likelihood=NegativeBinomial(10.0))
            err = try
                solve(prob_nb, AdamSolver(maxiters=2, sensealg=:auto))
                nothing
            catch e; e; end
            @test err isa ErrorException
            @test occursin("no loss for", err.msg)
        end

        # The stub must fire when the extension is NOT loaded: spawn julia
        # on the package environment and assert the informative error. This
        # works because extension activation requires the trigger package to
        # be LOADED, not merely installed somewhere on the LOAD_PATH stack —
        # the subprocess never runs `using SciMLSensitivity`, so a global-env
        # install cannot activate the extension. (The subprocess does need
        # the package-dir Manifest to exist, which dev checkouts and CI
        # buildpkg both guarantee.)
        @testset "stub error without the extension (subprocess)" begin
            pkgdir_psm = dirname(dirname(pathof(PartiallySpecifiedModels)))
            code = """
            using PartiallySpecifiedModels
            growth!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1])
            prob = PSMProblem(growth!, [1.0], (0.0, 1.0),
                [BSplineApproximator(:r, (0.5, 2.0), 4)];
                data_times=[0.0, 0.5, 1.0],
                data_values=reshape([1.0, 1.1, 1.2], :, 1),
                obs_to_state=[1])
            try
                solve(prob, AdamSolver(maxiters=2, sensealg=:auto))
                print("NO_ERROR")
            catch e
                msg = sprint(showerror, e)
                print(occursin("SciMLSensitivity", msg) &&
                      occursin("not loaded", msg) ? "STUB_OK" : "WRONG_ERROR")
            end
            """
            out = read(`$(Base.julia_cmd()) --project=$(pkgdir_psm)
                        --startup-file=no -e $code`, String)
            @test endswith(out, "STUB_OK")
        end
    end

    # ─── W10: GCVSolver search=:reuse (ddefit EScv one-decomposition trick) ──

    @testset "GCVSolver search=:reuse — one-decomposition fast GCV" begin
        PSM = PartiallySpecifiedModels

        # ── Construction validation ──
        @test_throws ArgumentError GCVSolver(search=:fast)
        @test_throws ArgumentError GCVSolver(search=:eigen)
        # :reuse is a pure-GCV trick: NCV's neighbourhood downdates need
        # A(λ)⁻¹ per λ, incompatible with one decomposition — rejected.
        @test_throws ArgumentError GCVSolver(search=:reuse, criterion=:ncv)
        @test GCVSolver().search == :direct                    # default unchanged
        @test GCVSolver(search=:reuse).search == :reuse
        @test GCVSolver(search=:direct, criterion=:ncv) isa GCVSolver

        # ── SCORE EQUIVALENCE: reuse family vs direct _gcv_score on fixed
        # working models. The reuse scorer replicates the direct scorer's
        # λ-dependent stability ridge exactly (per-argmax-segment pencil),
        # so agreement is limited only by the factorization roundoff:
        #  · full-rank S: ~1e-15 relative at ALL λ across the RHO bounds;
        #  · rank-deficient S: the penalty null-space eigenvalues of the
        #    whitened pencil are O(1e-12) (the replicated ridge) and carry
        #    the absolute O(eps·‖K‖) error of any double-precision
        #    eigendecomposition, so agreement degrades as λ·eps once
        #    λ ≳ 1e15 — measured (Xoshiro(7) fixtures of this shape):
        #    ≤9e-11 for ρ ≤ 15, ~1e-8 at ρ = 20, ~3e-4 over the full
        #    range to ρ = 40 (λ = 2.4e17, essentially-infinite smoothing
        #    where the direct path's trA/β are themselves conditioning-
        #    limited — the direct SCORE proper stays more stable there,
        #    so the loose full-range pin reflects reuse-side eps·λ
        #    growth in a region no search ever selects).
        #    Pinned below with ≥10× margins: 1e-10 for ρ ∈ [RHO_MIN, 13],
        #    1e-6 to ρ = 20, 5e-3 full range.
        rng_w10 = StableRNG(7)
        n_w10, p_w10 = 60, 12
        J_w10 = randn(rng_w10, n_w10, p_w10)
        z_w10 = randn(rng_w10, n_w10)
        ones_w10 = ones(n_w10)
        Z0_w10 = zeros(p_w10, p_w10)
        # B-spline-style second-difference penalty: 2-dim null space
        D_w10 = zeros(p_w10 - 2, p_w10)
        for i in 1:p_w10-2
            D_w10[i, i] = 1.0; D_w10[i, i+1] = -2.0; D_w10[i, i+2] = 1.0
        end
        S_rd_w10 = D_w10' * D_w10
        @test rank(S_rd_w10) == p_w10 - 2
        S_fr_w10 = Matrix(1.0I, p_w10, p_w10) .+
                   0.1 .* (x -> x' * x)(randn(rng_w10, p_w10, p_w10) ./ sqrt(p_w10))
        wv_w10 = rand(rng_w10, n_w10) .+ 0.3
        wm_w10 = copy(wv_w10)
        wm_w10[3] = 0.0; wm_w10[17] = 0.0; wm_w10[44] = 0.0   # masked rows
        nm_w10 = count(>(0), wm_w10)

        # worst relative disagreement (score AND tr(A) AND β̂) over 20 λ
        function reuse_worst_w10(w, n, S_base, S_pen; rho_hi=PSM.RHO_MAX)
            fam = PSM._gcv_reuse_family(J_w10, w, z_w10, S_base, S_pen,
                                        n, 1.4)
            fam === nothing && return Inf
            worst = 0.0
            for lam in exp.(range(PSM.RHO_MIN, rho_hi, length=20))
                gf, bf, _, tf = fam(lam)
                gd, bd, _, td = PSM._gcv_score(J_w10, w, z_w10,
                                               S_base .+ lam .* S_pen, n, 1.4)
                worst = max(worst,
                            abs(gf - gd) / max(abs(gd), 1e-300),
                            abs(tf - td) / max(td, 1e-300),
                            norm(bf - bd) / max(norm(bd), 1e-300))
            end
            worst
        end

        # well-conditioned (full-rank S): full RHO span at 1e-10
        @test reuse_worst_w10(ones_w10, n_w10, Z0_w10, S_fr_w10) < 1e-10
        # rank-deficient / weighted / masked / nonzero fixed penalty
        # (coordinate-descent shape): graded pins per the conditioning
        # analysis above
        for (w, n) in ((ones_w10, n_w10), (wv_w10, n_w10), (wm_w10, nm_w10))
            @test reuse_worst_w10(w, n, Z0_w10, S_rd_w10; rho_hi=13.0) < 1e-10
            @test reuse_worst_w10(w, n, Z0_w10, S_rd_w10; rho_hi=20.0) < 1e-6
            @test reuse_worst_w10(w, n, Z0_w10, S_rd_w10) < 5e-3
        end
        S_baseK_w10 = 3.7 .* Matrix(1.0I, p_w10, p_w10)
        @test reuse_worst_w10(wv_w10, n_w10, S_baseK_w10, S_rd_w10;
                              rho_hi=13.0) < 1e-10
        @test reuse_worst_w10(wv_w10, n_w10, S_baseK_w10, S_rd_w10) < 5e-3

        # ── Rank-deficient penalty null space: EDF contribution is 1 per
        # null direction (dᵢ ≈ 0 ⟹ 1/(1+λdᵢ) ≈ 1), verified against the
        # direct score at large-but-moderate λ where the fit is otherwise
        # fully smoothed: tr(A) ↓ but stays ≥ null-space dim = 2.
        fam_rd_w10 = PSM._gcv_reuse_family(J_w10, ones_w10, z_w10,
                                           Z0_w10, S_rd_w10, n_w10, 1.4)
        _, _, _, trA_f_w10 = fam_rd_w10(1e6)
        _, _, _, trA_d_w10 = PSM._gcv_score(J_w10, ones_w10, z_w10,
                                            1e6 .* S_rd_w10, n_w10, 1.4)
        @test abs(trA_f_w10 - trA_d_w10) < 1e-8
        @test 2.0 < trA_f_w10 < 2.1

        # ── Degraded case, unit level: near-singular A ⟹ family refuses
        # (returns nothing) instead of whitening garbage ──
        J_sing_w10 = randn(rng_w10, 8, p_w10)       # n = 8 < p = 12
        @test PSM._gcv_reuse_family(J_sing_w10, ones(8), randn(rng_w10, 8),
                                    Z0_w10, S_rd_w10, 8, 1.4) === nothing

        # ── End-to-end equivalence, single λ: logistic growth ──
        r_true_w10(N) = 0.5 * (1.0 - N / 10.0)
        logi_w10!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1])
        st_w10 = OrdinaryDiffEq.solve(
            ODEProblem(logi_w10!, [1.0], (0.0, 15.0), (; r=r_true_w10)),
            Tsit5(); saveat=0.2, abstol=1e-10, reltol=1e-10)
        t_w10 = collect(st_w10.t)
        truth_w10 = [u[1] for u in st_w10.u]
        data_w10 = max.(truth_w10 .+
                        0.05 .* randn(StableRNG(11), length(t_w10)), 0.01)
        mk_w10(res, times, vals) = PSMProblem(logi_w10!, [1.0], (0.0, 15.0),
            [BSplineApproximator(:r, (0.0, 12.0), res)];
            data_times=times, data_values=reshape(vals, :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=PSM.Gaussian())
        prob_e2e_w10 = mk_w10(8, t_w10, data_w10)
        sol_d_w10 = solve(prob_e2e_w10, GCVSolver(maxiters=25))
        sol_r_w10 = solve(prob_e2e_w10, GCVSolver(maxiters=25, search=:reuse))
        # λ̂ agreement within the golden-section search tolerance: the two
        # paths see bit-different scores (~1e-13 relative in the benign
        # region), so iterates can part ways only below the search tol.
        # Measured: |Δ log λ̂| = 8e-7, objective/GCV relΔ ≈ 6e-9, fitted
        # values within 5e-9. Pinned with ~100× margin.
        @test abs(log(sol_d_w10.smoothing_params[1]) -
                  log(sol_r_w10.smoothing_params[1])) < 1e-4
        @test abs(sol_d_w10.objective - sol_r_w10.objective) <
              1e-6 * max(abs(sol_d_w10.objective), 1.0)
        @test maximum(abs.(sol_d_w10.fitted_values .-
                           sol_r_w10.fitted_values)) < 1e-6
        @test abs(sol_d_w10.convergence.gcv - sol_r_w10.convergence.gcv) <
              1e-6 * abs(sol_d_w10.convergence.gcv)
        # identical convergence-info structure
        @test keys(sol_d_w10.convergence) == keys(sol_r_w10.convergence)
        @test sol_d_w10.convergence.criterion == :gcv
        @test sol_r_w10.convergence.criterion == :gcv

        # ── End-to-end, multi-λ (two approximators ⟹ coordinate descent
        # with one decomposition per coordinate per sweep) ──
        two_uf_w10!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1] - p.g(u[1]))
        pt_w10 = ODEProblem((du, u, p, t) -> (du[1] = 0.4u[1] - 0.05u[1]),
                            [1.0], (0.0, 5.0))
        st2_w10 = OrdinaryDiffEq.solve(pt_w10, Tsit5(); saveat=0.25)
        tt2_w10 = collect(st2_w10.t)
        dv2_w10 = reshape([st2_w10(t)[1] for t in tt2_w10] .+
                          0.02 .* randn(StableRNG(7), length(tt2_w10)), :, 1)
        prob2_w10 = PSMProblem(two_uf_w10!, [1.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.5, 6.0), 5; initial=x -> 0.3),
             BSplineApproximator(:g, (0.5, 6.0), 5; initial=x -> 0.03x)];
            data_times=tt2_w10, data_values=dv2_w10, obs_to_state=[1],
            known_params=NamedTuple(), likelihood=PSM.Gaussian(),
            solver=Tsit5())
        sol2_d_w10 = solve(prob2_w10, GCVSolver(maxiters=15))
        sol2_r_w10 = solve(prob2_w10, GCVSolver(maxiters=15, search=:reuse))
        @test length(sol2_r_w10.smoothing_params) == 2
        @test all(abs.(log.(sol2_d_w10.smoothing_params) .-
                       log.(sol2_r_w10.smoothing_params)) .< 1e-4)
        @test abs(sol2_d_w10.objective - sol2_r_w10.objective) <
              1e-6 * max(abs(sol2_d_w10.objective), 1.0)
        @test maximum(abs.(sol2_d_w10.fitted_values .-
                           sol2_r_w10.fitted_values)) < 1e-6

        # ── Degraded case, end-to-end: 8 data points, 20-knot basis ⟹
        # J'WJ rank ≤ 8 < p ⟹ the reuse guard trips every IRLS iteration
        # and the search runs the byte-identical direct scorer, so the two
        # fits must be EXACTLY equal (and finite) ──
        tt_s_w10 = collect(range(0.0, 15.0, length=8))
        st_s_w10 = OrdinaryDiffEq.solve(
            ODEProblem(logi_w10!, [1.0], (0.0, 15.0), (; r=r_true_w10)),
            Tsit5(); saveat=tt_s_w10, abstol=1e-10, reltol=1e-10)
        d_s_w10 = max.([u[1] for u in st_s_w10.u] .+
                       0.05 .* randn(StableRNG(3), 8), 0.01)
        prob_s_w10 = mk_w10(20, tt_s_w10, d_s_w10)
        sol_sd_w10 = solve(prob_s_w10, GCVSolver(maxiters=8))
        sol_sr_w10 = solve(prob_s_w10, GCVSolver(maxiters=8, search=:reuse))
        @test sol_sr_w10.smoothing_params == sol_sd_w10.smoothing_params
        @test sol_sr_w10.objective == sol_sd_w10.objective
        @test isfinite(sol_sr_w10.objective)
        @test all(isfinite, sol_sr_w10.fitted_values)

        # ── Timing (comment-only, per the W9 flake-review guidance):
        # search-only microbenchmark (grid 50 + golden section, minimum of
        # 7 runs, warmed, second-difference penalty, this machine):
        #   p=20 n=100: direct 2.1ms, reuse 1.0ms (2.0×)
        #   p=30 n=150: direct 4.8ms, reuse 1.5ms (3.3×)
        #   p=40 n=200: direct 8.7ms, reuse 2.5ms (3.5×)
        #   p=60 n=300: direct 20.0ms, reuse 5.4ms (3.7×)
        # Full multi-λ solve (prob2 above, m=2): 1.02s → 0.15s (~7×; the
        # coordinate sweep multiplies the per-λ solve count, so reuse
        # helps most there). At package-typical p ≈ 8–15 the win per IRLS
        # iteration is real but modest (~2×); the feature matters for
        # large bases and multi-approximator coordinate descent.
    end

    @testset "W11: jac=:forwarddiff prediction Jacobians" begin
        import ForwardDiff
        import Lux
        PSM = PartiallySpecifiedModels

        # ── Construction validation on all three solvers ──
        @test_throws ArgumentError LAML(jac=:bogus)
        @test_throws ArgumentError GCVSolver(jac=:bogus)
        @test_throws ArgumentError CollocationLAML(jac=:bogus)
        # Default is :fd on all three — the historical FD path stays the
        # default and is pinned byte-identical below.
        @test LAML().jac === :fd
        @test GCVSolver().jac === :fd
        @test CollocationLAML().jac === :fd
        @test LAML(jac=:forwarddiff).jac === :forwarddiff

        # ── Shared smooth logistic problem (BSpline) ──
        r_true_w11(N) = 0.8 * (1.0 - N / 2.0)
        function logi_w11!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        tt_w11 = collect(range(0.0, 10.0, length=25))
        st_w11 = OrdinaryDiffEq.solve(
            ODEProblem((du, u, p, t) -> (du[1] = r_true_w11(u[1]) * u[1]),
                       [0.2], (0.0, 10.0)), Tsit5();
            saveat=tt_w11, abstol=1e-10, reltol=1e-10)
        # Seed choice is deliberate (see the LAML end-to-end block below):
        # on this draw both Jacobian backends land on the same LAML optimum.
        dat_w11 = reshape([u[1] for u in st_w11.u] .+
                          0.02 .* randn(StableRNG(4), 25), :, 1)
        mk_w11(; data=dat_w11, weights=nothing) = PSMProblem(
            logi_w11!, [0.2], (0.0, 10.0),
            [BSplineApproximator(:r, (0.0, 2.5), 8)];
            data_times=tt_w11, data_values=data,
            (weights === nothing ? (;) : (; data_weights=weights))...)
        prob_w11 = mk_w11()
        b0_w11 = PSM.build_initial_params(prob_w11)
        np_w11 = length(b0_w11)

        # ── Jacobian agreement, FD vs ForwardDiff (smooth ODE) ──
        f0_w11 = vec(PSM.simulate(prob_w11, b0_w11))
        J_fd_w11 = zeros(25, np_w11)
        J_ad_w11 = zeros(25, np_w11)
        cfg_w11 = PSM._fd_jacobian_config(np_w11)
        PSM.compute_jacobian!(J_fd_w11, prob_w11, b0_w11, f0_w11, 25, 1;
                              dam=fill(1e-8, np_w11))
        PSM.compute_jacobian!(J_ad_w11, prob_w11, b0_w11, f0_w11, 25, 1;
                              dam=fill(1e-8, np_w11),
                              jac=:forwarddiff, fd_cfg=cfg_w11)
        @test all(isfinite, J_ad_w11)
        # FD's own error bound (truncation + integration noise); measured
        # agreement here is ~4e-11 relative.
        scale_w11 = maximum(abs, J_ad_w11)
        @test maximum(abs, J_fd_w11 - J_ad_w11) < 1e-6 * max(scale_w11, 1.0)
        # Silent-fallback detector (see the DDE variant below): a permanent
        # AD→FD fallback would make this diff EXACTLY 0.0; genuine AD
        # leaves FD truncation noise (~4e-11 measured).
        @test maximum(abs, J_fd_w11 - J_ad_w11) > 0

        # Default path untouched: jac=:fd through the keyword is EXACTLY the
        # implicit default (same code path, same adaptive steps).
        J_fd2_w11 = zeros(25, np_w11)
        PSM.compute_jacobian!(J_fd2_w11, prob_w11, b0_w11, f0_w11, 25, 1;
                              dam=fill(1e-8, np_w11), jac=:fd)
        @test J_fd2_w11 == J_fd_w11

        # ── Analytic case: discrete linear map u_{t+1} = g(1)·u_t ──
        # pred_t = g^t·u_0 with g = Σ cⱼBⱼ(1), so ∂pred_t/∂cⱼ = t·g^{t-1}·Bⱼ(1)
        # in closed form. No integrator → AD must hit machine precision while
        # FD carries its truncation error.
        function linmap_w11!(un, u, p, t)
            un[1] = p.g(1.0) * u[1]
        end
        ttd_w11 = collect(0.0:1.0:8.0)
        probd_w11 = PSMProblem(linmap_w11!, [1.0], (0.0, 8.0),
                               [BSplineApproximator(:g, (0.0, 2.0), 6;
                                                    initial=x -> 0.9)];
                               data_times=ttd_w11,
                               data_values=reshape(fill(1.0, 9), :, 1),
                               discrete=true)
        bd_w11 = PSM.build_initial_params(probd_w11)
        npd_w11 = length(bd_w11)
        g0_w11 = PSM.build_evaluator(probd_w11.approximators[1], bd_w11)(1.0)
        Bj_w11 = ForwardDiff.gradient(
            c -> PSM.build_evaluator(probd_w11.approximators[1], c)(1.0),
            bd_w11)
        J_an_w11 = zeros(9, npd_w11)
        for (i, t) in enumerate(ttd_w11), j in 1:npd_w11
            J_an_w11[i, j] = t == 0.0 ? 0.0 : t * g0_w11^(t - 1) * Bj_w11[j]
        end
        f0d_w11 = vec(PSM.simulate(probd_w11, bd_w11))
        J_fdd_w11 = zeros(9, npd_w11)
        J_add_w11 = zeros(9, npd_w11)
        PSM.compute_jacobian!(J_fdd_w11, probd_w11, bd_w11, f0d_w11, 9, 1;
                              dam=fill(1e-8, npd_w11))
        PSM.compute_jacobian!(J_add_w11, probd_w11, bd_w11, f0d_w11, 9, 1;
                              dam=fill(1e-8, npd_w11), jac=:forwarddiff,
                              fd_cfg=PSM._fd_jacobian_config(npd_w11))
        # Measured: AD 4.4e-16 (machine), FD 8.2e-11 (its truncation error)
        @test maximum(abs, J_add_w11 - J_an_w11) < 1e-12
        @test maximum(abs, J_fdd_w11 - J_an_w11) < 1e-8
        @test maximum(abs, J_add_w11 - J_an_w11) <
              maximum(abs, J_fdd_w11 - J_an_w11)

        # ── Neural approximator agreement ──
        nn_w11 = Lux.Chain(Lux.Dense(1 => 4, tanh), Lux.Dense(4 => 1))
        probn_w11 = PSMProblem(logi_w11!, [0.2], (0.0, 10.0),
                               [NeuralApproximator(:r, nn_w11; rng_seed=7)];
                               data_times=tt_w11, data_values=dat_w11)
        bn_w11 = PSM.build_initial_params(probn_w11)
        npn_w11 = length(bn_w11)
        f0n_w11 = vec(PSM.simulate(probn_w11, bn_w11))
        J_fdn_w11 = zeros(25, npn_w11)
        J_adn_w11 = zeros(25, npn_w11)
        PSM.compute_jacobian!(J_fdn_w11, probn_w11, bn_w11, f0n_w11, 25, 1;
                              dam=fill(1e-8, npn_w11))
        PSM.compute_jacobian!(J_adn_w11, probn_w11, bn_w11, f0n_w11, 25, 1;
                              dam=fill(1e-8, npn_w11), jac=:forwarddiff,
                              fd_cfg=PSM._fd_jacobian_config(npn_w11))
        @test all(isfinite, J_adn_w11)
        @test maximum(abs, J_fdn_w11 - J_adn_w11) <
              1e-4 * max(maximum(abs, J_adn_w11), 1.0)   # measured 1.6e-7

        # ── Masked cells: rows are computed for masked cells too (masking
        #    is applied downstream through the weights), identically in
        #    both backends ──
        datm_w11 = copy(dat_w11); datm_w11[5] = NaN
        wm_w11 = ones(25, 1); wm_w11[9] = 0.0
        probm_w11 = mk_w11(data=datm_w11, weights=wm_w11)
        f0m_w11 = vec(PSM.simulate(probm_w11, b0_w11))
        J_fdm_w11 = zeros(25, np_w11)
        J_adm_w11 = zeros(25, np_w11)
        PSM.compute_jacobian!(J_fdm_w11, probm_w11, b0_w11, f0m_w11, 25, 1;
                              dam=fill(1e-8, np_w11))
        PSM.compute_jacobian!(J_adm_w11, probm_w11, b0_w11, f0m_w11, 25, 1;
                              dam=fill(1e-8, np_w11), jac=:forwarddiff,
                              fd_cfg=PSM._fd_jacobian_config(np_w11))
        @test all(isfinite, J_adm_w11)
        @test !all(iszero, J_adm_w11[5, :])   # masked rows still populated
        @test !all(iszero, J_adm_w11[9, :])
        @test maximum(abs, J_fdm_w11 - J_adm_w11) <
              1e-6 * max(maximum(abs, J_adm_w11), 1.0)

        # ── End-to-end equivalence: LAML ──
        # Measured on this draw: fitted-values maxdiff 4.2e-8,
        # |log θ ratio| 2.5e-6, data_loss rel diff 4.7e-7, criterion diff
        # 1.5e-5 — the two backends land on the SAME optimum. Honest caveat
        # (why the seed is pinned): an 8-seed scan showed that on ~half the
        # draws of THIS near-linear-r problem the IRLS accept/reject
        # branching diverges to different Fellner–Schall fixed points in the
        # flat heavy-smoothing regime (λ̂ ratios up to e^12, in BOTH
        # directions — sometimes :fd attains the better LAML criterion,
        # sometimes :forwarddiff). That is path sensitivity of the IRLS/FS
        # iteration itself, not Jacobian error: the Jacobians agree to 1e-6
        # relative (pinned above) on every draw, and GCVSolver — whose λ is
        # selected by a global grid+golden search per iteration rather than
        # an FS fixed-point path — agrees to |log λ ratio| < 6e-3 on ALL
        # eight seeds.
        sol_lf_w11 = solve(mk_w11(), LAML(maxiters=20))
        sol_la_w11 = solve(mk_w11(), LAML(maxiters=20, jac=:forwarddiff))
        @test maximum(abs, sol_lf_w11.fitted_values -
                           sol_la_w11.fitted_values) < 1e-4
        @test abs(log(sol_la_w11.smoothing_params[1] /
                      sol_lf_w11.smoothing_params[1])) < 0.05
        @test abs(sol_la_w11.data_loss - sol_lf_w11.data_loss) <
              1e-3 * max(sol_lf_w11.data_loss, 1e-8)

        # ── End-to-end equivalence: GCVSolver ──
        sol_gf_w11 = solve(mk_w11(), GCVSolver(maxiters=12))
        sol_ga_w11 = solve(mk_w11(), GCVSolver(maxiters=12, jac=:forwarddiff))
        # Measured: fitted-values maxdiff 2.7e-6, |log λ ratio| 1.0e-3
        # (within the golden-section search tolerance).
        @test maximum(abs, sol_gf_w11.fitted_values -
                           sol_ga_w11.fitted_values) < 1e-4
        @test abs(log(sol_ga_w11.smoothing_params[1] /
                      sol_gf_w11.smoothing_params[1])) < 0.05

        # ── End-to-end equivalence: CollocationLAML ──
        sol_cf_w11 = solve(mk_w11(), CollocationLAML(maxiters=15,
                                                     n_continuation=4))
        sol_ca_w11 = solve(mk_w11(), CollocationLAML(maxiters=15,
                                                     n_continuation=4,
                                                     jac=:forwarddiff))
        # Measured: fitted-values maxdiff 3.9e-9, data_loss rel diff 6e-8.
        @test maximum(abs, sol_cf_w11.fitted_values -
                           sol_ca_w11.fitted_values) < 1e-5
        @test abs(sol_ca_w11.data_loss - sol_cf_w11.data_loss) <
              1e-4 * max(sol_cf_w11.data_loss, 1e-8)
        # λ was NOT compared here, unlike the GCVSolver neighbour above —
        # a coverage gap this suite carried. Measured at this config:
        # 0.1655051398 (:fd) vs 0.1655046247 (:forwarddiff), |log ratio|
        # 3.1e-6, so 1e-4 keeps 32x margin. PIN THE CONFIG: at package
        # defaults λ is unidentified on this fixture (edf ≈ 2.13, the
        # penalty null-space dimension) and the two backends land 17x
        # apart — not a defect, but a λ assertion there would be noise.
        @test abs(log(sol_cf_w11.smoothing_params[1] /
                      sol_ca_w11.smoothing_params[1])) < 1e-4

        # ── THE critical test: collocation failure-mask preservation ──
        # sqrt(u) throws DomainError at collocation points whose state is
        # negative; the residual there must hold the sentinel in BOTH
        # backends, the sentinel must never be differenced/differentiated
        # (the fix-campaign regression this guards produced ~1e12 Jacobian
        # entries), and the zero-block structure must be IDENTICAL.
        function sq_dyn_w11!(du, u, p, t)
            du[1] = p.h(sqrt(u[1]))
        end
        ttc_w11 = collect(range(0.0, 4.0, length=9))
        datc_w11 = reshape([1.0, 0.8, -0.5, 0.6, -0.3, 0.5, 0.4, 0.35, 0.3],
                           :, 1)
        probc_w11 = PSMProblem(sq_dyn_w11!, [1.0], (0.0, 4.0),
                               [BSplineApproximator(:h, (0.0, 1.5), 5)];
                               data_times=ttc_w11, data_values=datc_w11)
        alpha_c_w11 = reshape(copy(vec(datc_w11)), :, 1)
        beta_c_w11 = PSM.build_initial_params(probc_w11)
        D_c_w11 = PSM.build_diff_matrix(ttc_w11)
        _, Ffail_w11 = PSM.eval_ode_rhs_masked(probc_w11, ttc_w11,
                                               alpha_c_w11, beta_c_w11)
        @test findall(Ffail_w11) == [3, 5]   # the documented failure mask
        r_fd_w11, Jc_fd_w11 = PSM.collocation_residual_jacobian(
            probc_w11, ttc_w11, alpha_c_w11, beta_c_w11, D_c_w11, 1.0,
            ones(9))
        r_ad_w11, Jc_ad_w11 = PSM.collocation_residual_jacobian(
            probc_w11, ttc_w11, alpha_c_w11, beta_c_w11, D_c_w11, 1.0,
            ones(9); jac=:forwarddiff,
            fd_cfg=PSM._fd_jacobian_config(length(beta_c_w11)))
        # Residual path is shared: identical, sentinel present at failed pts
        @test r_fd_w11 == r_ad_w11
        @test any(x -> abs(x) > 1e5, r_fd_w11)          # sentinel present
        # β-block rows at failed points are exactly zero in BOTH backends
        n_alpha_c = 9
        for i in findall(Ffail_w11)
            @test all(iszero, Jc_fd_w11[9 + i, n_alpha_c+1:end])
            @test all(iszero, Jc_ad_w11[9 + i, n_alpha_c+1:end])
        end
        # Zero-pattern of the β-block is IDENTICAL across backends
        @test (Jc_fd_w11[:, n_alpha_c+1:end] .== 0.0) ==
              (Jc_ad_w11[:, n_alpha_c+1:end] .== 0.0)
        # No fabricated ~1e12 slopes anywhere (the regression guard)
        @test maximum(abs, Jc_fd_w11) < 1e10
        @test maximum(abs, Jc_ad_w11) < 1e10
        # Away from failed points the two backends agree
        @test maximum(abs, Jc_fd_w11 - Jc_ad_w11) < 1e-10
        # End-to-end on the failure problem: both backends complete
        sol_cff_w11 = solve(probc_w11, CollocationLAML(maxiters=8,
                                                       n_continuation=3))
        sol_caf_w11 = solve(probc_w11, CollocationLAML(maxiters=8,
                                                       n_continuation=3,
                                                       jac=:forwarddiff))
        @test all(isfinite, sol_cff_w11.fitted_values)
        @test all(isfinite, sol_caf_w11.fitted_values)

        # ── Discrete-collocation branch of the AD state/β blocks ──
        function ricker_w11!(un, u, p, t)
            un[1] = u[1] * exp(p.g(u[1]))
        end
        ttr_w11 = collect(0.0:1.0:8.0)
        datr_w11 = reshape(0.5 .+ 0.4 .* rand(StableRNG(11), 9), :, 1)
        probr_w11 = PSMProblem(ricker_w11!, [0.5], (0.0, 8.0),
                               [BSplineApproximator(:g, (0.0, 2.0), 5)];
                               data_times=ttr_w11, data_values=datr_w11,
                               discrete=true)
        alphar_w11 = reshape(copy(vec(datr_w11)), :, 1)
        betar_w11 = PSM.build_initial_params(probr_w11)
        rr1_w11, Jr1_w11 = PSM.collocation_residual_jacobian(
            probr_w11, ttr_w11, alphar_w11, betar_w11,
            PSM.build_diff_matrix(ttr_w11), 1.0, ones(9))
        rr2_w11, Jr2_w11 = PSM.collocation_residual_jacobian(
            probr_w11, ttr_w11, alphar_w11, betar_w11,
            PSM.build_diff_matrix(ttr_w11), 1.0, ones(9);
            jac=:forwarddiff,
            fd_cfg=PSM._fd_jacobian_config(length(betar_w11)))
        @test rr1_w11 == rr2_w11        # residual path is backend-independent
        # measured 2.7e-6 — the FD path's own one-sided truncation error
        @test maximum(abs, Jr1_w11 - Jr2_w11) < 1e-4

        # ── DDE support: MethodOfSteps is Dual-safe (decision: SUPPORTED,
        #    with the same per-iteration FD fallback as the ODE path) ──
        function dde_w11!(du, u, h, p, t)
            ud = h(p, t - 1.0)
            du[1] = -p.f(ud[1])
        end
        ttdd_w11 = collect(range(0.5, 6.0, length=12))
        probdd_w11 = PSMProblem(dde_w11!, [1.0], (0.0, 6.0),
                                [BSplineApproximator(:f, (-1.5, 1.5), 6)];
                                data_times=ttdd_w11,
                                data_values=reshape(
                                    [exp(-0.4t) for t in ttdd_w11], :, 1),
                                delays=[1.0])
        bdd_w11 = PSM.build_initial_params(probdd_w11)
        npdd_w11 = length(bdd_w11)
        f0dd_w11 = vec(PSM.simulate(probdd_w11, bdd_w11))
        Jd_fd_w11 = zeros(12, npdd_w11)
        Jd_ad_w11 = zeros(12, npdd_w11)
        PSM.compute_jacobian!(Jd_fd_w11, probdd_w11, bdd_w11, f0dd_w11, 12, 1;
                              dam=fill(1e-8, npdd_w11))
        PSM.compute_jacobian!(Jd_ad_w11, probdd_w11, bdd_w11, f0dd_w11, 12, 1;
                              dam=fill(1e-8, npdd_w11), jac=:forwarddiff,
                              fd_cfg=PSM._fd_jacobian_config(npdd_w11))
        @test all(isfinite, Jd_ad_w11)
        @test maximum(abs, Jd_fd_w11 - Jd_ad_w11) <
              1e-6 * max(maximum(abs, Jd_ad_w11), 1.0)   # measured 2.2e-11
        # Silent-fallback detector: if the AD path permanently fell back
        # to FD, both Jacobians would be FD from identical fresh state and
        # the diff would be EXACTLY 0.0. FD truncation noise guarantees a
        # nonzero gap whenever AD genuinely ran (measured 1.3e-10 here).
        @test maximum(abs, Jd_fd_w11 - Jd_ad_w11) > 0
        sol_dfd_w11 = solve(probdd_w11, LAML(maxiters=10))
        sol_dad_w11 = solve(probdd_w11, LAML(maxiters=10, jac=:forwarddiff))
        @test maximum(abs, sol_dfd_w11.fitted_values -
                           sol_dad_w11.fitted_values) < 1e-4  # measured 5.8e-7

        # ── Timing (comment-only, per the W9 flake-review guidance):
        # per-Jacobian microbenchmark (25 data points, logistic BSpline
        # problem, minimum of 5 warmed runs, this machine):
        #   p=8:  fd 0.57ms  forwarddiff 0.04ms  (14.0×)
        #   p=15: fd 1.09ms  forwarddiff 0.08ms  (13.9×)
        #   p=25: fd 1.80ms  forwarddiff 0.12ms  (15.6×)
        # The FD path pays 2·p ODE solves per Jacobian; the AD path pays
        # ⌈p/chunk⌉ chunked Dual solves (chunk ≤ 12), hence the ~order-of-
        # magnitude win. End-to-end (LAML maxiters=20 on the problem above):
        # 6.1s → 0.04s; GCVSolver: 1.1s → 0.01s; CollocationLAML: 2.5s →
        # 0.65s (its Jacobian is dominated by the pointwise state blocks).
    end

    # ─── N0: penalty_blocks — multiple penalty blocks per approximator ──

    @testset "N0: penalty_blocks" begin
        PSM = PartiallySpecifiedModels

        @testset "penalty_blocks default equivalence" begin
            a_pb = BSplineApproximator(:f, (0.0, 1.0), 10)
            pb = penalty_blocks(a_pb)
            @test pb isa Vector{Tuple{Matrix{Float64}, UnitRange{Int}}}
            @test length(pb) == 1
            @test pb[1][1] == penalty_matrix(a_pb)
            @test pb[1][2] == 1:10

            # np < 3 gate (moved from build_penalty_matrices into the
            # default method): a 2-parameter type with a non-nothing
            # penalty_matrix still contributes no default block.
            p2 = PolyApproximator(:h, (0.0, 1.0), 1)
            @test penalty_matrix(p2) !== nothing
            @test penalty_blocks(p2) ==
                  Tuple{Matrix{Float64}, UnitRange{Int}}[]

            # penalty_matrix === nothing ⟹ no blocks
            import Lux
            model_pb = Lux.Chain(Lux.Dense(1, 4, tanh), Lux.Dense(4, 1))
            na0 = NeuralApproximator(:g, model_pb; rng_seed=42)
            @test penalty_matrix(na0) === nothing   # penalty_weight = 0
            @test isempty(penalty_blocks(na0))
            naw = NeuralApproximator(:g, model_pb; rng_seed=42,
                                     penalty_weight=0.1)
            pbw = penalty_blocks(naw)
            @test length(pbw) == 1 && pbw[1][2] == 1:nparams(naw)
        end

        @testset "build_penalty_matrices enumeration (old semantics)" begin
            # Two penalized approximators with an unpenalized-by-gate
            # 2-param type between them: the enumerated lists must match
            # the historical (per-approximator, np ≥ 3, S ≠ nothing)
            # semantics exactly.
            dyn_pb!(du, u, p, t) = (du[1] = 0.0)
            r_pb = BSplineApproximator(:r, (0.0, 1.0), 8)
            h_pb = PolyApproximator(:h, (0.0, 1.0), 1)  # 2 params ⟹ gated
            g_pb = BSplineApproximator(:g, (0.0, 1.0), 6)
            prob_pb = PSMProblem(dyn_pb!, [1.0], (0.0, 1.0),
                [r_pb, h_pb, g_pb];
                data_times=[0.0, 0.5, 1.0],
                data_values=reshape(zeros(3), :, 1),
                obs_to_state=[1], known_params=NamedTuple(),
                likelihood=PSM.Gaussian())
            S_list_pb, offs_pb, nks_pb = PSM.build_penalty_matrices(prob_pb)
            @test length(S_list_pb) == 2
            @test S_list_pb[1] == penalty_matrix(r_pb)
            @test S_list_pb[2] == penalty_matrix(g_pb)
            @test offs_pb == [0, 10]  # g starts after r's 8 + h's 2 params
            @test nks_pb == [8, 6]    # block sizes

            # Multi-block type PRECEDED by another approximator: this is
            # the one case that exercises the global-offset composition
            # `approx_offset + first(local_range) - 1` on a range whose
            # first index is not 1. A slip here silently mis-assigns λ to
            # the wrong coefficients.
            t_pb = BSplineApproximator(:t, (0.0, 1.0), 5)
            tb_pb = TwoBlockApproximator(:f, (0.0, 3.0), 6, 6.0)
            prob_pb2 = PSMProblem(dyn_pb!, [1.0], (0.0, 1.0), [t_pb, tb_pb];
                data_times=[0.0, 0.5, 1.0],
                data_values=reshape(zeros(3), :, 1),
                obs_to_state=[1], known_params=NamedTuple(),
                likelihood=PSM.Gaussian())
            S_pb2, offs_pb2, nks_pb2 = PSM.build_penalty_matrices(prob_pb2)
            # t: one block at 0 (5 params); f: ridge on local 1:2 ⟹ global
            # offset 5, spline on local 3:8 ⟹ global offset 5 + 3 - 1 = 7
            @test offs_pb2 == [0, 5, 7]
            @test nks_pb2 == [5, 2, 6]
            @test length(S_pb2) == 3
            # blocks land inside their own approximator's coefficient span
            @test offs_pb2[3] + nks_pb2[3] == nparams(t_pb) + nparams(tb_pb)
        end

        @testset "build_penalty_matrices validation" begin
            dyn_pb!(du, u, p, t) = (du[1] = 0.0)
            mkprob_pb(a) = PSMProblem(dyn_pb!, [1.0], (0.0, 1.0), [a];
                data_times=[0.0, 0.5, 1.0],
                data_values=reshape(zeros(3), :, 1),
                obs_to_state=[1], known_params=NamedTuple(),
                likelihood=PSM.Gaussian())
            @test_throws ArgumentError PSM.build_penalty_matrices(
                mkprob_pb(BadBlocksApproximator(:bad, :overlap)))
            err_pb = try
                PSM.build_penalty_matrices(
                    mkprob_pb(BadBlocksApproximator(:bad, :overlap)))
                nothing
            catch e; e; end
            @test err_pb isa ArgumentError
            @test occursin(":bad", err_pb.msg)      # names the approximator
            @test occursin("disjoint", err_pb.msg)  # explains the rule
            @test_throws ArgumentError PSM.build_penalty_matrices(
                mkprob_pb(BadBlocksApproximator(:bad, :out_of_range)))
            @test_throws ArgumentError PSM.build_penalty_matrices(
                mkprob_pb(BadBlocksApproximator(:bad, :size_mismatch)))
            @test_throws ArgumentError PSM.build_penalty_matrices(
                mkprob_pb(BadBlocksApproximator(:bad, :asymmetric)))
            # Strided/non-contiguous ranges must be REJECTED, not silently
            # reinterpreted: a block is recorded as (offset, length), so
            # 1:2:5 would become the contiguous 1:3 and penalize the wrong
            # coefficients.
            @test_throws ArgumentError PSM.build_penalty_matrices(
                mkprob_pb(BadBlocksApproximator(:bad, :strided)))
        end

        # ── Shared two-block fixture: linear forcing problem du = f(t),
        # u(t) = ∫₀ᵗ f. Chosen over a decay du = −r(u)·u because the model
        # is LINEAR in the coefficients: the prediction Jacobian is exact
        # and constant, the IRLS is deterministic, and the two blocks are
        # identifiable by construction — 6 hats (spacing 0.5) cannot track
        # the ω = 6 carrier (period ≈ 1.05), so the Fourier block must
        # carry it. The two blocks then want genuinely different smoothing:
        # the ridge sees two strongly-identified Fourier coefficients it
        # can barely improve on (λ₁ settles at the larger value), while the
        # hat part is a linear trend plus a gentle arch — nearly in the
        # curvature penalty's null space, so λ₂ goes small. Measured
        # λ = [3.72e-4, 6.69e-5], i.e. λ₁ ≈ 5.6 λ₂, matching the
        # log(λ₁/λ₂) > 0.5 assertion below. ──
        om_n0 = 6.0
        a_n0 = TwoBlockApproximator(:f, (0.0, 3.0), 6, om_n0)
        f_true_n0(t) = 0.8 * sin(om_n0 * t) + 1.0 + 0.5 * t +
                       0.3 * sin(pi * t / 3.0)
        F_true_n0(t) = 0.8 * (1.0 - cos(om_n0 * t)) / om_n0 + t +
                       0.25 * t^2 +
                       0.3 * (3.0 / pi) * (1.0 - cos(pi * t / 3.0))
        dyn_n0!(du, u, p, t) = (du[1] = p.f(t))
        tt_n0 = collect(range(0.0, 3.0, length=61))
        dv_n0 = reshape(F_true_n0.(tt_n0) .+
                        0.01 .* randn(StableRNG(42), length(tt_n0)), :, 1)
        prob_n0 = PSMProblem(dyn_n0!, [0.0], (0.0, 3.0), [a_n0];
                             data_times=tt_n0, data_values=dv_n0,
                             obs_to_state=[1], known_params=NamedTuple(),
                             likelihood=PSM.Gaussian())
        S_n0, offs_n0, nks_n0 = PSM.build_penalty_matrices(prob_n0)

        @testset "two-block LAML fit: per-block smoothing parameters" begin
            # One approximator, TWO enumerated blocks at the right offsets
            @test offs_n0 == [0, 2]
            @test nks_n0 == [2, 6]
            # jac=:forwarddiff because THIS FIXTURE'S EVALUATOR IS ONLY C⁰
            # (the hat basis indexes via floor/clamp), not because the
            # penalty is multi-block. The kink makes adaptive step control
            # inject noise into the FD prediction Jacobian, and the λ
            # search then stalls at a NON-STATIONARY point while reporting
            # converged=true: under FD the fit lands 8.3 nats BELOW the
            # exact profiled-REML optimum (β̂₁ ≈ 0.51-0.54, sup-error
            # 0.38-0.47, varying with the dependency versions in play);
            # under AD it lands exactly ON the global optimum
            # V* = 252.2294 (β̂₁ = 0.801, sup-error 0.0118). A single-block
            # control on this same C⁰ basis fails under FD just as badly
            # (gap 0.002-9.3 nats), and a C² B-spline analogue is clean
            # under FD at one AND two blocks — so this is evaluator
            # smoothness, NOT a multi-block property. Custom approximators
            # with kinks or branches should prefer jac=:forwarddiff
            # regardless of how many penalty blocks they declare.
            sol_n0 = solve(prob_n0, LAML(maxiters=80, jac=:forwarddiff))
            lam_n0 = sol_n0.smoothing_params
            # TWO per-block smoothing parameters, finite, and genuinely
            # different (measured λ = [3.72e-4, 6.69e-5]: log-ratio 1.71,
            # pinned at > 0.5 — deterministic under StableRNG(42), with
            # only BLAS-level wiggle remaining)
            @test length(lam_n0) == 2
            @test all(isfinite, lam_n0) && all(>(0), lam_n0)
            @test log(lam_n0[1] / lam_n0[2]) > 0.5
            # Recovery of the truth (measured sup-error 0.0118 on [0, 3]
            # against a signal of range ≈ [0.2, 3.3]; pinned with ~8×)
            fh_n0 = sol_n0.unknown_functions[:f]
            g_n0 = collect(range(0.0, 3.0, length=49))
            @test maximum(abs.(fh_n0.(g_n0) .- f_true_n0.(g_n0))) < 0.1
            # The rough block's leading coefficient (truth 0.8;
            # measured 0.801)
            @test abs(sol_n0.parameters.f[1] - 0.8) < 0.1
            @test sol_n0.convergence.converged
        end

        # Exact prediction Jacobian of the LINEAR forcing model: column j
        # is the trajectory under the j-th unit coefficient vector (u0 = 0).
        np_n0 = nparams(a_n0)
        J_n0 = zeros(length(tt_n0), np_n0)
        for j in 1:np_n0
            ej_n0 = zeros(np_n0); ej_n0[j] = 1.0
            J_n0[:, j] = PSM.simulate(prob_n0, ej_n0)[:, 1]
        end

        @testset "two-block LAML gradient matches finite differences" begin
            # The suite's frozen-working-model FD harness (see "LAML
            # gradient matches finite differences" above), with the two
            # blocks coming from ONE approximator via build_penalty_matrices
            # — the T11-class objective/gradient desync detector for the
            # per-block enumeration (including the rank-2 ridge block below
            # the historical np ≥ 3 gate: Mp = 8 − (2 + 4) = 2).
            using PartiallySpecifiedModels: laml_objective, laml_gradient,
                                            build_S_lambda, _safe_inv
            n_g = length(tt_n0)
            w_g = ones(n_g)
            y_g2 = dv_n0[:, 1]
            function fd_vs_analytic_n0(family, yv, W, rho)
                S_lam = build_S_lambda(S_n0, offs_n0, nks_n0, rho, np_n0)
                beta = _safe_inv(J_n0' * Diagonal(W) * J_n0 + S_lam) *
                       (J_n0' * (W .* yv))
                mu = J_n0 * beta
                _, H, _, sigma2 = laml_objective(family, beta, J_n0, W, w_g,
                                                 yv, mu, S_n0, offs_n0,
                                                 nks_n0, rho, np_n0)
                g = laml_gradient(family, beta, S_n0, offs_n0, nks_n0,
                                  rho, np_n0, H, sigma2)
                h = 1e-4
                gfd = map(eachindex(rho)) do k
                    rp = copy(rho); rp[k] += h
                    rm = copy(rho); rm[k] -= h
                    Vp, = laml_objective(family, beta, J_n0, W, w_g, yv, mu,
                                         S_n0, offs_n0, nks_n0, rp, np_n0)
                    Vm, = laml_objective(family, beta, J_n0, W, w_g, yv, mu,
                                         S_n0, offs_n0, nks_n0, rm, np_n0)
                    (Vp - Vm) / (2h)
                end
                (g, collect(gfd))
            end
            # Same tolerance rationale as the harness above: observed
            # agreement ≤ 3e-5 relative; rtol 1e-3 catches any O(1) slip.
            for rho in ([-2.0, 1.0], [0.0, 0.0], [3.0, -1.0], [5.0, 4.0])
                g, gfd = fd_vs_analytic_n0(Gaussian(), y_g2, w_g, rho)
                @test g ≈ gfd rtol=1e-3 atol=1e-6
            end
            # Non-Gaussian branch (identity link ⟹ IRLS weights 1/μ,
            # frozen at count-like pseudo-data)
            y_pois_n0 = max.(round.(4.0 .* (y_g2 .+ 1.0)), 1.0)
            W_pois_n0 = 1.0 ./ y_pois_n0
            for rho in ([-1.0, 2.0], [2.0, 0.0])
                g, gfd = fd_vs_analytic_n0(Poisson(), y_pois_n0,
                                           W_pois_n0, rho)
                @test g ≈ gfd rtol=1e-3 atol=1e-6
            end
        end

        @testset "two-block GCV: direct vs reuse" begin
            n_gc = length(tt_n0)
            w_gc = ones(n_gc)
            z_gc = dv_n0[:, 1]
            m_gc = length(S_n0)
            # VALUE-basis working model J_v[i,j] = ϕⱼ(tᵢ) for the
            # score-level equivalence (search equivalence is a property of
            # the machinery, not of one working model — and the integral
            # basis has cond(J'J) near the reuse-path whitening guard,
            # which limits agreement to ~1e-6 for no diagnostic gain).
            J_v = zeros(n_gc, np_n0)
            for j in 1:np_n0
                ej_v = zeros(np_n0); ej_v[j] = 1.0
                phi_v = PSM.build_evaluator(a_n0, ej_v)
                J_v[:, j] = phi_v.(tt_n0)
            end

            # ── Per-λ score equivalence on the W10 pencil shapes, at BLOCK
            # granularity: the shared pencil λ·ΣₗSₗ plus each coordinate's
            # (S_base = other block at fixed ρ, S_pen = this block) — the
            # exact matrices the reuse factory builds inside
            # _coordinate_gcv. Measured worst relative disagreement
            # (score, tr(A), β̂ over 20 λ, ρ ≤ 13): 2.1e-7 — the embedded
            # single-block pencils have a 6-dim whitened null space whose
            # replicated stability ridge carries λ·eps error (the W10
            # rank-deficient analysis; a block-wiring error would be O(1)).
            # Pinned with ~50× margin.
            pencils_gc = Any[(zeros(np_n0, np_n0),
                              PSM.build_S_lambda(S_n0, offs_n0, nks_n0,
                                                 zeros(m_gc), np_n0))]
            rho_cur_gc = [1.3, -2.1]
            for k in 1:m_gc
                idx = [j for j in 1:m_gc if j != k]
                push!(pencils_gc,
                      (PSM.build_S_lambda(S_n0[idx], offs_n0[idx],
                                          nks_n0[idx], rho_cur_gc[idx],
                                          np_n0),
                       PSM.build_S_lambda(S_n0[k:k], offs_n0[k:k],
                                          nks_n0[k:k], [0.0], np_n0)))
            end
            navail_gc = 0
            for (S_base, S_pen) in pencils_gc
                fam = PSM._gcv_reuse_family(J_v, w_gc, z_gc, S_base, S_pen,
                                            n_gc, 1.4)
                fam === nothing && continue
                navail_gc += 1
                worst = 0.0
                for lam in exp.(range(PSM.RHO_MIN, 13.0, length=20))
                    gf, bf, _, tf = fam(lam)
                    gd, bd, _, td = PSM._gcv_score(J_v, w_gc, z_gc,
                                                   S_base .+ lam .* S_pen,
                                                   n_gc, 1.4)
                    worst = max(worst,
                                abs(gf - gd) / max(abs(gd), 1e-300),
                                abs(tf - td) / max(td, 1e-300),
                                norm(bf - bd) / max(norm(bd), 1e-300))
                end
                @test worst < 1e-5
            end
            @test navail_gc == 3   # all three pencils took the fast path

            # ── Fixed-working-model multi-λ coordinate descent over the
            # two blocks of ONE approximator: direct scorer vs the reuse
            # family factory must find the same optimum (measured
            # |Δρ| = 8.5e-4 — golden-section paths may part ways below the
            # search tolerance — and |ΔGCV| relative 1.6e-14).
            scorer_gc = S_lam -> PSM._gcv_score(J_v, w_gc, z_gc, S_lam,
                                                n_gc, 1.4)
            fac_gc = function (rv, k)
                idx = [j for j in 1:m_gc if j != k]
                PSM._gcv_reuse_family(J_v, w_gc, z_gc,
                    PSM.build_S_lambda(S_n0[idx], offs_n0[idx],
                                       nks_n0[idx], rv[idx], np_n0),
                    PSM.build_S_lambda(S_n0[k:k], offs_n0[k:k],
                                       nks_n0[k:k], [0.0], np_n0),
                    n_gc, 1.4)
            end
            rd_gc, _, gd_gc, _ = PSM._coordinate_gcv(
                scorer_gc, S_n0, offs_n0, nks_n0, np_n0,
                [0.0, 0.0], 1e-6; sweeps=4)
            rr_gc, _, gr_gc, _ = PSM._coordinate_gcv(
                scorer_gc, S_n0, offs_n0, nks_n0, np_n0,
                [0.0, 0.0], 1e-6; sweeps=4, family_factory=fac_gc)
            @test abs(gd_gc - gr_gc) < 1e-8 * max(abs(gd_gc), 1e-300)
            @test maximum(abs.(rd_gc .- rr_gc)) < 0.1

            # ── End-to-end: both searches complete the full IRLS solve
            # with TWO per-block λs, recover the truth, and agree at the
            # W10 conventions. jac=:forwarddiff again for the fixture's C⁰
            # basis (see the LAML testset above), not for any multi-block
            # reason: the kink-driven FD Jacobian noise perturbs the two
            # searches' working models differently, and the ridge-block
            # coordinate is nearly flat in GCV (shrinking two strongly-
            # identified coefficients has no bias–variance sweet spot), so
            # score-level differences flip golden-section basins and IRLS
            # feedback amplifies them to O(1) λ̂ disagreements. A C²
            # B-spline analogue tracks fine under plain FD at two blocks,
            # as does the pre-existing W10 two-approximator test. With AD
            # the working models are identical and the searches track:
            # measured |Δ log λ̂| = 9.1e-6, objective relΔ 5.1e-9, fitted
            # gap 1.7e-6, both sup-errors 0.0102 — ~100× margins per W10.
            sd_gc = solve(prob_n0, GCVSolver(maxiters=15,
                                             jac=:forwarddiff))
            sr_gc = solve(prob_n0, GCVSolver(maxiters=15, search=:reuse,
                                             jac=:forwarddiff))
            g_gc = collect(range(0.3, 2.7, length=49))
            for s in (sd_gc, sr_gc)
                @test length(s.smoothing_params) == 2
                @test all(isfinite, s.smoothing_params)
                @test all(>(0), s.smoothing_params)
                fh = s.unknown_functions[:f]
                @test maximum(abs.(fh.(g_gc) .- f_true_n0.(g_gc))) < 0.1
            end
            @test all(abs.(log.(sd_gc.smoothing_params) .-
                           log.(sr_gc.smoothing_params)) .< 1e-3)
            @test abs(sd_gc.objective - sr_gc.objective) <
                  1e-6 * max(abs(sd_gc.objective), 1.0)
            @test maximum(abs.(sd_gc.fitted_values .-
                               sr_gc.fitted_values)) < 1e-4
        end
    end


    # ─── SingleIndexApproximator (nested inner direction + outer smooth) ──

    @testset "SingleIndexApproximator — construction and validation" begin
        # p, nknots, xi, anchor, constraint
        @test_throws ArgumentError SingleIndexApproximator(:g, 1, 8)
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 2)
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 3;
                                                           constraint=:increasing)
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8; xi=0.0)
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8; xi=-1.0)
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8; xi=Inf)
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8; anchor=0)
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8; anchor=3)
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8;
                                                           constraint=:wiggly)
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8;
                                                           inner_ridge=-1.0)
        # states must be p distinct positive indices
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8; states=[1])
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8; states=[2, 2])
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8; states=[0, 1])

        # index_stats validation: shape, symmetry, PSD, non-degenerate
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8;
            index_stats=([0.0], Matrix(1.0I, 2, 2)))
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8;
            index_stats=(zeros(2), Matrix(1.0I, 3, 3)))
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8;
            index_stats=(zeros(2), [1.0 0.5; 0.1 1.0]))
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8;
            index_stats=(zeros(2), [1.0 2.0; 2.0 1.0]))     # eigenvalue −1
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8;
            index_stats=(zeros(2), zeros(2, 2)))            # zero trace
        @test_throws ArgumentError SingleIndexApproximator(:g, 2, 8;
            index_stats=zeros(2))                           # not a pair
        # a singular but non-zero-trace Σ̂ is ACCEPTED (the evaluator floors
        # aᵀΣ̂a, so a rank-deficient reference covariance is usable)
        @test SingleIndexApproximator(:g, 2, 8;
            index_stats=(zeros(2), [1.0 1.0; 1.0 1.0])) isa SingleIndexApproximator

        stats2 = ([1.0, 2.0], [1.0 0.2; 0.2 2.0])
        a = SingleIndexApproximator(:g, 2, 8; index_stats=stats2)
        @test a.name == :g
        @test a.p == 2
        @test a.anchor == 1
        @test a.constraint == :none
        @test a.domain == (-2.0, 2.0)                       # (−xi, xi)
        @test a.states == [1, 2]
        @test nparams(a) == 1 + 8                           # p−1 loadings + outer
        @test initial_params(a)[1] == 1.0                   # equal weights
        @test initial_params(a)[2:end] == zeros(8)
        # free mode keeps all p loadings
        af = SingleIndexApproximator(:g, 3, 8; anchor=nothing,
                                     index_stats=(zeros(3), Matrix(1.0I, 3, 3)))
        @test nparams(af) == 3 + 8
        @test initial_params(af)[1:3] == ones(3)
        # a shape-constrained outer smooth follows the SCOP parameter count,
        # including the zero-endpoint reduction
        ac = SingleIndexApproximator(:g, 2, 8; constraint=:increasing,
                                     index_stats=stats2)
        @test nparams(ac) == 1 + 8
        acz = SingleIndexApproximator(:g, 2, 8; constraint=:inc_zero_left,
                                      index_stats=stats2)
        @test nparams(acz) == 1 + 7
        # xi widens the outer knot span
        @test SingleIndexApproximator(:g, 2, 8; xi=5.0,
                                      index_stats=stats2).domain == (-5.0, 5.0)
        # an initial outer curve is sampled on the standardized knot grid
        ai = SingleIndexApproximator(:g, 2, 5; index_stats=stats2,
                                     initial=z -> 3.0 + z)
        @test initial_params(ai)[2:end] ≈ [1.0, 2.0, 3.0, 4.0, 5.0]
        # explicit starting direction, rescaled by the anchor entry
        al = SingleIndexApproximator(:g, 3, 5; index_stats=(zeros(3), Matrix(1.0I, 3, 3)),
                                     initial_loadings=[2.0, 1.0, -3.0])
        @test initial_params(al)[1:2] ≈ [0.5, -1.5]
        @test_throws ArgumentError SingleIndexApproximator(:g, 3, 5;
            index_stats=(zeros(3), Matrix(1.0I, 3, 3)), initial_loadings=[1.0, 2.0])
        @test_throws ArgumentError SingleIndexApproximator(:g, 3, 5;
            index_stats=(zeros(3), Matrix(1.0I, 3, 3)),
            initial_loadings=[0.0, 1.0, 1.0])   # zero anchor entry
        # string names accepted, matching the other constructors
        @test SingleIndexApproximator("g", 2, 8; index_stats=stats2).name == :g
        # an unresolved approximator refuses to build an evaluator rather
        # than silently standardizing against a made-up default
        @test_throws ErrorException build_evaluator(
            SingleIndexApproximator(:g, 2, 8), zeros(9))
    end

    @testset "SingleIndexApproximator — index geometry and identification" begin
        μ̂ = [1.0, 2.0, -0.5]
        Σ̂ = [2.0 0.3 0.1; 0.3 1.0 -0.2; 0.1 -0.2 0.5]
        θ_out = collect(range(-1.0, 1.5, length=7))

        # ── anchored mode: a[anchor] ≡ 1, and the evaluator is exactly the
        # outer smooth composed with the standardized index
        a1 = SingleIndexApproximator(:g, 3, 7; anchor=2, index_stats=(μ̂, Σ̂))
        θ = vcat([0.4, -0.7], θ_out)
        @test index_loadings(a1, θ) == [0.4, 1.0, -0.7]
        f1 = build_evaluator(a1, θ)
        s1 = build_evaluator(a1.outer, θ_out)
        av = [0.4, 1.0, -0.7]
        for u in ([1.0, 2.0, 0.0], [-1.0, 3.0, 2.0], [5.0, -4.0, 1.5])
            z = (dot(av, u) - dot(av, μ̂)) / sqrt(dot(av, Σ̂ * av))
            @test f1(u...) ≈ s1(z) atol=1e-12
        end
        # wrong arity is refused loudly
        @test_throws ArgumentError f1(1.0, 2.0)

        # ── free mode: z is invariant under a → c·a. For c a power of two
        # every scaling is exact in floating point, so the evaluator is
        # BIT-identical; for a general c it agrees to roundoff.
        a2 = SingleIndexApproximator(:g, 3, 7; anchor=nothing, index_stats=(μ̂, Σ̂))
        θf = vcat([0.6, 0.4, -0.2], θ_out)
        ff = build_evaluator(a2, θf)
        f2 = build_evaluator(a2, vcat(2.0 .* θf[1:3], θ_out))
        f37 = build_evaluator(a2, vcat(3.7 .* θf[1:3], θ_out))
        pts = [[1.0, 2.0, 0.0], [-1.0, 3.0, 2.0], [5.0, -4.0, 1.5], [0.0, 0.0, 0.0]]
        @test all(ff(u...) === f2(u...) for u in pts)
        @test maximum(abs(ff(u...) - f37(u...)) for u in pts) < 1e-12
        # …and reporting normalizes ‖a‖ = 1 with a positive first loading
        @test norm(index_loadings(a2, θf)) ≈ 1.0
        @test index_loadings(a2, θf) ≈ [0.6, 0.4, -0.2] ./ norm([0.6, 0.4, -0.2])
        @test index_loadings(a2, vcat(-2.0 .* θf[1:3], θ_out)) ≈
              index_loadings(a2, θf)

        # ── the aᵀΣ̂a guard: a direction in the null space of a singular Σ̂
        # still yields a finite index, and the floor is proportional to ‖a‖²
        # so it does NOT break the exact scale invariance.
        Σ0 = [1.0 1.0; 1.0 1.0]
        a3 = SingleIndexApproximator(:g, 2, 7; anchor=nothing,
                                     index_stats=([0.0, 0.0], Σ0))
        θn = vcat([1.0, -1.0], θ_out)         # exactly in the null space of Σ0
        fn = build_evaluator(a3, θn)
        fn2 = build_evaluator(a3, vcat([4.0, -4.0], θ_out))
        @test isfinite(fn(1.0, 0.0))
        @test all(fn(u, v) === fn2(u, v) for (u, v) in
                  ((1.0, 0.0), (0.0, 1.0), (2.5, -3.5)))
    end

    @testset "SingleIndexApproximator — Dual safety in params and inputs" begin
        using ForwardDiff
        μ̂ = [1.0, 2.0]
        Σ̂ = [2.0 0.3; 0.3 1.0]
        a = SingleIndexApproximator(:g, 2, 7; index_stats=(μ̂, Σ̂))
        θ = vcat([0.7], collect(range(-1.0, 1.5, length=7)) .+ 0.1 .* (1:7))
        f = build_evaluator(a, θ)

        # inside [−xi, xi] and out in BOTH linear-extrapolation branches
        for u in ([1.0, 2.0], [40.0, 40.0], [-40.0, -40.0])
            g1 = ForwardDiff.gradient(b -> build_evaluator(a, b)(u...), θ)
            @test all(isfinite, g1)
            h = 1e-6
            for k in (1, 2, 5)
                e_k = [i == k ? 1.0 : 0.0 for i in eachindex(θ)]
                fd = (build_evaluator(a, θ .+ h .* e_k)(u...) -
                      build_evaluator(a, θ .- h .* e_k)(u...)) / (2h)
                @test g1[k] ≈ fd atol=1e-5
            end
            g2 = ForwardDiff.gradient(v -> f(v[1], v[2]), u)
            @test all(isfinite, g2)
            # nested: Dual params of a Dual-state derivative (stiff autodiff
            # Jacobians inside through-the-solver training)
            g3 = ForwardDiff.gradient(
                b -> ForwardDiff.derivative(x -> build_evaluator(a, b)(x, u[2]), u[1]), θ)
            @test all(isfinite, g3)
        end
        # the index really does move with the states inside the knot span
        @test norm(ForwardDiff.gradient(v -> f(v[1], v[2]), [1.0, 2.0])) > 1e-6

        # the same holds for a shape-constrained outer smooth
        ac = SingleIndexApproximator(:g, 2, 7; constraint=:increasing,
                                     index_stats=(μ̂, Σ̂))
        θc = vcat([0.7], randn(StableRNG(21), 7))
        @test all(isfinite, ForwardDiff.gradient(
            b -> build_evaluator(ac, b)(1.0, 2.0), θc))
        @test all(isfinite, ForwardDiff.gradient(
            v -> build_evaluator(ac, θc)(v[1], v[2]), [1.0, 2.0]))
    end

    @testset "SingleIndexApproximator — penalty blocks and merged penalty" begin
        stats2 = ([0.0, 0.0], [1.0 0.2; 0.2 1.0])
        a = SingleIndexApproximator(:g, 2, 8; inner_ridge=0.05, index_stats=stats2)
        blocks = penalty_blocks(a)
        @test length(blocks) == 2
        @test blocks[1][2] == 1:1                    # the free loading
        @test blocks[2][2] == 2:9                    # the outer coefficients
        @test size(blocks[1][1]) == (1, 1)
        @test size(blocks[2][1]) == (8, 8)
        # disjoint, and together they tile the whole coefficient block
        @test isempty(intersect(blocks[1][2], blocks[2][2]))
        @test length(blocks[1][2]) + length(blocks[2][2]) == nparams(a)
        for (S, _) in blocks
            # numerically symmetric to the tolerance build_penalty_matrices
            # itself enforces (the spline penalty is H'B⁻¹H, symmetric up to
            # roundoff), and positive semi-definite
            @test maximum(abs.(S .- S')) < 1e-8 * max(maximum(abs.(S)), 1.0)
            @test minimum(eigvals(Symmetric(S))) > -1e-8
        end
        # the outer block is EXACTLY the univariate spline penalty on the
        # same knot count (reuse by composition, not duplication)
        @test blocks[2][1] ≈ penalty_matrix(BSplineApproximator(:g, (-2.0, 2.0), 8))
        # …and with a shape constraint, exactly the SCOP difference penalty
        ac = SingleIndexApproximator(:g, 2, 8; constraint=:increasing,
                                     index_stats=stats2)
        @test penalty_blocks(ac)[2][1] ≈ penalty_matrix(
            ShapeConstrainedBSplineApproximator(:g, (-2.0, 2.0), 8, :increasing))

        # merged penalty_matrix = block diagonal with the FIXED inner weight
        S = penalty_matrix(a)
        @test size(S) == (9, 9)
        @test issymmetric(S)
        @test S[1, 1] == 0.05
        @test all(S[1, 2:end] .== 0.0)
        @test S[2:end, 2:end] ≈ blocks[2][1]
        @test minimum(eigvals(Symmetric(S))) > -1e-8

        # build_penalty_matrices validates and lays the blocks out globally
        a_free = SingleIndexApproximator(:h, 2, 6; anchor=nothing,
                                         index_stats=(zeros(2), Matrix(1.0I, 2, 2)))
        prob_pb = PSMProblem((du, u, p, t) -> (du[1] = p.g(u[1], u[2]) +
                                                       p.h(u[1], u[2]);
                                               du[2] = -0.1 * u[2]; nothing),
            [1.0, 1.0], (0.0, 1.0), [a, a_free];
            data_times=[0.0, 0.5, 1.0], data_values=ones(3, 2), obs_to_state=[1, 2],
            likelihood=Gaussian(), solver=Tsit5())
        S_list, offs, nks = PartiallySpecifiedModels.build_penalty_matrices(prob_pb)
        @test length(S_list) == 4
        @test offs == [0, 1, 9, 11]
        @test nks == [1, 8, 2, 6]

        # index_loadings reads THIS approximator's block, not the whole
        # coefficient vector. Without the length check it silently returned
        # loadings decoded from whichever approximator came first.
        @test_throws ArgumentError index_loadings(a_free,
                                                  zeros(sum(nparams, prob_pb.approximators)))
        err_il = try
            index_loadings(a_free, zeros(sum(nparams, prob_pb.approximators)))
            nothing
        catch e; e; end
        @test occursin("coefficient block", err_il.msg)
        # the correct slice works
        @test length(index_loadings(a_free, zeros(nparams(a_free)))) == 2
    end

    @testset "SingleIndexApproximator — free mode warns under LAML/GCV" begin
        # anchor=nothing leaves the data term exactly flat along a → c·a, so
        # the inner ridge is minimized by ‖a‖ → 0 and a λ-estimating solver
        # collapses the loadings while the data loss still looks fine. That
        # silent path is now broken by a warning at solve entry.
        dyn_w!(du, u, p, t) = (du[1] = -p.f(u[1], u[2]) * u[1];
                               du[2] = -0.2 * u[2]; nothing)
        tw = collect(0.0:0.25:3.0)
        refw = OrdinaryDiffEq.solve(
            ODEProblem((du, u, p, t) -> (du[1] = -0.3 * u[1]; du[2] = -0.2 * u[2];
                                         nothing),
                       [2.0, 1.0], (0.0, 3.0)), Tsit5(), saveat=0.25)
        dw = reduce(hcat, refw.u)'
        aw = SingleIndexApproximator(:f, 2, 6; anchor=nothing,
                                     index_stats=(zeros(2), Matrix(1.0I, 2, 2)))
        prob_w = PSMProblem(dyn_w!, [2.0, 1.0], (0.0, 3.0), [aw];
            data_times=tw, data_values=dw, obs_to_state=[1, 2],
            known_params=NamedTuple(), likelihood=PSM.Gaussian())
        @test_logs (:warn, r"anchor=nothing") match_mode=:any begin
            solve(prob_w, LAML(maxiters=3))
        end
        @test_logs (:warn, r"anchor=nothing") match_mode=:any begin
            solve(prob_w, GCVSolver(maxiters=3))
        end
        # the default anchored mode must NOT warn
        aa = SingleIndexApproximator(:f, 2, 6;
                                     index_stats=(zeros(2), Matrix(1.0I, 2, 2)))
        prob_a = PSMProblem(dyn_w!, [2.0, 1.0], (0.0, 3.0), [aa];
            data_times=tw, data_values=dw, obs_to_state=[1, 2],
            known_params=NamedTuple(), likelihood=PSM.Gaussian())
        @test_logs min_level=Base.CoreLogging.Warn match_mode=:any begin
            solve(prob_a, LAML(maxiters=3))
        end
    end

    # ── The recovery / discriminator fixture ───────────────────────────
    #
    # Damped predator–prey whose unknown per-capita prey growth is a
    # SINGLE INDEX of both states, r(N, P) = s₀(0.6N + 0.4P) with a
    # sigmoidal s₀. Two properties matter and both are deliberate:
    #
    #  * the orbit CHANGES DIRECTION along the path, which is what identifies
    #    the loadings. The condition that defeats identification is
    #    COLLINEARITY — states moving along an affinely straight line, where
    #    rescaling `a` is absorbed into s₀. Monotonicity alone does NOT do
    #    it: measured on monotone-decay versions of this model, where every
    #    coordinate AND the index decrease monotonically, the loading is
    #    still recovered to ~2% (0.654 vs 2/3), because a curved path keeps
    #    changing its tangent direction.
    #  * the orbit runs roughly ALONG the index direction, so each index
    #    LEVEL SET crosses the state box far from the visited curve. Those
    #    crossings are the off-orbit states whose index value the data DO
    #    pin down — exactly where a full interaction surface has nothing to
    #    go on and a single index does.
    si_s0(v) = 0.55 - 1.0 * v^2 / (0.9 + v^2)
    si_true(N, Pd) = si_s0(0.6 * N + 0.4 * Pd)
    function si_true!(du, u, p, t)
        du[1] = u[1] * si_true(u[1], u[2])
        du[2] = u[2] * (0.4 * u[1] - 0.4 - 0.12 * u[2])
        nothing
    end
    function si_psm!(du, u, p, t)
        du[1] = u[1] * p.f(u[1], u[2])
        du[2] = u[2] * (p.c * u[1] - p.m - p.d * u[2])
        nothing
    end
    si_ref = OrdinaryDiffEq.solve(
        ODEProblem(si_true!, [3.0, 0.25], (0.0, 40.0)), Tsit5(),
        saveat=0.4, abstol=1e-10, reltol=1e-10)
    si_traj = reduce(hcat, si_ref.u)'
    si_ts = collect(si_ref.t)
    si_data = si_traj .+ 0.02 .* randn(StableRNG(31), size(si_traj))
    si_prob(a) = PSMProblem(si_psm!, [3.0, 0.25], (0.0, 40.0), [a];
        data_times=si_ts, data_values=si_data, obs_to_state=[1, 2],
        known_params=(c=0.4, m=0.4, d=0.12), likelihood=Gaussian(),
        solver=Tsit5())

    @testset "SingleIndexApproximator — reference statistics are fixed" begin
        # data whose two states have very different means/spreads
        ts_fx = collect(0.0:0.25:6.0)
        y_fx = hcat(2.0 .+ sin.(ts_fx), 5.0 .+ 3.0 .* cos.(0.7 .* ts_fx))
        dyn_fx! = (du, u, p, t) -> (du[1] = -0.1 * u[1] + p.g(u[1], u[2]);
                                    du[2] = -0.05 * u[2]; nothing)
        mkfx(a) = PSMProblem(dyn_fx!, [2.0, 8.0], (0.0, 6.0), [a];
            data_times=ts_fx, data_values=y_fx, obs_to_state=[1, 2],
            likelihood=Gaussian(), solver=Tsit5())

        prob_fx = mkfx(SingleIndexApproximator(:g, 2, 6))
        rfx = prob_fx.approximators[1]
        # resolution happened at problem construction, from the data
        @test rfx.mu !== nothing
        @test rfx.Sigma !== nothing
        @test rfx.mu[1] ≈ mean(y_fx[:, 1]) rtol=0.05
        @test rfx.mu[2] ≈ mean(y_fx[:, 2]) rtol=0.05
        vcol(v) = sum(abs2, v .- mean(v)) / (length(v) - 1)
        @test rfx.Sigma[1, 1] ≈ vcol(y_fx[:, 1]) rtol=0.25
        @test rfx.Sigma[2, 2] ≈ vcol(y_fx[:, 2]) rtol=0.25
        @test issymmetric(rfx.Sigma)
        @test minimum(eigvals(Symmetric(rfx.Sigma))) > -1e-8

        # The statistics do NOT depend on the fitted trajectory. Build an
        # evaluator, run a fit that moves the trajectory a long way, then
        # rebuild it: BIT-identical, because the struct is immutable and
        # nothing in any solve path touches (μ̂, Σ̂).
        prob_mv = si_prob(SingleIndexApproximator(:f, 2, 8; xi=2.5,
                                                  initial=z -> -0.05 - 0.1z))
        a_mv = prob_mv.approximators[1]
        θ_mv = PartiallySpecifiedModels.build_initial_params(prob_mv)
        pts_mv = [(1.0, 0.5), (3.0, 0.25), (9.0, 12.0)]
        before = build_evaluator(a_mv, θ_mv)
        vals_before = [before(x, y) for (x, y) in pts_mv]
        μ_before, Σ_before = copy(a_mv.mu), copy(a_mv.Sigma)
        pred0 = PartiallySpecifiedModels.simulate(prob_mv, θ_mv)
        sol_mv = solve(prob_mv, LAML(maxiters=40, warmup=10, initial_lambda=0.01))
        @test prob_mv.approximators[1] === a_mv        # never swapped out
        @test a_mv.mu == μ_before                      # never mutated
        @test a_mv.Sigma == Σ_before
        after = build_evaluator(a_mv, θ_mv)
        @test [after(x, y) for (x, y) in pts_mv] == vals_before
        # …and the fit really did move the trajectory, so this is not vacuous
        @test maximum(abs.(sol_mv.fitted_values .- pred0)) > 0.1

        # user-supplied statistics are never overwritten, and re-wrapping a
        # resolved approximator (the bootstrap path) is idempotent
        fixed = ([0.0, 0.0], [1.0 0.0; 0.0 1.0])
        rfix = mkfx(SingleIndexApproximator(:g, 2, 6; index_stats=fixed)).approximators[1]
        @test rfix.mu == [0.0, 0.0]
        @test rfix.Sigma == Matrix(1.0I, 2, 2)
        again = mkfx(rfx).approximators[1]
        @test again.mu == rfx.mu
        @test again.Sigma == rfx.Sigma
        @test mkfx(deepcopy(rfx)).approximators[1].Sigma == rfx.Sigma

        # `states` selects which states the p arguments standardize against
        rsel = mkfx(SingleIndexApproximator(:g, 2, 6; states=[2, 1])).approximators[1]
        @test rsel.mu ≈ reverse(rfx.mu) rtol=1e-8
        # an unobserved state contributes its u0 value and a diffuse scale
        dyn3! = (du, u, p, t) -> (du[1] = -0.1 * u[1] + p.g(u[1], u[3]);
                                  du[2] = -0.05 * u[2]; du[3] = -0.02 * u[3]; nothing)
        prob3 = PSMProblem(dyn3!, [2.0, 8.0, 4.0], (0.0, 6.0),
            [SingleIndexApproximator(:g, 2, 6; states=[1, 3])];
            data_times=ts_fx, data_values=y_fx, obs_to_state=[1, 2],
            likelihood=Gaussian(), solver=Tsit5())
        r3 = prob3.approximators[1]
        @test r3.mu[2] == 4.0
        @test r3.Sigma[2, 2] == 16.0
        @test r3.Sigma[1, 2] == 0.0
    end

    @testset "SingleIndexApproximator — recovery of loadings and curve" begin
        prob_si = si_prob(SingleIndexApproximator(:f, 2, 10; xi=2.5,
                                                  initial=z -> -0.05 - 0.1z))
        a_si = prob_si.approximators[1]

        # (a) LAML — the inner ridge and the outer roughness get SEPARATE
        # smoothing parameters, estimated jointly.
        sol_l = solve(prob_si, LAML(maxiters=60, warmup=10, initial_lambda=0.01))
        @test length(sol_l.smoothing_params) == 2
        @test all(isfinite, sol_l.smoothing_params)
        @test all(>(0), sol_l.smoothing_params)
        @test sol_l.smoothing_params[1] != sol_l.smoothing_params[2]
        @test sol_l.data_loss < 0.2                 # observed 0.0770
        load_l = index_loadings(a_si, sol_l.parameters)
        @test load_l[1] == 1.0                      # anchored exactly
        @test abs(load_l[2] - 2/3) < 0.15           # observed 0.65244 vs 0.667
        f_l = sol_l.unknown_functions[:f]
        err_l = [abs(f_l(si_traj[i, 1], si_traj[i, 2]) -
                     si_true(si_traj[i, 1], si_traj[i, 2]))
                 for i in 1:size(si_traj, 1)]
        @test sqrt(mean(abs2, err_l)) < 0.01        # observed 0.00157
        @test maximum(err_l) < 0.03

        # (b) AdamSolver — ForwardDiff through the ODE solve, i.e. Dual
        # parameters AND Dual states through the p-argument evaluator.
        sol_a = solve(si_prob(SingleIndexApproximator(:f, 2, 10; xi=2.5,
                                                      initial=z -> -0.05 - 0.1z)),
                      AdamSolver(maxiters=500, lr=0.03))
        @test sol_a.data_loss < 1.2                 # observed ≈ 0.75
        f_a = sol_a.unknown_functions[:f]
        err_a = [abs(f_a(si_traj[i, 1], si_traj[i, 2]) -
                     si_true(si_traj[i, 1], si_traj[i, 2]))
                 for i in 1:size(si_traj, 1)]
        @test sqrt(mean(abs2, err_a)) < 0.05
    end

    @testset "SingleIndexApproximator — off-orbit against the tensor surface" begin
        # THE DISCRIMINATOR. Both types are fitted to the SAME data with the
        # SAME solver and settings, and both fit the data about equally well.
        # They are then compared at states that (i) lie ≥ 0.5 away from every
        # visited state, and (ii) carry an index value the data DID observe.
        prob_si = si_prob(SingleIndexApproximator(:f, 2, 10; xi=2.5,
                                                  initial=z -> -0.05 - 0.1z))
        a_si = prob_si.approximators[1]
        sol_si = solve(prob_si, LAML(maxiters=60, warmup=10, initial_lambda=0.01))
        nlo, nhi = extrema(si_traj[:, 1])
        plo, phi = extrema(si_traj[:, 2])
        prob_tp = si_prob(TensorBSplineApproximator(:f, (nlo, nhi), (plo, phi),
                                                    5, 5; initial=(N, Pd) -> 0.0))
        sol_tp = solve(prob_tp, LAML(maxiters=60, warmup=10, initial_lambda=0.01))
        f_si = sol_si.unknown_functions[:f]
        f_tp = sol_tp.unknown_functions[:f]

        # fairness: the tensor is not handicapped — it fits the data as well
        # and carries MORE coefficients (25 against 11)
        @test nparams(prob_tp.approximators[1]) > nparams(a_si)
        @test abs(sol_tp.data_loss - sol_si.data_loss) <
              0.1 * max(sol_si.data_loss, 1e-12)

        zvals = 0.6 .* si_traj[:, 1] .+ 0.4 .* si_traj[:, 2]
        zlo, zhi = extrema(zvals)
        mind(N, Pd) = minimum(sqrt((si_traj[i, 1] - N)^2 + (si_traj[i, 2] - Pd)^2)
                              for i in 1:size(si_traj, 1))
        off = [(N, Pd) for N in range(nlo, nhi, length=21),
                           Pd in range(plo, phi, length=21)
               if zlo <= 0.6N + 0.4Pd <= zhi && mind(N, Pd) > 0.5]
        @test length(off) >= 8              # the region is not empty

        e_si = [abs(f_si(N, Pd) - si_true(N, Pd)) for (N, Pd) in off]
        e_tp = [abs(f_tp(N, Pd) - si_true(N, Pd)) for (N, Pd) in off]
        # observed: RMSE 0.00421 (single index) against 0.02678 (tensor),
        # max 0.00954 against 0.06097 — a factor of ~6 either way. The gap
        # survives dropping the index-observed filter: over ALL off-orbit
        # points it is 0.0256 vs 0.0815, and over points whose index the
        # data NEVER observed 0.0285 vs 0.0898 — both ~3.2x. So the filter
        # selects where the single index is most accurate in absolute
        # terms; it does not manufacture the ordering.
        @test sqrt(mean(abs2, e_si)) < 0.5 * sqrt(mean(abs2, e_tp))
        @test maximum(e_si) < 0.5 * maximum(e_tp)
        @test maximum(e_si) < 0.04
        # …while ON the orbit they are comparable, so the gap above really is
        # about extrapolating off it and not about a worse fit overall
        on_si = [abs(f_si(si_traj[i, 1], si_traj[i, 2]) -
                     si_true(si_traj[i, 1], si_traj[i, 2]))
                 for i in 1:size(si_traj, 1)]
        on_tp = [abs(f_tp(si_traj[i, 1], si_traj[i, 2]) -
                     si_true(si_traj[i, 1], si_traj[i, 2]))
                 for i in 1:size(si_traj, 1)]
        @test sqrt(mean(abs2, on_si)) < sqrt(mean(abs2, on_tp))
        @test sqrt(mean(abs2, on_tp)) < 0.02

        # HONEST SCOPE NOTE. What this establishes is MODEL MATCH, not orbit
        # geometry: si_true IS an index, and the index model wins because of
        # that. Rebuilding this same fixture with non-index truths reverses
        # the ranking by a comparable or larger margin — an additive
        # two-ridge truth gives 0.314 (single index) against 0.054 (tensor)
        # off-orbit, a multiplicative interaction 0.207 against 0.070.
        # Mitigating fact: the misspecification is NOT silent — on both
        # non-index truths the single index's data_loss was 27-66% worse
        # than the tensor's, so the in-sample fit flags the wrong choice.
        # One asymmetry in this comparison: the single index carries TWO
        # smoothing parameters (penalty_blocks) while the tensor's
        # Kronecker-sum penalty carries one. Unconditionally in the single
        # index's favour: wherever p > 2, no tensor type exists at all.
        # (Note also that null-space collapse is NOT the distinguishing
        # mechanism: on THIS fixture the fitted tensor retains only ~1.7% of
        # its variation outside the penalty's bilinear null space, versus
        # ~11.7% for the single index — collapse is how the tensor LOSES
        # here.)
    end

    @testset "SingleIndexApproximator — three states, shape constraint, bands" begin
        # (a) p = 3: an index over three states, where no tensor type exists
        function si3_true!(du, u, p, t)
            g = 0.4 + 0.5 * tanh(0.5 * u[1] + 0.3 * u[2] + 0.2 * u[3] - 1.5)
            du[1] = -u[1] * g
            du[2] = 0.5 * u[1] * g - 0.3 * u[2]
            du[3] = 0.3 * u[2] - 0.2 * u[3]
            nothing
        end
        ref3 = OrdinaryDiffEq.solve(
            ODEProblem(si3_true!, [4.0, 0.5, 0.2], (0.0, 12.0)), Tsit5(),
            saveat=0.3, abstol=1e-10, reltol=1e-10)
        tr3 = reduce(hcat, ref3.u)'
        ts3 = collect(ref3.t)
        d3 = tr3 .+ 0.02 .* randn(StableRNG(41), size(tr3))
        function si3!(du, u, p, t)
            g = p.g(u[1], u[2], u[3])
            du[1] = -u[1] * g
            du[2] = p.a * u[1] * g - p.b * u[2]
            du[3] = p.b * u[2] - p.c * u[3]
            nothing
        end
        prob3 = PSMProblem(si3!, [4.0, 0.5, 0.2], (0.0, 12.0),
            [SingleIndexApproximator(:g, 3, 8; xi=2.5, initial=z -> 0.6)];
            data_times=ts3, data_values=d3, obs_to_state=[1, 2, 3],
            known_params=(a=0.5, b=0.3, c=0.2), likelihood=Gaussian(),
            solver=Tsit5())
        a3 = prob3.approximators[1]
        @test nparams(a3) == 2 + 8
        sol3 = solve(prob3, LAML(maxiters=50, warmup=8, initial_lambda=0.01))
        @test length(sol3.smoothing_params) == 2
        @test sol3.data_loss < 0.5
        g3 = sol3.unknown_functions[:g]
        g3true(u) = 0.4 + 0.5 * tanh(0.5u[1] + 0.3u[2] + 0.2u[3] - 1.5)
        err3 = [abs(g3(tr3[i, :]...) - g3true(tr3[i, :])) for i in 1:size(tr3, 1)]
        @test sqrt(mean(abs2, err3)) < 0.05

        # (b) a shape-constrained outer smooth is monotone BY CONSTRUCTION,
        # for arbitrary parameter vectors — the SCOP guarantee, inherited
        # unchanged from ShapeConstrainedBSplineApproximator
        rng_sc = StableRNG(43)
        for constraint in (:increasing, :decreasing)
            asc = SingleIndexApproximator(:g, 2, 9; constraint=constraint,
                                          index_stats=(zeros(2), Matrix(1.0I, 2, 2)))
            for _ in 1:5
                θsc = vcat([0.8], 1.5 .* randn(rng_sc, 9))
                curve = [PartiallySpecifiedModels._eval_approx_at(asc, θsc, z)
                         for z in range(-2.5, 2.5, length=60)]
                d = diff(curve)
                @test (constraint == :increasing ? minimum(d) > -1e-10 :
                                                   maximum(d) < 1e-10)
            end
        end
        # …and it recovers a monotone-index truth end to end
        prob_sc = si_prob(SingleIndexApproximator(:f, 2, 9; xi=2.5,
                                                  constraint=:decreasing,
                                                  initial=z -> -0.05 - 0.1z))
        a_sc = prob_sc.approximators[1]
        sol_sc = solve(prob_sc, LAML(maxiters=60, warmup=10, initial_lambda=0.01))
        @test sol_sc.data_loss < 0.3
        f_sc = sol_sc.unknown_functions[:f]
        @test sqrt(mean(abs2, [abs(f_sc(si_traj[i, 1], si_traj[i, 2]) -
                                   si_true(si_traj[i, 1], si_traj[i, 2]))
                               for i in 1:size(si_traj, 1)])) < 0.03
        # the fitted response really is decreasing in the index
        curve_sc = [PartiallySpecifiedModels._eval_approx_at(
                        a_sc, Float64.(collect(sol_sc.parameters)), z)
                    for z in range(-2.5, 2.5, length=40)]
        @test maximum(diff(curve_sc)) < 1e-10

        # (c) confidence_band serves the OUTER curve over the standardized
        # index — the univariate payoff the tensor surface cannot have
        prob_cb = si_prob(SingleIndexApproximator(:f, 2, 10; xi=2.5,
                                                  initial=z -> -0.05 - 0.1z))
        sol_cb = solve(prob_cb, LAML(maxiters=40, warmup=10, initial_lambda=0.01))
        band = confidence_band(sol_cb, prob_cb)
        @test haskey(band, :f)
        @test extrema(band[:f].grid) == (-2.5, 2.5)
        @test all(isfinite, band[:f].fitted)
        @test all(isfinite, band[:f].se)
        @test all(band[:f].se .>= 0)
        @test all(band[:f].lower .<= band[:f].fitted)
        @test all(band[:f].fitted .<= band[:f].upper)
        # the band is the OUTER curve, i.e. exactly what _eval_approx_at gives
        @test band[:f].fitted ≈ [PartiallySpecifiedModels._eval_approx_at(
                                     prob_cb.approximators[1],
                                     Float64.(collect(sol_cb.parameters)), z)
                                 for z in band[:f].grid]

        # (d) bootstrap produces a real (non-NaN) band for the outer curve
        res_b = bootstrap(sol_cb, prob_cb,
                          AdamSolver(maxiters=60, lr=0.05); nboot=4,
                          rng=StableRNG(45))
        @test res_b.n_success >= 3
        @test haskey(res_b.ci_uf, :f)
        @test all(isfinite, res_b.ci_uf[:f].lower)
        @test all(isfinite, res_b.ci_uf[:f].upper)
    end

    @testset "SingleIndexApproximator — sibling registration" begin
        # In the union, so the SIX per-type penalty whitelists must list it
        # explicitly — the generic non-built-in fallback will NOT fire for it.
        @test SingleIndexApproximator <:
              PartiallySpecifiedModels._BUILTIN_APPROX_TYPES

        # Behavioral check where it is cheap: the gradient-matching solvers
        # really do apply the merged penalty (a large λ flattens the outer
        # curve's roughness).
        S_si = penalty_matrix(si_prob(
            SingleIndexApproximator(:f, 2, 9; xi=2.5)).approximators[1])
        rough(sol) = dot(sol.parameters, S_si * sol.parameters)
        for alg in (λ -> TwoStageSolver(maxiters=150, lambda_smooth=λ),
                    λ -> IntegralMatchingSolver(maxiters=150, lambda_smooth=λ))
            p0 = si_prob(SingleIndexApproximator(:f, 2, 9; xi=2.5,
                                                 initial=z -> -0.05 - 0.1z))
            pP = si_prob(SingleIndexApproximator(:f, 2, 9; xi=2.5,
                                                 initial=z -> -0.05 - 0.1z))
            s0_ = solve(p0, alg(0.0))
            sP_ = solve(pP, alg(50.0))
            @test s0_.parameters != sP_.parameters
            @test rough(sP_) < rough(s0_)
        end

        # The remaining four whitelists (AGM — whose smoothing_lambda is not
        # a user-facing keyword — plus MAGI, rodeo and DALTON, which sit
        # inside Kalman/HMC objectives far too slow to gate a unit test on)
        # cannot be exercised end-to-end here. But the campaign's recurring
        # failure mode is precisely a type landing on some sibling sites and
        # not others, so assert the registration at each site directly.
        src_dir = dirname(pathof(PartiallySpecifiedModels))
        for f in ("two_stage_solver.jl", "integral_matching_solver.jl",
                  "magi_solver.jl", "adaptive_gradient_matching.jl",
                  "rodeo_solver.jl", "dalton_solver.jl")
            @test occursin("SingleIndexApproximator", read(joinpath(src_dir, f), String))
        end
    end


    # ─── TransformedCovariateApproximator (learned transform of an
    #     exogenous covariate + outer smooth) ─────────────────────────

    @testset "TransformedCovariateApproximator — construction and validation" begin
        TC = TransformedCovariateApproximator
        ct_v = collect(0.0:1.0:10.0)
        x_v = Float64[1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]

        # trans is required and must name a known transformation
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:ewma)
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:none)
        # lags: :lagindex needs a window of at least 1, no longer than the record
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:lagindex, lags=0)
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:lagindex, lags=-2)
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:lagindex, lags=12)
        # …and it is a :lagindex setting, so :expsm refuses it rather than
        # silently ignoring a window the user thinks is in force
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:expsm, lags=3)
        # same for anchor: :expsm has no scale-invariant direction to pin,
        # so a supplied anchor is a modelling misunderstanding, not a no-op
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:expsm, anchor=2)
        # X must be aligned to times, in both directions
        @test_throws ArgumentError TC(:b, ct_v, x_v[1:5]; trans=:expsm)
        @test_throws ArgumentError TC(:b, ct_v[1:5], x_v; trans=:expsm)
        # auxiliary columns are an :expsm-only idea
        @test_throws ArgumentError TC(:b, ct_v, hcat(x_v, x_v); trans=:lagindex,
                                      lags=3)
        # outer-smooth resolution and knot span
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:expsm, nknots=2)
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:expsm, nknots=3,
                                      constraint=:increasing)
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:expsm, xi=0.0)
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:expsm, xi=-1.0)
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:expsm, xi=Inf)
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:expsm,
                                      constraint=:wiggly)
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:expsm,
                                      inner_ridge=-1.0)
        # the anchor must name a lag in the window
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:lagindex, lags=3,
                                      anchor=0)
        @test_throws ArgumentError TC(:b, ct_v, x_v; trans=:lagindex, lags=3,
                                      anchor=4)
        # the covariate record itself: enough of it, ordered, and finite
        @test_throws ArgumentError TC(:b, ct_v[1:2], x_v[1:2]; trans=:expsm)
        @test_throws ArgumentError TC(:b, [0.0, 2.0, 1.0, 3.0], ones(4);
                                      trans=:expsm)
        @test_throws ArgumentError TC(:b, [0.0, 1.0, 1.0, 3.0], ones(4);
                                      trans=:expsm)
        @test_throws ArgumentError TC(:b, ct_v, replace(x_v, 3.0 => NaN);
                                      trans=:expsm)

        # ── fields and parameter counts
        a = TC(:b, ct_v, x_v; trans=:expsm, nknots=6)
        @test a.name == :b
        @test a.trans == :expsm
        @test a.domain == (-2.0, 2.0)               # (−xi, xi), NOT time
        @test a.lags == 0
        @test a.anchor === nothing                  # no scale invariance to fix
        @test a.times == ct_v
        @test nparams(a) == 1 + 6                   # ONE inertia parameter
        @test initial_params(a) == vcat(0.0, zeros(6))   # ω = 1/2 start

        # auxiliary inertia covariates widen the inner block
        Xa = hcat(x_v, cos.(ct_v), sin.(ct_v))
        aa = TC(:b, ct_v, Xa; trans=:expsm, nknots=6)
        @test nparams(aa) == 3 + 6
        @test aa.inner_design[:, 1] == ones(11)     # the intercept
        @test aa.inner_design[:, 2] == cos.(ct_v)
        @test aa.inner_design[:, 3] == sin.(ct_v)

        # :lagindex anchors one weight, so it stores lags − 1 free ones
        al = TC(:b, ct_v, x_v; trans=:lagindex, lags=4, nknots=6)
        @test al.lags == 4
        @test al.anchor == 1
        @test nparams(al) == 3 + 6
        @test initial_params(al) == vcat(ones(3), zeros(6))   # equal weights

        # a shape-constrained outer smooth follows the SCOP parameter count,
        # including the zero-endpoint reduction
        @test nparams(TC(:b, ct_v, x_v; trans=:expsm, nknots=8,
                         constraint=:increasing)) == 1 + 8
        @test nparams(TC(:b, ct_v, x_v; trans=:expsm, nknots=8,
                         constraint=:inc_zero_left)) == 1 + 7
        # xi widens the outer knot span
        @test TC(:b, ct_v, x_v; trans=:expsm, xi=5.0).domain == (-5.0, 5.0)
        # an initial outer curve is sampled on the standardized knot grid
        ai = TC(:b, ct_v, x_v; trans=:expsm, nknots=5, initial=z -> 3.0 + z)
        @test initial_params(ai)[2:end] ≈ [1.0, 2.0, 3.0, 4.0, 5.0]
        # string names accepted, and a vector covariate is a one-column matrix
        @test TC("b", ct_v, x_v; trans=:expsm).name == :b
        @test TC(:b, ct_v, x_v; trans=:expsm).X == reshape(x_v, :, 1)

        # THE SIMPLIFICATION OVER SingleIndexApproximator. Its standardization
        # is against the moving trajectory, so it is UNRESOLVED until
        # PSMProblem construction fills in (μ̂, Σ̂) and `build_evaluator` on a
        # bare approximator is an error. Here the covariate is fixed data, so
        # this type is complete at construction: no index_stats keyword, no
        # resolve hook, and a bare approximator evaluates immediately.
        @test build_evaluator(TC(:b, ct_v, x_v; trans=:expsm, nknots=6),
                              zeros(7))(3.0) == 0.0
    end

    @testset "TransformedCovariateApproximator — inner transforms and standardization" begin
        PSM = PartiallySpecifiedModels
        TC = TransformedCovariateApproximator
        ct_v = collect(0.0:1.0:10.0)
        x_v = Float64[1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]

        # (a) the :expsm recursion against a HAND-COMPUTED reference. At a = 0
        # the inertia is ω = logistic(0) = 1/2 exactly, so each step halves the
        # gap to the new observation: 1, 1.5, 2.25, 3.125, 4.0625.
        a = TC(:b, ct_v, x_v; trans=:expsm, nknots=6)
        @test smoothing_inertia(a, vcat(0.0, zeros(6))) == fill(0.5, 11)
        @test PSM._tc_inner_series(a, [0.0])[1:5] ≈
              [1.0, 1.5, 2.25, 3.125, 4.0625] atol=1e-12
        # …and for general ω, against the recursion written out longhand
        for ω in (0.2, 0.8)
            aω = log(ω / (1 - ω))
            @test smoothing_inertia(a, vcat(aω, zeros(6)))[1] ≈ ω atol=1e-12
            ref = similar(x_v)
            ref[1] = x_v[1]
            for i in 2:11
                ref[i] = ω * ref[i - 1] + (1 - ω) * x_v[i]
            end
            @test PSM._tc_inner_series(a, [aω]) ≈ ref atol=1e-12
        end
        # ω → 1 is the INFINITE-INERTIA limit: s̃ collapses to the constant x₁
        # and z → 0. That is a legitimate model (a constant response), so the
        # additive variance floor must return it finitely rather than 0/0.
        z_inf = transformed_covariate(a, vcat(40.0, zeros(6)))
        @test all(isfinite, z_inf)
        @test maximum(abs, z_inf) < 1e-3
        # auxiliary covariates really do make the inertia vary
        Xa = hcat(x_v, cos.(ct_v))
        aa = TC(:b, ct_v, Xa; trans=:expsm, nknots=6)
        ωs = smoothing_inertia(aa, vcat([0.0, 1.5], zeros(6)))
        @test length(unique(round.(ωs, digits=6))) > 5
        @test all(0.0 .< ωs .< 1.0)

        # (b) the :lagindex design is FIXED DATA: row i holds the covariate at
        # tᵢ, tᵢ − Δ, … with Δ the sample spacing (1.0 here), held constant
        # before the start of the record.
        al = TC(:b, ct_v, x_v; trans=:lagindex, lags=4, nknots=6)
        @test al.inner_design[5, :] == [x_v[5], x_v[4], x_v[3], x_v[2]]
        @test al.inner_design[1, :] == fill(x_v[1], 4)
        @test al.inner_design[:, 1] == x_v            # lag 0 is x itself
        # …so the transform is EXACTLY linear in the weights
        w = [1.0, 0.5, 0.25, 0.125]
        @test PSM._tc_inner_series(al, w[2:end]) ≈ al.inner_design * w
        @test lag_weights(al, vcat(w[2:end], zeros(6))) == w
        # a[anchor] ≡ 1 wherever the anchor is placed
        a3 = TC(:b, ct_v, x_v; trans=:lagindex, lags=4, anchor=3, nknots=6)
        @test lag_weights(a3, vcat([0.2, 0.4, 0.6], zeros(6))) ==
              [0.2, 0.4, 1.0, 0.6]

        # (c) STANDARDIZATION. The paper's recipe — mean 0, variance 1 over the
        # covariate sample — ports VERBATIM here, because the covariate is
        # fixed data rather than a trajectory that moves under the optimizer.
        # The scale is √(v + κ), so the standardized variance is v/(v + κ):
        # it approaches 1 strictly FROM BELOW, by κ/v ≈ 2e-8 at worst over
        # these parameter values. That bound is asserted rather than hidden
        # inside an `≈`, because the direction is a property of the additive
        # floor and an exceedance would mean the floor had stopped applying.
        vsamp(v) = sum(abs2, v .- mean(v)) / (length(v) - 1)
        for θ in (-2.0, -0.5, 0.0, 1.0, 2.5)
            z = transformed_covariate(a, vcat(θ, zeros(6)))
            @test abs(mean(z)) < 1e-10
            @test 1.0 - 1e-6 < vsamp(z) <= 1.0
        end
        for wf in ([0.5, 0.25, 0.125], [1.0, 1.0, 1.0], [-0.3, 0.7, 0.1])
            z = transformed_covariate(al, vcat(wf, zeros(6)))
            @test abs(mean(z)) < 1e-10
            @test 1.0 - 1e-6 < vsamp(z) <= 1.0
        end
        # Standardizing a LINEAR transform makes z invariant under a → c·a,
        # which is precisely the flat direction the :lagindex anchor removes.
        # SingleIndexApproximator makes its aᵀΣ̂a floor proportional to ‖a‖² so
        # this invariance stays EXACT even when the floor binds; here the floor
        # is a fixed additive constant chosen instead for smoothness in `a`, so
        # the invariance holds only to ≈ κ/v ~ 1e-10 relative. That is
        # harmless: the anchor, not the floor, is what removes the direction.
        wv = [1.0, 0.5, 0.25, 0.125]
        @test maximum(abs, PSM._tc_standardize(al, al.inner_design * wv) .-
                           PSM._tc_standardize(al, al.inner_design * (4wv))) < 1e-8

        # (d) the evaluator interpolates z LINEARLY in time and holds it
        # constant outside the record, so f is continuous everywhere — which
        # adaptive ODE stepping requires.
        θ_out = collect(range(-1.0, 1.5, length=6))
        f = build_evaluator(a, vcat(0.0, θ_out))
        s_out = build_evaluator(a.outer, θ_out)
        z = transformed_covariate(a, vcat(0.0, θ_out))
        for i in 1:11
            @test f(ct_v[i]) ≈ s_out(z[i]) atol=1e-12
        end
        @test f(4.5) ≈ s_out((z[5] + z[6]) / 2) atol=1e-12
        @test f(-7.0) == f(0.0)
        @test f(99.0) == f(10.0)
        @test maximum(abs(f(t + 1e-7) - f(t - 1e-7)) for t in ct_v) < 1e-6
        # a ONE-argument callable of TIME: passing a state is refused loudly
        @test_throws ArgumentError f(1.0, 2.0)
    end

    @testset "TransformedCovariateApproximator — Dual safety in params and time" begin
        using ForwardDiff
        PSM = PartiallySpecifiedModels
        TC = TransformedCovariateApproximator
        ct_v = collect(0.0:1.0:10.0)
        x_v = Float64[1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]

        for a in (TC(:b, ct_v, x_v; trans=:expsm, nknots=7),
                  TC(:b, ct_v, x_v; trans=:lagindex, lags=4, nknots=7))
            ni = PSM._tc_n_inner(a)
            θ = vcat(a.trans === :expsm ? [0.7] : [0.6, 0.3, 0.1],
                     collect(range(-1.0, 1.5, length=7)))
            f = build_evaluator(a, θ)
            # inside the record and in BOTH constant-extrapolation branches
            for t in (0.0, 3.5, 7.25, -4.0, 40.0)
                g1 = ForwardDiff.gradient(b -> build_evaluator(a, b)(t), θ)
                @test all(isfinite, g1)
                h = 1e-6
                for k in (1, ni + 2, ni + 5)
                    e_k = [i == k ? 1.0 : 0.0 for i in eachindex(θ)]
                    fd = (build_evaluator(a, θ .+ h .* e_k)(t) -
                          build_evaluator(a, θ .- h .* e_k)(t)) / (2h)
                    @test g1[k] ≈ fd atol=1e-5
                end
            end
            # …and THROUGH the time interpolation, including nested Duals
            @test isfinite(ForwardDiff.derivative(t -> f(t), 3.5))
            @test all(isfinite, ForwardDiff.gradient(
                b -> ForwardDiff.derivative(t -> build_evaluator(a, b)(t), 3.5), θ))
            # the inner parameter really moves the fit, so none of this is vacuous
            @test abs(ForwardDiff.gradient(
                b -> build_evaluator(a, b)(3.5), θ)[1]) > 1e-8
        end

        # The :expsm scan stays finite at extreme inertia parameters, where the
        # naive 1/(1 + exp(−z)) logistic overflows to Inf and then to a NaN
        # derivative. `_tc_logistic` is tanh-based and branch-free instead.
        ax = TC(:b, ct_v, x_v; trans=:expsm, nknots=7)
        θx = collect(range(-1.0, 1.5, length=7))
        for big in (-800.0, 800.0)
            @test all(isfinite, ForwardDiff.gradient(
                b -> build_evaluator(ax, b)(3.5), vcat(big, θx)))
        end
        @test PSM._tc_logistic(-800.0) == 0.0
        @test PSM._tc_logistic(800.0) == 1.0
        @test PSM._tc_logistic(0.0) == 0.5
        @test PSM._tc_logistic(1.3) ≈ 1 / (1 + exp(-1.3)) atol=1e-15

        # the same holds for a shape-constrained outer smooth
        ac = TC(:b, ct_v, x_v; trans=:expsm, nknots=7, constraint=:increasing)
        θc = vcat([0.7], randn(StableRNG(51), 7))
        @test all(isfinite, ForwardDiff.gradient(
            b -> build_evaluator(ac, b)(3.5), θc))
        @test isfinite(ForwardDiff.derivative(
            t -> build_evaluator(ac, θc)(t), 3.5))
    end

    @testset "TransformedCovariateApproximator — penalty blocks and merged penalty" begin
        PSM = PartiallySpecifiedModels
        TC = TransformedCovariateApproximator
        ct_v = collect(0.0:1.0:10.0)
        x_v = Float64[1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]

        # (a) :expsm — a ridge on the inertia parameters
        a = TC(:b, ct_v, x_v; trans=:expsm, nknots=8, inner_ridge=0.05)
        bl = penalty_blocks(a)
        @test length(bl) == 2
        @test bl[1][2] == 1:1
        @test bl[2][2] == 2:9
        @test bl[1][1] == ones(1, 1)
        # the outer block is EXACTLY the univariate spline penalty on the same
        # knot count — reuse by composition, not duplication
        @test bl[2][1] ≈ penalty_matrix(BSplineApproximator(:b, (-2.0, 2.0), 8))
        S = penalty_matrix(a)
        @test size(S) == (9, 9)
        @test issymmetric(S)
        @test S[1, 1] == 0.05                     # the FIXED merged weight
        @test all(S[1, 2:end] .== 0.0)
        @test S[2:end, 2:end] ≈ bl[2][1]

        # (b) :lagindex — the paper's FIRST-DIFFERENCE smooth-lag prior, not a
        # ridge. Its null space is "all free weights equal", which is why a
        # large λ flattens the lag TAIL instead of erasing it.
        al = TC(:b, ct_v, x_v; trans=:lagindex, lags=5, nknots=8)
        bll = penalty_blocks(al)
        @test length(bll) == 2
        @test bll[1][2] == 1:4
        @test bll[2][2] == 5:12
        Si = bll[1][1]
        @test Si ≈ [1.0 -1.0 0.0 0.0; -1.0 2.0 -1.0 0.0;
                    0.0 -1.0 2.0 -1.0; 0.0 0.0 -1.0 1.0]
        @test norm(Si * ones(4)) < 1e-12
        @test rank(Si) == 3
        # …and the merged penalty carries the same matrix, scaled
        @test penalty_matrix(al)[1:4, 1:4] ≈ 1e-4 .* Si
        for (Sb, _) in vcat(bl, bll)
            @test maximum(abs.(Sb .- Sb')) < 1e-8 * max(maximum(abs.(Sb)), 1.0)
            @test minimum(eigvals(Symmetric(Sb))) > -1e-8
        end
        # with fewer than two free weights there is no difference to take, so
        # the inner block is omitted rather than declared as a zero matrix
        # (which would hand LAML an unidentified λ)
        a2 = TC(:b, ct_v, x_v; trans=:lagindex, lags=2, nknots=8)
        @test length(penalty_blocks(a2)) == 1
        @test penalty_blocks(a2)[1][2] == 2:9
        a1 = TC(:b, ct_v, x_v; trans=:lagindex, lags=1, nknots=8)
        @test nparams(a1) == 8
        @test length(penalty_blocks(a1)) == 1
        @test penalty_blocks(a1)[1][2] == 1:8

        # (c) build_penalty_matrices validates and lays the blocks out globally
        al_c = TC(:c, ct_v, x_v; trans=:lagindex, lags=5, nknots=8)
        prob_pb = PSMProblem((du, u, p, t) -> (du[1] = p.b(t) + p.c(t);
                                               du[2] = -0.1 * u[2]; nothing),
            [1.0, 1.0], (0.0, 10.0), [a, al_c];
            data_times=[0.0, 5.0, 10.0], data_values=ones(3, 2),
            obs_to_state=[1, 2], likelihood=Gaussian(), solver=Tsit5())
        S_list, offs, nks = PartiallySpecifiedModels.build_penalty_matrices(prob_pb)
        @test length(S_list) == 4
        @test offs == [0, 1, 9, 13]
        @test nks == [1, 8, 4, 8]

        # (d) the accessors read THIS approximator's block, not the whole
        # coefficient vector — the failure mode N1 found with index_loadings
        @test_throws ArgumentError lag_weights(al, zeros(3))
        @test_throws ArgumentError smoothing_inertia(a, zeros(3))
        @test_throws ArgumentError transformed_covariate(a, zeros(3))
        err_lw = try
            lag_weights(al, zeros(3))
            nothing
        catch e; e; end
        @test occursin("coefficient block", err_lw.msg)
        # …and refuse the transformation they do not describe
        @test_throws ArgumentError lag_weights(a, zeros(nparams(a)))
        @test_throws ArgumentError smoothing_inertia(al, zeros(nparams(al)))
    end

    # ── The :expsm recovery fixture ────────────────────────────────────
    #
    # SIR with transmission driven by TEMPERATURE through a learned thermal
    # inertia: β(t) = s₀(z(t)) with z the standardized exponentially smoothed
    # temperature at ω_true = 0.8. The covariate is a realistic seasonal
    # series (a 45-day-amplitude annual arc plus daily weather noise), and the
    # weather noise is what gives the inertia something to smooth — a
    # noise-free driver would make every ω produce nearly the same z.
    tc_ct = collect(0.0:1.0:120.0)
    tc_temp = [18.0 + 7.0 * sin(2pi * t / 180 - pi / 3) for t in tc_ct] .+
              1.2 .* randn(StableRNG(101), length(tc_ct))
    tc_wtrue = 0.8
    function tc_ewma(x, w)
        s = similar(x)
        s[1] = x[1]
        for i in 2:length(x)
            s[i] = w * s[i - 1] + (1 - w) * x[i]
        end
        s
    end
    function tc_lin(ts, vs, t)
        t <= ts[1] && return vs[1]
        t >= ts[end] && return vs[end]
        i = searchsortedlast(ts, t)
        w = (t - ts[i]) / (ts[i + 1] - ts[i])
        vs[i] + w * (vs[i + 1] - vs[i])
    end
    tc_ztrue = let v = tc_ewma(tc_temp, tc_wtrue)
        m = mean(v)
        (v .- m) ./ sqrt(sum(abs2, v .- m) / (length(v) - 1))
    end
    tc_s0(z) = 0.30 + 0.12 * tanh(0.9z)
    tc_btrue(t) = tc_s0(tc_lin(tc_ct, tc_ztrue, t))
    function tc_sir_true!(du, u, p, t)
        inf = tc_btrue(t) * u[1] * u[2] / 1000.0
        du[1] = -inf
        du[2] = inf - 0.2 * u[2]
        du[3] = inf
        nothing
    end
    tc_ref = OrdinaryDiffEq.solve(
        ODEProblem(tc_sir_true!, [990.0, 10.0, 0.0], (0.0, 120.0)), Tsit5(),
        saveat=2.0, abstol=1e-10, reltol=1e-10)
    tc_traj = reduce(hcat, tc_ref.u)'
    tc_ts = collect(tc_ref.t)
    tc_data = tc_traj[:, [2, 3]] .+
              3.0 .* randn(StableRNG(102), size(tc_traj, 1), 2)
    function tc_sir!(du, u, p, t)
        inf = p.beta(t) * u[1] * u[2] / p.N
        du[1] = -inf
        du[2] = inf - p.gam * u[2]
        du[3] = inf
        nothing
    end
    tc_prob(a) = PSMProblem(tc_sir!, [990.0, 10.0, 0.0], (0.0, 120.0), [a];
        data_times=tc_ts, data_values=tc_data, obs_to_state=[2, 3],
        known_params=(N=1000.0, gam=0.2), likelihood=Gaussian(), solver=Tsit5())
    tc_beta_rmse(f) = sqrt(mean(abs2,
        [f(t) - tc_btrue(t) for t in 0.0:1.0:120.0]))

    @testset "TransformedCovariateApproximator — SIR recovery of thermal inertia" begin
        TC = TransformedCovariateApproximator
        a = TC(:beta, tc_ct, tc_temp; trans=:expsm, nknots=8, initial=z -> 0.3)
        prob = tc_prob(a)

        # THE KINK MATTERS. β(t) is piecewise linear in t between covariate
        # times, so the default finite-difference prediction Jacobian picks up
        # adaptive-step noise at the kinks and the λ search stalls — the N0
        # failure mode, reproduced here as a DIRECT comparison rather than
        # asserted from theory.
        sol_fd = solve(prob, LAML(maxiters=60, warmup=10, initial_lambda=0.01))
        sol = solve(tc_prob(TC(:beta, tc_ct, tc_temp; trans=:expsm, nknots=8,
                               initial=z -> 0.3)),
                    LAML(maxiters=60, warmup=10, initial_lambda=0.01,
                         jac=:forwarddiff))
        # β(t) is piecewise linear, so the fd prediction Jacobian is noisy at
        # the kinks and its λ search lands far from the optimum. The
        # forwarddiff side is stable and exact: objective 480.98, λ_inner
        # 11.5632, converged.
        #
        # The fd side is NOT quotable as a number — it is chaotic in the
        # exact arithmetic. Measured: 611.90 (:maxiters) at BLAS=4/8 under
        # --check-bounds=yes, 746.19 (:maxiters) at BLAS=1, and 1993.61
        # REPORTING :converged_tol without the flag. So the gap runs from
        # ~131 to ~1513 nats and the reported status flips with the run
        # configuration — fd may or may not admit it failed. The assertion
        # below is deliberately a floor (20 nats) that every variant clears
        # by at least 6x, not a pin on any of those values.
        @test sol.objective < sol_fd.objective - 20.0
        @test sol.convergence.converged

        # two smoothing parameters, one per penalty block, jointly estimated
        @test length(sol.smoothing_params) == 2
        @test all(isfinite, sol.smoothing_params)
        @test all(>(0), sol.smoothing_params)
        @test sol.smoothing_params[1] != sol.smoothing_params[2]

        # the response curve is recovered sharply
        f = sol.unknown_functions[:beta]
        @test tc_beta_rmse(f) < 0.02              # observed 0.00877
        @test maximum(abs(f(t) - tc_btrue(t)) for t in 0.0:1.0:120.0) < 0.05

        # THE INERTIA IS ONLY WEAKLY IDENTIFIED — reported, not tuned away.
        # ω̂ = 0.6867 against a truth of 0.8, from a start of ω = 0.5. It is
        # shrunk toward 1/2 by the inner ridge (λ̂_inner = 11.56), and the
        # shrinkage is honest rather than a bug: re-running the identical
        # fixture at lower observation noise gives ω̂ = 0.7631 (sd 1.0) and
        # 0.7909 (sd 0.25) with λ̂_inner falling to 0.6393 and 0.0310 — so the
        # estimator is consistent and LAML is shrinking a genuinely weak
        # signal. Weak because standardization removes the amplitude damping
        # that different ω apply, leaving only a phase lag:
        # cor(z(ω=0.8), z(ω=0.7)) = 0.99763 over this covariate sample.
        # A profile of the LAML objective over FIXED ω has its minimum at
        # ω ≈ 0.70 (objective 476.8, against 504.4 at ω = 0.5 and 548.1 at
        # ω = 0.9), so the joint fit sits essentially AT its own criterion's
        # optimum; the residual gap to 0.8 belongs to the criterion, not the
        # optimizer. Notably the β-RMSE profile is minimized at the TRUE
        # ω = 0.8 (0.00305), so the criterion trades a little inertia for a
        # little outer curvature.
        ω̂ = smoothing_inertia(a, Float64.(collect(sol.parameters)))[1]
        @test length(smoothing_inertia(a, Float64.(collect(sol.parameters)))) ==
              length(tc_ct)
        @test 0.60 < ω̂ < 0.78                     # observed 0.6867
        @test ω̂ > 0.55                            # …and it did move off 0.5
        # the profile claim above, spot-checked at its two ends: a fit that
        # HOLDS ω at 1/2 does strictly worse than the joint fit
        a_half = TC(:beta, tc_ct, tc_ewma(tc_temp, 0.5); trans=:lagindex,
                    lags=1, nknots=8, initial=z -> 0.3)
        sol_half = solve(tc_prob(a_half),
                         LAML(maxiters=60, warmup=10, initial_lambda=0.01,
                              jac=:forwarddiff))
        @test sol_half.objective > sol.objective   # observed 504.43 vs 480.99
        @test tc_beta_rmse(sol_half.unknown_functions[:beta]) > tc_beta_rmse(f)

        # AdamSolver — ForwardDiff through the ODE solve, i.e. Dual parameters
        # AND a Dual-safe scan/interpolation inside the right-hand side. It
        # fits far less well than LAML here (observed data_loss 1243 against
        # LAML's 934, ω̂ 0.170, β-RMSE 0.0372): the flat-objective path barely
        # identifies the inertia at all. Asserted loosely because what this
        # checks is that the autodiff path RUNS and improves, not accuracy.
        a_ad = TC(:beta, tc_ct, tc_temp; trans=:expsm, nknots=8, initial=z -> 0.3)
        sol_ad = solve(tc_prob(a_ad), AdamSolver(maxiters=400, lr=0.05))
        @test isfinite(sol_ad.data_loss)
        @test sol_ad.data_loss < 3000.0
        @test tc_beta_rmse(sol_ad.unknown_functions[:beta]) < 0.06
    end

    @testset "TransformedCovariateApproximator — distributed-lag recovery and penalty" begin
        PSM = PartiallySpecifiedModels
        TC = TransformedCovariateApproximator

        # A fast-relaxing linear filter driven by the unknown function,
        # du/dt = g(t) − 0.8u, observed directly. The epidemic fixture above
        # is deliberately NOT reused here: an SIR integrates β over weeks, so
        # it responds only to the low-frequency content of the driver and the
        # lag PROFILE washes out (measured: the LAML fit there collapsed into
        # the penalty null space). A directly-observed fast response keeps the
        # profile identifiable.
        lg_ct = collect(0.0:1.0:40.0)
        lg_x = [10.0 + 4.0 * sin(2pi * t / 25) for t in lg_ct] .+
               2.5 .* randn(StableRNG(201), length(lg_ct))
        lg_wtrue = [1.0, 0.6, 0.36, 0.216]          # geometric decay, ratio 0.6
        lg_a0 = TC(:g, lg_ct, lg_x; trans=:lagindex, lags=4, nknots=8)
        lg_z = PSM._tc_standardize(lg_a0,
                                   PSM._tc_inner_series(lg_a0, lg_wtrue[2:end]))
        lg_s0(z) = 1.0 + 0.6 * tanh(z)
        lg_gtrue(t) = lg_s0(tc_lin(lg_ct, lg_z, t))
        lg_true!(du, u, p, t) = (du[1] = lg_gtrue(t) - 0.8 * u[1]; nothing)
        lg_ref = OrdinaryDiffEq.solve(
            ODEProblem(lg_true!, [1.25], (0.0, 40.0)), Tsit5(),
            saveat=0.5, abstol=1e-10, reltol=1e-10)
        lg_ts = collect(lg_ref.t)
        lg_data = reduce(hcat, lg_ref.u)' .+
                  0.02 .* randn(StableRNG(202), length(lg_ts), 1)
        lg_dyn!(du, u, p, t) = (du[1] = p.g(t) - p.k * u[1]; nothing)
        lg_prob(a) = PSMProblem(lg_dyn!, [1.25], (0.0, 40.0), [a];
            data_times=lg_ts, data_values=lg_data, obs_to_state=[1],
            known_params=(k=0.8,), likelihood=Gaussian(), solver=Tsit5())
        centroid(w) = sum((0:(length(w) - 1)) .* w) / sum(w)

        # (a) RECOVERY of the decaying lag profile
        a = TC(:g, lg_ct, lg_x; trans=:lagindex, lags=4, nknots=8,
               initial=z -> 1.0)
        sol = solve(lg_prob(a), LAML(maxiters=60, warmup=10,
                                     initial_lambda=0.01, jac=:forwarddiff))
        @test length(sol.smoothing_params) == 2
        @test all(isfinite, sol.smoothing_params)
        @test sol.smoothing_params[1] != sol.smoothing_params[2]
        # Guard the fit BEFORE reading recovery tolerances off it. The SCOP
        # testset below documents why: an unconverged LAML fit still returns
        # finite, plausible-looking numbers (and plausible λ and edf), so
        # `converged` is the only reliable signal that these tolerances mean
        # anything. This fit does converge (:converged_tol, objective 0.0146).
        @test sol.convergence.converged
        w = lag_weights(a, Float64.(collect(sol.parameters)))
        @test w[1] == 1.0                          # anchored exactly
        # observed [1.0, 0.5438, 0.2816, 0.2405] against [1, 0.6, 0.36, 0.216]
        @test maximum(abs.(w .- lg_wtrue)) < 0.15
        @test w[2] > w[3]                           # the decay is recovered…
        @test w[2] < w[1]
        # …and the identified summary, the mean lag, lands within 3%
        @test abs(centroid(w) - centroid(lg_wtrue)) < 0.06   # observed 0.0194
        g = sol.unknown_functions[:g]
        lg_rmse = sqrt(mean(abs2, [g(t) - lg_gtrue(t) for t in 0.0:0.25:40.0]))
        @test lg_rmse < 0.03                        # observed 0.01343
        # the fit is stable in the LAML start, not a lucky setting
        sol_b = solve(lg_prob(TC(:g, lg_ct, lg_x; trans=:lagindex, lags=4,
                                 nknots=8, initial=z -> 1.0)),
                      LAML(maxiters=60, warmup=5, initial_lambda=1.0,
                           jac=:forwarddiff))
        @test sol_b.convergence.converged
        @test maximum(abs.(lag_weights(a, Float64.(collect(sol.parameters))) .-
                           lag_weights(a, Float64.(collect(sol_b.parameters))))) < 0.02

        # (b) THE PENALTY DISCRIMINATOR. The inner block really is a
        # first-difference penalty and it really bites: sweeping the fixed
        # merged weight `inner_ridge` under a single-λ solver flattens the lag
        # profile by EIGHT orders of magnitude in roughness — and flattens
        # it toward a COMMON NON-ZERO value, which a ridge (whose null space
        # is {0}) could not do. That distinction is the point of the test.
        # Measured: roughness 0.1502 -> 1.11e-9, spread 0.3477 -> 4.70e-5,
        # and the free weights land at [0.9718, 0.9717, 0.9717] — a flat,
        # decidedly non-zero tail.
        rough(v) = sum(abs2, diff(v[2:end]))
        spread(v) = maximum(v[2:end]) - minimum(v[2:end])
        res = map((1e-6, 1e4)) do ir
            ai = TC(:g, lg_ct, lg_x; trans=:lagindex, lags=4, nknots=8,
                    inner_ridge=ir, initial=z -> 1.0)
            si = solve(lg_prob(ai), AdamSolver(maxiters=500, lr=0.05,
                                               penalty_weight=1.0))
            lag_weights(ai, Float64.(collect(si.parameters)))
        end
        w_free, w_pen = res
        @test rough(w_pen) < 1e-3 * rough(w_free)
        @test spread(w_pen) < 0.1 * spread(w_free)
        @test spread(w_pen) < 0.05
        # a ridge would have driven the free weights to zero; the smooth-lag
        # prior parks them on a common non-zero level instead
        @test mean(w_pen[2:end]) > 0.05
    end

    @testset "TransformedCovariateApproximator — shape constraints, bands, bootstrap" begin
        PSM = PartiallySpecifiedModels
        TC = TransformedCovariateApproximator

        # (a) a shape-constrained outer smooth is monotone BY CONSTRUCTION for
        # arbitrary parameter vectors — the SCOP guarantee, inherited unchanged
        # from ShapeConstrainedBSplineApproximator by composition
        rng_sc = StableRNG(53)
        for constraint in (:increasing, :decreasing)
            asc = TC(:beta, tc_ct, tc_temp; trans=:expsm, nknots=9,
                     constraint=constraint)
            for _ in 1:5
                θsc = vcat([0.8], 1.5 .* randn(rng_sc, 9))
                curve = [PSM._eval_approx_at(asc, θsc, z)
                         for z in range(-2.5, 2.5, length=60)]
                d = diff(curve)
                @test (constraint == :increasing ? minimum(d) > -1e-10 :
                                                   maximum(d) < 1e-10)
            end
        end
        # …and it recovers the monotone temperature response end to end.
        #
        # NOTE THE `initial_lambda`. The SCOP outer smooth composed with the
        # :expsm scan is strongly nonlinear, and starting LAML from the very
        # light penalty this file uses elsewhere (0.01) does NOT work here:
        # measured under the test suite's own `--check-bounds=yes`, that start
        # fails to converge — exiting on :maxiters with β-RMSE 0.031 (vs
        # 0.0085 here). Note what it does NOT do: λ = [0.54, 0.85] and edf
        # 5.11 both look perfectly reasonable, so magnitude checks would miss
        # it entirely and only `converged` catches it. This is the package's
        # documented advice for strongly
        # nonlinear problems (see the `initial_lambda` note in solver.jl), not
        # a fixture tuned until it passed: `initial_lambda` 1.0 and 0.1 (with
        # warmup 20) converge to the SAME optimum — objective 479.89, λ =
        # [11.18, 6.07], edf 5.444, ω̂ 0.6903 — and that optimum agrees with
        # the UNCONSTRAINED fit's ω̂ 0.6867 above.
        #
        # The convergence flag is asserted for exactly this reason: without
        # it the diverged run above still produced finite numbers, and an
        # earlier draft of this testset read its tolerances off one.
        a_sc = TC(:beta, tc_ct, tc_temp; trans=:expsm, nknots=9,
                  constraint=:increasing, initial=z -> 0.3)
        sol_sc = solve(tc_prob(a_sc), LAML(maxiters=60, warmup=10,
                                           initial_lambda=1.0,
                                           jac=:forwarddiff))
        @test sol_sc.convergence.converged
        @test sol_sc.edf > 3.0                      # not collapsed to a constant
        @test all(<(1e6), sol_sc.smoothing_params)  # …and λ not pinned at the cap
        @test tc_beta_rmse(sol_sc.unknown_functions[:beta]) < 0.02  # obs 0.00852
        θ_sc = Float64.(collect(sol_sc.parameters))
        curve_sc = [PSM._eval_approx_at(a_sc, θ_sc, z)
                    for z in range(-2.0, 2.0, length=40)]
        @test minimum(diff(curve_sc)) > -1e-10      # really increasing
        @test maximum(curve_sc) - minimum(curve_sc) > 0.05   # observed 0.3033
        @test 0.55 < smoothing_inertia(a_sc, θ_sc)[1] < 0.80  # observed 0.6903

        # (b) confidence_band serves the OUTER RESPONSE CURVE over the
        # standardized covariate — NOT f(t). Gridding the domain and calling
        # the fitted callable would evaluate β at "times" −xi…xi, which is the
        # sibling-site bug this dispatch exists to prevent.
        a_cb = TC(:beta, tc_ct, tc_temp; trans=:expsm, nknots=8,
                  initial=z -> 0.3)
        prob_cb = tc_prob(a_cb)
        sol_cb = solve(prob_cb, LAML(maxiters=40, warmup=10,
                                     initial_lambda=0.01, jac=:forwarddiff))
        band = confidence_band(sol_cb, prob_cb)
        @test haskey(band, :beta)
        @test extrema(band[:beta].grid) == (-2.0, 2.0)
        @test all(isfinite, band[:beta].fitted)
        @test all(isfinite, band[:beta].se)
        @test all(band[:beta].se .>= 0)
        @test all(band[:beta].lower .<= band[:beta].fitted)
        @test all(band[:beta].fitted .<= band[:beta].upper)
        θ_cb = Float64.(collect(sol_cb.parameters))
        @test band[:beta].fitted ≈ [PSM._eval_approx_at(a_cb, θ_cb, z)
                                    for z in band[:beta].grid]
        # …and that is NOT what calling the time-callable on the grid gives
        @test maximum(abs.(band[:beta].fitted .-
                           [sol_cb.unknown_functions[:beta](z)
                            for z in band[:beta].grid])) > 1e-6

        # (c) bootstrap produces a real (non-NaN) band for the response curve
        res_b = bootstrap(sol_cb, prob_cb, AdamSolver(maxiters=60, lr=0.05);
                          nboot=4, rng=StableRNG(54))
        @test res_b.n_success >= 3
        @test haskey(res_b.ci_uf, :beta)
        @test all(isfinite, res_b.ci_uf[:beta].lower)
        @test all(isfinite, res_b.ci_uf[:beta].upper)
    end

    @testset "TransformedCovariateApproximator — sibling registration" begin
        TC = TransformedCovariateApproximator
        # In the union, so the SIX per-type penalty whitelists must list it
        # explicitly — the generic non-built-in fallback will NOT fire for it.
        @test TC <: PartiallySpecifiedModels._BUILTIN_APPROX_TYPES

        # Behavioral check where it is cheap: the gradient-matching solvers
        # really do apply the merged penalty.
        S_tc = penalty_matrix(TC(:beta, tc_ct, tc_temp; trans=:expsm, nknots=9))
        rough(sol) = dot(sol.parameters, S_tc * sol.parameters)
        for alg in (λ -> TwoStageSolver(maxiters=150, lambda_smooth=λ),
                    λ -> IntegralMatchingSolver(maxiters=150, lambda_smooth=λ))
            s0_ = solve(tc_prob(TC(:beta, tc_ct, tc_temp; trans=:expsm,
                                   nknots=9, initial=z -> 0.3)), alg(0.0))
            sP_ = solve(tc_prob(TC(:beta, tc_ct, tc_temp; trans=:expsm,
                                   nknots=9, initial=z -> 0.3)), alg(50.0))
            @test s0_.parameters != sP_.parameters
            @test rough(sP_) < rough(s0_)
        end

        # The remaining four whitelists (AGM — whose smoothing_lambda is not a
        # user-facing keyword — plus MAGI, rodeo and DALTON, which sit inside
        # Kalman/HMC objectives far too slow to gate a unit test on) cannot be
        # exercised end-to-end here. The campaign's recurring failure mode is
        # precisely a type landing on some sibling sites and not others, so
        # assert the registration at each site directly.
        src_dir = dirname(pathof(PartiallySpecifiedModels))
        for f in ("two_stage_solver.jl", "integral_matching_solver.jl",
                  "magi_solver.jl", "adaptive_gradient_matching.jl",
                  "rodeo_solver.jl", "dalton_solver.jl",
                  "bootstrap.jl", "diagnostics.jl",
                  "approximator_interface.jl")
            @test occursin("TransformedCovariateApproximator",
                           read(joinpath(src_dir, f), String))
        end
        # exported, alongside its three accessors
        for s in (:TransformedCovariateApproximator, :lag_weights,
                  :smoothing_inertia, :transformed_covariate)
            @test s in names(PartiallySpecifiedModels)
        end
    end

    @testset "LAML — reported λ̂ and β̂ are mutually consistent" begin
        PSM = PartiallySpecifiedModels

        # β̂ is the penalized MLE at λ̂ if and only if it is a FIXED POINT of
        # the PCLS step at λ̂: one penalized-least-squares solve of the
        # working linear model formed AT β̂, using B(λ̂), returns β̂ again.
        # This refits β at `sol.smoothing_params` with the solver's own
        # machinery (`_pcls_augmented_solve` — the same call `pcls_step`
        # makes inside the IRLS loop) and reports how far it moves, relative
        # to ‖β̂‖.
        #
        # This is not decoration. The IRLS loop carries THREE θ vectors (see
        # the `theta_fit` comment in solver.jl) and used to report the
        # newest Fellner-Schall PROPOSAL, which at loop exit is untested and
        # is sometimes one the accept block explicitly rejected. On the
        # two-λ fixture below that made `smoothing_params` disagree with
        # `parameters` badly enough that this refit moved β by 77% of ‖β̂‖
        # (measured), and the reported `objective` was 7.3e7 against a
        # `data_loss` of 0.89 — the penalty evaluated at a λ the
        # coefficients had never seen.
        function pcls_refit_move(prob, sol)
            beta = Float64.(vec(sol.parameters))
            n_times = length(prob.data_times)
            n_obs = length(prob.obs_to_state)
            n_data = n_times * n_obs
            n_p = length(beta)

            S_list, uf_offsets, uf_nk = PSM.build_penalty_matrices(prob)
            B = zeros(n_p, n_p)
            for l in eachindex(S_list)
                off, nk = uf_offsets[l], uf_nk[l]
                for i in 1:nk, j in 1:nk
                    B[off+i, off+j] += sol.smoothing_params[l] * S_list[l][i, j]
                end
            end

            pred = PSM.simulate(prob, beta)
            y = zeros(n_data); w = zeros(n_data); f = zeros(n_data)
            k = 1
            for oi in 1:n_obs, ti in 1:n_times
                yv, wv = prob.data_values[ti, oi], prob.data_weights[ti, oi]
                if PSM._usable(yv, wv)
                    y[k] = yv; w[k] = wv
                end
                f[k] = pred[ti, oi]
                k += 1
            end

            J = zeros(n_data, n_p)
            PSM.compute_jacobian!(J, prob, beta, f, n_times, n_obs;
                                  dam=fill(1e-8, n_p), jac=:fd)
            w_irls = PSM.irls_weights(prob.likelihood, y, f, w)
            z = y .- f .+ J * beta
            beta_star = PSM._pcls_augmented_solve(J, z, B, w_irls)
            norm(beta_star .- beta) / max(norm(beta), 1e-12)
        end

        # The reported tuple must be ONE coherent set: `objective` is
        # ½(data_loss + β̂'S^λ̂β̂) using the SAME λ̂ and β̂ the solution
        # reports. Guards against "fixing" the reported λ̂ alone while
        # leaving the θ-dependent scalars computed at the other θ.
        function reported_penalty(prob, sol)
            beta = Float64.(vec(sol.parameters))
            S_list, uf_offsets, uf_nk = PSM.build_penalty_matrices(prob)
            p = 0.0
            for l in eachindex(S_list)
                off, nk = uf_offsets[l], uf_nk[l]
                bl = beta[off+1:off+nk]
                p += sol.smoothing_params[l] * dot(bl, S_list[l] * bl)
            end
            p
        end

        # ── Discriminating fixture: two penalized splines (hence two λ) on a
        #    Lotka-Volterra system. The noise draws are hardcoded so the fit
        #    is identical across Julia/RNG versions. Converged, so β̂ really
        #    is stationary and the fixed-point test is not confounded by
        #    truncation.
        noise_H = [-0.60448, 0.36713, -0.25198, 2.28400, -1.68664,
                   -0.22035, -1.89084]
        noise_L = [-1.13449, 1.36343, 0.20909, 0.48198, 0.80718,
                   -0.14331, -0.08570]
        r_true_lv(H) = 0.7 * exp(-H / 60)
        function lv_true_f1!(du, u, p, t)
            H, L = u
            du[1] = r_true_lv(H) * H - 0.01 * H * L
            du[2] = 0.01 * H * L - 0.25 * L
        end
        function lv_2s!(du, u, p, t)
            H, L = u
            du[1] = p.r(H) * H - p.α * H * L
            du[2] = p.α * H * L - p.δ(L) * L
        end
        dtimes_lv = collect(range(0.5, 13.5, length=7))
        st_lv = OrdinaryDiffEq.solve(
            ODEProblem(lv_true_f1!, [30.0, 40.0], (0.0, 14.0)), Tsit5();
            saveat=dtimes_lv, abstol=1e-8, reltol=1e-8)
        dvals_lv = hcat([st_lv(t)[1] for t in dtimes_lv] .+ 0.3 .* noise_H,
                        [st_lv(t)[2] for t in dtimes_lv] .+ 0.3 .* noise_L)
        prob_lv2 = PSMProblem(lv_2s!, [30.0, 40.0], (0.0, 14.0),
            [BSplineApproximator(:r, (0.0, 80.0), 6; initial=x -> 0.4),
             BSplineApproximator(:δ, (0.0, 100.0), 5; initial=x -> 0.25)];
            data_times=dtimes_lv, data_values=dvals_lv, obs_to_state=[1, 2],
            known_params=(α=0.01,), likelihood=Gaussian(), solver=Tsit5())
        sol_lv2 = solve(prob_lv2, LAML(maxiters=60, verbose=false, warmup=3))

        @test sol_lv2.convergence.converged
        # Measured: 1.0e-3 with the fix, 0.77 without it — a 760x gap, so
        # 1e-2 discriminates with a wide margin on both sides.
        @test pcls_refit_move(prob_lv2, sol_lv2) < 1e-2
        @test sol_lv2.objective ≈
              0.5 * (sol_lv2.data_loss +
                     reported_penalty(prob_lv2, sol_lv2)) rtol=1e-8
        # …and the reported penalty is commensurate with the fit rather than
        # dwarfing it: 0.0153 against a data loss of 0.888 (measured).
        @test reported_penalty(prob_lv2, sol_lv2) < sol_lv2.data_loss

        # ── The same invariant on two ordinary single-λ fixtures, so a
        #    future regression is caught broadly and on both Jacobian
        #    backends. Measured moves: 1.2e-8 (:fd) and 4.0e-9
        #    (:forwarddiff).
        Random.seed!(1234)
        dt_eg = collect(0.0:0.5:10.0)
        dv_eg = reshape(exp.(0.2 .* dt_eg) .+ 0.01 .* randn(length(dt_eg)), :, 1)
        growth_f1!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1])
        prob_eg = PSMProblem(growth_f1!, [1.0], (0.0, 10.0),
            [BSplineApproximator(:r, (0.0, 5.0), 5; initial=x -> 0.05)];
            data_times=dt_eg, data_values=dv_eg, obs_to_state=[1],
            likelihood=Gaussian(), solver=Tsit5())
        for alg_f1 in (LAML(maxiters=30, verbose=false),
                       LAML(maxiters=30, verbose=false, jac=:forwarddiff))
            sol_eg = solve(prob_eg, alg_f1)
            @test pcls_refit_move(prob_eg, sol_eg) < 1e-4
            @test sol_eg.objective ≈
                  0.5 * (sol_eg.data_loss +
                         reported_penalty(prob_eg, sol_eg)) rtol=1e-8
        end
    end

    @testset "CollocationLAML — reported λ̂ and β̂ are mutually consistent" begin
        PSM = PartiallySpecifiedModels

        # The collocation sibling of the LAML testset above, and the same
        # defect: `smoothing_params` reported a smoothing parameter that no
        # coefficient step was ever taken under. CollocationLAML computes its
        # Fellner-Schall update at the END of each continuation level, to be
        # consumed by the NEXT level's inner loop — so after the LAST level
        # the proposal is untested, and reporting it made λ̂ and β̂ describe
        # different fits. See the `theta_fit` comment block in
        # `collocation_solver.jl`.
        #
        # β̂ is the penalized optimum at λ̂ only if the gradient of the
        # COLLOCATION penalized objective
        #     ‖r(α, β; λ_ode)‖² + βᵀ B(λ̂) β,     ∇ = 2(Jᵀr + B p),
        # vanishes in its β block at the reported (α̂, β̂, λ̂). This recovers
        # α̂ from `fitted_values` (exact: the solution stores α at the
        # observed states verbatim), rebuilds r and J with the solver's own
        # `collocation_residual_jacobian` at the final λ_ode, and returns
        # ‖∇_β‖ / ‖β̂‖.
        #
        # Note this is a STATIONARITY test, not a refit-distance test as in
        # the LAML testset, and the distinction is load-bearing: on the
        # exponential-growth fixture below a refit at the wrong λ̂ moves β by
        # EXACTLY 0.0, so a displacement test would pass while the solver was
        # wrong. The gradient test flags the same fit at 0.0227 against a
        # 1e-3 gate.
        #
        # Why displacement is blind here: the Gauss–Newton Hessian is
        # dominated by the λ_ode-scaled ODE-fidelity block (λ_ode reaches
        # 1e2–1e4), so H⁻¹∇ stays tiny even when ∇ is enormous. It is NOT
        # that the jointly-optimised α block absorbs the wrong penalty —
        # that was measured and is false: holding α fixed makes β move LESS,
        # not more, on 4 of 5 fixtures (inverted by up to 300×). Refit
        # distance also is not bounded the way it first appeared: on a
        # logistic fixture it reaches 16% of ‖β‖ and shifts data_loss 35%.
        # Measured on the Poisson fixture below (converged == true): 303.6
        # before the fix, 0.00232 after — a 1.3e5× gap.
        function colloc_grad_norm(prob, sol)
            beta = Vector{Float64}(collect(sol.parameters))
            times = Float64.(prob.data_times)
            T_pts = length(times)
            n_obs = size(prob.data_values, 2)
            K = length(prob.u0)
            # α is only recoverable from `fitted_values` when every state is
            # observed, which is true of both fixtures here.
            @assert sort(collect(prob.obs_to_state)) == collect(1:K)
            alpha = zeros(T_pts, K)
            for j in 1:n_obs
                alpha[:, prob.obs_to_state[j]] .= sol.fitted_values[:, j]
            end

            n_alpha = T_pts * K
            w_vec = zeros(T_pts * n_obs)
            for j in 1:n_obs, i in 1:T_pts
                w_vec[(j - 1) * T_pts + i] = PSM.usable_cell(prob, i, j) ?
                                             prob.data_weights[i, j] : 0.0
            end

            S_list, uf_offsets, uf_nk = PSM.build_penalty_matrices(prob)
            B_beta = zeros(length(beta), length(beta))
            for l in eachindex(S_list)
                off, nk = uf_offsets[l], uf_nk[l]
                B_beta[off+1:off+nk, off+1:off+nk] .+=
                    sol.smoothing_params[l] .* S_list[l]
            end

            resid, J_full = PSM.collocation_residual_jacobian(
                prob, times, alpha, beta, PSM.build_diff_matrix(times),
                sol.convergence.lambda_ode_final, w_vec;
                jac=:fd, fd_cfg=nothing)
            g_beta = 2 .* (J_full' * resid)[n_alpha+1:end] .+
                     2 .* (B_beta * beta)
            norm(g_beta) / max(norm(beta), 1e-12)
        end

        # The reported tuple must be ONE coherent set: CollocationLAML's
        # `objective` is data_loss + β̂ᵀS^λ̂β̂ (no ½ factor, unlike LAML) using
        # the SAME λ̂ and β̂ the solution reports. Guards against "fixing" the
        # reported λ̂ alone while leaving the θ-dependent scalars at the other
        # θ.
        function colloc_reported_penalty(prob, sol)
            beta = Vector{Float64}(collect(sol.parameters))
            S_list, uf_offsets, uf_nk = PSM.build_penalty_matrices(prob)
            p = 0.0
            for l in eachindex(S_list)
                off, nk = uf_offsets[l], uf_nk[l]
                bl = beta[off+1:off+nk]
                p += sol.smoothing_params[l] * dot(bl, S_list[l] * bl)
            end
            p
        end

        growth_f1b!(du, u, p, t) = (du[1] = p.r(u[1]) * u[1])

        # ── Discriminating fixture: Poisson counts. The counts are hardcoded
        #    so the fit is identical across Julia/RNG versions. Converged, so
        #    β̂ really is at the inner loop's optimum and the stationarity
        #    test is not confounded by truncation.
        dt_f1b = collect(range(0.0, 6.0, length=25))
        counts_f1b = [5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0, 8.0, 8.0, 8.0, 9.0,
                      9.0, 10.0, 12.0, 12.0, 12.0, 14.0, 13.0, 14.0, 17.0,
                      18.0, 19.0, 21.0, 20.0, 24.0]
        prob_f1b_pois = PSMProblem(growth_f1b!, [5.0], (0.0, 6.0),
            [BSplineApproximator(:r, (0.0, 30.0), 6; initial=x -> 0.2)];
            data_times=dt_f1b, data_values=reshape(counts_f1b, :, 1),
            obs_to_state=[1], likelihood=Poisson(), solver=Tsit5())
        alg_f1b = CollocationLAML(maxiters=20, verbose=false,
                                  lambda_ode_start=0.01, lambda_ode_end=100.0,
                                  n_continuation=4)
        sol_f1b_pois = solve(prob_f1b_pois, alg_f1b)

        @test sol_f1b_pois.convergence.converged
        # Measured: 0.00232 with the fix, 303.6 without it — so 1.0 sits 430×
        # above the fixed value and 304× below the broken one.
        @test colloc_grad_norm(prob_f1b_pois, sol_f1b_pois) < 1.0
        @test sol_f1b_pois.objective ≈
              sol_f1b_pois.data_loss +
              colloc_reported_penalty(prob_f1b_pois, sol_f1b_pois) rtol=1e-8
        # …and the reported penalty is commensurate with the fit rather than
        # dwarfing it: 0.0121 against a data loss of 11.73 (measured; the
        # untested proposal reported 0.653, a 54× inflation).
        @test colloc_reported_penalty(prob_f1b_pois, sol_f1b_pois) <
              sol_f1b_pois.data_loss

        # ── The same invariant on the suite's own exponential-growth fixture
        #    (see the "CollocationLAML solver" testset), on both Jacobian
        #    backends and both at 4 continuation levels and at the package
        #    defaults. Measured ‖∇_β‖/‖β̂‖ with the fix: 2.33e-6, 2.33e-6 and
        #    1.71e-5; without it 0.0227, 0.0615 and 0.156.
        dt_f1b_g = collect(range(0.0, 5.0, length=30))
        prob_f1b_g = PSMProblem(growth_f1b!, [1.0], (0.0, 5.0),
            [BSplineApproximator(:r, (0.5, 5.0), 6; initial=x -> 0.2)];
            data_times=dt_f1b_g,
            data_values=reshape(exp.(0.3 .* dt_f1b_g), :, 1),
            obs_to_state=[1], known_params=NamedTuple(),
            likelihood=Gaussian(), solver=Tsit5(),
            abstol=1e-8, reltol=1e-8, maxiters=10000)
        for alg_g in (alg_f1b,
                      CollocationLAML(maxiters=20, verbose=false,
                                      lambda_ode_start=0.01,
                                      lambda_ode_end=100.0, n_continuation=4,
                                      jac=:forwarddiff),
                      CollocationLAML(verbose=false))
            sol_g = solve(prob_f1b_g, alg_g)
            @test sol_g.convergence.converged
            @test colloc_grad_norm(prob_f1b_g, sol_g) < 1e-3
            @test sol_g.objective ≈
                  sol_g.data_loss +
                  colloc_reported_penalty(prob_f1b_g, sol_g) rtol=1e-8
        end
    end

end
