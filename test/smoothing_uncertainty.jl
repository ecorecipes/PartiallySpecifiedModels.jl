module SmoothingUncertaintyTests

using Test, LinearAlgebra, ForwardDiff, Random
using PartiallySpecifiedModels, Lux, FluxKAN
using PartiallySpecifiedModels: solve
const PSM = PartiallySpecifiedModels

struct RidgeRate <: AbstractApproximator
    name::Symbol
end
PSM.nparams(::RidgeRate) = 1
PSM.initial_params(::RidgeRate) = [0.1]
PSM.penalty_matrix(::RidgeRate) = ones(1, 1)
PSM.penalty_blocks(::RidgeRate) = [(ones(1, 1), 1:1)]
PSM.build_evaluator(::RidgeRate, b) = x -> b[1]

struct BareRate <: AbstractApproximator
    name::Symbol
end
PSM.nparams(::BareRate) = 1
PSM.initial_params(::BareRate) = [0.1]
PSM.penalty_matrix(::BareRate) = nothing
PSM.build_evaluator(::BareRate, b) = x -> b[1]

struct TwoPenaltyPolynomial <: AbstractApproximator
    name::Symbol
end
PSM.nparams(::TwoPenaltyPolynomial) = 3
PSM.initial_params(::TwoPenaltyPolynomial) = [0.1, 0.0, 0.0]
PSM.penalty_matrix(::TwoPenaltyPolynomial) = Matrix(Diagonal([0.0, 1.0, 2.0]))
PSM.penalty_blocks(::TwoPenaltyPolynomial) = [(ones(1, 1), 2:2), (fill(2.0, 1, 1), 3:3)]
PSM.build_evaluator(::TwoPenaltyPolynomial, b) = x -> b[1] + b[2]*x + b[3]*x^2

function replace_solution(sol; convergence=sol.convergence, parameters=sol.parameters,
                           smoothing_params=sol.smoothing_params)
    PSMSolution(parameters, sol.objective, sol.data_loss, sol.edf, smoothing_params,
        sol.fitted_values, sol.data_values, sol.data_times, sol.unknown_functions, convergence)
end

function linear_fixture()
    t = collect(range(-1.0, 1.0, length=15))
    X = hcat(ones(length(t)), t, t.^2)
    w = collect(range(0.7, 1.8, length=length(t)))
    y = 0.8 .+ 0.7t .+ 0.4t.^2 .+ 0.1sin.(collect(1:length(t)))
    A, rhs = X' * Diagonal(w) * X, X' * (w .* y)
    S = [Matrix(Diagonal([0.0, 1.0, 0.0])), Matrix(Diagonal([0.0, 0.0, 2.0]))]
    rho = log.([0.4, 0.9])
    B = exp.(rho) .* S
    H = A + sum(B)
    beta = H \ rhs
    (; X, w, y, A, rhs, S, rho, B, H, beta, n_eff=length(t)-1)
end

@testset "Analytic smoothing covariance" begin
    @testset "Scalar mean and inverse-Cholesky root terms" begin
        A, B, beta, sigma2, Vrho = 3.0, 8.0, [0.7], 2.5, [0.3;;]
        H = A + B
        d = PSM._smoothing_derivatives(beta, [H;;], [[B;;]])
        c = PSM._smoothing_delta_covariance(d.sensitivity, d.root_derivatives, sigma2, Vrho)
        # Closed-form probes measured <=1.39e-17 in these four quantities;
        # 1e-12 leaves >70000x floating-point headroom.
        @test isapprox(only(d.sensitivity), -B*only(beta)/H; atol=1e-12, rtol=1e-12)
        @test isapprox(only(only(d.root_derivatives)), -B/(2H^1.5); atol=1e-12, rtol=1e-12)
        @test isapprox(only(c.mean_correction), only(Vrho)*(B*only(beta)/H)^2; atol=1e-12, rtol=1e-12)
        @test isapprox(only(c.root_correction), only(Vrho)*sigma2*B^2/(4H^3); atol=1e-12, rtol=1e-12)

        zero = PSM._smoothing_derivatives([0.0], [H;;], [[B;;]])
        z = PSM._smoothing_delta_covariance(zero.sensitivity, zero.root_derivatives, sigma2, Vrho)
        @test iszero(only(z.mean_correction))
        @test z.root_correction == c.root_correction
        @test only(z.root_correction) > 0

        scaled = PSM._smoothing_delta_covariance(d.sensitivity, d.root_derivatives, 4sigma2, Vrho)
        @test scaled.mean_correction == c.mean_correction
        @test scaled.root_correction == 4c.root_correction
        fixed = PSM._smoothing_delta_covariance(d.sensitivity, d.root_derivatives, sigma2, zeros(1, 1))
        @test iszero(only(fixed.mean_correction)) && iszero(only(fixed.root_correction))
    end

    @testset "Independent profile objective and noncommuting matrix derivatives" begin
        f = linear_fixture()
        (; X, w, y, A, rhs, S, rho, B, H, beta, n_eff) = f
        d = PSM._smoothing_derivatives(beta, H, B)
        rss = sum(w .* (y - X*beta).^2)
        # Hold the numerical determinant ridge fixed, as the correction does.
        determinant_ridge = 1e-10maximum(abs, diag(H)) + 1e-15
        function objective(r)
            penalty = sum(exp.(r) .* S)
            h = A + penalty
            bh = h \ rhs
            Q = sum(w .* (y - X*bh).^2) + dot(bh, penalty*bh)
            -n_eff/2*log(Q/n_eff) + sum(r)/2 -
                logdet(Symmetric(h + determinant_ridge*I))/2
        end
        profile = PSM._smoothing_profile_curvature(beta, H, B, [1, 1], d.sensitivity,
                                                   rss, n_eff; determinant_ridge)
        exact_D = ForwardDiff.jacobian(r -> (A + sum(exp.(r).*S)) \ rhs, rho)
        # Independent AD probes measured D <=2.78e-17, Hessian <=1.89e-15,
        # gradient <=4.89e-15; 1e-11 provides >2000x headroom.
        @test isapprox(d.sensitivity, exact_D; atol=1e-11, rtol=1e-11)
        @test isapprox(profile.hessian, -ForwardDiff.hessian(objective, rho); atol=1e-11, rtol=1e-11)
        @test isapprox(profile.gradient, ForwardDiff.gradient(objective, rho); atol=1e-11, rtol=1e-11)
        @test norm((H\B[1])*(H\B[2]) - (H\B[2])*(H\B[1])) > 0

        step = 1e-4
        for k in eachindex(rho)
            plus, minus = copy(rho), copy(rho)
            plus[k] += step
            minus[k] -= step
            fd = (inv(cholesky(Symmetric(A + sum(exp.(plus).*S))).U) -
                  inv(cholesky(Symmetric(A + sum(exp.(minus).*S))).U)) / (2step)
            # Independent perturbed Cholesky probes measured <=5.79e-11;
            # 1e-8 allows >170x differencing/roundoff headroom.
            @test isapprox(d.root_derivatives[k], fd; atol=1e-8, rtol=1e-8)
        end
        Vrho = [0.4 0.17; 0.17 0.8]
        c = PSM._smoothing_delta_covariance(d.sensitivity, d.root_derivatives, 2.5, Vrho)
        direct_root = sum(2.5Vrho[j,k] * d.root_derivatives[j] * d.root_derivatives[k]'
                          for j in 1:2, k in 1:2)
        # These are Float64 matrix identities; the derivative-oracle
        # discrepancy above was <=4.89e-15, with >2000x headroom here.
        @test isapprox(c.mean_correction, exact_D*Vrho*exact_D'; atol=1e-11, rtol=1e-11)
        @test isapprox(c.root_correction, direct_root; atol=1e-11, rtol=1e-11)
        for term in (c.mean_correction, c.root_correction, c.mean_correction+c.root_correction)
            @test minimum(eigvals(Symmetric(term))) >= -1e-11
        end
        @test !isapprox(c.root_correction,
            sum(2.5Vrho[j,j] * d.root_derivatives[j] * d.root_derivatives[j]' for j in 1:2))

        L = Diagonal([2.0, 0.5, 3.0])
        rescaled = PSM._smoothing_derivatives(L*beta, inv(L)*H*inv(L),
                                             [inv(L)*b*inv(L) for b in B])
        cs = PSM._smoothing_delta_covariance(rescaled.sensitivity, rescaled.root_derivatives, 2.5, Vrho)
        @test isapprox(cs.mean_correction, L*c.mean_correction*L; atol=1e-11, rtol=1e-11)
        @test isapprox(cs.root_correction, L*c.root_correction*L; atol=1e-11, rtol=1e-11)

        @test_throws DomainError PSM._smoothing_profile_curvature(zeros(3), H, B, [1,1],
            zeros(3,2), 0.0, n_eff)
        @test_throws DomainError PSM._smoothing_profile_curvature(beta, H, B, [1,1],
            d.sensitivity, rss, 0)
        @test_throws DomainError PSM._smoothing_cholesky(zeros(2,2))
        @test_throws DomainError PSM._smoothing_cholesky([1.0 0.2; 0.3 1.0])
    end

    @testset "Explicit curvature and covariance policies" begin
        @test_throws DomainError PSM._smoothing_rho_covariance(zeros(1,1), 0.0)
        @test_throws DomainError PSM._smoothing_rho_covariance([-1.0;;], 0.0)
        @test_throws DomainError PSM._smoothing_rho_covariance([NaN;;], 0.0)
        regularized = PSM._smoothing_rho_covariance(zeros(1,1), 0.25)
        @test regularized.eigenvalues == [0.0]
        @test regularized.covariance == [4.0;;]
        shifted = PSM._smoothing_rho_covariance([-1.0;;], 2.0)
        @test shifted.covariance == [1.0;;] && shifted.eigenvalues == [-1.0]
        for reg in (-1.0, NaN, Inf)
            @test_throws ArgumentError PSM._smoothing_rho_covariance(ones(1,1), reg)
        end
    end

    @testset "Gaussian LAML, unchanged means/defaults, and no re-simulation" begin
        times = collect(0.0:0.2:2.0)
        values = 1.5times + 0.15sin.(collect(1:length(times)))
        calls = Ref(0)
        rhs! = (du,u,p,t) -> (calls[] += 1; du[1]=p.r(t); nothing)
        prob = PSMProblem(rhs!, [0.0], (0.0,2.0), [RidgeRate(:r)];
            data_times=times, data_values=reshape(values,:,1), abstol=1e-11, reltol=1e-11)
        queries = Dict(:r => [0.2,0.8])
        for jac in (:fd, :forwarddiff)
            sol = solve(prob, LAML(maxiters=100, tol=1e-10, jac=jac))
            @test sol.convergence.smoothing_advanced && !sol.convergence.smoothing_fixed
            before = deepcopy(sol.convergence.smoothing_state)
            saved = (copy(sol.parameters), copy(sol.smoothing_params), copy(sol.convergence.V_beta))
            ncalls = calls[]
            correction = smoothing_covariance_correction(sol, prob)
            @test calls[] == ncalls
            @test correction.method == :wps_local_gaussian && correction.rho_source == :profile_reml
            @test correction.jac == jac && correction.rho_regularization == 0.0
            @test correction.n_eff == length(times)
            @test only(correction.eigenvalues) > 0
            @test correction.conditional_covariance == sol.convergence.sigma2 .* sol.convergence.V_beta
            @test correction.covariance == correction.conditional_covariance +
                correction.mean_correction + correction.root_correction
            @test only(correction.mean_correction) > 0 && only(correction.root_correction) > 0
            A, lambda = dot(times,times), only(sol.smoothing_params)
            beta = dot(times,values)/(A+lambda)
            # Linear-ODE ridge probes measured coefficient error <=1.89e-11;
            # 1e-7 permits >5000x numerical-solve/FD headroom without pinning lambda.
            @test isapprox(only(sol.parameters), beta; atol=1e-7, rtol=1e-7)
            @test isapprox(only(correction.coefficient_sensitivity), -lambda*beta/(A+lambda);
                           atol=1e-7, rtol=1e-7)
            @test isapprox(only(correction.root_derivatives[1]),
                           -lambda/(2(A+lambda+correction.covariance_ridge)^1.5);
                           atol=1e-7, rtol=1e-7)

            plain = confidence_band(sol, prob; uf_points=queries)[:r]
            explicit = confidence_band(sol, prob; uf_points=queries, unconditional=false)[:r]
            @test plain == explicit && keys(plain) == (:points,:fitted,:lower,:upper,:se)
            rng = Random.Xoshiro(31)
            saved_rng = copy(rng)
            point = confidence_band(sol, prob; uf_points=queries, unconditional=true, rng)[:r]
            @test rand(rng) == rand(saved_rng)
            @test point.fitted == plain.fitted && point.conditioning == :smoothing_corrected
            @test point.scope == :pointwise && point.n_draws == 0
            @test point.rho_source == :profile_reml && point.covariance_method == :wps_local_gaussian
            @test all(point.se .> plain.se)
            @test isapprox(point.se, fill(sqrt(only(correction.covariance)),2); atol=1e-7, rtol=1e-7)
            sim = confidence_band(sol, prob; uf_points=queries, unconditional=true,
                interval=:simultaneous, nsim=2000, rng=Random.Xoshiro(61))[:r]
            @test sim.scope == :function_grid && sim.conditioning == :smoothing_corrected
            @test sim.fitted == plain.fitted && sim.n_draws == 2000
            @test sim.critical >= PSM._qnorm(0.975)
            @test isapprox(sim.se, point.se; atol=1e-7, rtol=1e-7)
            @test calls[] == ncalls
            @test isequal(sol.convergence.smoothing_state, before)
            @test saved == (sol.parameters, sol.smoothing_params, sol.convergence.V_beta)

            regularized = smoothing_covariance_correction(sol, prob; rho_regularization=0.25)
            @test regularized.rho_regularization == 0.25
            @test only(regularized.rho_covariance) < only(correction.rho_covariance)
            @test regularized.conditional_covariance == correction.conditional_covariance
            @test_throws ArgumentError confidence_band(sol, prob; uf_points=queries, rho_regularization=0.25)
            @test_throws ArgumentError confidence_band(sol, prob; uf_points=queries, rho_covariance=ones(1,1))
            @test_throws ArgumentError smoothing_covariance_correction(sol, prob;
                rho_covariance=ones(1,1), rho_regularization=0.25)
            @test_throws DimensionMismatch smoothing_covariance_correction(sol, prob; rho_covariance=ones(2,2))
            @test_throws ArgumentError smoothing_covariance_correction(sol, prob; rho_covariance=[1.0])
            for bad in ([-1.0;;], [NaN;;], [Inf;;])
                @test_throws DomainError smoothing_covariance_correction(sol, prob; rho_covariance=bad)
            end
            for (key,value) in ((:solver,:GCVSolver), (:smoothing_state,nothing), (:smoothing_advanced,false))
                changed = replace_solution(sol; convergence=merge(sol.convergence, NamedTuple{(key,)}((value,))))
                @test_throws ArgumentError smoothing_covariance_correction(changed, prob)
            end
            @test_throws ArgumentError smoothing_covariance_correction(
                replace_solution(sol; parameters=2sol.parameters), prob)
            @test_throws ArgumentError smoothing_covariance_correction(
                replace_solution(sol; smoothing_params=2sol.smoothing_params), prob)
            @test_throws DomainError smoothing_covariance_correction(replace_solution(sol;
                convergence=merge(sol.convergence,(V_beta=2sol.convergence.V_beta,))), prob)
            for bound in (PSM.RHO_MIN, PSM.RHO_MAX)
                lambda_bound = [exp(bound)]
                state = merge(sol.convergence.smoothing_state, (lambda=lambda_bound,))
                at_bound = replace_solution(sol; smoothing_params=lambda_bound,
                    convergence=merge(sol.convergence, (smoothing_state=state,)))
                failure = try
                    smoothing_covariance_correction(at_bound, prob)
                catch e
                    e
                end
                @test failure isa DomainError
                @test occursin("optimization bound", sprint(showerror, failure))
            end

            fixed = solve(prob, LAML(maxiters=60, tol=1e-10, jac=jac, fixed_lambda=0.25lambda))
            @test_throws ArgumentError smoothing_covariance_correction(fixed, prob)
            supplied = smoothing_covariance_correction(fixed, prob; rho_covariance=correction.rho_covariance)
            @test supplied.rho_source == :supplied && supplied.rho_hessian === nothing
            @test supplied.smoothing_fixed && !supplied.smoothing_advanced
            @test supplied.rho_covariance == correction.rho_covariance
            @test supplied.coefficient_sensitivity != correction.coefficient_sensitivity
            @test confidence_band(fixed, prob; uf_points=queries, unconditional=true,
                rho_covariance=correction.rho_covariance)[:r].rho_source == :supplied
        end
        count_prob = PSMProblem(rhs!, [0.0], (0.0,2.0), [RidgeRate(:r)];
            data_times=times, data_values=reshape(values,:,1), likelihood=Poisson())
        unpen = PSMProblem(rhs!, [0.0], (0.0,2.0), [BareRate(:r)];
            data_times=times, data_values=reshape(values,:,1))
        bare = solve(unpen, LAML(maxiters=20))
        @test bare.convergence.smoothing_state === nothing
        @test_throws ArgumentError smoothing_covariance_correction(bare, unpen)
        @test_throws ArgumentError smoothing_covariance_correction(bare, count_prob)
    end

    @testset "Masked data, multiple penalties, offsets and owned matrices" begin
        times = collect(0.0:0.1:2.0)
        a, r = BareRate(:a), TwoPenaltyPolynomial(:r)
        y = hcat(0.8times, 0.5times + 0.4times.^2 - 0.1times.^3)
        y .+= 0.05sin.(reshape(collect(1:length(y)),size(y)))
        w = ones(size(y))
        y[2,1] = NaN
        w[4,2] = 0.0
        rhs! = (du,u,p,t) -> (du[1]=p.a(t); du[2]=p.r(t); nothing)
        prob = PSMProblem(rhs!, [0.0,0.0], (0.0,2.0), [a,r];
            data_times=times, data_values=y, data_weights=w, abstol=1e-11, reltol=1e-11)
        sol = solve(prob, LAML(maxiters=60, tol=1e-10, fixed_lambda=0.4, jac=:forwarddiff))
        Vrho = [0.3 0.1; 0.1 0.4]
        correction = smoothing_covariance_correction(sol, prob; rho_covariance=Vrho)
        @test correction.n_eff == length(y)-4
        @test size(correction.coefficient_sensitivity) == (4,2)
        @test all(iszero, correction.coefficient_sensitivity[1,:])
        @test all(isfinite, correction.coefficient_score)
        @test sol.convergence.smoothing_state.offsets == [2,3]
        @test_throws DomainError smoothing_covariance_correction(sol, prob;
            rho_covariance=[0.3 0.2; 0.1 0.4])
        points = [0.2,0.6,1.4]
        band = confidence_band(sol, prob; unconditional=true, rho_covariance=Vrho,
            uf_points=Dict(:a=>points,:r=>points))[:r]
        J = hcat(ones(3),points,points.^2)
        expected = sqrt.(diag(J * correction.covariance[2:4,2:4] * J'))
        # Same linear ODE algebra as the scalar probe (<=1.89e-11);
        # 1e-7 retains >5000x headroom for integration and function differencing.
        @test isapprox(band.se, expected; atol=1e-7, rtol=1e-7)
        sim = confidence_band(sol, prob; unconditional=true, rho_covariance=Vrho,
            uf_points=Dict(:a=>points,:r=>points), interval=:simultaneous,
            nsim=2000, rng=Random.Xoshiro(9))[:r]
        @test isapprox(sim.se, expected; atol=1e-7, rtol=1e-7)
        @test sim.fitted == band.fitted
        @test minimum(eigvals(Symmetric(correction.covariance-correction.conditional_covariance))) >= -1e-7
        correction.rho_covariance[1,1] = 99
        @test Vrho == [0.3 0.1; 0.1 0.4]
        correction.conditional_covariance[1,1] = 99
        @test sol.convergence.sigma2*sol.convergence.V_beta[1,1] != 99
        changed = deepcopy(prob)
        changed.data_weights[1,1] = 2.0
        @test_throws ArgumentError smoothing_covariance_correction(sol, changed; rho_covariance=Vrho)
    end

    @testset "Mixed multivariate KAN corrected bands" begin
        s = BSplineApproximator(:s, (0.0,2.0), 4; initial=x->0.2)
        a = KANApproximator(:g, LuxKANLinear(2,1;grid_size=3,standalone_spline_scale=false);
            input_domains=((0.0,2.0),(0.0,2.0)), penalty=:ridge)
        times = collect(0.0:0.1:2.0)
        y = hcat(0.4times, 0.3times + 0.1times.^2)
        y .+= 0.02sin.(reshape(collect(1:length(y)),size(y)))
        rhs! = (du,u,p,t) -> (du[1]=p.s(t); du[2]=p.g(t,t^2/2); nothing)
        prob = PSMProblem(rhs!, [0.0,0.0], (0.0,2.0), [s,a];
            data_times=times, data_values=y)
        sol = solve(prob, LAML(maxiters=40, fixed_lambda=0.2, jac=:forwarddiff))
        Vrho = [0.2 0.08; 0.08 0.3]
        correction = smoothing_covariance_correction(sol, prob; rho_covariance=Vrho)
        points = [0.2 0.1; 0.8 1.2; 1.7 0.4]
        band = confidence_band(sol, prob; uf_points=Dict(:g=>points), unconditional=true,
            rho_covariance=Vrho, uf_ngrid=3)[:g]
        ids = 5:length(sol.parameters)
        grid = only(a.layers).knots
        for (i,point) in enumerate(eachrow(points))
            x = point .- 1
            design = [FluxKAN.SiLU.(x); reduce(vcat, [
                vec(FluxKAN.bspline_basis(reshape([x[j]],1,1), vec(grid[j,:]),3)) for j in 1:2])]
            se = sqrt(dot(design, correction.covariance[ids,ids]*design))
            # Raw-basis KAN covariance oracles measured <=5.56e-17 in
            # kan_uncertainty.jl; 1e-10 permits >1e6x roundoff headroom.
            @test isapprox(band.se[i], se; atol=1e-10, rtol=1e-10)
        end
        @test band.points == points && band.conditioning == :smoothing_corrected
        @test band.rho_source == :supplied
    end
end

end
