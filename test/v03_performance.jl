using Test
using PartiallySpecifiedModels
using PartiallySpecifiedModels: build_evaluator
using PartiallySpecifiedModels: solve, simulate_psm, build_evaluator,
    _bspline_basis_vector, _scam_knot_vector, gamma_to_knot_values,
    build_bspline_evaluator, laml_hessian, _safe_inv, _compute_elbo,
    _kl_gaussian_penalized, _vi_loglik, _variational_simulate, _build_penalty_info
using LinearAlgebra
using Random
using ForwardDiff
using Statistics
using OrdinaryDiffEq

const PSM_PERF = PartiallySpecifiedModels

@testset "M2 — Zero-allocation approximators & performance engineering" begin

    @testset "TensorBSplineApproximator: exact parity, Dual-safe, 0 B/call" begin
        a = TensorBSplineApproximator(:f, (0.0, 2.0), (1.0, 5.0), 6, 5)
        rng = MersenneTwister(123)
        C_vec = randn(rng, 30)
        ev = build_evaluator(a, C_vec)

        # Reference nested CubicSpline evaluator
        kx = collect(range(0.0, 2.0, length=6))
        ky = collect(range(1.0, 5.0, length=5))
        C_mat = reshape(C_vec, 6, 5)
        col_splines = [build_bspline_evaluator(kx, C_mat[:, j]) for j in 1:5]
        ref_eval(x, y) = build_bspline_evaluator(ky, [s(x) for s in col_splines])(y)

        # Check interior, edges, and all 4 extrapolated corners
        xs = [-0.7, 0.0, 0.35, 1.0, 1.73, 2.0, 2.8]
        ys = [-0.5, 1.0, 1.9, 3.2, 4.8, 5.0, 6.4]
        max_err = 0.0
        for x in xs, y in ys
            max_err = max(max_err, abs(ev(x, y) - ref_eval(x, y)))
        end
        @test max_err < 1e-12

        # ForwardDiff Dual in inputs (x, y) and in parameters
        g_in = ForwardDiff.gradient(v -> ev(v[1], v[2]), [0.73, 2.61])
        g_in_ref = ForwardDiff.gradient(v -> ref_eval(v[1], v[2]), [0.73, 2.61])
        @test isapprox(g_in, g_in_ref; rtol=1e-11, atol=1e-12)

        g_p = ForwardDiff.gradient(p -> build_evaluator(a, p)(0.73, 2.61), C_vec)
        @test all(isfinite, g_p) && norm(g_p) > 0

        # Zero allocations for Float64 evaluation
        ev(0.73, 2.61) # warmup
        alloc_inner = @allocated ev(0.73, 2.61)
        alloc_extrap = @allocated ev(-0.5, 6.2)
        @test alloc_inner == 0
        @test alloc_extrap == 0
    end

    @testset "ShapeConstrainedBSplineApproximator: exact parity, Dual-safe, 0 B/call" begin
        rng = MersenneTwister(456)
        for constr in (:increasing, :decreasing, :inc_convex, :dec_concave,
                       :inc_zero_left, :dec_zero_right, :positive)
            a = ShapeConstrainedBSplineApproximator(:g, (0.0, 10.0), 8, constr)
            gamma = randn(rng, PSM_PERF.nparams(a))
            ev = build_evaluator(a, gamma)
            beta = gamma_to_knot_values(a, gamma)
            xk = _scam_knot_vector(a.domain, 8)
            off = constr in (:inc_zero_left, :dec_zero_left) ?
                  dot(_bspline_basis_vector(0.0, xk, 4), beta) :
                  constr in (:inc_zero_right, :dec_zero_right) ?
                  dot(_bspline_basis_vector(10.0, xk, 4), beta) : 0.0

            for x in [0.0, 0.3, 2.7, 5.0, 8.9, 10.0]
                ref = dot(_bspline_basis_vector(x, xk, 4), beta) - off
                @test isapprox(ev(x), ref; atol=1e-12, rtol=1e-12)
            end

            # Dual safety in x and gamma
            d_x = ForwardDiff.derivative(ev, 3.7)
            @test isfinite(d_x)
            g_gam = ForwardDiff.gradient(g -> build_evaluator(a, g)(3.7), gamma)
            @test all(isfinite, g_gam)

            # Zero allocation
            ev(3.7)
            @test (@allocated ev(3.7)) == 0
            @test (@allocated ev(-1.0)) == 0
            @test (@allocated ev(12.0)) == 0
        end
    end

    @testset "COMONetApproximator: Dual-safe & 0 B/call" begin
        rng = MersenneTwister(789)
        for constr in (:increasing, :decreasing, :inc_convex, :inc_concave,
                       :dec_convex, :convex, :concave, :positive)
            a = COMONetApproximator(:c, (0.0, 5.0), [6, 4], constr;
                                    activation=:softplus, rng_seed=11)
            th = PSM_PERF.initial_params(a)
            ev = build_evaluator(a, th)

            # Warmup and check 0 allocations
            y0 = ev(2.3)
            @test isfinite(y0)
            @test (@allocated ev(2.3)) == 0

            # Dual safety in x and theta
            @test isfinite(ForwardDiff.derivative(ev, 2.3))
            g_th = ForwardDiff.gradient(p -> build_evaluator(a, p)(2.3), th)
            @test all(isfinite, g_th)
        end
    end

    @testset "SingleIndexApproximator & GPApproximator: Dual-safe & 0 B/call" begin
        # SingleIndex with unconstrained and shape-constrained outer smooths
        mu = [1.0, 2.0, 3.0]
        Sig = [1.0 0.2 0.0; 0.2 1.5 0.1; 0.0 0.1 0.8]
        for constr in (:none, :increasing)
            si = SingleIndexApproximator(:si, 3, 6; constraint=constr, index_stats=(mu, Sig))
            p0 = collect(range(-0.3, 0.5, length=PSM_PERF.nparams(si)))
            ev = build_evaluator(si, p0)
            v = ev(1.2, 2.4, 2.9)
            @test isfinite(v)
            @test (@allocated ev(1.2, 2.4, 2.9)) == 0
            g_in = ForwardDiff.gradient(u -> ev(u[1], u[2], u[3]), [1.2, 2.4, 2.9])
            @test all(isfinite, g_in)
            @test_throws ArgumentError ev(1.2, 2.4)
        end

        # GPApproximator across all 3 kernels
        for kern in (:sqexp, :matern32, :matern52)
            gp = GPApproximator(:gp, (0.0, 4.0), 7; kernel=kern)
            p0 = collect(range(0.1, 1.5, length=7))
            ev = build_evaluator(gp, p0)
            v = ev(1.7)
            @test isfinite(v)
            @test (@allocated ev(1.7)) == 0
            @test isfinite(ForwardDiff.derivative(ev, 1.7))
        end
    end

    @testset "Block-sparse laml_hessian matches dense reference" begin
        rng = MersenneTwister(321)
        S1 = PSM_PERF.spline_penalty_matrix(collect(range(0.0, 1.0, length=6)))
        S2 = PSM_PERF.spline_penalty_matrix(collect(range(0.0, 1.0, length=8)))
        S3 = PSM_PERF.spline_penalty_matrix(collect(range(0.0, 1.0, length=5)))
        S_list = [S1, S2, S3]
        offsets = [0, 6, 14]
        nks = [6, 8, 5]
        n_p = 19
        rho = [1.5, -2.0, 4.2]
        lambda = exp.(rho)
        beta = randn(rng, n_p)
        A = randn(rng, 40, n_p)
        JWJ = A' * A
        S_lam = PSM_PERF.build_S_lambda(S_list, offsets, nks, rho, n_p)
        H = JWJ + S_lam
        sigma2 = 0.25

        H_sparse = laml_hessian(GaussianLikelihood(), beta, S_list, offsets, nks, rho, n_p, H, sigma2)

        # Dense reference implementation
        H_inv = _safe_inv(H)
        m = length(S_list)
        HinvS = [begin
            Sf = zeros(n_p, n_p)
            Sf[offsets[k]+1:offsets[k]+nks[k], offsets[k]+1:offsets[k]+nks[k]] .= lambda[k] .* S_list[k]
            H_inv * Sf
        end for k in 1:m]
        H_dense = zeros(m, m)
        for j in 1:m, k in j:m
            tr_val = sum(HinvS[j][i, l] * HinvS[k][l, i] for i in 1:n_p, l in 1:n_p)
            if j == k
                bk = beta[offsets[k]+1:offsets[k]+nks[k]]
                bSb = dot(bk, S_list[k] * bk)
                H_dense[k, k] = -0.5 * lambda[k] * bSb / sigma2 - 0.5 * tr(HinvS[k]) + 0.5 * tr_val
            else
                H_dense[j, k] = H_dense[k, j] = 0.5 * tr_val
            end
        end
        @test isapprox(H_sparse, H_dense; rtol=1e-12, atol=1e-12)
    end

    @testset "LAML defaults (jac=:auto, exploratory_maxiters=5000) & speedup" begin
        alg_def = LAML()
        @test alg_def.jac === :auto
        @test alg_def.exploratory_maxiters == 5_000
        @test_throws ArgumentError LAML(jac=:invalid)
        @test_throws ArgumentError LAML(exploratory_maxiters=0)

        # Two-species Lotka-Volterra PSM (12 params)
        function lv!(du, u, p, t)
            x, y = u
            du[1] = p.r(x) * x - 0.1 * x * y
            du[2] = p.g(x) * y - 0.3 * y
        end
        ts = collect(0.0:0.5:10.0)
        u0 = [1.5, 1.0]
        # Generate clean synthetic data
        p_true = (r = x -> 0.5 * (1.0 - x / 6.0), g = x -> 0.25 * x / (1.0 + 0.3 * x))
        prob_ref = PSMProblem(
            lv!, u0, (0.0, 10.0),
            [BSplineApproximator(:r, (0.0, 6.0), 6),
             BSplineApproximator(:g, (0.0, 6.0), 6)];
            data_times=ts, data_values=zeros(length(ts), 2))
        # Simulate truth using ODEProblem directly
        ode_true = ODEProblem(lv!, u0, (0.0, 10.0), p_true)
        sol_true = OrdinaryDiffEq.solve(ode_true, Tsit5(); saveat=ts, abstol=1e-10, reltol=1e-10)
        rng = MersenneTwister(42)
        Y = hcat([u[1] for u in sol_true.u], [u[2] for u in sol_true.u]) .+ 0.01 .* randn(rng, length(ts), 2)
        prob = PSMProblem(
            lv!, u0, (0.0, 10.0),
            [BSplineApproximator(:r, (0.0, 6.0), 6),
             BSplineApproximator(:g, (0.0, 6.0), 6)];
            data_times=ts, data_values=Y)

        # Warmup both paths once before timing
        sol_auto = solve(prob, LAML(maxiters=40))
        sol_fd   = solve(prob, LAML(maxiters=40, jac=:fd))
        @test sol_auto.convergence.converged
        @test sol_fd.convergence.converged
        @test sol_auto.data_loss < 0.05
        @test isapprox(sol_auto.data_loss, sol_fd.data_loss; rtol=0.15)

        # Compare per-Jacobian timing (min of 5 calls, immune to GC jitter and
        # differing outer IRLS iteration counts between exact AD and noisy FD).
        beta0 = vcat(PSM_PERF.initial_params.(prob.approximators)...)
        f0 = vec(simulate_psm(prob, beta0))
        n_t, n_o = size(prob.data_values)
        J_buf = zeros(n_t * n_o, length(beta0))
        fd_cfg = ForwardDiff.JacobianConfig(nothing, beta0)
        dam = ones(length(f0))
        # Warmup both Jacobian backends once
        PSM_PERF.compute_jacobian!(J_buf, prob, beta0, f0, n_t, n_o; dam=dam, jac=:forwarddiff, fd_cfg=fd_cfg)
        PSM_PERF.compute_jacobian!(J_buf, prob, beta0, f0, n_t, n_o; dam=dam, jac=:fd)
        t_auto = minimum(@elapsed(PSM_PERF.compute_jacobian!(J_buf, prob, beta0, f0, n_t, n_o; dam=dam, jac=:forwarddiff, fd_cfg=fd_cfg)) for _ in 1:5)
        t_fd   = minimum(@elapsed(PSM_PERF.compute_jacobian!(J_buf, prob, beta0, f0, n_t, n_o; dam=dam, jac=:fd)) for _ in 1:5)
        # ForwardDiff default should be substantially faster than FD on 12-param ODE
        @test t_auto < t_fd
    end

    @testset "VariationalSolver n_p-partial chain rule matches 2*n_p ForwardDiff" begin
        function exp_dyn!(du, u, p, t)
            du[1] = p.r(u[1]) * u[1]
        end
        ts = collect(0.0:1.0:5.0)
        Y = reshape(exp.(0.1 .* ts), :, 1)
        prob = PSMProblem(exp_dyn!, [1.0], (0.0, 5.0),
                          [BSplineApproximator(:r, (0.5, 3.0), 5)];
                          data_times=ts, data_values=Y,
                          abstol=1e-11, reltol=1e-11)
        n_p = 5
        Λ = Matrix{Float64}(I, n_p, n_p)
        pens, offs, _ = _build_penalty_info(prob)
        Λ .+= pens[1]
        logdetΛ = logdet(cholesky(Symmetric(Λ)))
        rng = MersenneTwister(99)
        mu = fill(0.1, n_p)
        log_sigma = fill(-2.0, n_p)
        epsilons = randn(rng, n_p, 3)
        obs_var = 0.01

        # Brute-force 2*n_p ForwardDiff gradient
        phi = vcat(mu, log_sigma)
        g_ref = ForwardDiff.gradient(
            pv -> -_compute_elbo(prob, pv[1:n_p], pv[n_p+1:end], Λ, logdetΛ, epsilons, obs_var),
            phi)

        # Chain-rule n_p-partial gradient
        sigma = exp.(log_sigma)
        dΛ = diag(Λ)
        g_mu = zeros(n_p); g_ls = zeros(n_p)
        for s in 1:3
            th = mu .+ sigma .* epsilons[:, s]
            g_s = ForwardDiff.gradient(b -> _vi_loglik(prob.likelihood, _variational_simulate(prob, b),
                                                       prob.data_values, prob.data_weights, obs_var), th)
            g_mu .+= g_s
            g_ls .+= g_s .* sigma .* epsilons[:, s]
        end
        g_chain = vcat(-g_mu ./ 3 .+ Λ * mu, -g_ls ./ 3 .+ (dΛ .* sigma.^2 .- 1.0))
        @test isapprox(g_chain, g_ref; rtol=1e-6, atol=1e-6)
    end
end
