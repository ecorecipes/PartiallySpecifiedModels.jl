# ─── Multi-Trajectory / Hierarchical PSMs ───────────────────────────
#
# Joint LAML estimation across G replicate trajectories (different initial
# conditions, time grids, treatments, or observation masks) sharing smooth
# mechanistic unknown functions and optional group-level random effects
# (`RandomEffectApproximator`).

"""
    MultiTrajectoryPSMProblem(problems; group_ids=1:length(problems))
    MultiTrajectoryPSMProblem(dynamics!, u0s, tspans, approximators;
                              data_times_list, data_values_list, ...)

A hierarchical / multi-replicate PSM problem combining `G` individual
[`PSMProblem`](@ref)s that share the same unknown-function approximators
(and optionally group-level [`RandomEffectApproximator`](@ref)s indexed via
`p.group_id`).

Each trajectory `g ∈ 1:G` may have its own initial condition `u0`, `tspan`,
observation timestamps `data_times`, observation matrix `data_values`,
`data_weights`, `known_params`, and per-trajectory `estimate_u0`.
"""
struct MultiTrajectoryPSMProblem
    problems::Vector{PSMProblem}
    group_ids::Vector{Int}
end

function _inject_group_id(prob::PSMProblem, gid::Int)
    haskey(prob.known_params, :group_id) && return prob
    kp = merge(prob.known_params, (group_id=gid,))
    PSMProblem(prob.dynamics!, prob.u0, prob.tspan, prob.approximators,
               prob.data_times, prob.data_values, prob.data_weights,
               prob.obs_to_state, kp, prob.likelihood,
               prob.ode_solver, prob.ode_kwargs, prob.discrete,
               prob.delays, prob.history, prob.obs_fn, prob.estimate_u0)
end

function MultiTrajectoryPSMProblem(problems::AbstractVector{<:PSMProblem};
                                   group_ids::AbstractVector{<:Integer}=collect(1:length(problems)))
    G = length(problems)
    G >= 1 || throw(ArgumentError("MultiTrajectoryPSMProblem: need at least 1 trajectory"))
    length(group_ids) == G || throw(ArgumentError(
        "MultiTrajectoryPSMProblem: group_ids length ($(length(group_ids))) must match problems ($G)"))

    p1 = problems[1]
    names1 = [a.name for a in p1.approximators]
    nps1 = [nparams(a) for a in p1.approximators]
    for g in 2:G
        pg = problems[g]
        [a.name for a in pg.approximators] == names1 || throw(ArgumentError(
            "MultiTrajectoryPSMProblem: trajectory $g has different approximator names than trajectory 1"))
        [nparams(a) for a in pg.approximators] == nps1 || throw(ArgumentError(
            "MultiTrajectoryPSMProblem: trajectory $g has different approximator parameter counts than trajectory 1"))
        typeof(pg.likelihood) === typeof(p1.likelihood) || throw(ArgumentError(
            "MultiTrajectoryPSMProblem: all trajectories must share the same likelihood type"))
    end

    gids = Int.(collect(group_ids))
    wrapped = PSMProblem[_inject_group_id(problems[g], gids[g]) for g in 1:G]
    MultiTrajectoryPSMProblem(wrapped, gids)
end

function MultiTrajectoryPSMProblem(dynamics!, u0s::AbstractVector,
                                   tspans,
                                   approximators::Vector{<:AbstractApproximator};
                                   data_times_list::AbstractVector,
                                   data_values_list::AbstractVector,
                                   data_weights_list=nothing,
                                   obs_to_state=nothing,
                                   known_params_list=nothing,
                                   likelihood=GaussianLikelihood(),
                                   solver=Tsit5(),
                                   discrete::Bool=false,
                                   delays::Vector{Float64}=Float64[],
                                   history::Union{Nothing,Function}=nothing,
                                   obs_fn::Union{Nothing,Function,TrajectoryObservation}=nothing,
                                   estimate_u0::Union{Bool,AbstractVector{<:Integer}}=false,
                                   group_ids::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                                   solver_kwargs...)
    G = length(u0s)
    G >= 1 || throw(ArgumentError("MultiTrajectoryPSMProblem: u0s is empty"))
    length(data_times_list) == G || throw(ArgumentError("data_times_list length must match u0s"))
    length(data_values_list) == G || throw(ArgumentError("data_values_list length must match u0s"))
    gids = group_ids === nothing ? collect(1:G) : Int.(collect(group_ids))

    probs = PSMProblem[]
    for g in 1:G
        ts_g = tspans isa Tuple ? tspans : tspans[g]
        dw_g = data_weights_list === nothing ? nothing : data_weights_list[g]
        kp_g = known_params_list === nothing ? NamedTuple() : known_params_list[g]
        kp_g = haskey(kp_g, :group_id) ? kp_g : merge(kp_g, (group_id=gids[g],))
        o2s_g = obs_to_state === nothing ? collect(1:size(data_values_list[g], 2)) : obs_to_state
        p_g = PSMProblem(dynamics!, u0s[g], ts_g, approximators;
                         data_times=data_times_list[g],
                         data_values=data_values_list[g],
                         data_weights=dw_g,
                         obs_to_state=o2s_g,
                         known_params=kp_g,
                         likelihood=likelihood,
                         solver=solver,
                         discrete=discrete,
                         delays=delays,
                         history=history,
                         obs_fn=obs_fn,
                         estimate_u0=estimate_u0,
                         solver_kwargs...)
        push!(probs, p_g)
    end
    MultiTrajectoryPSMProblem(probs, gids)
end

# Parameter layout helpers for MultiTrajectoryPSMProblem
function _mt_param_layout(mp::MultiTrajectoryPSMProblem)
    n_ap = n_approx_params(mp.problems[1])
    G = length(mp.problems)
    u0_offs = Vector{Int}(undef, G)
    u0_lens = Vector{Int}(undef, G)
    off = n_ap
    for g in 1:G
        len_g = length(mp.problems[g].estimate_u0)
        u0_offs[g] = off
        u0_lens[g] = len_g
        off += len_g
    end
    (n_ap=n_ap, u0_offs=u0_offs, u0_lens=u0_lens, n_total=off)
end

n_total_params(mp::MultiTrajectoryPSMProblem) = _mt_param_layout(mp).n_total

function build_initial_params(mp::MultiTrajectoryPSMProblem)
    b = vcat([initial_params(a) for a in mp.problems[1].approximators]...)
    for pg in mp.problems
        if !isempty(pg.estimate_u0)
            b = vcat(b, Float64[Float64(pg.u0[idx]) for idx in pg.estimate_u0])
        end
    end
    b
end

function _mt_sub_beta(beta::AbstractVector, n_ap::Int, u0_off::Int, u0_len::Int)
    u0_len == 0 && return @view beta[1:n_ap]
    vcat(@view(beta[1:n_ap]), @view(beta[u0_off+1:u0_off+u0_len]))
end

"""
    simulate_psm(mp::MultiTrajectoryPSMProblem, beta) -> Vector{Matrix}

Simulate every trajectory in a [`MultiTrajectoryPSMProblem`](@ref) at shared
parameter vector `beta`.
"""
function simulate_psm(mp::MultiTrajectoryPSMProblem, beta::AbstractVector;
                      maxiters_override::Union{Nothing,Int}=nothing)
    lay = _mt_param_layout(mp)
    G = length(mp.problems)
    [simulate_psm(mp.problems[g],
                  _mt_sub_beta(beta, lay.n_ap, lay.u0_offs[g], lay.u0_lens[g]);
                  maxiters_override=maxiters_override) for g in 1:G]
end

"""
    solve(mp::MultiTrajectoryPSMProblem, alg::LAML; parallel=false) -> PSMSolution

Jointly fit a hierarchical / multi-trajectory PSM across all trajectories using
IRLS + LAML (Wood et al. 2016; Wood & Fasiolo 2017). Smooth unknown functions
and group-level [`RandomEffectApproximator`](@ref) penalties are shared across
trajectories; smoothing parameters (including random-effect variances) are
selected automatically by LAML.

Set `parallel=true` (when Julia runs with multiple threads) to parallelize
per-trajectory ODE solves and Jacobians across threads.
"""
function SciMLBase.solve(mp::MultiTrajectoryPSMProblem, alg::LAML;
                         parallel::Bool=false)
    G = length(mp.problems)
    for g in 1:G
        _validate_problem(mp.problems[g], "LAML(MultiTrajectory)")
    end
    p1 = mp.problems[1]
    fam = p1.likelihood
    lay = _mt_param_layout(mp)
    n_p = lay.n_total
    n_ap = lay.n_ap
    verbose = alg.verbose

    # Flatten observations and weights across trajectories (obs-major within each trajectory)
    traj_sizes = Int[length(pg.data_values) for pg in mp.problems]
    traj_offs = Int[0]
    for g in 1:(G - 1)
        push!(traj_offs, traj_offs[end] + traj_sizes[g])
    end
    n_data = sum(traj_sizes)
    y_vec = Vector{Float64}(undef, n_data)
    w_vec = Vector{Float64}(undef, n_data)
    for g in 1:G
        pg = mp.problems[g]
        nt_g = length(pg.data_times)
        no_g = size(pg.data_values, 2)
        k = traj_offs[g]
        for oi in 1:no_g, ti in 1:nt_g
            k += 1
            y_vec[k] = pg.data_values[ti, oi]
            w_vec[k] = pg.data_weights[ti, oi]
        end
    end
    n_used = _n_usable(y_vec, w_vec)
    n_used == 0 && error("MultiTrajectory LAML: every observation across all trajectories is masked.")

    # Penalty blocks from shared approximators (trailing u0 parameters are unpenalized)
    S_list, uf_offsets, uf_nk = build_penalty_matrices(p1)
    m = length(S_list)

    # Smoothing parameters initialization & fixed_lambda support
    theta = zeros(m)
    fixed_mask = falses(m)
    if alg.fixed_lambda !== nothing
        if alg.fixed_lambda isa Float64
            theta .= alg.fixed_lambda
            fixed_mask .= true
        else
            length(alg.fixed_lambda) == m || throw(ArgumentError(
                "LAML: fixed_lambda vector length ($(length(alg.fixed_lambda))) must match penalty blocks ($m)"))
            for l in 1:m
                if alg.fixed_lambda[l] !== nothing
                    theta[l] = Float64(alg.fixed_lambda[l])
                    fixed_mask[l] = true
                else
                    tr_S = tr(S_list[l])
                    theta[l] = tr_S > 0 ? 1.0 / tr_S : 1.0
                end
            end
        end
    elseif alg.initial_lambda !== nothing
        theta .= alg.initial_lambda
    else
        for l in 1:m
            tr_S = tr(S_list[l])
            theta[l] = tr_S > 0 ? 1.0 / tr_S : 1.0
        end
    end
    smoothing_fixed = all(fixed_mask) && m > 0

    function _mt_eval_f(beta_v::Vector{Float64})
        f_out = Vector{Float64}(undef, n_data)
        preds = Vector{Matrix{Float64}}(undef, G)
        if parallel && Threads.nthreads() > 1
            errs = Vector{Any}(nothing, G)
            Threads.@threads for g in 1:G
                try
                    bg = _mt_sub_beta(beta_v, n_ap, lay.u0_offs[g], lay.u0_lens[g])
                    preds[g] = Float64.(simulate_psm(mp.problems[g], bg))
                catch e
                    errs[g] = e
                end
            end
            for g in 1:G
                errs[g] !== nothing && throw(errs[g])
            end
        else
            for g in 1:G
                bg = _mt_sub_beta(beta_v, n_ap, lay.u0_offs[g], lay.u0_lens[g])
                preds[g] = Float64.(simulate_psm(mp.problems[g], bg))
            end
        end
        for g in 1:G
            pg = mp.problems[g]
            nt_g = length(pg.data_times)
            no_g = size(pg.data_values, 2)
            k = traj_offs[g]
            pr = preds[g]
            for oi in 1:no_g, ti in 1:nt_g
                k += 1
                f_out[k] = pr[ti, oi]
            end
        end
        f_out, preds
    end

    # Prebuild per-trajectory ForwardDiff configs
    sub_cfgs = [_fd_jacobian_config(n_ap + lay.u0_lens[g]) for g in 1:G]
    sub_dams = [fill(1e-5, n_ap + lay.u0_lens[g]) for g in 1:G]

    function _mt_jacobian!(J::Matrix{Float64}, beta_v::Vector{Float64}, f0::Vector{Float64})
        fill!(J, 0.0)
        function _fill_traj!(g::Int)
            pg = mp.problems[g]
            nt_g = length(pg.data_times)
            no_g = size(pg.data_values, 2)
            nd_g = nt_g * no_g
            np_g = n_ap + lay.u0_lens[g]
            Jg = zeros(nd_g, np_g)
            bg = Vector{Float64}(_mt_sub_beta(beta_v, n_ap, lay.u0_offs[g], lay.u0_lens[g]))
            f0_g = @view f0[(traj_offs[g]+1):(traj_offs[g]+nd_g)]
            compute_jacobian!(Jg, pg, bg, f0_g, nt_g, no_g;
                              dam=sub_dams[g], jac=alg.jac, fd_cfg=sub_cfgs[g])
            rows_g = (traj_offs[g]+1):(traj_offs[g]+nd_g)
            J[rows_g, 1:n_ap] .= @view Jg[:, 1:n_ap]
            if lay.u0_lens[g] > 0
                cols_u0 = (lay.u0_offs[g]+1):(lay.u0_offs[g]+lay.u0_lens[g])
                J[rows_g, cols_u0] .= @view Jg[:, (n_ap+1):np_g]
            end
        end
        if parallel && Threads.nthreads() > 1
            Threads.@threads for g in 1:G
                _fill_traj!(g)
            end
        else
            for g in 1:G
                _fill_traj!(g)
            end
        end
        J
    end

    function _mt_pen_obj(beta_v::Vector{Float64}, f_v::Vector{Float64}, B_mat::Matrix{Float64})
        pen = dot(beta_v, B_mat * beta_v)
        if fam isa GaussianLikelihood
            rss = 0.0
            @inbounds for k in 1:n_data
                _usable(y_vec[k], w_vec[k]) || continue
                rss += w_vec[k] * (y_vec[k] - f_v[k])^2
            end
            return 0.5 * (rss + pen)
        else
            return -log_likelihood(fam, y_vec, f_v, w_vec) + 0.5 * pen
        end
    end

    p_curr = build_initial_params(mp)
    f_curr, preds_curr = _mt_eval_f(p_curr)
    J = zeros(n_data, n_p)
    B_mat = build_S_lambda(S_list, uf_offsets, uf_nk, log.(max.(theta, 1e-300)), n_p)
    obj_curr = _mt_pen_obj(p_curr, f_curr, B_mat)

    conv_converged = false
    conv_reason = :maxiters
    conv_iters = 0

    for iter in 1:alg.maxiters
        conv_iters = iter
        _mt_jacobian!(J, p_curr, f_curr)
        W_irls = irls_weights(fam, y_vec, f_curr, w_vec)
        wrk = _working_residual(fam, y_vec, f_curr, w_vec)
        z_vec = wrk .+ J * p_curr

        # LAML smoothing update after warmup
        if m > 0 && iter > alg.warmup && !smoothing_fixed
            try
                theta_new, _ = estimate_smoothing_params(
                    J, W_irls, w_vec, y_vec, f_curr, p_curr,
                    S_list, uf_offsets, uf_nk, n_p;
                    family=fam, rho_init=log.(max.(theta, 1e-300)),
                    criterion=alg.criterion, fixed_mask=fixed_mask)
                theta .= theta_new
            catch e
                _is_program_error(e) && rethrow()
            end
        end

        B_mat = build_S_lambda(S_list, uf_offsets, uf_nk, log.(max.(theta, 1e-300)), n_p)
        obj_curr = _mt_pen_obj(p_curr, f_curr, B_mat)

        # PCLS step via stabilized normal equations + line search
        JWJ = J' * (W_irls .* J)
        H = Symmetric(0.5 .* (JWJ .+ JWJ') .+ B_mat)
        rhs = J' * (W_irls .* z_vec)
        p_target = try
            F_h = cholesky(H; check=false)
            issuccess(F_h) ? (F_h \ rhs) : (pinv(Matrix(H)) * rhs)
        catch
            pinv(Matrix(H)) * rhs
        end

        step_vec = p_target .- p_curr
        accepted = false
        alpha = 1.0
        for _ls in 1:15
            p_try = p_curr .+ alpha .* step_vec
            f_try, preds_try = try
                _mt_eval_f(p_try)
            catch e
                _is_program_error(e) && rethrow()
                alpha *= 0.5
                continue
            end
            obj_try = _mt_pen_obj(p_try, f_try, B_mat)
            if isfinite(obj_try) && obj_try <= obj_curr + 1e-12 * abs(obj_curr)
                rel_improv = abs(obj_curr - obj_try) / max(abs(obj_curr), 1.0)
                step_rel = norm(p_try .- p_curr) / max(norm(p_curr), 1.0)
                p_curr = p_try
                f_curr = f_try
                preds_curr = preds_try
                obj_curr = obj_try
                accepted = true
                if iter > alg.warmup && rel_improv < alg.tol && step_rel < 1e-4
                    conv_converged = true
                    conv_reason = :converged_tol
                end
                break
            end
            alpha *= 0.5
        end

        if verbose && (iter <= 3 || iter % 10 == 0 || conv_converged)
            @printf("MultiTrajectory LAML iter %d: obj=%.5e λ=%s\n",
                    iter, obj_curr, string(round.(theta; sigdigits=3)))
        end
        conv_converged && break
        if !accepted && iter > alg.warmup
            conv_converged = true
            conv_reason = :step_stalled
            break
        end
    end

    # Final Jacobian, covariance V_beta, EDF, and LAML value
    _mt_jacobian!(J, p_curr, f_curr)
    W_irls = irls_weights(fam, y_vec, f_curr, w_vec)
    B_final = build_S_lambda(S_list, uf_offsets, uf_nk, log.(max.(theta, 1e-300)), n_p)
    JWJ = J' * (W_irls .* J)
    H_final = 0.5 .* (JWJ .+ JWJ') .+ B_final
    V_beta = try
        _safe_inv(H_final)
    catch
        nothing
    end
    edf = V_beta !== nothing ? clamp(tr(V_beta * JWJ), 0.0, Float64(n_p)) : Float64(n_p)

    data_loss = 0.0
    for k in 1:n_data
        _usable(y_vec[k], w_vec[k]) || continue
        data_loss += w_vec[k] * (y_vec[k] - f_curr[k])^2
    end
    sigma2_hat = fam isa GaussianLikelihood ? data_loss / max(n_used - edf, 1.0) : 1.0
    laml_val = if m > 0
        try
            V, _, _, _ = laml_objective(fam, p_curr, J, W_irls, w_vec, y_vec, f_curr,
                                        S_list, uf_offsets, uf_nk,
                                        log.(max.(theta, 1e-300)), n_p)
            V
        catch
            NaN
        end
    else
        NaN
    end

    # Pack parameters & unknown-function evaluators
    syms = Symbol[a.name for a in p1.approximators]
    vals = Vector{Float64}[]
    uf_evals = Dict{Symbol, Any}()
    off = 0
    for approx in p1.approximators
        np = nparams(approx)
        pk = Float64.(p_curr[off+1:off+np])
        push!(vals, pk)
        uf_evals[approx.name] = build_evaluator(approx, pk)
        off += np
    end
    if n_p > n_ap
        push!(syms, :u0)
        push!(vals, Float64.(p_curr[(n_ap+1):n_p]))
    end
    params = ComponentArray(NamedTuple{Tuple(syms)}(Tuple(vals)))

    # Stack fitted_values / data_values / data_times when observation widths match
    same_nobs = all(size(pg.data_values, 2) == size(p1.data_values, 2) for pg in mp.problems)
    fitted_stack = same_nobs ? vcat(preds_curr...) : preds_curr[1]
    data_stack = same_nobs ? vcat([pg.data_values for pg in mp.problems]...) : p1.data_values
    times_stack = same_nobs ? vcat([pg.data_times for pg in mp.problems]...) : p1.data_times

    conv_info = (V_beta=V_beta, sigma2=sigma2_hat, converged=conv_converged,
                 iterations=conv_iters, reason=conv_reason,
                 method=:multi_trajectory_laml, solver=:LAML,
                 laml=laml_val, n_trajectories=G,
                 group_ids=copy(mp.group_ids),
                 trajectory_fits=preds_curr)

    PSMSolution(params, obj_curr, data_loss, edf, copy(theta),
                Float64.(fitted_stack), Float64.(data_stack),
                Float64.(times_stack), uf_evals, conv_info)
end
