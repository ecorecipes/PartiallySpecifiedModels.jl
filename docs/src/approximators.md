# Approximators

PartiallySpecifiedModels.jl provides eleven approximator types for representing unknown functions in dynamical systems. Each approximator is a callable object that maps scalar inputs to scalar outputs and is fitted as part of the model (the tensor-product approximator takes two scalar inputs; the single-index approximator takes `p`; the transformed-covariate approximator takes time).

## BSplineApproximator

Cubic B-spline basis with automatic smoothing via penalized least squares. The default and most commonly used approximator.

```@example approx
using PartiallySpecifiedModels # hide
approx = BSplineApproximator(:f, (0.0, 10.0), 8;
                              initial = x -> 0.3)
```

- **`:f`**: Symbol used to access the function in the parameter struct (`p.f(x)`)
- **`(0.0, 10.0)`**: Domain of the input variable
- **`8`**: Number of evenly-spaced knots (controls flexibility)
- **`initial`**: Function or constant for initial coefficient values

The smoothing parameter ``\lambda`` is estimated automatically by LAML or GCV.

```@docs
BSplineApproximator
```

## TensorBSplineApproximator

Bivariate tensor-product spline for unknown functions of **two** state variables — the flagship ecological case in Wood (2001), where a predation response ``g(P, N)`` depends on both predator and prey densities.

```@example approx
approx2d = TensorBSplineApproximator(:g, (0.0, 3.0), (0.5, 3.0), 5, 5;
                                     initial = (N, P) -> 0.3 * N * P)
```

- **`(0.0, 3.0)` / `(0.5, 3.0)`**: domains of the first and second argument
- **`5, 5`**: knots per margin (`nparams = 5 × 5 = 25` coefficients — the surface values on the knot grid)
- **`anisotropy`** (keyword, default 1.0): fixed relative weight of y-roughness against x-roughness in the Kronecker-sum penalty; a single smoothing parameter ``\lambda`` scales the whole penalty

In the dynamics the fitted surface is called with two arguments:

```julia
function predprey!(du, u, p, t)
    g = p.g(u[1], u[2])          # bivariate unknown interaction
    du[1] = u[1] * (1 - u[1] / p.K) - g
    du[2] = p.e * g - p.m * u[2]
end

prob = PSMProblem(predprey!, [1.0, 1.5], (0.0, 40.0), [approx2d];
                  data_times=ts, data_values=data, obs_to_state=[1, 2],
                  known_params=(K=6.0, e=0.5, m=0.25),
                  likelihood=Gaussian(), solver=Tsit5())
sol = solve(prob, AdamSolver(maxiters=400, lr=0.05))
sol.unknown_functions[:g](1.5, 2.0)   # fitted surface at (N, P)
```

Works with the through-the-solver and penalized-likelihood solvers (`AdamSolver`, `LAML`, `GCVSolver`, `DerivativeFreeSolver`, `MultipleShootingSolver`, …); for strongly nonlinear oscillatory systems, start `LAML` from a light penalty (e.g. `LAML(warmup=10, initial_lambda=0.01)`). A 1-D trajectory only visits a curve through the `(x, y)` plane, so the surface is identified near the visited region — evaluate it there. `confidence_band` and the bootstrap unknown-function bands are univariate and do not cover tensor surfaces.

```@docs
TensorBSplineApproximator
```

## SingleIndexApproximator

Unknown function of **several** states through **one learned direction**,
``f(u_1, \dots, u_p) = s(z)`` with ``z = (a^\top u - a^\top \hat\mu)/\sqrt{a^\top \hat\Sigma a}``.
The loadings ``a`` and the univariate outer smooth ``s`` are estimated jointly — the
nested-effects construction of Fasiolo et al. (arXiv:2511.19234), where the inner
transformation and the outer smooth carry separate smoothing parameters under LAML.

```@example approx
approx_si = SingleIndexApproximator(:f, 2, 10; xi = 2.5,
                                    initial = z -> -0.05 - 0.1 * z)
```

With `index_stats = nothing` (the default) this approximator is **unresolved**:
its standardization statistics are filled in at `PSMProblem` construction, which
stores a *resolved copy*. Read fitted results back through the problem's own
object — `prob.approximators[1]`, or `sol.unknown_functions[:f]` — since calling
`build_evaluator` on the bare `approx_si` above raises an error telling you so.
Pass `index_stats = (mu, Sigma)` if you would rather resolve it up front.

- **`2`**: number of index arguments (`p`), called as `p.f(u[1], u[2])`
- **`10`**: knots of the outer smooth, placed on ``[-\xi, \xi]`` in **standardized** units
- **`anchor`** (keyword, default `1`): `a[anchor] ≡ 1` is not a parameter, which fixes both the scale and the sign of the index. The anchor variable must genuinely load
- **`constraint`** (keyword, default `:none`): any of [`SHAPE_CONSTRAINTS`](@ref) makes the outer smooth the SCOP construction
- **`index_stats`** (keyword): `(mu, Sigma)`; `nothing` (default) derives them once from the data at `PSMProblem` construction and then holds them **fixed** for the whole fit

**When to prefer this to a tensor surface — and when not.** The advantage is
**model match**, not orbit geometry. On the damped predator–prey fixture in the test suite,
whose truth genuinely *is* an index, both types fit the data equally well but the single
index extrapolates far better off the orbit: RMSE 0.00421 against the tensor's 0.02678 at
off-orbit states whose index the data pin down, 0.0256 against 0.0815 over all off-orbit
states, and 0.0112 against 0.0369 over the whole state box — with 11 coefficients where the
tensor uses 25, recovering the true loadings to within 2%. Rebuild the same fixture with a
truth that is *not* an index and the ranking reverses by a comparable or larger margin: an
additive two-ridge truth gives 0.314 against the tensor's 0.054, a multiplicative
interaction 0.207 against 0.070.

So reach for a single index when you have reason to believe one direction drives the
response (and when ``p > 2``, where no tensor type exists at all); reach for the tensor
when you expect a genuine interaction. Usefully, **the misspecification is not silent**: on
both non-index truths above the single index's `data_loss` was 27–66% worse than the
tensor's, so a poor in-sample fit flags the wrong choice. One asymmetry to keep in mind when
comparing: the single index carries **two** smoothing parameters (inner and outer) while the
tensor's Kronecker-sum penalty carries **one**.

Two honest caveats. First, the loadings need the state path to change direction. What
defeats identification is **collinearity** — states moving along an (affinely) straight
line in state space, where a rescaled ``a`` is absorbed into ``s``. Mere monotonicity is
*not* enough to break it: on decay fixtures where every coordinate and the index itself
decrease monotonically, the loadings were still recovered to 2%, because a curved path keeps
changing its tangent direction. Second, ``\xi = 2`` covers ±2 standard deviations of the
index; a strongly skewed orbit needs a larger `xi` (outside the knots the outer smooth
extrapolates linearly, as everywhere else in the package).

`confidence_band` works, and returns the band of the **outer curve** over the standardized
index — the univariate payoff a tensor surface cannot offer.

```@docs
SingleIndexApproximator
index_loadings
```

## TransformedCovariateApproximator

Unknown function of **time** driven by an **exogenous covariate** through a learned
transformation, ``f(t) = s(z(t))`` with ``z`` the standardized transform of a
user-supplied series. Same nested-effects construction as
[`SingleIndexApproximator`](@ref) (Fasiolo et al., arXiv:2511.19234) — a smooth
composed with a learned inner statistic, each with its own smoothing parameter under
LAML — but with a different inner transformation and a much simpler standardization.

```@example approx
times_tc = collect(0.0:1.0:120.0)
temp_tc  = [18.0 + 7.0 * sin(2pi * t / 180 - pi / 3) for t in times_tc]
approx_tc = TransformedCovariateApproximator(:beta, times_tc, temp_tc;
                                             trans = :expsm, nknots = 8,
                                             constraint = :increasing)
nparams(approx_tc)   # 1 inertia parameter + 8 outer coefficients
```

In the dynamics it is a **one-argument callable of time**, the same convention every
other time-varying unknown function in the package uses:

```julia
sir!(du, u, p, t) = begin
    infection = p.beta(t) * u[1] * u[2] / p.N
    du[1] = -infection
    du[2] =  infection - p.gamma * u[2]
end
```

- **`trans = :expsm`** — adaptive exponential smoothing,
  ``\tilde s_i = \omega_i \tilde s_{i-1} + (1 - \omega_i) x_i`` with
  ``\omega_i = \mathrm{logistic}(\tilde w_i^\top a)``. By default **one** parameter, a
  constant inertia; extra columns of the covariate matrix are auxiliary covariates that
  make the inertia adaptive. The scientific case: transmission responding to temperature
  with a learned *thermal lag*.
- **`trans = :lagindex`** — a distributed lag ``\sum_\ell a_\ell x(t - \ell\Delta)`` over a
  window of `lags` samples, with the paper's **smooth-lag prior** (a first-difference
  penalty on the lag weights) and the same anchor identification the single index uses:
  `a[anchor] ≡ 1` fixes both the scale and the sign.
- **`constraint`** (keyword): any of [`SHAPE_CONSTRAINTS`](@ref) makes the outer response
  curve the SCOP construction — e.g. a response guaranteed monotone in smoothed temperature.
- **`xi`** (keyword, default 2.0): the outer smooth spans ``[-\xi, \xi]`` in **standardized**
  covariate units. So `domain` is *not* time, and [`confidence_band`](@ref) accordingly
  reports the band of the **response curve** ``s(z)``, not of ``f(t)``.

**Why this is the easy half of the nested construction.** The single index standardizes
against the fitted *trajectory*, which moves during the fit, so it must freeze reference
statistics at `PSMProblem` construction and lives in an "unresolved" state until then. Here
the covariate is **fixed data**, so the paper's recipe — standardize the transformed series
to mean 0 and variance 1 over the sample — applies verbatim, ``z`` depends smoothly on the
transform parameters alone, and none of that machinery is needed: the approximator is
complete at construction and `build_evaluator` works on a bare object.

**Two practical cautions, both measured.** First, ``f(t)`` is piecewise linear in ``t``
between covariate times, so prefer `jac = :forwarddiff` on `LAML`/`GCVSolver`/`CollocationLAML`:
on the SIR recovery fixture in the test suite the default finite-difference prediction
Jacobian **lands at a badly worse optimum, and in the default run configuration calls it a
success** — between 130 and 1500 nats worse than `jac = :forwarddiff`'s 481.0 (inner
smoothing parameter 11.56, converged), depending on the environment. The gap is not worth
quoting precisely: the finite-difference optimum is chaotic in the exact arithmetic, shifting
~130 nats and six orders of magnitude in the inner λ between BLAS thread counts, and it exits
`:maxiters` under `--check-bounds=yes` while claiming convergence without it. The lesson is
qualitative — use `:forwarddiff` here and check the convergence flag. Second, the inner
transformation is **much more weakly identified than the response curve**. Standardization
removes the amplitude damping that different ``\omega`` apply to the covariate, leaving only a
phase lag: on that fixture `cor(z(ω=0.8), z(ω=0.7)) = 0.9976`. The estimator is consistent —
``\hat\omega`` = 0.687, 0.763, 0.791 as the observation noise falls through 3.0, 1.0, 0.25
against a truth of 0.8 — but at realistic noise LAML shrinks it toward ``\omega = 1/2``, while
the response curve itself is recovered to an RMSE of 0.0088 on a ``\beta`` ranging over
0.185–0.395. Read the inner parameters as a regularized summary, not a sharp estimate.

```@docs
TransformedCovariateApproximator
lag_weights
smoothing_inertia
transformed_covariate
```

## ShapeConstrainedBSplineApproximator

SCOP-spline (Shape-Constrained P-spline) that enforces shape constraints on the fitted function via reparameterization of the B-spline coefficients. Supports 14 constraint types.

```@example approx
approx_sc = ShapeConstrainedBSplineApproximator(:f, (0.0, 10.0), 8, :increasing)
```

### Available constraints

| Constraint | Description |
|------------|-------------|
| `:increasing` | Monotonically increasing |
| `:decreasing` | Monotonically decreasing |
| `:convex` | Convex (curves upward) |
| `:concave` | Concave (curves downward) |
| `:inc_convex` | Increasing and convex |
| `:inc_concave` | Increasing and concave |
| `:dec_convex` | Decreasing and convex |
| `:dec_concave` | Decreasing and concave |
| `:positive` | Non-negative everywhere |
| `:dec_positive` | Decreasing and non-negative |
| `:inc_zero_left` | Increasing, zero at left endpoint |
| `:inc_zero_right` | Increasing, zero at right endpoint |
| `:dec_zero_left` | Decreasing, zero at left endpoint |
| `:dec_zero_right` | Decreasing, zero at right endpoint |

Zero-at-endpoint constraints fix one knot coefficient to zero, reducing the parameter count by 1.

Following Pya & Wood (2015), the monotone and monotone-plus-curvature constraints (`:increasing`, `:decreasing`, `:inc_*`, `:dec_*`) carry a free level, so the fitted function may cross zero. Use `:positive` or `:dec_positive` when positivity is required.

```@docs
ShapeConstrainedBSplineApproximator
```

## SPDEApproximator

Matérn SPDE (Stochastic Partial Differential Equation) approximator following [Lindgren et al. (2011)](https://doi.org/10.1111/j.1467-9868.2011.00777.x). Uses a finite element discretization of the Matérn SPDE to build a precision-based penalty matrix, providing an interpretable alternative to B-spline penalties.

```@example approx
approx_spde = SPDEApproximator(:f, (0.0, 10.0), 10;
                                nu = 1.5,
                                range_param = 3.0,
                                initial = x -> 0.0)
```

- **`nu`**: Matérn smoothness parameter — 0.5 (rough), 1.5 (once-differentiable, default), or 2.5 (twice-differentiable)
- **`range_param`**: Spatial correlation length ``\rho``. Controls the scale over which the function varies. Defaults to 1/3 of the domain width.
- The overall smoothing strength ``\tau^2`` is still estimated automatically via LAML/GCV, separately from the range.

Use [`optimize_spde_range`](@ref) to select the range parameter automatically via profile GCV.

```@docs
SPDEApproximator
```

## ShapeConstrainedSPDEApproximator

Combines the Matérn SPDE penalty with SCOP-spline reparameterization to enforce shape constraints at mesh nodes. Supports all 14 constraint types (same as [`ShapeConstrainedBSplineApproximator`](@ref)).

```@example approx
approx_scspde = ShapeConstrainedSPDEApproximator(:f, (0.0, 10.0), 10, :increasing;
                                                  nu = 1.5, range_param = 3.0)
```

Constraints are enforced at mesh nodes via a cumulative-sum reparameterization through `softplus`, and they hold **at the nodes only**: the cubic-spline interpolant between nodes has cardinal functions that take negative values (measured minimum −0.17), so the fitted function can violate the constraint between nodes — a `:positive` fixture with all-positive node values (alternating ≈5 / ≈0.007) dipped to −0.121. Adding mesh nodes does not cure this (the worst-case dip per unit node value grows slightly with `n_basis`, measured 0.24 → 0.27 for 6 → 40 nodes); it helps only indirectly, by letting the fitted node values vary more smoothly relative to the spacing. Audit a fitted result with [`check_constraints`](@ref), and use [`ShapeConstrainedBSplineApproximator`](@ref) when the constraint must hold everywhere — its B-spline convex-hull property makes all 14 constraints exact.

!!! tip
    Simple constraints (`:increasing`, `:decreasing`, `:concave`) tend to converge more reliably than combined constraints (`:inc_concave`, `:dec_positive`) which can trap the optimizer in local optima.

```@docs
ShapeConstrainedSPDEApproximator
```

## NeuralApproximator

Neural network approximator using [Lux.jl](https://github.com/LuxDL/Lux.jl). Suitable for complex or high-dimensional unknown functions. Compatible with gradient-based solvers (`AdamSolver`, `MultipleShootingSolver`).

```@example approx
import Lux
chain = Lux.Chain(Lux.Dense(1, 16, tanh), Lux.Dense(16, 1))
approx_nn = NeuralApproximator(:f, chain)
```

The network weights are fitted as part of the optimization. Network architecture is specified as a standard Lux `Chain`.

```@docs
NeuralApproximator
```

## GPApproximator

Gaussian process approximator. Uses a GP prior over the unknown function with learnable hyperparameters (lengthscale, signal variance). Particularly useful with gradient matching solvers.

```@example approx
approx_gp = GPApproximator(:f, (0.0, 10.0), 20)
```

```@docs
GPApproximator
```

## ShapeConstrainedGPApproximator

Combines the GP predictive-mean interpolation of [`GPApproximator`](@ref) with the SCOP-spline reparameterization to enforce shape constraints at the inducing points. Supports all 14 constraint types (same as [`ShapeConstrainedBSplineApproximator`](@ref)).

```@example approx
approx_scgp = ShapeConstrainedGPApproximator(:f, (0.0, 10.0), 10, :increasing)
```

Constraints are enforced at the inducing-point values via a cumulative-sum reparameterization through `softplus`, and they hold **at the inducing values only**: the kernel interpolant between them has cardinal functions that take negative values (measured minimum −0.26 for the default `:sqexp` kernel), so the fitted function can violate the constraint between points — a `:positive` fixture with all-positive inducing values (alternating ≈5 / ≈0.007) dipped to −0.505 against a maximum of 5.6. Adding inducing points does **not** cure this — the worst-case dip per unit inducing value grows with `n_inducing` (measured 0.35 → 0.57 for 6 → 40 points at default kernel settings). What measurably reduces it on that fixture: a rougher kernel (`kernel=:matern32` at its default lengthscale cut the dip from −0.505 to −0.023) or a shorter explicit `lengthscale` (0.5× the default eliminated it), both at the cost of wigglier interpolation. Audit a fitted result with [`check_constraints`](@ref), and use [`ShapeConstrainedBSplineApproximator`](@ref) when the constraint must hold everywhere. Kernel hyperparameters adapt during LAML/GCV fits exactly as for `GPApproximator` (pass an explicit `lengthscale` to fix them — required if you rely on a short lengthscale to limit violations).

```@docs
ShapeConstrainedGPApproximator
```

## COMONetApproximator

Constrained Monotone Network — a neural network architecture that guarantees shape constraints by construction. Each constraint uses the architecture matching its function class: monotone constraints use positive (`exp(W)`) weights with saturating tanh hidden units, curvature-only constraints use a two-branch input-convex form (twice the parameters), and `:positive` exponentiates an unconstrained MLP.

```@example approx
approx_comon = COMONetApproximator(:f, (0.0, 10.0), (16, 16), :increasing)
```

### Available constraints

| Constraint | Description |
|------------|-------------|
| `:increasing` | Monotonically increasing (tanh units — includes sigmoids and other monotone-nonconvex shapes) |
| `:decreasing` | Monotonically decreasing |
| `:convex` | Convex, possibly non-monotone (two-branch form, 2× parameters) |
| `:concave` | Concave, possibly non-monotone (two-branch form, 2× parameters) |
| `:inc_convex` | Increasing and convex |
| `:inc_concave` | Increasing and concave |
| `:dec_convex` | Decreasing and convex |
| `:dec_concave` | Decreasing and concave |
| `:positive` | Strictly positive output (`exp` of an unconstrained MLP) |

```@docs
COMONetApproximator
```

## Choosing an Approximator

| Criterion | BSpline | ShapeConstrained | SPDE | Neural | GP | COMONet |
|-----------|:-------:|:----------------:|:----:|:------:|:--:|:-------:|
| Few parameters | ✓ | ✓ | ✓ | | | |
| Automatic smoothing | ✓ | ✓ | ✓ | | | |
| Shape constraints | | ✓ | ✓ | | | ✓ |
| Interpretable range | | | ✓ | | | |
| High flexibility | | | | ✓ | ✓ | |
| Gradient matching | ✓ | ✓ | ✓ | | ✓ | |
| UDE-style training | | | | ✓ | | ✓ |
| Interpretability | ✓ | ✓ | ✓ | | | |

**General guidance:**
- Start with [`BSplineApproximator`](@ref) — it works well in most cases with automatic smoothing.
- Use [`TensorBSplineApproximator`](@ref) when the unknown function takes TWO state variables (e.g. a predation response `g(P, N)`).
- Use [`SPDEApproximator`](@ref) for an interpretable correlation-length parameter and Matérn-based smoothing.
- Use [`ShapeConstrainedBSplineApproximator`](@ref) or [`ShapeConstrainedSPDEApproximator`](@ref) when you have prior knowledge about monotonicity or convexity.
- Use [`NeuralApproximator`](@ref) when the unknown function may be complex or when using UDE-style solvers.
- Use [`GPApproximator`](@ref) with gradient matching or MAGI solvers.
- Use [`COMONetApproximator`](@ref) for neural network flexibility with monotonicity guarantees.
