# Cross-approximator calibration experiments

This suite compares **procedures** (representation, regularization, fitting
rule and interval construction), not just parameter counts or network
names. It is designed to find candidates for better calibration without
choosing them on their confirmation results.

Production solver defaults are unchanged. The experiment uses a common,
stable, finite-grid working-LAML fitter and retains numerical failures,
iteration-budget exhaustion and smoothing-grid boundary selections.

## Experiments included

| Experiment | Purpose |
|---|---|
| Five approximator families and 15 configurations | Separate family, resolution, kernel/range and prior effects |
| Nonlinear growth and linear integral observations | Separate nonlinear inverse-problem effects from an analytically tractable control |
| Conditional, percentile and studentized bootstrap intervals | Compare covariance approximations with full-refit sampling distributions |
| Pilot-generated bootstrap-t/max-t | Test a predeclared less-smoothed generating model, without giving the bootstrap the truth |
| Sample-split bias-aware one-step inference | Include a worst-case bias allowance over an explicit function class |
| Finite-bank and polynomial confidence-set inversion | Provide reference function envelopes without a coefficient-normal approximation |
| Density values, a grid-weighted mean and a contrast | Compare pointwise, simultaneous and functional targets |
| Weak-identification pairs | Find separated response functions with similar observation distributions |
| Frozen development selection and confirmation | Prevent selecting methods using confirmation coverage |

The initial scope is **unary responses with Gaussian observation errors**.
This does not implement full continuous-GP inference, Bayesian architecture
averaging, unknown covariance-shape estimation, uncertain initial conditions,
multivariate/index/shape-constrained response inference, or certified
confidence envelopes over an unrestricted nonlinear function class.

## Environment and a bounded pilot

From the repository root:

```bash
julia --project=benchmarks/kan -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

julia --project=benchmarks/kan --check-bounds=yes benchmarks/calibration/run.jl \
  --stage=smoke --plan-only=true

julia --project=benchmarks/kan --check-bounds=yes benchmarks/calibration/run.jl \
  --stage=smoke \
  --methods=conditional,bootstrap_percentile,bootstrap_t,bootstrap_t_pilot,split_bias_bank,split_bias_ellipsoid \
  --output=benchmarks/calibration/results/my-pilot

python3 -B benchmarks/calibration/analyze.py summarize \
  --inputs=benchmarks/calibration/results/my-pilot \
  --output=benchmarks/calibration/results/my-pilot-analysis
```

The default smoke cohort has two datasets (one per observation operator),
five model configurations, known Gaussian scale, three bootstrap attempts
per requested generator and a deliberately small coefficient budget.
It exercises the implementation; **it cannot rank or certify calibration**.
The selector rejects smoke results.

The retained `results/pilot-final/` run covers all five families, both
operators and all applicable methods: 10 full-data model fits and 3,402
interval records. All endpoints are available, but both deliberately
low-budget MLP fits exhaust their iteration budgets. A separate
`results/noise-shard-pilot/` run exercises estimated scale, heteroskedastic/
AR noise, alternative designs and a non-first shard, with three fits and
384 interval records. These are implementation pilots, not coverage
estimates suitable for choosing a winner.

`--plan-only=true` prints dataset, fit-selection and candidate-refit counts
without generating data or fitting models. Large runs are opt-in. Every
output directory must be new.

## Model configurations

| Family | IDs | Declared model |
|---|---|---|
| Cubic spline | `spline8`, `spline12`, `spline20` | Intrinsic curvature penalty, unpenalized affine component |
| GP interpolant | `gp12`, `gp20`, `gp12_short`, `gp12_long` | Fixed Matern-5/2 interpolant; lengthscale 0.4, 0.2 or 0.8; nodal curvature penalty |
| SPDE | `spde12`, `spde20`, `spde12_short`, `spde12_long` | Full-rank Matern precision, nu 1.5; range 0.8, 0.4 or 1.6 |
| MLP | `mlp13`, `mlp25` | One tanh hidden layer, widths 4 or 8, full-rank weight/bias ridge |
| KAN | `kan12`, `kan16` | One cubic edge, grid sizes 8 or 12; complete-edge curvature and free affine component |

The default comparison is `spline8,gp12,spde12,mlp13,kan12`.
Specify `--models=...` for resolution/range experiments. Priors are
explicitly **not identical**; marginal criteria are never compared across
representations to select a model.

Initial responses are 0.5 where representable. GP inducing values are 0.5,
which need not give an exactly constant interpolant between nodes. MLP
hidden features use seed 42, with zero output weights and output bias 0.5.
KAN base weights are zero and spline coefficients are 0.5. Initialization
is fixed across refits, not sampled to manufacture an ensemble interval.

The GP is the package's finite interpolation model. No conditional
continuous-GP process variance is added after fitting. Kernel/range and
resolution variants are independent experimental configurations; they are
not silently integrated hyperparameter uncertainty.

## Observation operators, designs and noise

`growth` observes trajectories of `N'_j = r(N_j) N_j`. Its bank responses
have baseline `0.9(1-N/2)` and positive smooth multipliers.

`integral` observes `u'_j = r(v_j t)`. Degree-five Legendre polynomial
controls have exact integrated means, enabling continuous-class reference
inference. Other responses use stored high-accuracy numerical means.
For an in-bank truth, the reference method and data generator use the same
stored mean vector.

All three designs have **40 scalar observations**:

| Design | Trajectories / times | Difference |
|---|---|---|
| `baseline` | 2 / 20 | Growth initial states 0.2 and 0.7 |
| `diverse` | 4 / 10 | Growth initial states 0.2, 0.55, 1.0, 1.45; varied input speeds for the integral operator |
| `replicated` | 4 / 10 | Two repeated trajectories with independent observation errors |

Available noise shapes are `iid`, `hetero` and `ar1`. Heteroskedastic SD
changes linearly from 0.7 to 1.3 times the common scale. AR(1) correlation
is 0.5 between adjacent observations within a trajectory, not a fixed
continuous-time correlation length. The covariance **shape is known** to
the fitter and resampler. `--noise-modes=known,estimated` distinguishes
known common scale from residual/EDF estimation of that scale.

Reference confidence sets and bias-aware bounds currently require the
`known` scale arm. They are not run under an estimated scale and then
mislabelled as exact known-noise inference.

`--sigmas=0.015,0.03` adds noise-level strata. Initial conditions and input
speeds remain known. This is observation noise, not unmodelled process noise.

## Inference methods and their scope

### Ordinary covariance and bootstrap methods

`conditional` uses local working covariance, with Gaussian maximum
calibration for a simultaneous density-grid band.

`bootstrap_percentile` is pointwise only. `bootstrap_t` uses each refit's
own standard errors:

```text
t_b(x) = (f_b(x) - f_generating(x)) / se_b(x)
```

Pointwise endpoints invert the signed empirical quantiles. Simultaneous
bands use the level-quantile of `max_x |t_b(x)|`. Smoothing and coefficients
are re-estimated from the declared starts in every replicate. Under an
estimated-noise arm, the scale is also re-estimated.

`bootstrap_t_pilot` generates data from a fixed-fraction smoothing refit
(`--pilot-fraction=0.25` by default), centers its bootstrap statistic on
that generating response, and still refits the original declared
estimator. This is an experimental bias-aware generating model, not a
coverage guarantee or a nested correction accounting for every source of
pilot-estimation error.

Bootstrap methods require complete finite results for **every attempted
curve and its required scales**. A failed or zero-scale/nonzero-deviation
replicate makes the affected interval method unavailable. It is not
silently removed. Attempts, seeds, selected smoothing, coefficients,
scales and failures are saved.

### Bias-aware one-step inference

`split_bias_bank` and `split_bias_ellipsoid` fit the pilot on alternating
observation rows within each trajectory. The held-out update is affine in
the inference observations; it does not depend on fitting the full dataset
successfully. For correlated Gaussian noise, the code uses conditional
innovations:

```text
y_innov = y_I - Sigma_IT Sigma_TT^(-1) y_T
```

These innovations are independent of the training errors. Prediction
means, Jacobians and candidate/reference means receive the same
transformation. The result has the form `estimate = offset + M*y_innov`,
with an exactly computed conditional sampling SE under the known Gaussian
covariance.

The finite-bank allowance is the maximum absolute conditional bias over
every declared candidate response. The polynomial allowance is the exact
support-function bound for

```text
beta = center + D^(-1) u,  ||u|| <= 1
center = [0.6, -0.2, 0, 0, 0, 0]
D = diag(1, 4, 9, 16, 25, 36)
```

For the integral operator this gives
`abs(offset + (M*A-G)*center) + norm((M*A-G)/D)` row by row.
This includes representation and regularization effects of the
one-step estimator relative to the declared class; it is not an estimate
of bias learned from the simulation truth.

Pointwise intervals add the bias allowance to a normal sampling radius.
Simultaneous intervals use a Bonferroni normal radius over the requested
grid plus the point-specific allowance. These are conservative
constructions, not necessarily efficient ones.

Coverage statements require the known Gaussian covariance, class
membership, and successful numerical construction. The function estimate
after the one-step update need not lie inside the original neural-network
parameterization. Thus comparisons concern the **complete procedure**.

### Confidence-set references

`bank_set` accepts bank members whose full whitened residual sum is at
most the level-quantile of chi-squared with the number of observations.
It returns the envelope of accepted function values and functionals.
An empty set is reported as such, not replaced by a plausible-looking band.

`linear_set` is the corresponding exact ellipsoidal inversion for the
degree-five polynomial integral model. QR gives the least-squares center,
residual sum and projected ellipsoid radii. Neither reference is eligible
to win the approximator ranking: its function-class information is different.

These constructions do **not** claim certified optimization over a
continuous nonlinear ODE function class. The nonlinear reference class
is finite and fully enumerated.

## Truth families and weak identification

The fixed bank contains affine, sinusoidally modified, broad-bump and
localized oscillatory responses. The integral bank additionally contains
two polynomial controls. Weak-identification cases select a pair with
target separation at least 0.02 and a small observation-distance/target-
separation ratio. Pair selection uses the precomputed bank, design and
noise level **before observing data and without consulting any approximator**.

`weak_pairs.csv` retains every pair's Mahalanobis distance, Gaussian KL
divergence and target separation, not just the chosen pair.

Confirmation repeats the development truths with independent noise and
adds narrow bumps, a rational/logistic shoulder, double features, a chirp
and a new polynomial coefficient vector as appropriate. Extra diverse/
replicated designs and heteroskedastic/AR noise are also declared before
development and added by the lock.

Class membership is explicit on every interval row. Out-of-bank or
out-of-polynomial-class truths are **not excluded** from the stress
assessment; the associated restricted-class guarantees simply do not apply.

## Checkpointed refinement and confirmation

### Approved full campaign

`campaigns/confirmation-20260910/` coordinates the user-approved
**refinement-first, 500-datasets-per-confirmation-stratum** workflow.
It adopts completed refinement shard 1, executes shards 2--320, aggregates
the matched development evidence, freezes procedures with
`--require-refinement`, and only then generates and fits confirmation data.
Selected bootstrap procedures use 999 attempts in confirmation.

The campaign runs one worker, pins the original Julia executable and
scientific source, and shares that verified read-only source across shards
to avoid copying or recompiling the package for every shard. Confirmation
is partitioned into shards of at most 100 datasets. The final cross-shard
aggregation is SQLite-backed rather than holding the complete interval
population in memory. Its statistics agree with the ordinary analyzer
to numerical roundoff on the qualification pilot.

Monitor `campaign.log`, `state.json` and the active shard's
`component-progress.toml`. The controller pauses rather than changing the
protocol if free space falls below 50 GiB or a worker fails. Both the
scientific worker and disk-backed aggregation respect the pause marker.
On completion, the report is published in `confirmation-analysis/`.

After interruption, resume the **whole campaign**, not individual phases:

```bash
python3 -B benchmarks/calibration/campaigns/confirmation-20260910/controllers/campaign.py run \
  --directory=benchmarks/calibration/campaigns/confirmation-20260910 \
  >> benchmarks/calibration/campaigns/confirmation-20260910/campaign.log 2>&1
```

Completed shards and frozen selection are reused. The campaign lock prevents
duplicate execution. A reboot still stops the process; saved checkpoints
allow this command to continue it. Merely launching this workflow does not
mean confirmation has finished or that any procedure is calibrated.

Use `stages.py` for new bootstrap and confirmation campaigns. The original
`run.jl` commands below remain useful for small runs, but checkpoint only
completed model fits; the staged runner also saves **every bootstrap
attempt**, including failures.

The completed screen has produced
`plans/bootstrap-refinement.toml`: one conditional-estimator choice per
family, observation operator and noise-scale arm, for **20 choices**.
The declared selection rule first minimizes worst-stratum simultaneous
coverage deficit, then pointwise coverage deficit, then unavailability and
width. The secondary criterion prevents tied zero whole-grid coverage
from selecting a model solely because its bands are narrow. This is a
development-based shortlist, not a confirmed ranking.

The refinement compares percentile, bootstrap-t/max-t and pilot-generated
bootstrap-t/max-t intervals. Percentile and ordinary bootstrap-t share one
set of draws. The pilot method uses a separate generating fit and draw set.
All existing conditional and bias-aware screen results remain comparison
controls; they are not overwritten or silently recomputed.

At 99 attempts per generator, the full plan contains **3,200 parent fits
and 633,600 bootstrap refits**. Each refit repeats the declared smoothing
search. That is a substantial workload, so execution is explicit and
supports independently restartable dataset shards.

```bash
# Regenerate a shortlist in a NEW file if repeating this workflow.
python3 -B benchmarks/calibration/stages.py shortlist \
  --screen=benchmarks/calibration/runs/develop-full-20260910-0906 \
  --target=density_simultaneous --top=1 --nboot=99 \
  --output=benchmarks/calibration/plans/my-refinement.toml

# Prepare an immutable scientific-source/environment/adapter snapshot.
python3 -B benchmarks/calibration/stages.py prepare \
  --plan=benchmarks/calibration/plans/bootstrap-refinement.toml \
  --shard=2/320 \
  --output=benchmarks/calibration/runs/bootstrap-refinement-002

python3 -B benchmarks/calibration/stages.py run \
  --directory=benchmarks/calibration/runs/bootstrap-refinement-002 --plan-only
```

The existing `runs/bootstrap-refinement-001/` has completed shard 1/320:
ten model/noise-arm jobs, 990 ordinary and 990 pilot-generated refits, and
1,674 finite interval records. Its audited summaries are in `analysis/`.
This is one dataset, not a completed calibration assessment. The same
command checks a completed shard or resumes an interrupted one:

```bash
python3 -B benchmarks/calibration/stages.py run \
  --directory=benchmarks/calibration/runs/bootstrap-refinement-001
```

The same command resumes after interruption. `--job-limit=N` pauses after
N newly completed model/noise-arm jobs; it does not truncate an estimator's
declared bootstrap budget. An OS file lock prevents duplicate execution.
`component-progress.toml` records the last committed original fit,
generating fit or bootstrap attempt.

Original screen coefficients and data are reused only when the scientific
code, fitting options, dataset and parent-file fingerprints match.
Because the old screen did not save a covariance root or forward prediction
in each fit record, these are numerically reconstructed at the exact saved
coefficients with the documented replay checks; there is no new optimization.
New runs save these quantities directly in their original-fit component.

Each component includes its plan/dataset/generator context and a payload
fingerprint. Restart skips completed attempts, preserves their exact RNG
identities, and never retries a recorded failure as though it were a new
draw. Completed fit-and-endpoint bundles and final aggregate tables use
atomic publication. Candidate fits, missing curves and unsuccessful
generating fits remain explicit.

An implementation pilot under `runs/bootstrap-checkpoint-pilot/` ran all
five families, both operators and known/estimated scale: 20 fit jobs, 120
bootstrap refits and 1,292 interval records. It was stopped after its first
job and resumed; all earlier component and fit files were byte-identical.
This pilot uses three draws and cannot select a confirmation procedure.

### Auditing refinement and freezing confirmation

Combine the completed screen with **all completed refinement shards**:

```bash
python3 -B benchmarks/calibration/analyze.py summarize \
  --inputs=benchmarks/calibration/runs/develop-full-20260910-0906/results,PATH_TO_REFINEMENT_1/results,PATH_TO_REFINEMENT_2/results \
  --output=benchmarks/calibration/results/refined-development-analysis

python3 -B benchmarks/calibration/analyze.py freeze \
  --analysis=benchmarks/calibration/results/refined-development-analysis \
  --require-refinement --target=density_simultaneous --top=1 \
  --confirmation-datasets=500 --confirmation-bootstrap=999 \
  --output=benchmarks/calibration/plans/confirmation-lock.toml
```

The auditor checks bootstrap attempt IDs, dimensions, RNG lineage and
failure handling in addition to interval/data fingerprints. The
`--require-refinement` gate rejects missing procedures or incomplete
matched refinement populations. Reference confidence-set controls are
deduplicated rather than counted as extra independent datasets.

The freezer compares complete procedures, including the retained screen
methods. It does not assume that a bootstrap must win. Its decision uses
development data only and is still a candidate for independent confirmation.

```bash
python3 -B benchmarks/calibration/stages.py confirmation-plan \
  --lock=benchmarks/calibration/plans/confirmation-lock.toml \
  --source-root=benchmarks/calibration/runs/develop-full-20260910-0906/source \
  --output=benchmarks/calibration/plans/confirmation.toml

python3 -B benchmarks/calibration/stages.py prepare \
  --plan=benchmarks/calibration/plans/confirmation.toml --shard=1/100 \
  --output=benchmarks/calibration/runs/confirmation-001

python3 -B benchmarks/calibration/stages.py run \
  --directory=benchmarks/calibration/runs/confirmation-001 --plan-only
```

Removing `--plan-only` executes that shard. Confirmation checks the locked
participants, fitting/inference controls and budgets before generating
observations. It uses the same checkpoint machinery for every refit,
independent confirmation RNG streams, and the held-out truth/design/noise
conditions. No scientific confirmation run or final calibrated winner has
been produced yet.

Restart contracts compare checkpointed draws, scales, diagnostics and
endpoints with the original uncheckpointed bootstrap. They also interrupt
after a committed attempt, preserve failed attempts without redrawing,
reject changed generator/plan contexts, and check confirmation locks.
Disabling component reuse causes six failures. The restored workflow,
legacy recovery and inference contracts pass. Execution inputs are frozen
separately from orchestration code, so the scientific protocol remains
`566d17b66810aecf04355c85f446d9bc4f4434d6619268d5d14ce40a18acda2d`.

The final staged implementation and plan archive
`results/source-checkpointed-stages-final.tar.gz` has SHA-256
`3a09500858e985fb18fb3a3f94aa0b143a5e516e5325189e63838212a588de96`.
The development shortlist SHA-256 is
`ffc91a774d1bccb542743b6e41d23d3b36844cc83fafb9178260b16d23e87704`.

## Original development runner

Start with a lower-cost development screen:

```bash
julia --project=benchmarks/kan --check-bounds=yes benchmarks/calibration/run.jl \
  --stage=develop --methods=conditional,split_bias_bank,split_bias_ellipsoid \
  --noise-modes=known,estimated --output=benchmarks/calibration/results/develop-screen \
  --plan-only=true
```

Remove `--plan-only=true` to execute. Development defaults to 20 datasets
per stratum, 40 coefficient iterations and a broader smoothing grid than
the pilot. Add model variants with `--models=...`.

Bootstrap development can be run separately on the **same** datasets:

```bash
julia --project=benchmarks/kan --check-bounds=yes benchmarks/calibration/run.jl \
  --stage=develop --methods=bootstrap_percentile,bootstrap_t,bootstrap_t_pilot \
  --noise-modes=known,estimated --shard=1/20 \
  --output=benchmarks/calibration/results/develop-bootstrap-01 --plan-only=true
```

The default development bootstrap budget is 99 attempts per generator.
Choose and execute the required shards explicitly; bootstrap profile
selection is expensive. `--shard=i/n` partitions datasets, keeping all
requested procedures paired within each dataset.

Summaries can combine disjoint model, method or dataset shards. Numerical
settings and paired observations must match; overlapping model/interval
records are rejected. Duplicate identical reference controls are counted once.

```bash
python3 -B benchmarks/calibration/analyze.py summarize \
  --inputs=PATH_TO_SCREEN,PATH_TO_BOOTSTRAP_SHARD_1,PATH_TO_BOOTSTRAP_SHARD_2 \
  --output=benchmarks/calibration/results/development-analysis

python3 -B benchmarks/calibration/analyze.py freeze \
  --analysis=benchmarks/calibration/results/development-analysis \
  --target=density_simultaneous --top=1 \
  --confirmation-datasets=500 --confirmation-bootstrap=999 \
  --output=benchmarks/calibration/results/development-lock.toml
```

The freezer requires development data, at least 20 datasets per stratum,
complete matched populations, sufficient declared bootstrap budget and
unchanged analysis artifacts. Selection is by worst-stratum observed
coverage deficit, then availability and width. It selects separately by
family, operator and noise-scale arm. This is **exploratory selection for
confirmation**, not a declaration of calibrated coverage.

```bash
julia --project=benchmarks/kan --check-bounds=yes benchmarks/calibration/run.jl \
  --stage=confirm --lock=benchmarks/calibration/results/development-lock.toml \
  --shard=1/100 --output=benchmarks/calibration/results/confirm-001 \
  --plan-only=true
```

Confirmation refuses changes to locked models, inference methods,
coefficient controls, smoothing grids, level, query grid, pilot fraction,
bootstrap budget or Gaussian simulation budget. Stage-separated RNG streams
give new data. Dataset prefixes and shards may be run incrementally;
analysis marks confirmation incomplete until the planned populations are present.
No large development or confirmation campaign is launched automatically.

## Outputs and interpretation

- `design.toml`: complete policy, model specifications, class assumptions,
  requested dataset jobs, shard identity and lock fingerprint.
- `datasets/`: observed and reference trajectories, target values,
  covariance shapes, seeds and class membership.
- `fits/`: selected coefficients, candidate diagnostics, bootstrap
  attempts/draws/scales and split-pilot inference matrices.
- `intervals.csv`: every requested endpoint, availability and coverage event.
- `weak_pairs.csv`: forward-map indistinguishability diagnostics.
- Analysis: per-dataset scores, per-point coverage, stratum summaries,
  width/interval-score tradeoffs and exploratory procedure rankings.

Pointwise density coverage is averaged **within a dataset** before Monte
Carlo errors are calculated. Simultaneous coverage is an all-points-covered
event. The functionals are the specified trapezoidal grid-weighted mean and
nearest-grid `r(0.6)-r(1.2)` contrast, not silently exact continuum integrals.
Unavailable intervals remain in the coverage denominator; available-only
widths are labelled. More parameters, higher coverage with much wider bands,
or an optimizer's stability flag do not establish a superior approximator.

All numerical source, environment, output and input fingerprints are
retained. Known truth enters generation, class-membership labels and scoring,
not the learned fit, smoothing choice or bootstrap generating response.

The focused Julia contracts cover every model factory, exact integral
controls, replica-specific studentization, finite-class/ellipsoid bias
bounds, correlated innovations, training/inference separation and immutable
confirmation settings. Removing the bias allowance, reusing the original
SE instead of each bootstrap SE, and fitting the pilot on held-out data
produces ten failures. A separate accounting reversal confirms that
simultaneous coverage requires all points, not merely any point. Reversals
are restored; the pilot is also independently audited from its saved rows.

The retained pilots use numerical protocol hash
`566d17b66810aecf04355c85f446d9bc4f4434d6619268d5d14ce40a18acda2d`.
The source archive `results/source-calibration-suite.tar.gz` has SHA-256
`7e653bcbe60fb0fdd8bdb8fcfd1613b1d51d4392d6203fadfb0d5317eeaef141`.
These identify the implemented experiments, not a selected or confirmed
calibration procedure.

## Resuming the interrupted full development screen

The run under `runs/develop-full-20260910-0906/` was interrupted by a
host reboot after saving 6,416 of 9,600 fit records. Its original runner
saved fitted coefficients and split-inference components, but held the
aggregate interval tables in memory.

The continuation preserves every original fit/data file and the frozen
numerical protocol. `resume.jl` reconstructs the lost non-bootstrap bands
without reoptimizing those saved fits. Pointwise and split-inference
endpoints use the original saved values; simultaneous Gaussian bands
reconstruct the working covariance root at the exact saved coefficients
and replay the original RNG seed. This is a numerical replay, not a
claim of bit-for-bit reconstruction of lost covariance factors.
Every replay is checked against the saved estimates, standard errors and
EDF, with its discrepancies recorded in the checkpoint.

All subsequent fit records and interval rows are checkpointed together
using a flushed temporary file and atomic rename. A file lock prevents
two continuations from running concurrently; another reboot releases the
lock without invalidating completed checkpoints.

```bash
bash benchmarks/calibration/runs/develop-full-20260910-0906/resume.sh \
  >> benchmarks/calibration/runs/develop-full-20260910-0906/resume.log 2>&1
```

The launcher uses the copied environment, original frozen scientific
code and a separately frozen recovery adapter. It skips completed jobs,
assembles the aggregate tables only when all expected jobs are present,
and publishes audited summaries after successful completion. It does not
start bootstrap refinement or confirmation. `progress.toml` records the
checkpoint count; `resume.toml` documents recovery provenance and limits.

The full non-bootstrap development screen has now completed: **320 datasets,
9,600 model/noise-arm fits and 1,064,960 interval records**. Results and
audited summaries are under that run's `results/` and `analysis/` directories.
All model fits have finite `status=ok`, but 736 do not meet the iteration-
stability convergence criterion and 2,165 select a smoothing-grid boundary.
They remain in the summaries. The reference confidence sets are empty on
some datasets (1,408 unavailable endpoint records), also retained.

An exploratory signal is the long-range 12-node SPDE configuration:
with estimated noise scale, its worst-stratum average pointwise coverage
is 95.3% for growth and 97.7% for the integral operator. Mean pointwise
widths across strata are 0.106 and 0.172, respectively, appreciably wider
than many competing conditional intervals. These are nominal 95% intervals,
20 datasets per stratum, and development-selected comparisons—not a
confirmation of nominal coverage or an approximator-independent guarantee.
No bootstrap refinement, selection lock or held-out confirmation was run.
