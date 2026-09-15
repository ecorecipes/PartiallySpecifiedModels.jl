# Fixed-grid KAN performance improvement

The evaluator now computes only the active cubic basis functions, accumulates
edge contributions directly, keeps scalar inputs in tuples, and returns the
last scalar without constructing an output matrix. Hidden layers allocate
only their output vector. No shared mutable scratch state or trainable
parameter cache was introduced.

The local recursion handles padded boundary intervals, exact knots,
nonuniform grids and values outside spline support. The parameterization,
fixed grids, base activation and curvature penalties are unchanged.

## Frozen-weight replay

The weights selected by the original 1000-iteration experiment were replayed
without refitting. The profile covers two cases, three seeds and both KAN
architectures, with an unchanged MLP as a control. Each scalar timing is the
median of five warmed 10,000-call loops. BLAS uses one thread.

| Model | Scalar ns before | Scalar ns after | Bytes/call before | Bytes/call after | Input-derivative bytes before | Input-derivative bytes after |
|---|---:|---:|---:|---:|---:|---:|
| Shallow KAN, 28 parameters | 1904 | 77.5 | 2688 | 0 | 4416 | 0 |
| Composed KAN, 28 parameters | 2523 | 210.6 | 3840 | 80 | 5200 | 96 |
| MLP control, 28 parameters | 429.5 | 443.3 | 960 | 960 | 1200 | 1200 |

This is approximately 24.6x faster scalar evaluation for the shallow KAN
and 12.0x for the composed KAN. The remaining composed allocation is the
hidden-layer output vector. Removing closure boxing was material: an
intermediate local-support version still allocated 704 bytes per scalar
call before that was corrected.

Maximum frozen-weight discrepancies across the profiles:

| Quantity | Maximum absolute difference |
|---|---:|
| Function values | 2.23e-16 |
| Input derivatives | 2.67e-15 |
| Parameter derivatives | 4.45e-16 |
| Integrated test trajectory | 1.29e-10 |

The arithmetic ordering changes slightly, so bitwise-identical optimization
paths are not claimed. These errors are far below the trajectory solver's
configured tolerances.

## Unchanged end-to-end experiment

The same seven models, three seeds, two problems, four validation candidates,
1000-iteration ceiling and learning rate were rerun. There were no changed
regularization selections. Non-KAN numerical outputs were unchanged.

Median nonlinear-case costs:

| Model | Fit seconds before | Fit seconds after | Speedup | Allocated MiB before | Allocated MiB after |
|---|---:|---:|---:|---:|---:|
| Shallow KAN | 2.514 | 0.145 | 17.4x | 8451 | 30.35 |
| Composed KAN | 4.665 | 0.694 | 6.72x | 14835 | 309.01 |
| MLP control | 0.621 | 0.664 | 0.93x | 1880 | 1880 |
| Spline control, 28 coefficients | 0.358 | 0.310 | 1.16x | 124.59 | 124.59 |

The composed KAN's allocation volume falls by about 48x, from roughly
14.5 GiB to 309 MiB. These are cumulative allocations, not peak resident
memory. Control timings indicate ordinary run-to-run variation; the KAN
reductions are much larger.

Maximum differences after refitting were 2.58e-8 in trajectory RMSE,
6.24e-10 in response RMSE, and 2.15e-8 in validation RMSE. The approximation
assessment therefore remains materially the same, but the cost conclusion
changes: the composed KAN is now close to the MLP's fit time on the nonlinear
fixture rather than about 7.5x slower. The optimized shallow KAN is faster
than the parameter-matched spline and MLP in that fixture.

No general advantage over splines is established. The eight-coefficient
spline remains a strong, cheap baseline, and the KAN's longer-horizon
prediction errors have not improved merely because evaluation is cheaper.
The next substantive assessment should be genuinely multivariate.

## Reproduce

From the repository root, using the environment described in [README.md](README.md):

```bash
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/profile.jl \
  --output=benchmarks/kan/results/profile-new \
  --reference=benchmarks/kan/results/profile-before/oracle.toml
julia --project=benchmarks/kan --check-bounds=yes benchmarks/kan/compare.jl \
  --iterations=1000 --output=benchmarks/kan/results/comparison-new
```

The archived baseline profile was generated before the evaluator change.
Fresh runs use the current implementation and must use a new output path.

- [Baseline profile](results/profile-before/profile.csv)
- [Optimized profile](results/profile-optimized/profile.csv)
- [Optimized frozen-weight oracles](results/profile-optimized/oracle.toml)
- [Optimized assessment summary](results/optimized-1000/summary.md)
- [Optimized selections](results/optimized-1000/selected.csv)
- [Optimized assessment metadata](results/optimized-1000/metadata.toml)
