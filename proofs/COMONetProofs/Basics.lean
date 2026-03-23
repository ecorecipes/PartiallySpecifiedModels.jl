import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Order.Monotone.Basic
import Mathlib.Topology.Order.Basic

/-!
# COMONet Basics

Fundamental definitions and properties for the COMONet architecture.

## Positive weights via exp

COMONet ensures positive weights by parameterizing weight matrices as
`W = exp(W̃)` where `W̃ ∈ ℝ` is unconstrained. Since `exp : ℝ → ℝ₊`,
this guarantees `W > 0` for all parameter values.

## Activation functions

- **ReLU**: `max(0, x)` — non-decreasing, convex
- **Negative ReLU**: `-max(0, -x) = min(0, x)` — non-decreasing, concave
-/

namespace COMONetProofs

/-! ### Positive weights -/

/-- The exponential function is strictly positive for all real inputs.
This is the foundational property ensuring COMONet weights are positive. -/
theorem exp_pos (w : ℝ) : Real.exp w > 0 :=
  Real.exp_pos w

/-- The exponential function is positive (non-strict), useful for
weight multiplication properties. -/
theorem exp_nonneg (w : ℝ) : Real.exp w ≥ 0 :=
  le_of_lt (Real.exp_pos w)

/-! ### ReLU activation function -/

/-- ReLU activation: `relu(x) = max(0, x)`. -/
noncomputable def relu (x : ℝ) : ℝ := max 0 x

/-- ReLU is non-negative. -/
theorem relu_nonneg (x : ℝ) : relu x ≥ 0 := by
  simp [relu]

/-- ReLU at non-negative input equals the input. -/
theorem relu_of_nonneg {x : ℝ} (hx : x ≥ 0) : relu x = x := by
  simp [relu, max_eq_right (by linarith : (0 : ℝ) ≤ x)]

/-- ReLU at negative input is zero. -/
theorem relu_of_neg {x : ℝ} (hx : x < 0) : relu x = 0 := by
  simp [relu, max_eq_left (by linarith : (0 : ℝ) ≥ x)]

/-- ReLU is monotone non-decreasing. -/
theorem relu_monotone : Monotone relu := by
  intro a b hab
  simp only [relu]
  exact max_le_max_left 0 hab

/-! ### Negative ReLU (concave unit activation) -/

/-- Negative ReLU: `nrelu(x) = -max(0, -x) = min(0, x)`.
Used in COMONet's concave units. -/
noncomputable def nrelu (x : ℝ) : ℝ := -relu (-x)

/-- Negative ReLU equals min(0, x). -/
theorem nrelu_eq_min (x : ℝ) : nrelu x = min 0 x := by
  simp only [nrelu, relu]
  -- -max(0, -x) = min(0, x) by the identity -max(a,b) = min(-a,-b)
  cases le_or_gt 0 x with
  | inl h =>
    -- x ≥ 0: max(0,-x) = 0, min(0,x) = 0
    have : -x ≤ 0 := neg_nonpos.mpr h
    rw [max_eq_left this, neg_zero, min_eq_left h]
  | inr h =>
    -- x < 0: max(0,-x) = -x, min(0,x) = x
    have : 0 ≤ -x := neg_nonneg.mpr (le_of_lt h)
    rw [max_eq_right this, neg_neg, min_eq_right (le_of_lt h)]

/-- Negative ReLU is non-positive. -/
theorem nrelu_nonpos (x : ℝ) : nrelu x ≤ 0 := by
  rw [nrelu_eq_min]
  exact min_le_left 0 x

/-- Negative ReLU is monotone non-decreasing. -/
theorem nrelu_monotone : Monotone nrelu := by
  intro a b hab
  simp only [nrelu]
  exact neg_le_neg (relu_monotone (neg_le_neg hab))

/-! ### Affine maps with positive weights -/

/-- An affine map `x ↦ w * x + b` with `w > 0` is strictly monotone. -/
theorem affine_pos_weight_strictMono {w b : ℝ} (hw : w > 0) :
    StrictMono (fun x => w * x + b) := by
  intro a c hac
  linarith [mul_lt_mul_of_pos_left hac hw]

/-- An affine map `x ↦ w * x + b` with `w > 0` is monotone non-decreasing. -/
theorem affine_pos_weight_mono {w b : ℝ} (hw : w > 0) :
    Monotone (fun x => w * x + b) :=
  StrictMono.monotone (affine_pos_weight_strictMono hw)

/-- An affine map with `w ≥ 0` is monotone non-decreasing. -/
theorem affine_nonneg_weight_mono {w b : ℝ} (hw : w ≥ 0) :
    Monotone (fun x => w * x + b) := by
  intro a c hac
  have : w * a ≤ w * c := mul_le_mul_of_nonneg_left hac hw
  linarith

end COMONetProofs
