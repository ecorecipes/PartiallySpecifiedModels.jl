import COMONetProofs.Basics
import Mathlib.Analysis.Convex.Function
import Mathlib.Analysis.Convex.SpecificFunctions.Basic

/-!
# Convexity Theorems for COMONet

## Key Results

1. **ReLU is convex**: `max(0, x)` is a convex function on ℝ.
2. **COMONet ConvUnit**: `relu(exp(w̃) * x + b)` is convex in `x`.
3. **Multi-layer convexity**: Composition of ConvUnit layers is convex.
-/

namespace COMONetProofs

/-! ### Convexity of ReLU -/

/-- ReLU is convex on ℝ. Follows from `max` of two convex functions. -/
theorem relu_convexOn : ConvexOn ℝ Set.univ relu := by
  apply ConvexOn.sup
  · exact convexOn_const 0 convex_univ
  · exact convexOn_id convex_univ

/-! ### ReLU applied to affine map -/

/-- Helper: affine precomposition preserves convexity when
the original function is convex. We prove `relu ∘ affine` is convex
directly from the definition. -/
theorem relu_affine_convexOn (w b : ℝ) :
    ConvexOn ℝ Set.univ (fun x => relu (w * x + b)) := by
  rw [ConvexOn]
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a c ha hc hac
  simp only [relu, smul_eq_mul]
  -- Goal: max 0 (w * (a * x + c * y) + b) ≤ a * max 0 (w * x + b) + c * max 0 (w * y + b)
  have h_lin : w * (a * x + c * y) + b = a * (w * x + b) + c * (w * y + b) := by
    have hac1 : a + c = 1 := hac
    have : b = (a + c) * b := by rw [hac1, one_mul]
    linarith
  rw [h_lin]
  exact max_le
    (add_nonneg (mul_nonneg ha (le_max_left 0 _)) (mul_nonneg hc (le_max_left 0 _)))
    (add_le_add (mul_le_mul_of_nonneg_left (le_max_right 0 _) ha)
                (mul_le_mul_of_nonneg_left (le_max_right 0 _) hc))

/-- **COMONet ConvUnit with exp-weight**: For any unconstrained weight
`w̃` and bias `b`, `relu(exp(w̃) * x + b)` is convex in `x`. -/
theorem relu_exp_weight_convexOn (w_tilde b : ℝ) :
    ConvexOn ℝ Set.univ (fun x => relu (Real.exp w_tilde * x + b)) :=
  relu_affine_convexOn (Real.exp w_tilde) b

/-! ### Two-layer convexity -/

/-- **Two-layer ConvUnit convexity**: `relu(w₂ * relu(w₁ * x + b₁) + b₂)`
is convex when `w₁, w₂ > 0`. -/
theorem two_layer_convUnit_convexOn (w₁ b₁ w₂ b₂ : ℝ)
    (hw₂ : w₂ ≥ 0) :
    ConvexOn ℝ Set.univ
      (fun x => relu (w₂ * relu (w₁ * x + b₁) + b₂)) := by
  rw [ConvexOn]
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a c ha hc hac
  simp only [relu, smul_eq_mul]
  -- Inner convexity
  have h_inner := (relu_affine_convexOn w₁ b₁).2
    (Set.mem_univ x) (Set.mem_univ y) ha hc hac
  simp only [relu, smul_eq_mul] at h_inner
  set ix := max 0 (w₁ * x + b₁)
  set iy := max 0 (w₁ * y + b₁)
  set iz := max 0 (w₁ * (a * x + c * y) + b₁)
  -- h_inner : iz ≤ a * ix + c * iy
  -- Need: max 0 (w₂ * iz + b₂) ≤ a * max 0 (w₂ * ix + b₂) + c * max 0 (w₂ * iy + b₂)
  have h_w2_iz : w₂ * iz ≤ w₂ * (a * ix + c * iy) :=
    mul_le_mul_of_nonneg_left h_inner hw₂
  have h_expand : w₂ * (a * ix + c * iy) = a * (w₂ * ix) + c * (w₂ * iy) := by ring
  have h_sum : w₂ * iz + b₂ ≤ a * (w₂ * ix + b₂) + c * (w₂ * iy + b₂) := by
    have hac1 : a + c = 1 := hac
    have h1 : w₂ * iz ≤ a * (w₂ * ix) + c * (w₂ * iy) := by linarith [h_w2_iz, h_expand]
    have h2 : a * (w₂ * ix + b₂) + c * (w₂ * iy + b₂)
            = a * (w₂ * ix) + c * (w₂ * iy) + (a + c) * b₂ := by ring
    rw [h2, hac1, one_mul]
    linarith
  exact max_le
    (add_nonneg (mul_nonneg ha (le_max_left 0 _)) (mul_nonneg hc (le_max_left 0 _)))
    (le_trans h_sum
      (add_le_add (mul_le_mul_of_nonneg_left (le_max_right 0 _) ha)
                  (mul_le_mul_of_nonneg_left (le_max_right 0 _) hc)))

/-- **Two-layer ConvUnit with exp-weights** is convex. -/
theorem two_layer_convUnit_exp_weight_convexOn (w₁ b₁ w₂ b₂ : ℝ) :
    ConvexOn ℝ Set.univ
      (fun x => relu (Real.exp w₂ * relu (Real.exp w₁ * x + b₁) + b₂)) :=
  two_layer_convUnit_convexOn (Real.exp w₁) b₁ (Real.exp w₂) b₂
    (le_of_lt (exp_pos w₂))

end COMONetProofs
