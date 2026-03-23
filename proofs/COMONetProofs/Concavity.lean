import COMONetProofs.Basics
import COMONetProofs.Convexity
import Mathlib.Analysis.Convex.Function

/-!
# Concavity Theorems for COMONet

## Key Results

1. **Negative ReLU is concave**: `nrelu(x) = -relu(-x) = min(0, x)` is concave.
2. **COMONet ConcUnit**: `nrelu(exp(w̃) * x + b)` is concave in `x`.
3. **Multi-layer concavity**: ConcUnit layers with positive weights preserve concavity.
-/

namespace COMONetProofs

/-! ### Concavity of negative ReLU -/

/-- Negative ReLU is concave on ℝ. -/
theorem nrelu_concaveOn : ConcaveOn ℝ Set.univ nrelu := by
  rw [ConcaveOn]
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a c ha hc hac
  simp only [nrelu, relu, smul_eq_mul]
  -- Need: a * (-max 0 (-x)) + c * (-max 0 (-y)) ≤ -(max 0 (-(a*x + c*y)))
  -- i.e., max 0 (-(a*x+c*y)) ≤ a * max 0 (-x) + c * max 0 (-y)
  -- This is convexity of relu applied to negated inputs
  suffices h : max 0 (-(a * x + c * y)) ≤ a * max 0 (-x) + c * max 0 (-y) by linarith
  have h_neg : -(a * x + c * y) = a * (-x) + c * (-y) := by ring
  rw [h_neg]
  exact max_le
    (add_nonneg (mul_nonneg ha (le_max_left 0 _)) (mul_nonneg hc (le_max_left 0 _)))
    (add_le_add (mul_le_mul_of_nonneg_left (le_max_right 0 _) ha)
                (mul_le_mul_of_nonneg_left (le_max_right 0 _) hc))

/-! ### ConcUnit -/

/-- **COMONet ConcUnit**: `nrelu(w * x + b)` is concave for any `w, b`. -/
theorem nrelu_affine_concaveOn (w b : ℝ) :
    ConcaveOn ℝ Set.univ (fun x => nrelu (w * x + b)) := by
  rw [ConcaveOn]
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a c ha hc hac
  simp only [nrelu, relu, smul_eq_mul]
  -- Need: a * (-max 0 (-(w*x+b))) + c * (-max 0 (-(w*y+b)))
  --     ≤ -(max 0 (-(w*(a*x+c*y)+b)))
  suffices h : max 0 (-(w * (a * x + c * y) + b))
      ≤ a * max 0 (-(w * x + b)) + c * max 0 (-(w * y + b)) by linarith
  have h_lin : -(w * (a * x + c * y) + b) = a * (-(w * x + b)) + c * (-(w * y + b)) := by
    have hac1 : a + c = 1 := hac
    have : b = (a + c) * b := by rw [hac1, one_mul]
    linarith
  rw [h_lin]
  exact max_le
    (add_nonneg (mul_nonneg ha (le_max_left 0 _)) (mul_nonneg hc (le_max_left 0 _)))
    (add_le_add (mul_le_mul_of_nonneg_left (le_max_right 0 _) ha)
                (mul_le_mul_of_nonneg_left (le_max_right 0 _) hc))

/-- **COMONet ConcUnit with exp-weight**: For any unconstrained weight
`w̃` and bias `b`, `nrelu(exp(w̃) * x + b)` is concave in `x`. -/
theorem nrelu_exp_weight_concaveOn (w_tilde b : ℝ) :
    ConcaveOn ℝ Set.univ (fun x => nrelu (Real.exp w_tilde * x + b)) :=
  nrelu_affine_concaveOn (Real.exp w_tilde) b

/-! ### Multi-layer concavity -/

/-- **Two-layer ConcUnit concavity** with positive outer weight. -/
theorem two_layer_concUnit_concaveOn (w₁ b₁ w₂ b₂ : ℝ)
    (hw₂ : w₂ ≥ 0) :
    ConcaveOn ℝ Set.univ
      (fun x => nrelu (w₂ * nrelu (w₁ * x + b₁) + b₂)) := by
  rw [ConcaveOn]
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a c ha hc hac
  simp only [nrelu, relu, smul_eq_mul]
  -- Inner concavity: nrelu(w₁*z+b₁) is concave
  have h_inner := (nrelu_affine_concaveOn w₁ b₁).2
    (Set.mem_univ x) (Set.mem_univ y) ha hc hac
  simp only [nrelu, relu, smul_eq_mul] at h_inner
  set ix := -max 0 (-(w₁ * x + b₁))
  set iy := -max 0 (-(w₁ * y + b₁))
  set iz := -max 0 (-(w₁ * (a * x + c * y) + b₁))
  -- h_inner : a * ix + c * iy ≤ iz
  -- Need outer nrelu to preserve: suffices to show
  -- max 0 (-(w₂*iz+b₂)) ≤ a * max 0 (-(w₂*ix+b₂)) + c * max 0 (-(w₂*iy+b₂))
  suffices h : max 0 (-(w₂ * iz + b₂))
      ≤ a * max 0 (-(w₂ * ix + b₂)) + c * max 0 (-(w₂ * iy + b₂)) by linarith
  -- Since iz ≥ a*ix + c*iy and w₂ ≥ 0:
  -- w₂*iz ≥ w₂*(a*ix + c*iy) = a*w₂*ix + c*w₂*iy
  -- So -(w₂*iz+b₂) ≤ -(a*w₂*ix + c*w₂*iy + b₂) = a*(-(w₂*ix+b₂)) + c*(-(w₂*iy+b₂))
  have h_w2_iz : w₂ * (a * ix + c * iy) ≤ w₂ * iz :=
    mul_le_mul_of_nonneg_left h_inner hw₂
  have h_neg : -(w₂ * iz + b₂) ≤ a * (-(w₂ * ix + b₂)) + c * (-(w₂ * iy + b₂)) := by
    have hac1 : a + c = 1 := hac
    have : b₂ = (a + c) * b₂ := by rw [hac1, one_mul]
    linarith
  calc max 0 (-(w₂ * iz + b₂))
      ≤ max 0 (a * (-(w₂ * ix + b₂)) + c * (-(w₂ * iy + b₂))) :=
        max_le_max_left 0 h_neg
    _ ≤ a * max 0 (-(w₂ * ix + b₂)) + c * max 0 (-(w₂ * iy + b₂)) :=
        max_le
          (add_nonneg (mul_nonneg ha (le_max_left 0 _)) (mul_nonneg hc (le_max_left 0 _)))
          (add_le_add (mul_le_mul_of_nonneg_left (le_max_right 0 _) ha)
                      (mul_le_mul_of_nonneg_left (le_max_right 0 _) hc))

/-- **Two-layer ConcUnit with exp-weights** is concave. -/
theorem two_layer_concUnit_exp_weight_concaveOn (w₁ b₁ w₂ b₂ : ℝ) :
    ConcaveOn ℝ Set.univ
      (fun x => nrelu (Real.exp w₂ * nrelu (Real.exp w₁ * x + b₁) + b₂)) :=
  two_layer_concUnit_concaveOn (Real.exp w₁) b₁ (Real.exp w₂) b₂
    (le_of_lt (exp_pos w₂))

end COMONetProofs
