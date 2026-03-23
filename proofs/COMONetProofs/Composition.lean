import COMONetProofs.Basics
import COMONetProofs.Monotonicity
import COMONetProofs.Convexity
import COMONetProofs.Concavity
import Mathlib.Analysis.Convex.Function
import Mathlib.Order.Monotone.Basic

/-!
# Composition Theorems for COMONet

## Key Results

1. **Monotone ∘ Monotone = Monotone**: Standard composition rule.
2. **Positive scaling preserves convexity/concavity**.
3. **COMONet full network guarantees**: 2-hidden + output layer.
4. **Summary theorem**: Collects all architectural guarantees.
-/

namespace COMONetProofs

/-! ### Composition rules -/

/-- **Monotone composition**: standard mathematical fact. -/
theorem comp_monotone {f g : ℝ → ℝ}
    (hf : Monotone f) (hg : Monotone g) :
    Monotone (g ∘ f) :=
  hg.comp hf

/-- Positive scaling preserves convexity. -/
theorem pos_scale_convexOn {f : ℝ → ℝ}
    (hf : ConvexOn ℝ Set.univ f) {α : ℝ} (hα : α ≥ 0) (b : ℝ) :
    ConvexOn ℝ Set.univ (fun x => α * f x + b) := by
  rw [ConvexOn]
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a c ha hc hac
  simp only [smul_eq_mul]
  have hf_ineq := hf.2 (Set.mem_univ x) (Set.mem_univ y) ha hc hac
  simp only [smul_eq_mul] at hf_ineq
  have h1 : α * f (a * x + c * y) ≤ α * (a * f x + c * f y) :=
    mul_le_mul_of_nonneg_left hf_ineq hα
  have hac1 : a + c = 1 := hac
  have : b = (a + c) * b := by rw [hac1, one_mul]
  linarith

/-- Positive scaling preserves concavity. -/
theorem pos_scale_concaveOn {f : ℝ → ℝ}
    (hf : ConcaveOn ℝ Set.univ f) {α : ℝ} (hα : α ≥ 0) (b : ℝ) :
    ConcaveOn ℝ Set.univ (fun x => α * f x + b) := by
  rw [ConcaveOn]
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a c ha hc hac
  simp only [smul_eq_mul]
  have hf_ineq := hf.2 (Set.mem_univ x) (Set.mem_univ y) ha hc hac
  simp only [smul_eq_mul] at hf_ineq
  have h1 : α * (a * f x + c * f y) ≤ α * f (a * x + c * y) :=
    mul_le_mul_of_nonneg_left hf_ineq hα
  have hac1 : a + c = 1 := hac
  have : b = (a + c) * b := by rw [hac1, one_mul]
  linarith

/-! ### COMONet output layer -/

/-- **COMONet linear output with exp-weight preserves convexity**. -/
theorem exp_weight_output_preserves_convexOn {f : ℝ → ℝ}
    (hf : ConvexOn ℝ Set.univ f) (w_tilde b : ℝ) :
    ConvexOn ℝ Set.univ (fun x => Real.exp w_tilde * f x + b) :=
  pos_scale_convexOn hf (le_of_lt (exp_pos w_tilde)) b

/-- **COMONet linear output with exp-weight preserves concavity**. -/
theorem exp_weight_output_preserves_concaveOn {f : ℝ → ℝ}
    (hf : ConcaveOn ℝ Set.univ f) (w_tilde b : ℝ) :
    ConcaveOn ℝ Set.univ (fun x => Real.exp w_tilde * f x + b) :=
  pos_scale_concaveOn hf (le_of_lt (exp_pos w_tilde)) b

/-! ### Full COMONet architecture theorems -/

/-- **COMONet monotone network (2 hidden + output)**: Monotone for any weights. -/
theorem comonet_monotone_network_3layer
    (w₁ b₁ w₂ b₂ w₃ b₃ : ℝ) :
    Monotone (fun x =>
      Real.exp w₃ * relu (Real.exp w₂ * relu (Real.exp w₁ * x + b₁) + b₂) + b₃) :=
  (affine_nonneg_weight_mono (le_of_lt (exp_pos w₃))).comp
    (relu_monotone.comp
      ((affine_nonneg_weight_mono (le_of_lt (exp_pos w₂))).comp
        (relu_monotone.comp (affine_nonneg_weight_mono (le_of_lt (exp_pos w₁))))))

/-- **COMONet convex network (2 hidden + output)**: Convex for any weights. -/
theorem comonet_convex_network_3layer
    (w₁ b₁ w₂ b₂ w₃ b₃ : ℝ) :
    ConvexOn ℝ Set.univ (fun x =>
      Real.exp w₃ * relu (Real.exp w₂ * relu (Real.exp w₁ * x + b₁) + b₂) + b₃) :=
  exp_weight_output_preserves_convexOn
    (two_layer_convUnit_exp_weight_convexOn w₁ b₁ w₂ b₂)
    w₃ b₃

/-- **COMONet concave network (2 hidden + output)**: Concave for any weights. -/
theorem comonet_concave_network_3layer
    (w₁ b₁ w₂ b₂ w₃ b₃ : ℝ) :
    ConcaveOn ℝ Set.univ (fun x =>
      Real.exp w₃ * nrelu (Real.exp w₂ * nrelu (Real.exp w₁ * x + b₁) + b₂) + b₃) :=
  exp_weight_output_preserves_concaveOn
    (two_layer_concUnit_exp_weight_concaveOn w₁ b₁ w₂ b₂)
    w₃ b₃

/-- **Summary theorem**: COMONet architectural constraints guarantee
shape properties for ALL parameter values. No constrained optimization
needed — the guarantees are structural. -/
theorem comonet_shape_guarantees :
    -- 1. Monotonicity: for all weights, the monotone network is monotone
    (∀ w₁ b₁ w₂ b₂ : ℝ,
      Monotone (fun x => relu (Real.exp w₂ * relu (Real.exp w₁ * x + b₁) + b₂))) ∧
    -- 2. Convexity: for all weights, the convex network is convex
    (∀ w₁ b₁ w₂ b₂ : ℝ,
      ConvexOn ℝ Set.univ
        (fun x => relu (Real.exp w₂ * relu (Real.exp w₁ * x + b₁) + b₂))) ∧
    -- 3. Concavity: for all weights, the concave network is concave
    (∀ w₁ b₁ w₂ b₂ : ℝ,
      ConcaveOn ℝ Set.univ
        (fun x => nrelu (Real.exp w₂ * nrelu (Real.exp w₁ * x + b₁) + b₂))) := by
  exact ⟨
    fun w₁ b₁ w₂ b₂ =>
      relu_monotone.comp
        ((affine_nonneg_weight_mono (le_of_lt (exp_pos w₂))).comp
          (relu_monotone.comp (affine_nonneg_weight_mono (le_of_lt (exp_pos w₁))))),
    fun w₁ b₁ w₂ b₂ => two_layer_convUnit_exp_weight_convexOn w₁ b₁ w₂ b₂,
    fun w₁ b₁ w₂ b₂ => two_layer_concUnit_exp_weight_concaveOn w₁ b₁ w₂ b₂⟩

end COMONetProofs
