import COMONetProofs.Basics
import Mathlib.Order.Monotone.Basic

/-!
# Monotonicity Theorems for COMONet

## Key Results

1. **Single neuron monotonicity**: If `w > 0` and `σ` is monotone
   non-decreasing, then `x ↦ σ(w * x + b)` is monotone non-decreasing.

2. **exp-weight monotonicity**: For any unconstrained `w̃ ∈ ℝ`,
   `x ↦ σ(exp(w̃) * x + b)` is monotone when σ is monotone.

3. **Composition preservation**: The composition of monotone non-decreasing
   functions is monotone non-decreasing.

These theorems form the basis of COMONet's monotonicity guarantee:
each layer uses `exp(W̃)` weights (positive by construction) and
monotone activations, so the overall network is monotone.
-/

namespace COMONetProofs

/-! ### Single neuron monotonicity -/

/-- **Core monotonicity theorem**: If `σ` is monotone non-decreasing and
`w > 0`, then the neuron `x ↦ σ(w * x + b)` is monotone non-decreasing.

This is the fundamental building block of COMONet's monotonicity guarantee.
The proof follows from:
1. `x ↦ w * x + b` is monotone when `w > 0`
2. Monotone ∘ Monotone = Monotone -/
theorem neuron_monotone_of_pos_weight
    {σ : ℝ → ℝ} (hσ : Monotone σ) {w b : ℝ} (hw : w > 0) :
    Monotone (fun x => σ (w * x + b)) :=
  hσ.comp (affine_pos_weight_mono hw)

/-- **exp-weight monotonicity**: For any unconstrained weight `w̃ ∈ ℝ`
and any monotone activation `σ`, the neuron `x ↦ σ(exp(w̃) * x + b)`
is monotone non-decreasing.

This is the key insight of COMONet: parameterizing weights through `exp`
guarantees positivity without constrained optimization, and combined
with monotone activations gives guaranteed monotone neurons. -/
theorem neuron_monotone_of_exp_weight
    {σ : ℝ → ℝ} (hσ : Monotone σ) (w_tilde b : ℝ) :
    Monotone (fun x => σ (Real.exp w_tilde * x + b)) :=
  neuron_monotone_of_pos_weight hσ (exp_pos w_tilde)

/-! ### ReLU neuron monotonicity -/

/-- A ReLU neuron with positive weight is monotone. -/
theorem relu_neuron_monotone {w b : ℝ} (hw : w > 0) :
    Monotone (fun x => relu (w * x + b)) :=
  neuron_monotone_of_pos_weight relu_monotone hw

/-- A ReLU neuron with exp-weight is monotone for any unconstrained weight. -/
theorem relu_neuron_exp_weight_monotone (w_tilde b : ℝ) :
    Monotone (fun x => relu (Real.exp w_tilde * x + b)) :=
  neuron_monotone_of_exp_weight relu_monotone w_tilde b

/-! ### Negative ReLU neuron monotonicity -/

/-- A negative ReLU neuron with positive weight is monotone. -/
theorem nrelu_neuron_monotone {w b : ℝ} (hw : w > 0) :
    Monotone (fun x => nrelu (w * x + b)) :=
  neuron_monotone_of_pos_weight nrelu_monotone hw

/-- A negative ReLU neuron with exp-weight is monotone. -/
theorem nrelu_neuron_exp_weight_monotone (w_tilde b : ℝ) :
    Monotone (fun x => nrelu (Real.exp w_tilde * x + b)) :=
  neuron_monotone_of_exp_weight nrelu_monotone w_tilde b

/-! ### Composition of monotone layers -/

/-- **Monotone composition**: The composition of two monotone non-decreasing
functions is monotone non-decreasing.

This is a standard result, but we state it explicitly for clarity in
the COMONet context: if layer L₁ and layer L₂ are both monotone,
then L₂ ∘ L₁ is monotone. -/
theorem monotone_comp_monotone
    {f g : ℝ → ℝ} (hf : Monotone f) (hg : Monotone g) :
    Monotone (g ∘ f) :=
  hg.comp hf

/-- **Multi-layer monotonicity (3 layers)**: Composition of three
monotone functions is monotone.

COMONet typically uses 2-3 hidden layers; this covers the 3-layer case
(input → hidden₁ → hidden₂ → output). -/
theorem monotone_comp3
    {f g h : ℝ → ℝ} (hf : Monotone f) (hg : Monotone g) (hh : Monotone h) :
    Monotone (h ∘ g ∘ f) :=
  hh.comp (hg.comp hf)

/-- **COMONet ExpUnit (single layer)**: A single COMONet monotone layer
with exp-parameterized weight, bias, and monotone activation. -/
theorem comonet_exp_unit_monotone
    {σ : ℝ → ℝ} (hσ : Monotone σ) (w_tilde b : ℝ) :
    Monotone (fun x => σ (Real.exp w_tilde * x + b)) :=
  neuron_monotone_of_exp_weight hσ w_tilde b

/-- **COMONet two-layer monotone network**: Two sequential exp-weight
layers with monotone activations produce a monotone function. -/
theorem comonet_two_layer_monotone
    {σ₁ σ₂ : ℝ → ℝ} (hσ₁ : Monotone σ₁) (hσ₂ : Monotone σ₂)
    (w₁ b₁ w₂ b₂ : ℝ) :
    Monotone (fun x =>
      σ₂ (Real.exp w₂ * σ₁ (Real.exp w₁ * x + b₁) + b₂)) :=
  (neuron_monotone_of_exp_weight hσ₂ w₂ b₂).comp
    (neuron_monotone_of_exp_weight hσ₁ w₁ b₁)

end COMONetProofs
