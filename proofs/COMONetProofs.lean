/-!
# COMONet Proofs

Formal verification of shape constraint guarantees for the
COMONet (Constrained Monotone Network) architecture.

These proofs establish that the COMONet architectural constraints
(positive weights via exp(W), ReLU activations) mathematically
guarantee monotonicity, convexity, and concavity properties
of the resulting neural network functions.

## References

- "A Novel Architecture for Monotone, Convex and Concave Neural Networks"
  (ICLR 2026 submission)
- Pya & Wood (2015) "Shape constrained additive models"
-/

import COMONetProofs.Basics
import COMONetProofs.Monotonicity
import COMONetProofs.Convexity
import COMONetProofs.Concavity
import COMONetProofs.Composition
