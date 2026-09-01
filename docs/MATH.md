# Backpropagation — Complete Derivation for Deep Neural Networks

This document explains how backpropagation works in a deep neural network with multiple hidden layers, using the actual architecture from this project as reference.

---

## Table of Contents

1. [Network Architecture](#1-network-architecture)
2. [What is Backpropagation?](#2-what-is-backpropagation)
3. [The Core Idea: Chain Rule](#3-the-core-idea-chain-rule)
4. [Forward Pass (Full Derivation)](#4-forward-pass-full-derivation)
5. [Loss Function](#5-loss-function)
6. [Backward Pass — Step by Step](#6-backward-pass--step-by-step)
7. [Deriving Gradients for Each Layer](#7-deriving-gradients-for-each-layer)
8. [Weight Updates](#8-weight-updates)
9. [Why This Works — Intuition](#9-why-this-works--intuition)

---

## 1. Network Architecture

For this project, we use a **3-hidden-layer network** with 30 neurons per layer:

```
Input Layer:     30 neurons (features from dataset)
Hidden Layer 1:  30 neurons (sigmoid activation)
Hidden Layer 2:  30 neurons (sigmoid activation)  
Hidden Layer 3:  30 neurons (sigmoid activation)
Output Layer:    1 neuron  (sigmoid activation)
```

### Weight Matrices Dimensions

```
W1: 30 × 30    (input → hidden 1)
W2: 30 × 30    (hidden 1 → hidden 2)
W3: 30 × 30    (hidden 2 → hidden 3)
W4: 30 × 1     (hidden 3 → output)

b1: 1 × 30
b2: 1 × 30
b3: 1 × 30
b4: 1 × 1
```

---

## 2. What is Backpropagation?

**Backpropagation** is the algorithm that computes **how much each weight contributed to the error**, so we can adjust weights to reduce error.

Think of it like this:

```
Forward pass:  Input → [Layer 1] → [Layer 2] → [Layer 3] → [Output] → Loss
                                                                    ↑
                                                                   How wrong?
                                                                   
Backward pass: Input ← [Layer 1] ← [Layer 2] ← [Layer 3] ← [Output]
                ∂J/∂W1   ∂J/∂W1     ∂J/∂W2     ∂J/∂W3     ∂J/∂W4
                ↑         ↑          ↑          ↑          ↑
                Each weight's "blame" for the error
```

**The problem:** We know the final error, but we need to figure out how to distribute "blame" across thousands of weights.

**The solution:** Use the **chain rule** from calculus to propagate error signals backward.

---

## 3. The Core Idea: Chain Rule

### Simple Example of Chain Rule

If `y` depends on `u`, and `u` depends on `x`:

```
        dy          du
dx  =  ----  ·  ----
        du          dx
```

### In Neural Networks

The loss `J` depends on output `ŷ`, which depends on `Z4`, which depends on `W4`...

```
∂J     ∂J     ∂ŷ     ∂Z4     ∂A3
---- = ---- · ---- · ---- · ---- · ... · ∂J/∂W1
∂W1    ∂ŷ     ∂Z4    ∂A3     ∂Z1
```

Instead of computing this huge chain directly, backpropagation computes it **one layer at a time**, reusing previous computations.

---

## 4. Forward Pass (Full Derivation)

### Layer-by-Layer Computation

```
INPUT: X (shape: batch_size × 30)

Layer 1:
  Z1 = X · W1 + b1        (batch_size × 30)
  A1 = σ(Z1)              (batch_size × 30)

Layer 2:
  Z2 = A1 · W2 + b2       (batch_size × 30)
  A2 = σ(Z2)              (batch_size × 30)

Layer 3:
  Z3 = A2 · W3 + b3       (batch_size × 30)
  A3 = σ(Z3)              (batch_size × 30)

Output Layer:
  Z4 = A3 · W4 + b4       (batch_size × 1)
  A4 = σ(Z4) = ŷ          (batch_size × 1)
```

### Why Store A1, A2, A3, Z1, Z2, Z3?

**These are needed for backpropagation!** The chain rule requires knowing the activations from the forward pass to compute gradients.

---

## 5. Loss Function

### Binary Cross-Entropy

For binary classification (malignant vs benign):

```
J = -(1/m) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
```

Where:
- `m` = number of samples in batch
- `y` = true label (0 or 1)
- `ŷ` = predicted probability (output of last layer)

### Derivative of Loss w.r.t. Prediction

```
∂J     1     ŷ - y
---- = --- · -------
∂ŷ     m     ŷ(1-ŷ)
```

This tells us: "How much does the loss change if I change the prediction?"

---

## 6. Backward Pass — Step by Step

### Key Definitions

Let's define **δ (delta)** as the "error signal" at each layer:

```
δ_l = ∂J/∂Z_l   (how much loss changes with pre-activation Z at layer l)
```

Our goal: Find `δ_4, δ_3, δ_2, δ_1` and then compute gradients.

---

### Step 1: Output Layer (Layer 4)

```
δ4 = ∂J/∂Z4 = (∂J/∂ŷ) · (∂ŷ/∂Z4)
```

**Computing ∂J/∂ŷ:**
```
∂J/∂ŷ = (1/m) · (ŷ - y) / (ŷ(1-ŷ))
```

**Computing ∂ŷ/∂Z4 (sigmoid derivative):**
```
ŷ = σ(Z4)
∂ŷ/∂Z4 = σ'(Z4) = σ(Z4)·(1-σ(Z4)) = ŷ·(1-ŷ)
```

**Combining:**
```
δ4 = (1/m) · (ŷ-y)/(ŷ(1-ŷ)) · ŷ(1-ŷ)
   = (1/m) · (ŷ - y)
```

**Beautiful simplification!** The `ŷ(1-ŷ)` terms cancel out.

For a single example:
```
δ4 = ŷ - y    (scalar)
```

**Interpretation:** The error at the output is simply "prediction minus truth."

---

### Step 2: Hidden Layer 3 (Layer 3)

Now we propagate the error backward:

```
δ3 = ∂J/∂Z3 = (∂J/∂Z4) · (∂Z4/∂A3) · (∂A3/∂Z3)
```

Breaking this down:

**∂J/∂Z4 = δ4** (we just computed this)

**∂Z4/∂A3:**
```
Z4 = A3 · W4 + b4
∂Z4/∂A3 = W4^T    (shape: 30 × 1)
```

**∂A3/∂Z3 (sigmoid derivative):**
```
A3 = σ(Z3)
∂A3/∂Z3 = σ'(Z3) = A3 ⊙ (1 - A3)
```
(⊙ = element-wise multiplication)

**Combining:**
```
δ3 = (δ4 · W4^T) ⊙ A3 ⊙ (1 - A3)
```

**Shape check:**
```
δ4:    1 × 1
W4^T: 30 × 1
δ4·W4^T: 1 × 30
A3:    1 × 30
δ3:    1 × 30  ✓
```

**Interpretation:** Each neuron in layer 3 gets error proportional to:
1. How much it contributed to the output (via W4)
2. How "active" it was (via sigmoid derivative)

---

### Step 3: Hidden Layer 2 (Layer 2)

```
δ2 = ∂J/∂Z2 = (∂J/∂Z3) · (∂Z3/∂A2) · (∂A2/∂Z2)
```

**∂J/∂Z3 = δ3** (from previous step)

**∂Z3/∂A2:**
```
Z3 = A2 · W3 + b3
∂Z3/∂A2 = W3^T    (shape: 30 × 30)
```

**∂A2/∂Z2:**
```
A2 = σ(Z2)
∂A2/∂Z2 = A2 ⊙ (1 - A2)
```

**Combining:**
```
δ2 = (δ3 · W3^T) ⊙ A2 ⊙ (1 - A2)
```

**Shape check:**
```
δ3:    1 × 30
W3^T: 30 × 30
δ3·W3^T: 1 × 30
A2:    1 × 30
δ2:    1 × 30  ✓
```

---

### Step 4: Hidden Layer 1 (Layer 1)

```
δ1 = ∂J/∂Z1 = (∂J/∂Z2) · (∂Z2/∂A1) · (∂A1/∂Z1)
```

**∂J/∂Z2 = δ2** (from previous step)

**∂Z2/∂A1:**
```
Z2 = A1 · W2 + b2
∂Z2/∂A1 = W2^T    (shape: 30 × 30)
```

**∂A1/∂Z1:**
```
A1 = σ(Z1)
∂A1/∂Z1 = A1 ⊙ (1 - A1)
```

**Combining:**
```
δ1 = (δ2 · W2^T) ⊙ A1 ⊙ (1 - A1)
```

---

## 7. Deriving Gradients for Each Layer

Now that we have error signals δ, we compute weight gradients:

### General Formula

```
∂J/∂W_l = A_{l-1}^T · δ_l
∂J/∂b_l = δ_l    (summed over batch)
```

**Why?** Because:
```
Z_l[i,j] = Σ_k A_{l-1}[i,k] · W_l[k,j] + b_l[j]

∂Z_l[i,j]/∂W_l[k,j] = A_{l-1}[i,k]

∂J/∂W_l[k,j] = Σ_i (∂J/∂Z_l[i,j]) · (∂Z_l[i,j]/∂W_l[k,j])
             = Σ_i δ_l[i,j] · A_{l-1}[i,k]
```

In matrix form:
```
∂J/∂W_l = A_{l-1}^T · δ_l
```

### Gradients for Our Network

```
Layer 4 (Output):
  dW4 = A3^T · δ4    (30 × 1)
  db4 = δ4           (1 × 1)

Layer 3:
  dW3 = A2^T · δ3    (30 × 30)
  db3 = δ3           (1 × 30)

Layer 2:
  dW2 = A1^T · δ2    (30 × 30)
  db2 = δ2           (1 × 30)

Layer 1:
  dW1 = X^T · δ1     (30 × 30)
  db1 = δ1           (1 × 30)
```

### Shape Verification

```
W1: 30 × 30,  dW1 = X^T(30 × m) · δ1(m × 30) = 30 × 30  ✓
W2: 30 × 30,  dW2 = A1^T(30 × m) · δ2(m × 30) = 30 × 30  ✓
W3: 30 × 30,  dW3 = A2^T(30 × m) · δ3(m × 30) = 30 × 30  ✓
W4: 30 × 1,   dW4 = A3^T(30 × m) · δ4(m × 1)  = 30 × 1   ✓
```

---

## 8. Weight Updates

After computing gradients, we update weights using gradient descent:

```
W_l ← W_l - η · dW_l
b_l ← b_l - η · db_l
```

Where η is the learning rate (e.g., 0.0314).

### Intuition

```
If dW > 0: weight is increasing the loss → decrease weight
If dW < 0: weight is decreasing the loss → increase weight
```

---

## 9. Why This Works — Intuition

### The Credit Assignment Problem

In a deep network with 3 hidden layers:

```
Input → [30 neurons] → [30 neurons] → [30 neurons] → Output
```

If the output is wrong, which weight caused the error?

**Answer:** All of them, but in different proportions. Backpropagation figures out each weight's "share of blame."

### Visualizing Error Flow

```
Forward:  Error = Loss(ŷ, y)
              ↓
          ŷ = σ(Z4)
              ↓
      Z4 = A3·W4 + b4
              ↓
          A3 = σ(Z3)     ← Each neuron's activation affects next layer
              ↓
      Z3 = A2·W3 + b3
              ↓
          A2 = σ(Z2)
              ↓
      Z2 = A1·W2 + b2
              ↓
          A1 = σ(Z1)
              ↓
      Z1 = X·W1 + b1

Backward:  δ4 = ŷ - y           (how wrong is output?)
              ↓
          δ3 = δ4·W4^T ⊙ σ'(Z3) (propagate through W4)
              ↓
          δ2 = δ3·W3^T ⊙ σ'(Z2) (propagate through W3)
              ↓
          δ1 = δ2·W2^T ⊙ σ'(Z1) (propagate through W2)
```

### Key Insights

1. **Earlier layers get smaller gradients** because they're multiplied by more weights (vanishing gradient problem).

2. **The sigmoid derivative `σ'(z) = σ(z)(1-σ(z))`** has a maximum of 0.25 (at z=0), so gradients shrink as they propagate backward.

3. **This is why ReLU is often preferred** for hidden layers—its derivative is 1 for positive inputs, so gradients don't vanish as quickly.

---

## Complete Algorithm Summary

```python
# FORWARD PASS
Z1 = X @ W1 + b1
A1 = sigmoid(Z1)

Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)

Z3 = A2 @ W3 + b3
A3 = sigmoid(Z3)

Z4 = A3 @ W4 + b4
A4 = sigmoid(Z4)  # prediction ŷ

# COMPUTE LOSS
loss = -[y*log(A4) + (1-y)*log(1-A4)]

# BACKWARD PASS
# Output layer
delta4 = A4 - y                          # shape: (m, 1)
dW4 = A3.T @ delta4 / m                  # shape: (30, 1)
db4 = sum(delta4) / m                    # shape: (1,)

# Hidden layer 3
delta3 = (delta4 @ W4.T) * A3 * (1-A3)   # shape: (m, 30)
dW3 = A2.T @ delta3 / m                  # shape: (30, 30)
db3 = sum(delta3) / m                    # shape: (30,)

# Hidden layer 2
delta2 = (delta3 @ W3.T) * A2 * (1-A2)   # shape: (m, 30)
dW2 = A1.T @ delta2 / m                  # shape: (30, 30)
db2 = sum(delta2) / m                    # shape: (30,)

# Hidden layer 1
delta1 = (delta2 @ W2.T) * A1 * (1-A1)   # shape: (m, 30)
dW1 = X.T @ delta1 / m                   # shape: (30, 30)
db1 = sum(delta1) / m                    # shape: (30,)

# UPDATE WEIGHTS
W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2
W3 -= lr * dW3
b3 -= lr * db3
W4 -= lr * dW4
b4 -= lr * db4
```

---

## Visual Diagram of Gradient Flow

```
                    FORWARD PASS
    ┌──────────────────────────────────────────┐
    │                                          │
    ▼                                          │
   X ──►[W1]──► Z1 ──► σ ──► A1              │
                                          │
              ──►[W2]──► Z2 ──► σ ──► A2     │
                                          │
              ──►[W3]──► Z3 ──► σ ──► A3     │
                                          │
              ──►[W4]──► Z4 ──► σ ──► ŷ ──► Loss
                                          │
    ◄──────────────────────────────────────────┘
                    BACKWARD PASS

   dW1 ◄── δ1 ◄─────────────────────────────┘
                    ▲
              δ1 = (δ2·W2ᵀ) ⊙ σ'(Z1)

   dW2 ◄── δ2 ◄────────────────────┘
                    ▲
              δ2 = (δ3·W3ᵀ) ⊙ σ'(Z2)

   dW3 ◄── δ3 ◄───────────┘
                    ▲
              δ3 = (δ4·W4ᵀ) ⊙ σ'(Z3)

   dW4 ◄── δ4 ◄─────┘
                    ▲
              δ4 = ŷ - y
```

---

## Common Questions

### Q: Why multiply by W^T during backprop?

Because the forward pass was `Z_next = A_current · W`, so the gradient flows back through `W^T`. This is the "transpose" pattern in backpropagation.

### Q: Why multiply by the activation derivative?

The activation function is a "gate" that controls how much signal passes through. Its derivative tells us how much changing the input would change the output. If a neuron is saturated (sigmoid near 0 or 1), its derivative is near 0, meaning "this neuron can't learn more."

### Q: What happens if gradients vanish?

If many layers have derivatives < 1 (like sigmoid's max 0.25), multiplying them makes the gradient exponentially small for early layers. The network stops learning. Solutions: ReLU, residual connections, batch normalization.

---

## Further Reading

- [3Blue1Brown: What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [CS231n Notes on Backpropagation](https://cs231n.github.io/optimization-2/)
- [Stanford Deep Learning Tutorial](http://deeplearning.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/)
