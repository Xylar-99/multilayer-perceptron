# Multilayer Perceptron — Concept Guide

> Deep explanation of the neural-network concepts this project demands.
> **No solutions** — the goal is that you could re-derive everything on a whiteboard, because the defense explicitly grades your *explanations* of feedforward, backpropagation, and gradient descent.

---

## 1. The big picture

You will build a **neural network from scratch** (no TensorFlow/PyTorch/Keras — only linear algebra libraries like numpy allowed) that reads 30 numeric measurements of a breast-mass cell nucleus and predicts **malignant (M) or benign (B)**. This is the natural next step after DSLR:

- ft_linear_regression → 1 neuron, identity activation (a line).
- dslr → 1 neuron, sigmoid activation (logistic regression).
- **multilayer_perceptron → many neurons in layers** — a composition of logistic-regression-like units that can learn *non-linear* decision boundaries.

Everything you learned before (gradient descent, cross-entropy loss, feature scaling, train/validation splits) reappears here, generalized.

---

## 2. The neuron (perceptron), precisely

One neuron does two things:

1. **Weighted sum:** `z = Σ (xₖ · wₖ) + b` — a dot product between inputs and weights, plus a bias.
2. **Activation:** `a = f(z)` — a non-linear squashing of z.

- The **weights** decide how much each input matters (learned).
- The **bias** shifts the activation threshold — without it, every neuron's decision boundary would be forced through the origin (learned too).
- The **activation function** is what gives the network its power. Key fact to internalize: **without non-linear activations, stacking layers is pointless** — a composition of linear maps is still one linear map; a 100-layer linear network can't do more than logistic regression. The non-linearity is the entire reason hidden layers help.

Activations you should know (and compare at defense):

| Function | Formula | Range | Notes |
|---|---|---|---|
| Sigmoid | 1/(1+e⁻ᶻ) | (0,1) | Historical default; saturates → vanishing gradients in deep nets |
| Tanh | (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ) | (−1,1) | Zero-centered sigmoid; usually trains better than sigmoid |
| ReLU | max(0, z) | [0,∞) | Cheap, no saturation for z>0; "dead neuron" risk for z<0 |
| Softmax | eᶻⁱ/Σeᶻʲ | outputs sum to 1 | **Required on your output layer** — turns scores into a probability distribution over classes |

## 3. The architecture — layers and matrices

A **dense (fully connected) layer**: every neuron receives *all* outputs of the previous layer. The whole layer computes, in one matrix operation:

```
Zˡ = Wˡ · Aˡ⁻¹ + bˡ        (weighted sums for the whole layer at once)
Aˡ = f(Zˡ)                  (activations)
```

This is why the subject says linear algebra is "indispensable": a layer *is* a matrix multiplication. Understand the **shapes** cold — if layer l−1 has n neurons and layer l has m, then W is (m, n), b is (m, 1), and you should be able to trace shapes through the whole network (the subject's example: 30 inputs → 24 → 24 → 24 → 2 outputs). Most bugs in this project are shape bugs; most defense questions start with "what size is this matrix?"

Requirements from the subject: **at least 2 hidden layers**, softmax output, and a **modular** design where layer count/sizes/activations come from a config or CLI args — think of a layer as an object with its own W, b, f, and the network as a list of layers.

## 4. Feedforward

"Feedforward" = running data through the network to get a prediction: take the input vector, apply layer 1's matrix + activation, feed the result to layer 2, and so on until the output layer's softmax gives you `[P(malignant), P(benign)]`. That's it — a chain of function compositions:

```
ŷ = softmax(W³ · f(W² · f(W¹x + b¹) + b²) + b³)
```

You must be able to narrate this flow on the whiteboard at defense, with shapes.

## 5. The loss — cross-entropy again

With softmax outputs and one-hot labels (M = [1,0], B = [0,1]), the loss is **categorical cross-entropy**; for two classes it reduces to the **binary cross-entropy** the subject prints:

```
E = -(1/N) Σ [ yₙ log(pₙ) + (1−yₙ) log(1−pₙ) ]
```

Same intuition as DSLR: reward confident correct probabilities, punish confident wrong ones brutally. Numerical caution to understand: log(0) = −∞, so you must think about clipping probabilities (e.g., to [ε, 1−ε]) — know *why*, not just that you did it.

## 6. Backpropagation — the heart of the project

Gradient descent needs ∂E/∂w for *every* weight in the network. Backpropagation is just an **efficient application of the chain rule** to compute all of them in one backward sweep.

### The intuition

The error is measured at the output. Each weight's "blame" for that error depends on everything between it and the output. The chain rule lets you propagate blame backward, layer by layer:

1. **Output layer:** with softmax + cross-entropy, the error signal simplifies to the beautiful `δᴸ = ŷ − y` (predicted probabilities minus true one-hot). Understand why this cancellation happens — cross-entropy was *designed* to pair with softmax/sigmoid; it's a classic defense question.
2. **Propagate backward:** `δˡ = (Wˡ⁺¹)ᵀ δˡ⁺¹ ⊙ f'(Zˡ)` — a layer's blame is the next layer's blame, pulled back through the weights, times the local slope of the activation.
3. **Gradients:** `∂E/∂Wˡ = δˡ (Aˡ⁻¹)ᵀ` and `∂E/∂bˡ = δˡ` — each weight's gradient is (blame of its target neuron) × (activation of its source neuron).
4. **Update:** standard gradient descent, `W ← W − α ∂E/∂W`.

Why "backward"? Because computing gradients output→input reuses each δ; computing them independently would be exponentially wasteful. Backprop = dynamic programming over the chain rule.

You should be able to derive the sigmoid case by hand: `f'(z) = f(z)(1−f(z))`.

### Training loop vocabulary

- **Epoch** — one full pass over the training set.
- **Batch size** — how many examples you average gradients over before one update (mini-batch GD; the subject example uses 8).
- **Learning rate** — same role and same failure modes as in ft_linear_regression.
- **Weight initialization** — you cannot start at zero: all neurons in a layer would compute identical outputs and receive identical gradients forever ("symmetry breaking" problem). Random init is mandatory; schemes like **He** (the `heUniform` in the subject) or **Xavier/Glorot** scale randomness by layer size to keep signals from exploding/vanishing. Know why the scale matters.

## 7. Dataset & preprocessing concepts

The data is the **Wisconsin Diagnostic Breast Cancer** dataset: 32 columns = ID + diagnosis (M/B) + 30 real-valued features (mean/SE/worst of 10 nucleus measurements: radius, texture, perimeter, area, smoothness, ...).

Concepts you must apply and justify:

- **The CSV has no header row** — check before parsing.
- **Drop the ID** — it's not a feature (and would leak nothing but noise).
- **Encode labels** — M/B → one-hot vectors matching your 2 softmax outputs.
- **Standardize features** (z-score, with means/stds computed on the *training* set only and saved with the model — the predict program must reuse them). Features range from ~0.05 to ~2500; unscaled, the big ones dominate every gradient.
- **Train/validation split** — your own separate program. The validation set is *never trained on*; it estimates performance on unknown data. A seed makes the split reproducible.
- **Class imbalance awareness** — ~63% benign / 37% malignant; accuracy alone can flatter you. (Good bonus territory: precision/recall/F1 — for cancer detection, a false negative is far worse than a false positive.)

## 8. Overfitting, learning curves, early stopping

You must plot **loss and accuracy for both train and validation per epoch**. Learn to *read* these curves:

- Both losses falling together → healthy learning.
- Training loss falls, validation loss rises → **overfitting**: the network memorizes training noise instead of generalizing. The gap between curves is the diagnostic.
- Both stuck high → underfitting (too small a network, too low a learning rate, bad preprocessing).

**Early stopping** (bonus): keep training while validation loss improves; stop (and keep the best weights) when it hasn't improved for k epochs ("patience"). It's the simplest regularizer and pairs naturally with the curves you already plot.

## 9. Bonus concepts — modern optimizers

All are refinements of `W ← W − α·∇`:

- **Momentum / Nesterov** — accumulate a velocity vector: dampens zig-zag, powers through plateaus. Nesterov "looks ahead" before computing the gradient.
- **RMSprop** — per-parameter adaptive learning rate: divide by a running average of squared gradients, so rarely-updated weights get bigger steps.
- **Adam** — momentum + RMSprop combined (with bias correction); the modern default. If you implement one bonus optimizer, this is the one — and be ready to explain its two moment estimates.

## 10. What you must be able to explain at defense (explicitly graded)

1. **Feedforward** — narrate one input flowing through the network, with matrix shapes.
2. **Backpropagation** — the chain rule story: where δ comes from, how it flows backward, how each W gets its gradient.
3. **Gradient descent** — the update rule, learning rate, batch vs mini-batch vs stochastic.
4. Why non-linear activations are essential; what softmax does and why the output layer needs it.
5. Why cross-entropy (and why it pairs so well with softmax that δᴸ = ŷ − y).
6. Why random weight initialization is required (symmetry breaking).
7. Train vs validation: what each curve means, how you'd recognize overfitting.
8. Your preprocessing pipeline and why predict must reuse training-time statistics.

## 11. Recommended videos 🎥

The 3Blue1Brown series is *the* preparation for this project — watch all four before coding:

| # | Video | Why |
|---|-------|-----|
| 1 | **3Blue1Brown — "But what is a neural network?"** — [youtube.com/watch?v=aircAruvnKk](https://www.youtube.com/watch?v=aircAruvnKk) | Layers, weights, biases, activations — the exact mental model the subject's diagrams assume. |
| 2 | **3Blue1Brown — "Gradient descent, how neural networks learn"** — [youtube.com/watch?v=IHZwWFHWa-w](https://www.youtube.com/watch?v=IHZwWFHWa-w) | Cost surfaces and why the gradient tells each weight how to change. |
| 3 | **3Blue1Brown — "What is backpropagation really doing?"** — [youtube.com/watch?v=Ilg3gGewQ5U](https://www.youtube.com/watch?v=Ilg3gGewQ5U) | The intuition of blame propagating backward — this is the defense question. |
| 4 | **3Blue1Brown — "Backpropagation calculus"** — [youtube.com/watch?v=tIeHLnjs5U8](https://www.youtube.com/watch?v=tIeHLnjs5U8) | The chain-rule derivation, symbol by symbol — turns intuition into the equations you'll implement. |
| 5 | **Samson Zhang — "Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)"** — [youtube.com/watch?v=w8yWXqWQYmU](https://www.youtube.com/watch?v=w8yWXqWQYmU) | Shows what a from-scratch numpy implementation involves (watch for *concepts*: shapes, vectorization — design your own code). |
| 6 | **StatQuest — "Neural Networks Pt. 2: Backpropagation Main Ideas"** — [youtube.com/watch?v=IN2XmBhILt4](https://www.youtube.com/watch?v=IN2XmBhILt4) | Slower, numerical walkthrough if 3B1B's calculus felt fast. |
| 7 | **StatQuest — "The SoftMax Derivative"** + **"Neural Networks with SoftMax"** — [youtube.com/watch?v=KpKog-L9veg](https://www.youtube.com/watch?v=KpKog-L9veg) | Softmax + cross-entropy pairing, required for your output layer. |
| 8 | *(bonus)* **"Adam Optimizer, Clearly Explained" / momentum & RMSprop videos by StatQuest or Sourish Kundu** | Only if you attack the optimizer bonus. |

Extra reading: **Michael Nielsen — *Neural Networks and Deep Learning*** (free online book, [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com)) — chapters 1–3 are the written equivalent of this whole project, including the four backprop equations in exactly the form you need.

## 12. Concept checklist before coding

- [ ] I can compute one neuron's output by hand (weighted sum → activation).
- [ ] I can explain why hidden layers are useless without non-linear activations.
- [ ] I can trace matrix shapes through a 30→24→24→2 network.
- [ ] I can state the 4 backprop equations and say what each term *means*.
- [ ] I know why softmax + cross-entropy gives δ = ŷ − y.
- [ ] I know why zero-initialization fails and what He/Xavier init fixes.
- [ ] I can read a loss curve and point at where overfitting starts.
- [ ] I know what must be saved with the model (topology, weights, normalization stats).
- [ ] I know which libraries are forbidden (NN frameworks) vs allowed (numpy, matplotlib).
