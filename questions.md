# Neural Net — Concepts & Questions

> The why behind the implementation. Questions worked through building the first neural net from scratch.

---

**How many weights and biases do you initialize?**
Determined by how many layers you have, not how big they are. 2 connections (input→hidden, hidden→output) = 2 weight matrices + 2 bias vectors = 4 total. Scaling from (4,8,1) to (32,64,1) doesn't add more matrices, just makes them bigger.

---

**The input of each layer is the output of the previous one**
`A1` is the output of the hidden layer and the input to the output layer. That's literally what "connected layers" means. The chain just keeps going for however deep you build it.

---

**Why do you initialize weights randomly but biases as zeros?**
Weights need randomness so neurons learn different things — if all weights are equal they all update identically and the hidden layer is useless. Biases don't have this problem so zeros is fine.

---

**What does `global` do?**
Lets a function modify a variable defined outside of it. Without `global`, Python creates a new local variable instead of updating the real one. You need it in `update_params` so weight changes actually stick between epochs.

---

**Which loss function do you use?**
- MSE → regression (predicting a number)
- Binary cross-entropy → binary classification (0 or 1)
- Categorical cross-entropy → multi-class (one of N classes)

MSE is the right starting point — simplest derivative, same backprop mechanics as the others.

---

**Why does backprop go backwards?**
Each layer's gradient depends on the layer ahead of it. You can't compute `dW1` until you know `dA1`, and you can't know `dA1` until you've computed `dZ2`. The chain rule forces the order.

---

**Backprop calculates gradients. Weight update applies them. Two separate steps.**
Don't conflate them. Backprop tells you which direction to move. Weight update actually moves.

---

**Why was loss flat for hundreds of epochs then suddenly dropped?**
Normal for XOR. The hidden layer has to learn a non-linear representation first before the output layer can click into place. The flat region is the network figuring out internal structure, not a sign something is broken.

---

**Why `* 0.01` on weight initialization?**
Keeps initial weights small. Large initial weights push activations into the saturated regions of sigmoid (near 0 or 1) where the gradient is nearly zero — the network can't learn. Small weights keep you in the active region early on.

---

**Why do you return Z1 and Z2 from the forward pass?**
Backprop needs them to compute the activation derivatives (`relu_derivative(Z1)`, `sigmoid_derivative(Z2)`). If you don't return them you'd have to recompute them inside backprop, which defeats the point.

---

**Why store both Z and A?**
A is what goes forward to the next layer. Z is what you save for the way back. They serve different purposes — don't confuse them.

---

**What is loss actually telling you?**
A single number representing how wrong your predictions are across all samples. Everything in backprop exists to reduce this number. When loss stops dropping meaningfully, the network has converged.