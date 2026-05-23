# Neural Net From Scratch — Zac's Reference

> Built this from scratch using numpy only. No PyTorch, no TensorFlow. This is the real thing.

---

## The Mental Model

A neural net is just a function approximator. You feed it inputs, it does matrix math, and you compare its output to reality. The gap between output and reality is **loss**. You use that loss to nudge every weight slightly. Repeat millions of times — it learns.

**Everything flows in this order, every single training step:**
```
forward pass → loss → backward pass → weight update
```

---

## Step 1 — Define Your Architecture

This is the first thing you write. You're declaring the shape of the network.

```python
input_size  = 4   # number of features in your data
hidden_size = 8   # neurons in the hidden layer (you pick this)
output_size = 1   # what you're predicting
```

The number of weights/biases is determined by **how many layers you have**, not how big they are. 2 connections = 4 matrices, always.

---

## Step 2 — Initialize Weights and Biases

```python
W1 = np.random.randn(input_size, hidden_size) * 0.01   # (4, 8)
b1 = np.zeros((1, hidden_size))                         # (1, 8)
W2 = np.random.randn(hidden_size, output_size) * 0.01  # (8, 1)
b2 = np.zeros((1, output_size))                         # (1, 1)
```

- `W` shape is always `(left_layer_size, right_layer_size)`
- `b` shape is always `(1, right_layer_size)`
- `* 0.01` keeps weights small — large initial weights cause exploding gradients
- Biases start at zero, weights need randomness so neurons learn differently

---

## Step 3 — Activation Functions

Write these once, call them anywhere. These are what make the network non-linear.

```python
def relu(x):
    return np.maximum(0, x)           # hidden layers

def sigmoid(x):
    return 1 / (1 + np.exp(-x))      # output layer (binary classification)

def relu_derivative(x):
    return (x > 0).astype(float)     # 1 if x > 0, else 0

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)               # computed from output you already have
```

| Function | Use | Formula | Derivative |
|----------|-----|---------|-----------|
| ReLU | Hidden layers | max(0, x) | 1 if x > 0 else 0 |
| Sigmoid | Output (binary) | 1/(1+e^-x) | σ(x) * (1 - σ(x)) |

---

## Step 4 — Forward Pass

The input of each layer is the output of the previous one. Store Z values — you need them in backprop.

```python
def forward(X):
    Z1 = X @ W1 + b1      # linear combination, shape (n, 8)
    A1 = relu(Z1)          # activation, shape (n, 8)
    Z2 = A1 @ W2 + b2     # linear combination, shape (n, 1)
    A2 = sigmoid(Z2)       # final prediction, shape (n, 1)
    return A1, A2, Z1, Z2  # return Z's for backprop
```

- `Z` = raw pre-activation value → **save for backprop**
- `A` = activated output → **pass forward to next layer**
- `@` is matrix multiply, preferred over `.dot()` or `.matmul()`

---

## Step 5 — Loss Function

Measures how wrong you are. Backprop starts with the derivative of this.

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]
```

| Loss | Use When |
|------|----------|
| MSE | Regression (predicting a number) |
| Binary Cross-Entropy | Binary classification (0 or 1) |
| Categorical Cross-Entropy | Multi-class (one of N classes) |

---

## Step 6 — Backpropagation

Go **strictly backwards**. Chain rule: `dL/dW = dL/dA * dA/dZ * dZ/dW`

The 3 operations you need:
```
elementwise multiply  →  *
matrix multiply       →  @
transpose             →  .T
```

```python
def backward(X, y_true, A1, A2, Z1, Z2):
    # --- OUTPUT LAYER (start here) ---
    dA2 = mse_loss_derivative(y_true, A2)   # entry point into backprop
    dZ2 = dA2 * sigmoid_derivative(Z2)      # through activation
    dW2 = A1.T @ dZ2                         # gradient for W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) # gradient for b2

    # --- HIDDEN LAYER ---
    dA1 = dZ2 @ W2.T                         # pass error back
    dZ1 = dA1 * relu_derivative(Z1)          # through activation
    dW1 = X.T @ dZ1                          # gradient for W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) # gradient for b1

    return dW1, db1, dW2, db2
```

**Why backwards?** Each layer's gradient depends on the one ahead of it. You can't compute `dW1` until you know `dA1`, and you can't know `dA1` until you've computed `dZ2`.

---

## Step 7 — Weight Update (Gradient Descent)

You have the gradients from backprop. Now apply them. Need `global` because you're modifying variables defined outside the function.

```python
def update_params(dW1, db1, dW2, db2, learning_rate):
    global W1, b1, W2, b2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
```

- `learning_rate` is a small float (0.01 to 0.1 typically)
- Too small → learns too slowly
- Too large → overshoots, loss bounces around or explodes

---

## Step 8 — Training Loop

Wraps everything. This is the engine.

```python
def train(X, y_true, epochs, learning_rate):
    for epoch in range(epochs):
        A1, A2, Z1, Z2 = forward(X)
        loss = mse_loss(y_true, A2)
        dW1, db1, dW2, db2 = backward(X, y_true, A1, A2, Z1, Z2)
        update_params(dW1, db1, dW2, db2, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

---

## Test It — XOR Problem

XOR is the classic first test. Not linearly separable — you actually need the hidden layer to solve it.

```python
X = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 1, 1]
])

y = np.array([[0], [1], [1], [0]])

if __name__ == "__main__":
    train(X, y, epochs=10000, learning_rate=0.1)
    A1, A2, Z1, Z2 = forward(X)
    print("predictions: ", A2)
```

Expected output after training:
```
[0, 0, 0, 0] → ~0.03   ✓
[0, 1, 0, 1] → ~0.97   ✓
[1, 0, 1, 0] → ~0.97   ✓
[1, 1, 1, 1] → ~0.03   ✓
```

---

## Things That Will Break You (And Why)

**Loss not dropping** → learning rate too small, try 0.1 instead of 0.01

**Loss exploding** → learning rate too large, or forgot `* 0.01` on weight init

**Shape errors** → print `.shape` on everything, the matrix dimensions have to line up: `(n, 4) @ (4, 8) → (n, 8)`

**Weights not updating** → forgot `global W1, b1, W2, b2` in update function

**Flat loss then sudden drop** → normal for XOR, hidden layer has to learn a representation first

---

## Key Rules To Memorize

- Always store `Z1` and `Z2` in forward pass — backprop needs them
- Backprop goes strictly reverse of forward pass — output layer first
- Number of weight matrices = number of connections between layers
- `A` is what goes forward, `Z` is what you save for the way back
- Weight update is separate from backprop — backprop calculates gradients, update applies them