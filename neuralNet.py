import numpy as np
from sklearn.datasets import make_moons

input_size, hidden_size, output_size = 2, 8, 1

# initialize weights and biases
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# forward pass
def forward(X):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    return A1, A2, Z1, Z2

# loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

# backprop
def backward(X, y_true, A1, A2, Z1, Z2):
    dA2 = mse_loss_derivative(y_true, A2)
    dZ2 = dA2 * sigmoid_derivative(Z2)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# adjust weights and biases 
def update_params(dW1, db1, dW2, db2, learning_rate):
    global W1, b1, W2, b2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# training loop
def train(X, y_true, epochs, learning_rate):
    for epoch in range(epochs):
        A1, A2, Z1, Z2 = forward(X)
        loss = mse_loss(y_true, A2)
        dW1, db1, dW2, db2 = backward(X, y_true, A1, A2, Z1, Z2)
        update_params(dW1, db1, dW2, db2, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# example usage
if __name__ == "__main__":
   
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    y = y.reshape(-1, 1) 

    train(X, y, epochs=10000, learning_rate=0.1)
    # A1, A2, Z1, Z2 = forward(X)
    # print("predictions: ", A2)
