# Neural Net Visualizer

A neural network built from scratch using only NumPy — no PyTorch, no TensorFlow. Implements forward pass, backpropagation, and gradient descent by hand.

![Neural Net Visualizer](neural-net-demo.gif)

## What it does

- Trains a 2→8→1 network on 9 different datasets (XOR, moons, circle, AND/OR gates, etc.)
- Visualizes the decision boundary as a color map updating after training
- Shows weight strengths as a network flow diagram (green = positive, red = negative)
- Plots the loss curve and final predictions

## Architecture

- **Input layer:** 2 neurons (x, y coordinates)
- **Hidden layer:** 8 neurons with ReLU activation
- **Output layer:** 1 neuron with Sigmoid activation
- **Loss:** Mean Squared Error
- **Optimizer:** Vanilla gradient descent

## How to run

```bash
pip install numpy scikit-learn
python app.py
```

Open `http://127.0.0.1:8000` in your browser.