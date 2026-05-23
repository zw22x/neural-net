DEFINITION

A neural net is just repeated matrix math with a squeeze function (activation) applied after each step. The whole point is: do forward pass, measure how wrong you are, go backwards and figure out which weights caused the error, nudge them. Repeat until loss is low.

** INPUT LAYER **
x, y, weights, biases
input_size = 4 # number of features
hidden_size = 8 # neurons in hidden layer
outpur_size = 1 # neurons in output layer

weights: random small values (np.random.randn * 0.01)
biases: zeros
shape of W1 = (input_size, hidden_size)
shape of W2 = (hidden_size, output_size)

code snippet
# input_size, hidden_size, output_size = 4, 8, 1
# W1 = np.random.randn(input_size, hidden_size) * 0.01
# b1 = np.zeros((1, hidden_size))
# W2 = np.random.randn(hidden_size, output_size) * 0.01
# b2 = np.zeros((1, output_size))

** ACTIVATION FUNCTIONS **
ReLU, sigmoid and derivative of each
ReLU: max(0,x)    derivative: 1 if x > 0 else 0
Sigmoid: 1/(1+e^-x)   derivative: sigma(x) * (1 - sigma(x)) 

** HIDDEN LAYER ** 
send a forward pass of inputs thru the activation functions and interconnected weights
Z = X @ W + b      # .matmul() / @ preferred over .dot()
A = activation(Z)  # ReLU for hidden layers
send A to next layer

** LOSS FUNCTION **
MSE: mean((y_pred - y_true)^2)
derivative: 2 * (y_pred - y_true) / n

** OUTPUT LAYER **
receive output of predictions and loss and use loss to adjust the weights
Z = A_hidden @ W2 + b2
y_pred = activation(Z) # sigmoid for binary, none for regression compute loss


** BACKPROPAGATION **
chain rule: dL/dW = dL/dA * dA/dZ * dZ/dW
go layer by layer backwards
compute dZ, dW, db for each layer

** WEIGHT UPDATE **
W = W - lr * dL/dW
b = b - lr * dL/db
lr = learning rate (e.g. 0.01)

** TRAINING LOOP **
for epoch in range(n):
    forward pass -> loss -> backward pass -> update weights
