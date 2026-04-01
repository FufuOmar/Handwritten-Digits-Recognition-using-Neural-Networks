import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    data = np.loadtxt(filepath)
    X = data[:,:-1] # All columns but last
    y = data[:,-1].astype(int) # Last column
    
    y = encode_labels(y)

    return X, y

def encode_labels(y):
    one_hot = np.zeros((len(y), 10))
    for row in range(len(y)):
        one_hot[row][y[row]] = 1
    return one_hot

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Pass input through each layer applying sigmoid activation, store all activations for backprop
def forward_prop(X, weights):
    activations = [X]
    for W in weights:
        bias = np.ones((activations[-1].shape[0], 1))  # column of 1s, one per sample
        A_with_bias = np.hstack([bias, activations[-1]])  # prepend it
        z = np.matmul(A_with_bias, W)
        a = sigmoid(z)
        activations.append(a)
    return activations

def back_prop(activations, weights, y):
    gradients = []
    m = y.shape[0]
    K = y.shape[1]  # number of output units (10)

    delta = (activations[-1] - y) * sigmoid_derivative(activations[-1]) / (m * K)
    for i in reversed(range(len(weights))):
        bias = np.ones((activations[i].shape[0], 1))
        A_prev_with_bias = np.hstack([bias, activations[i]])  # match what forward prop saw
        dW = np.matmul(A_prev_with_bias.T, delta)
        gradients.insert(0, dW)
        if i != 0:
            W_no_bias = weights[i][1:, :]  # strip the bias row
            delta = (delta @ W_no_bias.T) * sigmoid_derivative(activations[i])

    return gradients

# Create weights for each link between layers
def init_weights(layer_sizes):
    weights = []
    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i] + 1  # +1 for bias
        n_out = layer_sizes[i + 1]
        W = np.random.uniform(-1, 1, (n_in, n_out))
        weights.append(W)
    return weights

def proxy_error(y_pred, y_true):
    return np.sum((y_true - y_pred) ** 2) / (y_true.shape[0] * y_true.shape[1] * 2)

def misclassification_error(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) != np.argmax(y_true, axis=1))

def gradient_descent(X_train, y_train, X_test, y_test, layer_sizes, lr, R):
    weights = init_weights(layer_sizes)
    best_error = float('inf')
    best_weights = None
    history = {'train_proxy': [], 'test_proxy': [], 'train_misc': [], 'test_misc': []}
    
    for _ in range(R):
        # Forward prop on train
        activations = forward_prop(X_train, weights)
        
        # Compute and record train errors
        train_proxy = proxy_error(activations[-1], y_train)
        train_misc = misclassification_error(activations[-1], y_train)
        history['train_proxy'].append(train_proxy)
        history['train_misc'].append(train_misc)
        
        # Forward prop on test and record
        test_activations = forward_prop(X_test, weights)
        test_proxy = proxy_error(test_activations[-1], y_test)
        test_misc = misclassification_error(test_activations[-1], y_test)
        history['test_proxy'].append(test_proxy)
        history['test_misc'].append(test_misc)
        
        # Save best weights by train misclassification error
        if train_misc < best_error:
            best_error = train_misc
            best_weights = [W.copy() for W in weights]
        
        # Backprop and update weights
        gradients = back_prop(activations, weights, y_train)
        for i in range(len(weights)):
            weights[i] -= lr * gradients[i]
    
    return best_weights, history


def main():
    X, y = load_data('optdigits_train.dat')
    print(X.shape)  # should be (1934, 1024)
    print(y.shape)  # should be (1934, 10)
    print(y[0])     # should look like one-hot vector

if __name__ == "__main__":
    main()