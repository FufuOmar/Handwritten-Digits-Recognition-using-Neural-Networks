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

def main():
    X, y = load_data('optdigits_train.dat')
    print(X.shape)  # should be (1934, 1024)
    print(y.shape)  # should be (1934, 10)
    print(y[0])     # should look like one-hot vector

if __name__ == "__main__":
    main()