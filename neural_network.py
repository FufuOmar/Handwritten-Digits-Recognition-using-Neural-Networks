class NeuralNetwork:
    def __init__(self, layers, lr=0.1):
        self.layers = layers
        self.lr = lr
        self.weights = []