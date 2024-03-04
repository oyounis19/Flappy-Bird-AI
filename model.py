import numpy as np
import pickle

class NeuralNetwork:
    '''
    This class represents the neural network used for the Flappy Bird game.
    Model architecture:
    - Input layer: 5 nodes (vertical position of bird, horizontal distance to next pipe, Height of the next gap, bird velocity, Gap's center position)
    - Hidden layer: 8 nodes (ReLU activation function)
    - Output layer: 1 node (Sigmoid activation function)
    '''
    def __init__(self, input_size=5, hidden_size=8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1

        self.weights = [np.random.randn(self.input_size, self.hidden_size), # Default: (6x8) random weights matrix from input to hidden layer
                        np.random.randn(self.hidden_size, self.output_size)] # Default: (8x1) random weights matrix from hidden to output layer

    def forward(self, X: np.ndarray, thresh: float) -> int:
        '''
        Forward pass of the neural network.
        param X: input data (5x1)
        return: output of the neural network (0 or 1)
        '''
        if X.shape[0] != self.input_size:
            raise ValueError(f'Input size must be {self.input_size}, got {X.shape[0]} instead.')
        
        self.z = np.dot(X, self.weights[0])
        self.z2 = self.relu(self.z)
        self.z3 = np.dot(self.z2, self.weights[1])
        o = self.sigmoid(self.z3)

        return 1 if o >= thresh else 0

    def sigmoid(self, s):
        return 1.0 / (1.0 + np.exp(-s))

    def relu(self, s):
        return np.maximum(0, s)
    
    def copy(self):
        '''
        Create a copy of the neural network.
        return: copy of the neural network
        '''
        nn = NeuralNetwork(self.input_size, self.hidden_size)
        nn.weights[0] = self.weights[0].copy()
        nn.weights[1] = self.weights[1].copy()
        return nn
    
    def save_weights(self, filename: str):
        '''
        Save the neural network to a file.
        param filename: name of the file
        '''
        np.savez(filename, *self.weights)
        return True

    def load_weights(self, filename: str):
        '''
        Load the neural network from a file.
        param filename: name of the file
        '''
        weights = np.load(filename)
        self.weights = [weights[f'arr_{i}'] for i in range(len(weights.files))]
        return True