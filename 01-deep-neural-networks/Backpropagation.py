import numpy as np
from matplotlib import pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """ Initialize the neural network with input size, hidden size, and learning rate. """
        self.learning_rate = learning_rate

        # initialize weights and biases [input->hidden]
        self.W1 = np.random.randn(input_size, hidden_size)*0.1
        self.b1 = np.zeros((1, hidden_size))

        # initialize weights and biases [hidden->output]
        self.W2 = np.random.randn(hidden_size, output_size)*0.1
        self.b2 = np.zeros((1, output_size))

        #store values for backward pass [gradient]
        self.z1 = None  # Hidden layer pre-activation
        self.a1 = None  # Hidden layer activation
        self.z2 = None  # Output layer pre-activation
        self.a2 = None  # Output layer activation

    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(x):
        """ Derivative of the sigmoid function """
        return x*(1-x)
    

    def forward(self, X):
        # hidden layer computation
        self.z1 = np.dot(X, self.W1)+self.b1    # inpt [] * weights []
        self.a1 = self.sigmoid(self.z1)    

        # output layer computation
        self.z2 = np.dot(self.a1, self.W2)+self.b2  # activation * weights of hidden layer
        self.a2 = self.sigmoid(self.z2)

        return self.a2
    
    def compute_loss(self, y_pred, y_true):
        loss = 1/2 * np.mean((y_pred-y_true)**2)
        return loss

    def backward(self, y_pred, y_true):
        pass

    def train(self, X, y, epochs=1000, verbose=True):
        losses = []

        for epoch in range(epochs):
            # forward
            prediction = self.forward(X)

            # loss
            loss = self.compute_loss(prediction, y)
            losses.append(loss)

            # backward
            self.backward(prediction, y)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")

        return losses
    
    def predict(self, X):
        return self.forward(X)
    
