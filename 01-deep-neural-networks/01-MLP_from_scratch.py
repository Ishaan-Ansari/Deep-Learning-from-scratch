import numpy as np

class Neuron:
    def __init__(self):
        self.activation = None
        self.weight = None
        self.bias = None

    def set_activation(self, value: float)->None:
        """Set the activation of the neuron"""
        self.activation = value

    def set_bias(self, value: float)->None:
        """Set the bias of the neuron"""
        self.bias = value

    def set_weight(self, value: float)->None:
        """Set the weight of the neuron"""
        self.weight = value

class Layer:
    def __init__(self, num_input:int, num_neurons: int, activation: str="relu"):
        """ Initialize a layer
        Args:
        - num_input (int): Number of inputs to the layer.
        - num_neurons (int): Number of neurons in the layer.
        - activation (str): Activation function ('ReLU' or 'Linear').
        """

        # Matrix of weights initialized with He initialization for ReLU
        if activation == "relu":
            self.weights = np.random.randn(num_input, num_neurons)*np.sqrt(2/num_input)
        else:
            self.weights = np.random.randn(num_input, num_neurons)*0.01

        self.biases = np.zeros(num_neurons) # bias vector
        self.activation = activation

        self.inputs = None # inputs to the layer
        self.z = None # pre-activation (weighted sum)
        self.activations = None # post-activation output

    def forward(self, inputs: np.ndarray)->np.ndarray:
        """ Perform a forward pass through the layer """
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        if self.activation == "relu":
            self.activations = self.relu(self.z)
        else:
            self.activations = self.z

        return self.activations

    def backward(self, dA: np.ndarray, learning_rate: float)->np.ndarray:
        """Backpropagate through this layer, update parameters, and return gradient wrt inputs"""

        m = self.inputs.shape[0]  

        if self.activation == "relu":
            dZ = dA * self.relu_derivative(self.z)
        else:
            dZ = dA

        # Gradients w.r.t weights and biases      
        dw = (1/m)*np.dot(self.inputs.T, dZ)
        db = (1/m)*np.sum(dZ, axis=0)

        # Gradient w.r.t inputs (to pass to previous layer)
        dInputs = np.dot(dZ, self.weights.T)

        # Update weights and biases
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db

        return dInputs

    @staticmethod
    def relu(x: np.ndarray)->np.ndarray:
        """
        ReLU 
        If x > 0 return the input value
        If the input value == 0, return 0
        """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)  # Vectorized derivative

def loss(y_pred: np.ndarray, y_true: np.ndarray)->float:
    """ Calculate how correct the network's predictions are. """
    return np.mean(0.5*(y_pred-y_true)**2)

def loss_derivative(y_pred: np.ndarray, y_true: np.ndarray)->float:
    """ Calculate the detivative of loss function. """    
    return y_pred-y_true    
    
class Network:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_layer = Layer(input_dim, hidden_dim, activation="relu")
        self.output_layer = Layer(hidden_dim, output_dim, activation="linear")

    def forward(self, x):
        hidden_activations = self.hidden_layer.forward(x)
        output_activations = self.output_layer.forward(hidden_activations)
        return output_activations
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, learning_rate: float):
        # Compute loss derivative
        dLoss = loss_derivative(y_pred, y_true)

        # Backpropagate through output layer (linear activation)
        dHidden = self.output_layer.backward(dLoss, learning_rate)

        # Backpropagate through hidden layer (ReLU activation)
        self.hidden_layer.backward(dHidden, learning_rate)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float):
        # training loop
        for epoch in range(epochs):
            # forward pass
            y_pred = self.forward(x)
            # compute loss
            loss_val = loss(y_pred, y)
            # backward pass (using cached values from forward pass)
            self.backward(y_pred, y, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss_val}")


if __name__ == "__main__":
    np.random.seed(42) # For reproducibility

    # create data: x as input and y as output
    x = np.random.rand(100, 3) # 100 examples, 3 features (3 inputs)

    y = np.random.rand(100, 1) # 100 target outputs

    # initialize the network (3 inputs, 4 neurons in hidden layer, 1 output)
    net = Network(input_dim=3, hidden_dim=4, output_dim=1)

    # train the network
    net.train(x, y, epochs=100, learning_rate=0.01)
 