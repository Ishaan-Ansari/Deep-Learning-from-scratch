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
    def __init__(self, num_input:int, num_neurons: int):
        """ Initialize a layer
        Args:
        - num_input (int): Number of inputs to the layer.
        - num_neurons (int): Number of neurons in the layer.
        """
        self.neurons = [Neuron() for _ in range(num_neurons)] # List of neuron objects
        self.weights = np.random.randn(num_input, num_neurons) # Matrix of weights
        self.biases = np.zeros(num_neurons) # bias vector

        self.inputs = None
        self.z = None # pre-activation (weighted sum)
        self.activations = None # post-activation output

    
    def set_activations(self, inputs: np.ndarray)->None:
        """ Caclulate the weighted sum for a neuron in the layer
        (this is NOT the neuron's activation value)."""
        self.z = np.dot(inputs, self.weights)+self.biases

    def apply_activations(self)->None:
        """ Apply the activation function to the weighted sum of the inputs
        This is value of the neuron's activation. """
        self.activations = self.relu(self.z)

    def get_activations(self)->np.ndarray:
        """ Return the activations of the layer """
        return self.activations
    
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

    def forward(self, inputs: np.ndarray)->np.ndarray:
        """ Perform a forward pass through the layer """
        self.inputs = inputs
        self.set_activations(inputs)
        self.apply_activations()
        return self.activations
    
    def backward(self, dA: np.ndarray, learning_rate: float)->np.ndarray:
        """Backpropagate through this layer, update parameters, and return gradient wrt inputs"""
        dZ = dA * (self.z > 0).astype(float) # ReLU derivative
        # Gradients w.r.t weights and biases
        m = self.inputs.shape[0]        
        dw = (1/m)*np.dot(self.inputs.T, dZ)
        db = (1/m)*np.sum(dZ, axis=0)

        # Gradient w.r.t inputs (to pass to previous layer)
        dInputs = np.dot(dZ, self.weights.T)

        # Update weights and biases
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db

        return dInputs

def loss(y_pred: np.ndarray, y_true: np.ndarray)->float:
    """ Calculate how correct the network's predictions are. """
    return np.mean(0.5*(y_pred-y_true)**2)

def loss_derivative(y_pred: np.ndarray, y_true: np.ndarray)->float:
    """ Calculate the detivative of loss function. """    
    return y_pred-y_true    
    
class Network:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_layer = Layer(input_dim, hidden_dim)
        self.output_layer = Layer(hidden_dim, output_dim)

    def forward(self, x):
        hidden_activations = self.hidden_layer.forward(x)
        output_activations = self.output_layer.forward(hidden_activations)
        return output_activations
    
    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float):
        # perform a forward pass
        y_pred = self.forward(x)

        # compute loss derivative at the output
        dLoss = y_pred - y

        # backpropagate  throgh output layer
        dHidden = self.output_layer.backward(dLoss, learning_rate)

        # backpropagate through hidden layer
        self.hidden_layer.backward(dHidden, learning_rate)

    def train(self, x, y, epochs, learning_rate):
        # training loop
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss_val = loss(y_pred, y)
            self.backward(x, y, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss_val}")


if __name__ == "__main__":
    # create data: x as input and y as output
    x = np.random.rand(100, 3) # 100 examples, 3 features (3 inputs)

    y = np.random.rand(100, 1) # 100 target outputs

    # initialize the network (3 inputs, 4 neurons in hidden layer, 1 output)
    net = Network(input_dim=3, hidden_dim=4, output_dim=1)

    # train the network
    net.train(x, y, epochs=100, learning_rate=0.01)
 
