# coding: utf-8

'''
Library that implements a basic multilayer perceptron.
A multilayer perceptron (MLP) is a fully connected class of feedforward artificial neural network (ANN).
Was designed for teaching purposes, it can be used for neuroevolution optimization.
'''

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import typing
import warnings
import numpy as np


# suppress warnings
warnings.filterwarnings('ignore')


class NN:
    '''
    Neural Network (NN) helper class.

    It has been devised to be used in a neuroevolution training scheme.
    This means that the traditional backpropagation algorithm is not available.
    It offers two other methods namely ravel and update.
    Ravel reduces the NN to a 1D  vector.
    The update method receives a 1D vector and loads the values into a network.
    '''
    def __init__(self, nn_architecture:dict, seed:int=42) -> None:
        '''
        Constructor for a NN object.

        Requires the definition of the network as follows::
        
            NN_ARCHITECTURE = [
            {'input_dim': 4, 'output_dim': 2, 'activation': 'relu'},
            {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
            ]
        
        Args:
            nn_architecture (dict): neural network definition
            seed (int): seed for the random generator used in the network initialization
        '''
        self.nn_architecture = nn_architecture
        self.params_values = init_layers(self.nn_architecture, seed)

    def predict(self, X:np.ndarray) -> np.ndarray:
        '''
        Implements the forward propagation algorithm for a Neural Network.

        Args:
            X (np.ndarray): 1D vector that is the input of the network
        
        Returns:
            np.ndarray: the output of the network
        '''
        A_curr, _ = full_forward_propagation(X, self.params_values, self.nn_architecture)
        return A_curr
    
    def predict_activations(self, X:np.ndarray)->tuple:
        '''
        Implements the forward propagation algorithm for a Neural Network.
        It also returns a list with the activations of each node (used for visualization purposes).

        Args:
            X (np.ndarray): 1D vector that is the input of the network
        
        Returns:
            tuple: the output of the network, and the activations of each node
        '''
        A_curr, memory = full_forward_propagation(X, self.params_values, self.nn_architecture)
        activations = []
        
        for i in range(len(self.nn_architecture)):
            activations.append(compute_activations(memory[f'A{i}'], self.nn_architecture[i]['activation']))
        activations.append(compute_activations(A_curr, self.nn_architecture[-1]['activation']))

        return A_curr, activations
        
    def ravel(self) -> np.ndarray:
        '''
        Reduces the network into a 1D vector.

        Returns:
            np.ndarray: a 1D vector that represents the network
        '''
        array = []
        for idx, layer in enumerate(self.nn_architecture):
            # we number network layers from 1
            layer_idx = idx + 1
            # extraction of W for the current layer
            W_curr = self.params_values["W" + str(layer_idx)]
            # extraction of b for the current layer
            b_curr = self.params_values["b" + str(layer_idx)]
            array.extend(W_curr.ravel())
            array.extend(b_curr.ravel())
        return np.array(array)

    def update(self, params:np.ndarray) -> None:
        '''
        Loads a 1D vector into a network.

        Args:
            params (np.ndarray): 1D vector that contains the weights and bias of the network
        '''
        begin = 0
        for idx, layer in enumerate(self.nn_architecture):
            # we number network layers from 1
            layer_idx = idx + 1
            # extraction of W for the current layer
            W_curr = self.params_values["W" + str(layer_idx)]
            n_elems = W_curr.size
            head = params[begin:begin+n_elems]
            self.params_values["W" + str(layer_idx)] = head.reshape(W_curr.shape)
            begin += n_elems
            # extraction of b for the current layer
            b_curr = self.params_values["b" + str(layer_idx)]
            n_elems = b_curr.size
            head = params[begin:begin+n_elems]
            self.params_values["b" + str(layer_idx)] = head
            begin += n_elems
    
    def layers(self) -> list:
        '''
        Returns the number of nodes per layer.

        Returns:
            list: the number of nodes per layer
        '''
        rv = []

        # get the input
        rv.append(self.nn_architecture[0]['input_dim'])

        # get the hidden layers
        for i in range(1, len(self.nn_architecture)):
            rv.append(self.nn_architecture[1]['input_dim'])

        # get the output
        rv.append(self.nn_architecture[-1]['output_dim'])

        return rv
        
    def __str__(self) -> str:
        '''
        Generates a str with the structure of the network.

        Returns:
            str: a str with the structure of the network
        '''
        return str({'ARCHITECTURE':self.nn_architecture,'PARAMETERS':self.params_values})


def init_layers(nn_architecture:dict, seed:int = 42) -> dict:
    '''
    Given a network definition it creates and initializes a new network.

    Args:
        nn_architecture (dict): neural network definition
        seed (int): seed for the random generator used in the network initialization
    
    Returns:
        dict: with the network parameters
    '''
    # random seed initiation
    np.random.seed(seed)
    # number of layers in our neural network
    number_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        
        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size) * 0.1

        nin = layer_input_size
        nout = layer_output_size
        sd = math.sqrt(6.0 / (nin + nout))

        #params_values['W' + str(layer_idx)] = np.random.normal(loc=0, scale=sd, size=(layer_output_size, layer_input_size))
        #params_values['b' + str(layer_idx)] = np.random.normal(loc=0, scale=sd, size=(layer_output_size))
        
    return params_values


def sigmoid(Z:np.ndarray) -> np.ndarray:
    '''
    Computes the sigmoid function for a 1D vector.
    $$
        \\sigma(z) = \\frac{1} {1 + e^{-z}}
    $$

    Args:
        Z (np.ndarray): 1D input vector
    
    Returns:
        np.ndarray: 1D vector with the results of the sigmoid operation
    '''
    return 1/(1+np.exp(-Z))


def sigmoid_backward(dA:np.ndarray, Z:np.ndarray) -> np.ndarray:
    '''
    Computes the backpropagation step for the sigmoid function.

    Args:
        dA (np.ndarray): derivative of the previous layer
        Z (np.ndarray): 1D input vector
    
    Returns:
        np.ndarray: 1D vector with the results of the backpropagation step
    '''
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)


def relu(Z:np.ndarray) -> np.ndarray:
    '''
    Computes the relu function for a 1D vector.
    $$
        Relu(z) = max(0, z)
    $$

    Args:
        Z (np.ndarray): 1D input vector
    
    Returns:
        np.ndarray: 1D vector with the results of the relu operation
    '''
    return np.maximum(0,Z)


def relu_backward(dA:np.ndarray, Z:np.ndarray) -> np.ndarray:
    '''
    Computes the backpropagation step for the relu function.

    Args:
        dA (np.ndarray): derivative of the previous layer
        Z (np.ndarray): 1D input vector
    
    Returns:
        np.ndarray: 1D vector with the results of the backpropagation step
    '''
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ


def swish(Z:np.ndarray) -> np.ndarray:
    '''
    Computes the swish function for a 1D vector.

    Args:
        Z (np.ndarray): 1D input vector
    
    Returns:
        np.ndarray: 1D vector with the results of the swish operation
    '''
    return Z*sigmoid(Z)


def swish_backward(dA:np.ndarray, Z:np.ndarray) -> np.ndarray:
    '''
    Computes the backpropagation step for the swish function.

    Args:
        dA (np.ndarray): derivative of the previous layer
        Z (np.ndarray): 1D input vector
    
    Returns:
        np.ndarray: 1D vector with the results of the backpropagation step
    '''
    sig = sigmoid(Z)
    return dA * (sig * (1+Z*(1-sig)))


def single_layer_forward_propagation(A_prev:np.ndarray, W_curr:np.ndarray,
b_curr:np.ndarray, activation:str='relu') -> np.ndarray:
    '''
    Computes a single step of the forward propagation algorithm.

    Args:
        A_prev (np.ndarray): Result from the previous layer 
        W_curr (np.ndarray): The weights 
        b_curr (np.ndarray): The bias
        activation (str): the activation function

    Returns:
        np.ndarray: the forward pass vector
    
    Raises:
        Exception: on non-supported activation function
    '''
    # calculation of the input value for the activation function
    Z_curr = np.dot(W_curr, A_prev) 
    Z_curr += b_curr
    
    # selection of activation function
    if activation == 'relu':
        activation_func = relu
    elif activation == 'sigmoid':
        activation_func = sigmoid
    elif activation == 'swish':
        activation_func = swish
    else:
        raise Exception('Non-supported activation function')
        
    # return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr


def full_forward_propagation(X:np.ndarray, params_values:dict, nn_architecture:dict) -> tuple:
    '''
    Computes a single step of the forward propagation algorithm.

    Args:
        X (np.ndarray): 1D vector that is the input of the network
        params_values (dict): the network parameters (weights and bias)
        nn_architecture (dict): neural network definition
       
    Returns:
        tuples: network output and the memory dict(for debug)
    
    Raises:
        Exception: on non-supported activation function
    '''
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr
        
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer

        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        # saving calculated values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory


def get_cost_value(Y_hat:np.ndarray, Y:np.ndarray) -> np.ndarray:
    '''
    Binary crossentropy cost function
    
    $$
        J(W,b) = \\frac{1}{m} \\sum{m}{i=1} L(\\hat{y}^{(i)}, y^{(i)});
        L(\\hat{y},y) = -{(y\\log(\\hat{y}) + (1 - y)\\log(1 - \\hat{y}))}
    $$

    Args:
        Y_hat (np.ndarray): the predicted values
        Y (np.ndarray): the expected values (outcome)

    Returns:
        np.ndarray: binary crossentropy cost
    '''
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)


def convert_prob_into_class(probs:np.ndarray) -> np.ndarray:
    '''
    An auxiliary function that converts probability into class

    Args:
        probs (np.ndarray): the binary class probabilities

    Returns:
        np.ndarray: binary classes
    '''
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(Y_hat:np.ndarray, Y:np.ndarray) -> float:
    '''
    Computes the accuracy value

    Args:
        Y_hat (np.ndarray): the predicted values
        Y (np.ndarray): the expected values (outcome)

    Returns:
        float: accuracy
    '''
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def update(params_values:dict, grads_values:dict, nn_architecture:dict, learning_rate:float) -> dict:
    '''
    The goal of this method is to update network parameters using gradient optimisation.

    Args:
        params_values (dict): the network parameters (weights and bias)
        grad_values (dict): stores cost function derivatives calculated with respect to these parameters
        nn_architecture (dict): neural network definition
        learning_rate (float): the learning rate for the gradient descent optimization method
    
    Returns:
        dict: the network parameters

    '''
    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]
    return params_values


def single_layer_backward_propagation(dA_curr:np.ndarray, W_curr:np.ndarray, 
b_curr:np.ndarray, Z_curr:np.ndarray, A_prev:np.ndarray, activation:str='relu') -> tuple:
    '''
    The essence of this algorithm is the recursive use of a chain rule known 
    from differential calculus — calculate a derivative of functions created 
    by assembling other functions, whose derivatives we already know.

    Args:
        dA_curr (np.ndarray): the derivative of the current activation
        W_curr (np.ndarray): The weights 
        b_curr (np.ndarray): The bias
        Z_curr (np.ndarray): the previous values
        A_prev (np.ndarray): the previous activation values 
        activation (str): the activation function (default relu)
    
    Returns:
        tuple: the derivate for the next layer, and the derivatives of the weights and bias
    '''
    # number of examples
    m = A_prev.shape[1]
    
    # selection of activation function
    if activation == 'relu':
        backward_activation_func = relu_backward
    elif activation == 'sigmoid':
        backward_activation_func = sigmoid_backward
    elif activation == 'swish':
        backward_activation_func = swish_backward
    else:
        raise Exception('Non-supported activation function')
    
    # calculation of the activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    
    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat:np.ndarray, Y:np.ndarray, 
memory:dict, params_values:dict, nn_architecture:dict) -> dict:
    '''
    Backward propagation algorithm (for binary classification)

    We start by calculating a derivative of the cost function with 
    respect to the prediction vector — result of forward propagation. 
    Then iterate through the layers of the network starting from the 
    end and calculate the derivatives with respect to all parameters.
    Ultimately, function returns a python dictionary containing the gradient we are looking for.

    Args:
        Y_hat (np.ndarray): the predicted values
        Y (np.ndarray): the expected values (outcome)
        memory (dict): intermediate values (debug)
        params_values (dict): the network parameters (weights and bias)
        nn_architecture (dict): neural network definition

    Returns:
        dict: stores cost function derivatives calculated with respect to these parameters
    '''
    grads_values = {}
    
    # number of examples
    m = Y.shape[1]
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(Y_hat.shape)
    
    # initiation of gradient descent algorithm
    # TODO: for binary classification
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values


def train(X:np.ndarray, Y:np.ndarray, nn_architecture:dict, 
epochs:int=100, learning_rate:float=0.01, 
verbose:bool=False, callback:typing.Callable=None) -> dict:
    '''
    The function returns optimized weights obtained as a 
    result of the training and the history of the metrics 
    change during the training.

    Args:
        X (np.ndarray): 1D vector that is the input of the network
        Y (np.ndarray): the expected values (outcome)
        nn_architecture (dict): neural network definition
        epochs (int): the number of backwards propagation loops (default 100)
        learning_rate (float): the learning rate for the gradient descent optimization method (default 0.01)
        verbose (bool): control the output of debug information (default False)
        callback (typing.Callable): callback function that is called at each epoch (deafult None)
    
    Returns
        dict: the network parameters
    '''
    # initiation of neural net parameters
    params_values = init_layers(nn_architecture, 2)
    # initiation of lists storing the history 
    # of metrics calculated during the learning process 
    cost_history = []
    accuracy_history = []
    
    # performing calculations for subsequent iterations
    for i in range(epochs):
        # step forward
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        
        # calculating metrics and saving them in history
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        # updating model state
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
        if(i % 50 == 0):
            if(verbose):
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            if(callback is not None):
                callback(i, params_values)
    return params_values


def compute_activations(A:np.ndarray, activation:str) -> list:
    '''
    Check if the layer nodes are active or not (activation functions is high or low)

    Args:
        A (np.ndarray): the output of a layer
        activation (str): the activation function
    
    Returns:
        list: with 1 (high) for activa and 0 for inactive (low)
    '''
    if activation == 'relu':
        return [0 if e <= 0 else 1 for e in A]
    elif activation == 'sigmoid':
        return [0 if e < 0.5 else 1 for e in A]
    elif activation == 'swish':
        return [0 if e <= 0 else 1 for e in A]
    else:
        raise Exception('Non-supported activation function')


def network_size(nn_architecture:dict) -> int:
    '''
    Given a network topology it computnes the number of weigths and bias.

    Args:
        nn_architecture (dict): neural network definition
    
    Returns
        int: the number of weigths and bias 
    '''
    size = 0
    # iteration over network layers
    for layer in nn_architecture:
        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        size += layer_output_size*layer_input_size
        size += layer_output_size
    return size