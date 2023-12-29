import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1/(1+np.exp(-1*z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    train_data = np.vstack([mat['train' + str(i)] for i in range(10)])
    train_label = np.hstack([np.full(len(mat['train' + str(i)]), i) for i in range(10)])

    test_data = np.vstack([mat['test' + str(i)] for i in range(10)])
    test_label = np.hstack([np.full(len(mat['test' + str(i)]), i) for i in range(10)])

    # Shuffle the data
    train_indices = np.random.permutation(len(train_data))
    test_indices = np.random.permutation(len(test_data))

    train_data, train_label = train_data[train_indices], train_label[train_indices]
    test_data, test_label = test_data[test_indices], test_label[test_indices]

    # Feature selection: Remove columns with the same value across all samples
    uninformative_columns = np.all(train_data == train_data[0, :], axis=0)
    selected_features = np.where(uninformative_columns == False)[0]

    train_data = train_data[:, selected_features]
    test_data = test_data[:, selected_features]

    # Normalize pixel values to the range [0, 1]
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # Split into training and validation sets
    validation_size = int(0.1 * len(train_data))
    validation_data, validation_label = train_data[:validation_size], train_label[:validation_size]
    train_data, train_label = train_data[validation_size:], train_label[validation_size:]

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    m = len(training_data)
    training_data = np.column_stack((training_data, np.ones(m)))

    # Forward pass
    hidden_output = sigmoid(training_data.dot(w1.T))
    hidden_output_bias = np.column_stack((hidden_output, np.ones(m)))
    final_output = sigmoid(hidden_output_bias.dot(w2.T))

    # Convert labels to one-hot encoding
    outputclass = np.eye(n_class)[training_label.astype(int)]

    # Compute negative log likelihood
    obj_val = (-1 / m) * np.sum(outputclass * np.log(final_output) + (1 - outputclass) * np.log(1 - final_output))

    # Regularization term
    reg_term = (lambdaval / (2 * m)) * (np.sum(np.square(w1[:, :-1])) + np.sum(np.square(w2[:, :-1])))
    obj_val += reg_term

    # Backpropagation
    delta2 = final_output - outputclass
    gradient_w2 = (1 / m) * delta2.T.dot(hidden_output_bias)
    gradient_w2[:, :-1] += (lambdaval / m) * w2[:, :-1]

    delta1 = delta2.dot(w2[:, :-1]) * (hidden_output_bias[:, :-1] * (1 - hidden_output_bias[:, :-1]))
    gradient_w1 = (1 / m) * delta1.T.dot(training_data)
    gradient_w1[:, :-1] += (lambdaval / m) * w1[:, :-1]

    # Flatten gradients
    obj_grad = np.concatenate((gradient_w1.flatten(), gradient_w2.flatten()), 0)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    data = np.concatenate((data, np.ones(shape=(len(data), 1))), axis=1)
    hidden_output = sigmoid(data.dot(w1.T))
    hidden_output = np.concatenate((hidden_output, np.ones(shape=(len(hidden_output), 1))), axis=1)
    final_output = sigmoid(hidden_output.dot(w2.T))

    labels = np.argmax(final_output, axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""
now = time.time()

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularzation hyper-parameter
lambdaval = 60
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
print(" time to train", time.time() - now, "seconds")

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')