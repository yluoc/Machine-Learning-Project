'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1/(1+np.exp(-z))
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    m = len(training_data)
    training_data = np.column_stack((training_data, np.ones(m)))

    # Forward pass
    hidden_output = sigmoid(training_data.dot(w1.T))
    hidden_output = np.column_stack((hidden_output, np.ones(m)))
    final_output = sigmoid(hidden_output.dot(w2.T))

    # Convert labels to one-hot encoding
    outputclass = np.eye(n_class)[training_label.astype(int)]

    # Compute negative log likelihood
    obj_val = (-1 / m) * np.sum(outputclass * np.log(final_output) + (1 - outputclass) * np.log(1 - final_output))

    # Regularization term
    reg_term = (lambdaval / (2 * m)) * (np.sum(np.square(w1[:, :-1])) + np.sum(np.square(w2[:, :-1])))
    obj_val += reg_term

    # Backpropagation
    delta2 = final_output - outputclass
    gradient_w2 = (1 / m) * delta2.T.dot(hidden_output)
    gradient_w2[:, :-1] += (lambdaval / m) * w2[:, :-1]

    delta1 = delta2.dot(w2[:, :-1]) * (hidden_output[:, :-1] * (1 - hidden_output[:, :-1]))
    gradient_w1 = (1 / m) * delta1.T.dot(training_data)
    gradient_w1[:, :-1] += (lambdaval / m) * w1[:, :-1]

    # Flatten gradients
    obj_grad = np.concatenate((gradient_w1.flatten(), gradient_w2.flatten()))

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    return (obj_val, obj_grad)
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    # Your code here
    data = np.concatenate((data, np.ones(shape=(len(data), 1))), axis=1)
    hidden_output = sigmoid(data.dot(w1.T))
    hidden_output = np.concatenate((hidden_output, np.ones(shape=(len(hidden_output), 1))), axis=1)
    final_output = sigmoid(hidden_output.dot(w2.T))

    labels = np.argmax(final_output, axis=1)

    return labels
# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')