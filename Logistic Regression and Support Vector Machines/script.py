import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt
import time
import timeit


def preprocess():
    """ 
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    bias = np.ones((n_data, 1))
    new_features = n_features + 1
    weight = initialWeights.reshape((new_features, 1))
    train_data_bias = np.hstack((bias, train_data))
    theta_n = sigmoid(np.dot(train_data_bias, weight))

    x1 = labeli * np.log(theta_n)
    x2 = (1.0 - labeli) * np.log(1.0 - theta_n)

    x = theta_n.shape[0]

    error = (-1.0 / x) * np.sum(x1 + x2)
    error_grad = (1.0 / x) * np.sum(((theta_n - labeli) * train_data_bias), axis=0)
    
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    bias = np.ones((data.shape[0], 1))
    data_bias = np.hstack((bias, data))

    x1 = np.dot(data_bias, W)
    x = sigmoid(x1)

    label = np.argmax(x, axis=1).reshape((data.shape[0], 1))
    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    X_bias = np.hstack((np.ones((n_data, 1)), train_data))
    weights = params.reshape((n_feature + 1, n_class))
    scores = np.dot(X_bias, weights)
    exp_scores = np.exp(scores)

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    error = -np.sum(labeli * np.log(probs)) / n_data

    error_grad = np.dot(X_bias.T, (probs - labeli)) / n_data
    error_grad = error_grad.flatten()

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    data_bias = np.hstack((np.ones((data.shape[0], 1)), data))
    scores = np.dot(data_bias, W)
    label[:, 0] = np.argmax(scores, axis=1)

    return label

#Script for Logistic Regression

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
initialWeights = initialWeights.flatten()
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

#Script for Support Vector Machine

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
# linear kernel
print('using linear kernel')
svl = svm.SVC(kernel = 'linear')
svl.fit(train_data, train_label.ravel())
print('\n Training data Accuracy: {:.2f}%'.format(100 * svl.score(train_data, train_label))) # 97.286
print('\n Validation data Accuracy: {:.2f}%'.format(100 * svl.score(validation_data, validation_label))) # 93.64
print('\n Testing data Accuracy: {:.2f}%'.format(100 * svl.score(test_data, test_label))) # 93.78
print('\n')

# radial basis with gamma = 1
print('using radial basis with gamma = 1.0')
svr = svm.SVC(kernel = 'rbf', gamma = 1.0)
svr.fit(train_data, train_label.ravel())
print('\n Training data Accuracy: {:.2f}%'.format(100 * svr.score(train_data, train_label)))
print('\n Validation data Accuracy: {:.2f}%'.format(100 * svr.score(validation_data, validation_label)))
print('\n Testing data Accuracy: {:.2f}%'.format(100 * svr.score(test_data, test_label)))
print('\n')

# radial basis with gamma default
print('using radial basis with gamma default')
svr0 = svm.SVC(kernel = 'rbf')
svr0.fit(train_data, train_label.ravel())
print('\n Training data Accuracy: {:.2f}%'.format(100 * svr0.score(train_data, train_label)))
print('\n Validation data Accuracy: {:.2f}%'.format(100 * svr0.score(validation_data, validation_label)))
print('\n Testing data Accuracy: {:.2f}%'.format(100 * svr0.score(test_data, test_label)))
print('\n')

# radial basis function with value of gamma setting to default and varying value of C
print('radial basis function with value of gamma setting to default and varying value of C')
c_value = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
train_data_accuracy, validation_data_accuracy, testing_data_accuracy = np.zeros(11), np.zeros(11), np.zeros(11)

for i in range(len(c_value)):
    svC = svm.SVC(C = c_value[i], kernel = 'rbf')
    svC.fit(train_data, train_label.ravel())

    print("\nwhen c is " + str(c_value[i]))

    train_data_accuracy[i] = 100 * svC.score(train_data, train_label)
    print('\n Training data Accuracy: {:.2f}% '.format(train_data_accuracy[i]))

    validation_data_accuracy[i] = 100 * svC.score(validation_data, validation_label)
    print('\n Validation data Accuracy: {:.2f}%'.format(validation_data_accuracy[i]))
  
    testing_data_accuracy[i] = 100 * svC.score(test_data, test_label)
    print('\n Testing data Accuracy: {:.2f}%'.format(testing_data_accuracy[i]))

"""
#Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}
initialWeights_b = initialWeights_b.flatten()

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
