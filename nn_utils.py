import numpy as np
import tensorflow as tf
import math

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name = 'X')
    Y = tf.placeholder(tf.float32, shape=(None, n_y), name = 'Y')
    
    return X, Y

def nn_model(X, n_y, model, plain_layers, regularizer = None):
    """
    Create a chosen neural network model

    Arguments:
    X -- shape of the images of the dataset, array of shape (n_H, n_W, n_C)
    n_y -- scalar, number of classes
    model -- flag model, text string ("LeNet", VGG_16", "plain" etc.)
    plain_layers -- list specifying size of each hidden layer for plain model
    regularizer -- l2 regularization of weights

    Return:
    Z -- output of forward propagation (output of the last LINEAR unit), of shape (n_Y, number of examples)
    """

    # Choose model
    if model == "LeNet":
        print("LeNet neural network created!")
        Z = LeNet(X, n_y, regularizer)
    elif model == "VGG_16":
        Z = VGG_16(X, n_y, regularizer)
        print("VGG_16 neural network created!")
    elif model == "plain":
        Z = plain(X, n_y, plain_layers, regularizer)
        print("Plain neural network created!")
    elif model == "NgNet":
        Z = NgNet(X, n_y, regularizer)
        print("NgNet neural network created!")
    else:
        print("No new neural network created! Quaries must refer to trained model.")
        Z = None

    # Predict operation and probabilities
    if Z != None:
        predict_op = tf.argmax(Z, 1, name = 'predict_op')
        probas = tf.nn.softmax(Z)
        probas = tf.multiply(probas,1, name = 'probas') # create this copy variable to be able to name it

    return Z

def VGG_16(X, n_y, regularizer):
    """
    Creates a VGG_16 mode

    Arguments:
    X -- shape of the images of the dataset, array of shape (n_H, n_W, n_C)
    n_y -- scalar, number of classes
    regularizer -- l2 regularization of weights

    Return:
    Z -- output of forward propagation (output of the last LINEAR unit), of shape (n_Y, number of examples)
    """

    # CONV64->RELU x2
    for i in range(2):
        W = tf.get_variable("W" + str(i), [3,3,X.shape[3],64], initializer = tf.contrib.layers.xavier_initializer(), regularizer=regularizer) # X.shape[3] is n_c_prev
        X = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'SAME', name = 'conv' + str(i))
        X = tf.nn.relu(X, name = 'relu' + str(i))

    # MAXPOOL
    X = tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID', name = 'max_pool' + str(0))

    # CONV128->RELU x2
    for i in range(2,4):
        W = tf.get_variable("W" + str(i), [3,3,X.shape[3],128], initializer = tf.contrib.layers.xavier_initializer(), regularizer=regularizer) # X.shape[3] is n_c_prev
        X = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'SAME', name = 'conv' + str(i))
        X = tf.nn.relu(X, name = 'relu' + str(i))

    # MAXPOOL
    X = tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID', name = 'max_pool' + str(1))

    # CONV256->RELU x3
    for i in range(4,7):
        W = tf.get_variable("W" + str(i), [3,3,X.shape[3],256], initializer = tf.contrib.layers.xavier_initializer(), regularizer=regularizer) # X.shape[3] is n_c_prev
        X = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'SAME', name = 'conv' + str(i))
        X = tf.nn.relu(X, name = 'relu' + str(i))

    # MAXPOOL
    X = tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID', name = 'max_pool' + str(2))

    # CONV512->RELU x3
    for i in range(7,10):
        W = tf.get_variable("W" + str(i), [3,3,X.shape[3],512], initializer = tf.contrib.layers.xavier_initializer(), regularizer=regularizer) # X.shape[3] is n_c_prev
        X = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'SAME', name = 'conv' + str(i))
        X = tf.nn.relu(X, name = 'relu' + str(i))

    # MAXPOOL
    X = tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID', name = 'max_pool' + str(3))

    # CONV512->RELU x3
    for i in range(10,13):
        W = tf.get_variable("W" + str(i), [3,3,X.shape[3],512], initializer = tf.contrib.layers.xavier_initializer(), regularizer=regularizer) # X.shape[3] is n_c_prev
        X = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'SAME', name = 'conv' + str(i))
        X = tf.nn.relu(X, name = 'relu' + str(i))

    # MAXPOOL
    X = tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID', name = 'max_pool' + str(4))
        
    # FLATTEN
    X = tf.contrib.layers.flatten(X)

    # FULLY CONNECTED x2
    for i in range(2):
        X = tf.contrib.layers.fully_connected(X, num_outputs = 4096, activation_fn=tf.nn.relu, weights_regularizer=regularizer)

    # FULLY CONNECTED without non-linear activation function (not not call softmax)
    Z = tf.contrib.layers.fully_connected(X, num_outputs = n_y, activation_fn=None, weights_regularizer=regularizer)

    return Z

def LeNet(X, n_y, regularizer):
    """
    Creates a LeNet

    Arguments:
    X -- shape of the images of the dataset, array of shape (n_H, n_W, n_C)
    n_y -- scalar, number of classes
    regularizer -- l2 regularization of weights

    Return:
    Z -- output of forward propagation (output of the last LINEAR unit), of shape (n_Y, number of examples)
    """

    # CONV6->RELU
    for i in range(1):
        W = tf.get_variable("W" + str(i), [5,5,X.shape[3],6], initializer = tf.contrib.layers.xavier_initializer(), regularizer=regularizer) # X.shape[3] is n_c_prev
        X = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'VALID', name = 'conv' + str(i))
        X = tf.nn.relu(X, name = 'relu' + str(i))

    # MAXPOOL
    X = tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID', name = 'max_pool' + str(0))

    # CONV16->RELU
    for i in range(1,2):
        W = tf.get_variable("W" + str(i), [5,5,X.shape[3],16], initializer = tf.contrib.layers.xavier_initializer(), regularizer=regularizer) # X.shape[3] is n_c_prev
        X = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'SAME', name = 'conv' + str(i))
        X = tf.nn.relu(X, name = 'relu' + str(i))

    # MAXPOOL
    X = tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID', name = 'max_pool' + str(1))
    
    # FLATTEN
    X = tf.contrib.layers.flatten(X)

    # FULLY CONNECTED
    for i in range(1):
        X = tf.contrib.layers.fully_connected(X, num_outputs = 120, activation_fn=tf.nn.relu, weights_regularizer=regularizer)

    # FULLY CONNECTED
    for i in range(1,2):
        X = tf.contrib.layers.fully_connected(X, num_outputs = 84, activation_fn=tf.nn.relu, weights_regularizer=regularizer)

    # FULLY CONNECTED without non-linear activation function (not not call softmax)
    Z = tf.contrib.layers.fully_connected(X, num_outputs = n_y, activation_fn=None, weights_regularizer=regularizer)

    return Z

def NgNet(X, n_y, regularizer):
    """
    Creates a LeNet

    Arguments:
    X -- shape of the images of the dataset, array of shape (n_H, n_W, n_C)
    n_y -- scalar, number of classes
    regularizer -- l2 regularization of weights

    Return:
    Z -- output of forward propagation (output of the last LINEAR unit), of shape (n_Y, number of examples)
    """

    # CONV8->RELU
    W = tf.get_variable('W1', [4,4,X.shape[3],8], initializer = tf.contrib.layers.xavier_initializer(), regularizer=regularizer) # X.shape[3] is n_c_prev
    X = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'SAME')
    X = tf.nn.relu(X)

    # MAXPOOL
    X = tf.nn.max_pool(X, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')

    # CONV16->RELU
    W = tf.get_variable('W2', [2,2,X.shape[3],16], initializer = tf.contrib.layers.xavier_initializer(), regularizer=regularizer) # X.shape[3] is n_c_prev
    X = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'SAME')
    X = tf.nn.relu(X)

    # MAXPOOL
    X = tf.nn.max_pool(X, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    
    # FLATTEN
    X = tf.contrib.layers.flatten(X)

    # FULLY CONNECTED without non-linear activation function (not not call softmax)
    Z = tf.contrib.layers.fully_connected(X, num_outputs = n_y, activation_fn=None, weights_regularizer=regularizer)

    return Z

def plain(X, n_y, plain_layers, regularizer):
    """
    Creates a plain neural network

    Arguments:
    X -- shape of the images of the dataset, array of shape (n_H, n_W, n_C)
    n_y -- scalar, number of classes
    plain_layers -- list specifying size of each hidden layer for plain model
    regularizer -- l2 regularization of weights

    Return:
    Z -- output of forward propagation (output of the last LINEAR unit), of shape (n_Y, number of examples)
    """

    # FLATTEN
    X = tf.contrib.layers.flatten(X)
    
    # FULLY CONNECTED x2
    for i in plain_layers:
        X = tf.contrib.layers.fully_connected(X, num_outputs = i, activation_fn=tf.nn.relu, weights_regularizer=regularizer)

    # FULLY CONNECTED without non-linear activation function (not not call softmax)
    Z = tf.contrib.layers.fully_connected(X, num_outputs = n_y, activation_fn=None, weights_regularizer=regularizer)

    return Z

def compute_cost(X, Y, regularizer = None):
    """
    Computes the cost
    
    Arguments:
    X -- output of forward propagation (output of the last LINEAR unit), of shape (n_Y, number of examples)
    Y -- "true" labels vector placeholder, same shape as X
    regularizer -- l2 regularization of weights
    
    Returns:
    cost - Tensor of the cost function
    """

    # Regularization term
    if regularizer != None:
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    else:
        reg_term = 0

    # Cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = X, labels = Y))

    cost += reg_term

    return cost

def one_hot_matrix(y):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    y -- vector 'array containing the labels of shape (number of examples)
    
    Returns: 
    one_hot -- one hot matrix
    """

    m = y.shape[0]          # number of examples
    c = int(np.max(y))      # highest label value is equal to number of classes

    # initialize the matrix
    one_hot = np.zeros((c + 1, m))      # include "zero class"

    for i in range(m):
        one_hot[int(y[i]), i] = 1
    
    return one_hot

def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, n_Hi, n_Wi, n_Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
