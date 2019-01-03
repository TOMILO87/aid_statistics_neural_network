"""This module let you train and query a neural networks"""

import os
from nn_utils import *

class Neural_network(object):
    """
    Creates a neural network object:
    
    Arguments to initialize:
    input_shape -- shape of the images of the dataset, array of shape (m, n_H, n_W, n_C)
    output_shape -- number of classes, of shape (m, n_y), should be one-hot-matrix
    model -- flag model, text string ("LeNet", "VGG_16", "NgNet", "plain" etc.)
    plain_layers -- size of each hidden layer for plain model, a list e.g. [64, 32, 16, 8]
    beta -- L2 reqularization parameter
    
    Arguments to train:
    X_train -- training set, of shape (m, n_H, n_W, n_C)
    Y_train -- test set, of shape (m, n_y)
    X_dev -- development set set, of shape (m, n_H, n_W, n_C)
    Y_dev -- develpment set, of shape (m, n_y)
    learning_rate -- learning rate of the optimization
    exponential_decay -- True to enable exponential learning rate decay
    decay_steps -- how often learning_rate decay by decay rate, an integer
    decay_rate -- how much learning_rate decay
    staircase --  True to decay the learning rate at discrete intervals 
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost
    print_cost_freq -- frequency cost is printed with respect to number of epochs
    plot_cost -- True to plot the cost every 1 epoch
    plot_parameters -- True to plot names trained parameters and dimensions
    base_dir -- where to save the trained model

    Arguments to query:
    X_test -- test set, of shape (m, n_H, n_W, n_C)
    Y_test -- test set, of shape (m, n_y)
    X_unk -- unknown set, of shape (m, n_H, n_W, n_C), to predict class for
    trained_model -- flag trained model to query from, text string (e.g. "LeNet-1000")
    print_prob -- print probability class
    print_pred -- print predicted class
    print_class -- print true class if available

    Return for training:
    parameters -- a dictionary containing trained parameters

    Return for query:
    Y_pred -- predicted class given X_test, an array of shape (m, 1)
    """

    # initialize
    def __init__(self, input_shape, output_shape, model = None, plain_layers = [64, 32, 16], beta = 0):

        # Get shape
        self.n_H = input_shape[1]
        self.n_W = input_shape[2]
        self.n_C = input_shape[3]
        self.n_y = output_shape[1]

        # Define the input and output placeholder as a tensor with shapes input_shape and output_shape
        self.X, self.Y = create_placeholders(self.n_H, self.n_W, self.n_C, self.n_y)

        # Regularizer
        if beta!=0:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
        else:
            self.regularizer = None

        # Create chosen model
        self.model = model # to be able to refer to model during query
        self.Z = nn_model(self.X, self.n_y, self.model, plain_layers, self.regularizer) # used during training

    # train
    def train(self, X_train, Y_train, X_dev = None, Y_dev = None, learning_rate = 0.009,
              exponential_decay = False, decay_steps = 100, staircase = False,
              decay_rate = 0.96, num_epochs = 200, minibatch_size = 64, print_cost = False,
              print_cost_freq = 100, plot_cost = False, plot_parameters = False,
              base_dir = None):

        self.costs = [] # to keep track of the cost
        self.m = X_train.shape[0] # number of examples
        self.global_step = tf.Variable(0, trainable=False) # to use for the decay computation

        # Create saver object
        self.saver = tf.train.Saver(save_relative_paths=True)
        if base_dir == None:
            self.base_dir = os.path.abspath(os.path.dirname(__file__))
        else:
            self.base_dir = base_dir

        # Enable exponential decay to the learning rate
        if exponential_decay == True:
            self.starter_learning_rate = learning_rate
            self.learning_rate = tf.train.exponential_decay(learning_rate = self.starter_learning_rate,
                                                            global_step = self.global_step,
                                                            decay_steps = decay_steps, decay_rate = decay_rate,
                                                            staircase = staircase)
        else:
            self.learning_rate = learning_rate

        # Cost function: Add cost function to tensorflow graph
        self.cost = compute_cost(self.Z, self.Y, self.regularizer) #self.Z was defined during initialization

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss = self.cost)

        # Initialize all the variables globally
        self.init = tf.global_variables_initializer()

        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:

            # Run the initialization
            sess.run(self.init)

            # Do the training loop
            for epoch in range(num_epochs):

                minibatch_cost = 0 #removed . from 0.
                num_minibatches = max(1, int(self.m / minibatch_size)) # number of minibatches of size minibatch_size in the train set.
                #if m < minibatch_size, set num_minibatches equal to one to avoid division by zero below
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                    _ , temp_cost = sess.run([self.optimizer, self.cost], feed_dict={self.X: minibatch_X, self.Y: minibatch_Y})
                    minibatch_cost += temp_cost / num_minibatches
                
                # Print the cost
                if print_cost == True and epoch % print_cost_freq == 0:
                    print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                if print_cost == True and epoch % 1 == 0:
                    self.costs.append(minibatch_cost)

                # increment global step, which is used for exponential decay
                self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1)
                sess.run(self.increment_global_step_op)
        
            # Plot the cost
            if plot_cost == True:
                import matplotlib.pyplot as plt # import here because module unavailable raspberry
                plt.plot(np.squeeze(self.costs))
                plt.ylabel('cost')
                plt.xlabel('iterations (per tens)')
                plt.title("Learning rate =" + str(learning_rate))
                plt.show()

            # Calculate the correct predictions
            self.predict_op = tf.argmax(self.Z, 1)
            self.correct_prediction = tf.equal(self.predict_op, tf.argmax(self.Y, 1))

            # Calculate and print accuracy on the test set
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
            self.train_accuracy = self.accuracy.eval({self.X: X_train, self.Y: Y_train})
            print("Train Accuracy:", self.train_accuracy)

            # Get and plot parameters
            self.parameters = {}
            self.tvars = tf.trainable_variables()
            self.tvars_vals = sess.run(self.tvars)
            for var, val in zip(self.tvars, self.tvars_vals):
                if plot_parameters == True:
                    #print(tf.trainable_variables()) # Prints parameter dimensions -- repeats for each dimension
                    print(var.name, val)  # Prints the name of the parameter alongside its value.
                self.parameters[var.name] = val

            # Calculate and print accuracy on the development set if available
            try:
                self.dev_accuracy = self.accuracy.eval({self.X: X_dev, self.Y: Y_dev})
                print("Development Accuracy:", self.dev_accuracy)
            except ValueError:
                pass

            # Save the trained model
            self.saver.save(sess, self.base_dir + '/' + str(self.model), global_step = 1000)

        return self.parameters

    # Query the neural network
    def query(self, X_test, Y_test = None, trained_model = None,
              print_prob = False, print_pred = False, print_class = False):

        # Check if a model has been specified, default is objects's model
        if trained_model == None and self.model != None:
            trained_model = str(self.model) + "-1000"
        else:
            trained_model = str(trained_model)

        # Reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            # Restore variables
            self.saver = tf.train.import_meta_graph(trained_model + ".meta")
            self.saver.restore(sess,tf.train.latest_checkpoint('./')) # checkpoint is currently latest trained model regardless of model argument (!)
            graph = tf.get_default_graph()
            self.m = X_test.shape[0] # number of examples
            self.X = graph.get_tensor_by_name("X:0")
            self.Y = graph.get_tensor_by_name("Y:0")
            self.predict_op = graph.get_tensor_by_name("predict_op:0")
            self.probas = graph.get_tensor_by_name("probas:0")

            # Calculate and print accuracy on the test set if available
            if hasattr(Y_test, "__len__"): # true if Y_test is an array
                self.predict_op = sess.run(self.predict_op, {self.X: X_test, self.Y: Y_test})
                self.test_accuracy = np.mean(self.predict_op==np.argmax(Y_test, axis = 1)) # alternative method for accuracy since above unavailable
                print("Test Accuracy:", self.test_accuracy)

            # Calculate predicted class based on probabilities
            self.probas_test = sess.run(self.probas, {self.X: X_test})
            self.Y_pred = np.argmax(self.probas_test, axis = 1)

            # Print probabilities
            if print_prob == True:
                print("Proabilities:", self.probas_test)

            # Print predicted class
            if print_pred == True:
                print("Predicted class:", self.Y_pred)

            # Print true class if available
            if hasattr(Y_test, "__len__") and print_class == True:
                print("True class:", np.argmax(Y_test, axis = 1))

        return self.Y_pred
