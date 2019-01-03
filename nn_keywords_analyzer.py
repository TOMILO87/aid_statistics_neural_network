from nn import * # contains my custom neural network object
from sklearn.utils import shuffle

np.random.seed(1)
tf.set_random_seed(1)

n_key_words = 800
n_test_12 = 6565 # must match number for policy objective e.g. 6855 for gender

# Train neural network #

# load presense of keywords in training dataset
X = np.loadtxt('X_train.txt', dtype=int)
Y = np.loadtxt('Y_train.txt', dtype=int)

#!#
#Y = np.zeros((129893+246489, 1))
#Y[:129893,:] = 1

# make input data fit for neural network
temp = np.zeros((X.shape[0], n_key_words, 1, 1)) # because nn only accepts input of shape (m, n_H, n_W, n_C)
temp[:,:,0,0] = X
X = temp
Y = np.transpose(one_hot_matrix(Y))

# shuffle arrayes in sync
X, Y = shuffle(X, Y)

input_shape = X.shape
output_shape = Y.shape

# initialize nn
nn = Neural_network(input_shape = input_shape, output_shape = output_shape, model = 'plain',
                    plain_layers = [128, 64, 32, 16, 8])

_ = nn.train(X_train = X, Y_train = Y, print_cost = True, minibatch_size = 1024,
             plot_cost=True, learning_rate = 0.001, num_epochs = 200, print_cost_freq = 100)

# Quary/Test neural network and results #

# load presense of keywords in training dataset
X = np.loadtxt('X_test.txt', dtype=int)
Y = np.loadtxt('Y_test.txt', dtype=int)

#!#
#Y = np.zeros((6855+12984, 1))
#Y[:6855,:] = 1

# make input data fit for neural network
temp = np.zeros((X.shape[0], n_key_words, 1, 1)) # because nn only accepts input of shape (m, n_H, n_W, n_C)
temp[:,:,0,0] = X
X = temp
Y = np.transpose(one_hot_matrix(Y))
# don't shuffle; want results for each category
print("X shape", X.shape)
print("Y shape", Y.shape)

print("Average correct predictions for contributions whose policy marker is 1 or 2:")
_ = nn.query(X_test = X[:n_test_12], Y_test = Y[:n_test_12], print_prob = True, print_pred = True, print_class = True)
print("Average correct predictions for contributions whose policy marker is 0:")
_ = nn.query(X_test = X[n_test_12:], Y_test = Y[n_test_12:], print_prob = True, print_pred = True, print_class = True)