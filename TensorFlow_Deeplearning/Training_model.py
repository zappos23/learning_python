import numpy as np
from image_processing import *

def minibatcher(X, y, batch_size, shuffle):
    assert X.shape[0] == y.shape[0]
    n_samples = X.shape[0]
    if shuffle:
        idx = np.random.permutation(n_samples)
    else:
        idx = list(range(n_samples))

    for k in range(int(np.ceil(n_samples/batch_size))):
        from_idx = k * batch_size
        to_idx = (k + 1) * batch_size
        yield X[idx[from_idx:to_idx], :, :, :], y[idx[from_idx:to_idx], :]

for mb in minibatcher(X_train, y_train, 10000, True):
    print(mb[0].shape, mb[1].shape)


#### Building neural network
import tensorflow as tf

# Linear function Z
def fc_no_activation_layer(in_tensors, n_units):
    w = tf.get_variable('fc_w', [in_tensors.get_shape()[1], n_units], tf.float32,
                        tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('fc_B', [n_units, ], tf.float32, tf.constant_initializer(0.0))

    return tf.matmul(in_tensors, w) + b

# Activation function A
def fc_layer(in_tensors, n_units):
    return tf.nn.leaky_relu(fc_no_activation_layer(in_tensors, n_units))

# defining the neural layer network connection with convolutional layer
def conv_layer(in_tensors, kernal_size, n_units):
    w = tf.get_variable('conv_W', [kernal_size, kernal_size, in_tensors.get_shape()[3], n_units],
                        tf.float32, tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('conv_B', [n_units, ], tf.float32, tf.constant_initializer(0.0))

    return tf.nn.leaky_relu(tf.nn.conv2d(in_tensors, w, [1, 1, 1, 1], 'SAME') + b)

# Size of the window and strides are both squares(quadrates)
def maxpool_layer(in_tensors, sampling):
    return tf.nn.max_pool(in_tensors, [1, sampling, sampling, 1], [1, sampling, sampling, 1], 'SAME')

# defining the drop using regularization. This is only used during training the network.
def dropout(in_tensors, keep_proba, is_training):
    return tf.cond(is_training, lambda: tf.nn.dropout(in_tensors, keep_proba), lambda: in_tensors)

# model composed of the following layers:
#   1) 2D convolution, 5x5 32 filters
#   2) 2D convolution, 5x5 64 filters
#   3) Flattenizer
#   4) Fully connected layer, 1024 units
#   5) dropout 40%
#   6) fully connected layer, no activation
#   7) softmax output

def model(in_tensors, is_training):
    # First layer: 5x5 2d-conv, 32 filters, 2x maxpool, 20% dropout
    with tf.variable_scope('l1'):
        l1 = maxpool_layer(conv_layer(in_tensors, 5, 32), 2)
        l1_out = dropout(l1, 0.8, is_training)

    # Second layer: 5x5 2d-conv, 64 filters, 2x maxpool, 20% dropout
    with tf.variable_scope('l2'):
        l2 = maxpool_layer(conv_layer(l1_out, 5, 64), 2)
        l2_out = dropout(l2, 0.8, is_training)

    # Flattenizer
    with tf.variable_scope('flatten'):
        l2_out_flat = tf.layers.flatten(l2_out)

    # Fully collected layer, 1024 neurons, 40% dropout
    with tf.variable_scope('l3'):
        l3 = fc_layer(l2_out_flat, 1024)
        l3_out = dropout(l3, 0.6, is_training)

    # output
    with tf.variable_scope('out'):
        out_tensors = fc_no_activation_layer(l3_out, N_CLASSES)

    return out_tensors

###### Training the model
