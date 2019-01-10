"""
Use case: Deep learning for high content imaging
Description: Defining TF ops
"""


import tensorflow as tf

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer."""
    
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and convolve
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)
def predict(logits):
    return tf.nn.softmax(logits)
    
def acc(preds, true_labels, batch_size):
    """ Given prdictions and true labels, calculate accuracy """
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(true_labels, 1))
    score = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    return tf.divide(score,batch_size)

def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)


def deconv(x,  filter_height, filter_width, output_shape, name,
           stride_y, stride_x, stddev=0.02):
    with tf.variable_scope(name):
        in_C = x.get_shape().as_list()[-1]
        
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('weights', [filter_height, filter_width, output_shape[-1], in_C],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape,
                strides=[1, stide_y, stride_x, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.truncated_normal_initializer(stddev=stddev))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    return deconv
