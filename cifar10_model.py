import tensorflow as tf
import math as m
import cifar10_datafeed
from os.path import isfile

batch_size = 128  # SNumber of samples in mini-batch
n_classes = cifar10_datafeed.n_classes  # Number of class in dataset
image_size = cifar10_datafeed.image_size  # Size of image sent to model for input
n_channels = cifar10_datafeed.n_channels  # Number of input chanells per pixel (RGB)
n_inputs = image_size * image_size * n_channels  # Inputs per image (pixels * channel per pixel)
size_pooled = int(image_size / 4)  # Total number of inputs per image after max pooling

total_train_examples = 5000  # Total number of examples in training set
total_test_examples = cifar10_datafeed.total_test_examples  # Total number of examples in test set

variable_save_file = './test/saved_weights.cpkt'  # Location of file where weights will be stored


####################################################################################################
# variable_summaries
####################################################################################################
def variable_summaries(summary_name, tf_variable):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).

    INPUT:
        summary_name (str) Name of summary to use in tensorboard
        tf_variable (tf.Variable) Tensor object from Tensorflow to analyze
    """

    # Assigns the summaries for the variables
    mean = tf.reduce_mean(tf_variable)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(tf_variable - mean)))
    tf.summary.scalar(summary_name + '_mean', mean)
    tf.summary.scalar(summary_name + '_stddev', stddev)
    tf.summary.scalar(summary_name + '_max', tf.reduce_max(tf_variable))
    tf.summary.scalar(summary_name + '_min', tf.reduce_min(tf_variable))
    tf.summary.histogram(summary_name + '_histogram', tf_variable)

    #######
    return
    #######

#####################
# variable_summaries
#####################


####################################################################################################
# load_variable_value
####################################################################################################
def load_variable_value(saver_instance, session_instance, reset=False):
    """
    Initializes or loads init value for all variables in the system.

    INPUT:
        saver (tf.train.Saver) instance to save/restore variable values to disk
        session (tf.Session) session that runs the model
    """

    # Gets relevant dataset based (full training dataset or full test set)
    if not reset and isfile(variable_save_file + '.meta'):

        saver_instance.restore(session_instance, variable_save_file)
        print('Model restored')

    else:

        session_instance.run(tf.global_variables_initializer())
        print('Variable initialized')

    #######
    return
    #######

##########################
# END load_variable_value
##########################


####################################################################################################
# make_weights
####################################################################################################
def make_weights(variable_shape, tensor_inputs, weight_decay_lambda=0.0):
    """
    Initializes values for variable weight tensor. Initial values is taken as N(0, 0.1) truncated to
     +- 2 st. dev

    INPUT:
        variable_shape (int[4]) shape of tensor to initialize
        tensor_inputs (int) number of input neurons for the weight tensor
        weight_decay_lambda (float) weight factor to apply to weight tensot norm.

    OUTPUT:
        tf.Variable(int[4]) initialized tensor
    """

    # Sets variable value initializer (truncated normal distribution, with 1/sqrt(fan_in) as std dev
    weights_initializer = tf.truncated_normal_initializer(stddev=1./m.sqrt(tensor_inputs),
                                                          dtype=tf.float32)

    # Initializes weights variables
    weight_variable = tf.get_variable('weight', variable_shape, initializer=weights_initializer,
                                      dtype=tf.float32)

    # Adds the weight decay factor if necessary
    if weight_decay_lambda > 0.0:

        weight_decay = weight_decay_lambda * tf.nn.l2_loss(weight_variable)
        tf.add_to_collection('weight_decay', weight_decay)

    #######################
    return weight_variable
    #######################

###################
# END make_weights
###################


####################################################################################################
# make_biases
####################################################################################################
def make_biases(variable_shape, init_value=0.1):
    """
    Sets initial values for a tensor variable bias. Initial values is constant + according to input.

    INPUT:
        shape (int[1]) shape of the tensor to initialize
        init_value (float) initial value for all biases

    OUTPUT:
        tf.Variable(int[1]) initialized tensor
    """

    # Sets initializer for biases tensor
    bias_initializer = tf.constant_initializer(init_value)

    # Makes biases variable
    bias_variable = tf.get_variable('bias', variable_shape, initializer=bias_initializer,
                                    dtype=tf.float32)

    #####################
    return bias_variable
    #####################

##################
# END make_biases
##################


####################################################################################################
# conv2d
####################################################################################################
def conv2d(input_vector, weights):
    """
    Applies convolution to input vector using a (1, 1) stride and provided weights (patch size and
    channel_sizes)

    INPUT:
        input_vector (int[4]) tensor of shape [batch, height, width, tot_in_channels]
        weights (int[4]) filter of shape [height, width, tot_in_channels, tot_out_channels]
    """

    # Stride parameter must have stride[0] = stride[3] = 1. The 2nd-3rd parameter means the shift
    # between each patch.
    stride = [1, 1, 1, 1]

    # Sets padding to have input_size=output_size (though input_channel_size != output_channel size)
    padding = 'SAME'

    # Applies convolution
    convolved = tf.nn.conv2d(input_vector, weights, strides=stride, padding=padding)

    #################
    return convolved
    #################


#############
# END conv2d
#############


####################################################################################################
# max_pool_2x2
####################################################################################################
def max_pool_2x2(input_vector, sliding_window_size=2, stride_size=2):
    """
    Applies max pooling to input vector (convolved input) using a 2x2 pool.

    INPUT:
        input_vector (int[4]) tensor of shape [batch, height, width, tot_in_channels]
    """

    # Size of sliding window (ksize) + stride of sliding window (stride). We move by patches of 2x2
    ksize = [1, sliding_window_size, sliding_window_size, 1]
    stride = [1, stride_size, stride_size, 1]

    padding = 'SAME'

    # Applies max pooling
    pooled = tf.nn.max_pool(input_vector, ksize=ksize, strides=stride, padding=padding)

    ##############
    return pooled
    ##############

###################
# END max_pool_2x2
###################


####################################################################################################
# make_model
####################################################################################################
def make_model(data_batch, add_summary=False, reuse=False):
    """
    Creates the pipe from input batch to prediction

    INPUT:
        data_batch (tf.Tensor) tensor of input data with format [batch_size, image_width,
            image_height, n_channels]
        add_summary (bool) adds summary information to tensorboard
        reuse (bool) reuse previously created variable or creates new variable (cannot overwrite)

    RETURN
        (tf.Tensor) tensor of prediction with format [batch_size, n_classes]
    """

    del add_summary

    patch_size1 = 5
    out_per_patch1 = 64
    patch_size2 = 5
    out_per_patch2 = 64
    out_dense1 = 384
    out_dense2 = 192

    #
    # First convolutional layer, each input patch of 5x5 pixels with 1 channel (grey-scale) produce
    # 64 features
    #

    with tf.variable_scope('conv1', reuse=reuse):

        weights_conv1 = make_weights([patch_size1, patch_size1, n_channels, out_per_patch1],
                                     patch_size1 * patch_size1 * n_channels, 0.0)
        biases_conv1 = make_biases([out_per_patch1], init_value=0.0)

    # Computes first convolution and applies rectified linear unit on its output y = (max(0, x))
    h_conv1 = tf.nn.relu(conv2d(data_batch, weights_conv1) + biases_conv1)

    # Apply max pool. Pool is 2x2 => output dims halved => [batch_size, 14, 14, feature_per_patch1]
    h_pool1 = max_pool_2x2(h_conv1, sliding_window_size=3)

    # Normalizes the feature_per_patch1-length vector.
    h_pool1_normalized = tf.nn.lrn(h_pool1, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    #
    # Second convolutional layer, each input patch of 5x5 with 32 channel produces 64 features
    #

    with tf.variable_scope('conv2', reuse=reuse):

        weights_conv2 = make_weights([patch_size2, patch_size2, out_per_patch1, out_per_patch2],
                                     patch_size2 * patch_size2 * out_per_patch1, 0.0)
        biases_conv2 = make_biases([out_per_patch2])

    # Computes first convolution and applies rectified linear unit on its output y = (max(0, x))
    h_conv2 = tf.nn.relu(conv2d(h_pool1_normalized, weights_conv2) + biases_conv2)

    # Then apply max pool. Since pool is 2x2, output dimensions is halved => 7x7.
    h_pool2 = max_pool_2x2(h_conv2, sliding_window_size=3)

    # Normalizes the feature_per_patch1-length vector.
    h_pool2_normalized = tf.nn.lrn(h_pool2, bias=1.0, alpha=0.001/9.0, beta=0.75)

    #
    # Third layer : densely connected with dropout. Each of the 64 features from the 7x7 "neurons"
    # maps to 384 neurons with 1 feature
    #

    with tf.variable_scope('dense1', reuse=reuse):

        weights_dense1 = make_weights([size_pooled * size_pooled * out_per_patch2, out_dense1],
                                      size_pooled * size_pooled * out_per_patch2, 4*1e-3)
        biases_dense1 = make_biases([out_dense1])

    # In order to multiply h_pool2 output with weight_dense, flatten its output into one vector
    h_pool2_flattened = tf.reshape(h_pool2_normalized,
                                   [-1, size_pooled * size_pooled * out_per_patch2])

    # Computes values for next layer and apply rectified linear unit to output.
    h_dense1 = tf.nn.relu(tf.matmul(h_pool2_flattened, weights_dense1) + biases_dense1)

    #
    # Fourth layer : densely connected with dropout. Each of 384 neurons will map to 192 neurons
    #

    with tf.variable_scope('dense2', reuse=reuse):

        weights_dense2 = make_weights([out_dense1, out_dense2], out_dense1, 4*1e-3)
        biases_dense2 = make_biases([out_dense2])

    # Computes values for next layer and apply rectified linear unit to output.
    h_dense = tf.nn.relu(tf.matmul(h_dense1, weights_dense2) + biases_dense2)

    # Applies dropout on output
    h_dense2_dropped = tf.nn.dropout(h_dense, 0.5)

    #
    # Fifth (final) layer : applies readout layer, to convert the 1024 neurons value to the 10
    # output classes possible.
    #

    with tf.variable_scope('readout', reuse=reuse):

        weights_readout = make_weights(([out_dense2, n_classes]), out_dense2, 0.0)
        biases_readout = make_biases([n_classes], 0.0)

    # Computes predicion for given class (not probability). Softmax has not been applied.
    model_prediction = tf.matmul(h_dense2_dropped, weights_readout) + biases_readout

    ########################
    return model_prediction
    ########################

#################
# END make_model
#################
