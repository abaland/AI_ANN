import tensorflow as tf
import _pickle
import numpy as np
import math as m


n_classes = 10
n_input = 32*32*3
batch_size = 5000
batch_index = 0

train_filenames = ['Data/Cifar/data_batch_1',
                   'Data/Cifar/data_batch_2',
                   'Data/Cifar/data_batch_3',
                   'Data/Cifar/data_batch_4',
                   'Data/Cifar/data_batch_5']
test_filename = 'Data/Cifar/test_batch'

DEBUG = False  # If DEBUG is set to True, only Class 0 and 1 will be kept from train/test sets
INSPECT = False


def variable_summaries(name, var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """

    with tf.name_scope(name + '_summaries'):
        mean = tf.reduce_mean(var)

    tf.summary.scalar(name + '_mean', mean)

    with tf.name_scope(name + '_stddev'):

        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

    tf.summary.scalar(name + '_stddev', stddev)
    tf.summary.scalar(name + '_max', tf.reduce_max(var))
    tf.summary.scalar(name + '_min', tf.reduce_min(var))
    tf.summary.histogram(name + '_histogram', var)


########################################################################################################################
# read_dataset
########################################################################################################################
def read_dataset(dataset_filename):
    """
    Reads dataset file (Pickled) and returns the corresponding dataset.

    INPUT:
        dataset_filename (str) filename of the dataset file

    OUTPUT:

    """

    dataset_file = open(dataset_filename, 'rb')
    dataset_loaded = _pickle.load(dataset_file, encoding='latin1')
    dataset_file.close()

    ######################
    return dataset_loaded
    ######################

###################
# END read_dataset
###################


########################################################################################################################
# convert_loaded_dataset_cifar
########################################################################################################################
def convert_loaded_dataset_cifar(dataset_start):
    """
    Converts the read dataset to a usable format later on, version CIFAR. For Cifar, the class is in the 'label' key,
       and the RGB values are in 'data'. For each image in 'data', a 3072-length array first contains 1024 red values,
       then green1024 values, then 1024 blue values, ordered by row.

    INPUT:
        dataset_start (dict) dataset loaded (unpickled).

    OUTPUT:

    """

    if DEBUG:

        global n_classes
        n_classes = 2

    data = np.array(dataset_start['data'], dtype=np.float32)
    classes_raw = dataset_start['labels']
    classes_one_hot = list(map(lambda x: [1 if x==i else 0 for i in range(n_classes)], classes_raw))
    classes = np.array(classes_one_hot)

    if DEBUG:

        to_keep = np.array(classes_raw) < n_classes
        data = data[to_keep, ]
        classes = classes[to_keep]

    #####################
    return data, classes
    #####################

###################################
# END convert_loaded_dataset_cifar
###################################


########################################################################################################################
# normalize_dataset
########################################################################################################################
def normalize_dataset(train, test):
    """
    """

    # Compute (mean, stddev) of training dataset to normalize it and almost-normalize test dataset.
    #    We behave as if we did not possess the test set during training => that data cannot be used for mean/stddev
    train_mean_per_dimension = np.mean(train, axis=0)
    train_stddev_per_dimension = np.std(train, axis=0)

    # If stdev is 0, column is useless. x - mean(x) will already give a 0-column, so set std to 1.0 to avoid NaN.
    train_stddev_per_dimension[train_stddev_per_dimension == 0.] = 1.0

    # Normalizes datasets. This does not handle cases where a column is exclusively
    train = (train - train_mean_per_dimension) / train_stddev_per_dimension
    test = (test - train_mean_per_dimension) / train_stddev_per_dimension

    ###################
    return train, test
    ###################

########################
# END normalize_dataset
########################


########################################################################################################################
# get_next_batch
########################################################################################################################
def get_next_batch(dataset, labels):
    """
    """

    global batch_index

    if batch_index + batch_size >= dataset.shape[0]:

        batch_index = 0

    index_start = batch_index
    index_end = batch_index + batch_size
    batch_index = index_end

    #######################################################################
    return dataset[index_start:index_end,], labels[index_start:index_end,]
    #######################################################################

#####################
# END get_next_batch
#####################


########################################################################################################################
# initialize_variable_weights
########################################################################################################################
def initialize_variable_weights(shape, n_inputs):
    """
    Sets initial values for a tensor variable weights. Initial values is taken as N(0, 0.1) but values below/above 2 st.
    dev are dropped (truncated normal).

    INPUT:
        shape (int[4]) shape of the tensor to initialize

    OUTPUT:
        tf.Variable(int[4]) initialized tensor
    """

    # Sets initial value for weights
    init_weights = tf.truncated_normal(shape, stddev=2./m.sqrt(n_inputs))

    # Makes weights variable
    init_variable_weights = tf.Variable(init_weights)

    #############################
    return init_variable_weights
    #############################

##################################
# END initialize_variable_weights
##################################


########################################################################################################################
# initialize_variable_biases
########################################################################################################################
def initialize_variable_biases(shape, init_value=0.1):
    """
    Sets initial values for a tensor variable bias. Initial values is constant and according to input.

    INPUT:
        shape (int[1]) shape of the tensor to initialize
        init_value (float) initial value for all biases

    OUTPUT:
        tf.Variable(int[1]) initialized tensor
    """

    # Sets initialize value for biases
    init_biases = tf.constant(init_value, shape=shape)

    # Makes biases variable
    init_variable_biases = tf.Variable(init_biases)

    ############################
    return init_variable_biases
    ############################


#################################
# END initialize_variable_biases
#################################


########################################################################################################################
# conv2d
########################################################################################################################
def conv2d(input_vector, weights):
    """
    Applies convolution to input vector using a (1, 1) stride and provided weights (patch size and channel_sizes)

    INPUT:
        input_vector (int[4]) tensor of shape [batch, height, width, tot_in_channels]
        weights (int[4]) filter of shape [height, width, tot_in_channels, tot_out_channels]
    """

    # Stride parameter must have stride[0] = stride[3] = 1. The 2nd-3rd parameter means the shift between each patch.
    stride = [1, 1, 1, 1]

    # Sets padding to have input_size = output_size (though input_channel_size != output_channel size)
    padding = 'SAME'

    # Applies convolution
    convolved = tf.nn.conv2d(input_vector, weights, strides=stride, padding=padding)

    #################
    return convolved
    #################


#############
# END conv2d
#############


########################################################################################################################
# max_pool_2x2
########################################################################################################################
def max_pool_2x2(input_vector):
    """
    Applies max pooling to input vector (convolved input) using a 2x2 pool.

    INPUT:
        input_vector (int[4]) tensor of shape [batch, height, width, tot_in_channels]
    """

    # Size of sliding window (ksize) and strike of sliding window (stride). We move by patches of 2x2
    ksize = [1, 2, 2, 1]
    stride = [1, 2, 2, 1]

    padding = 'SAME'

    # Applies max pooling
    pooled = tf.nn.max_pool(input_vector, ksize=ksize, strides=stride, padding=padding)

    ##############
    return pooled
    ##############


###################
# END max_pool_2x2
###################


print('Module loaded')


cifar_train_datas = []
cifar_train_labels = []
for train_filename in train_filenames:

    dataset_train_loaded = read_dataset(train_filename)
    one_cifar_train_data, one_cifar_train_label = convert_loaded_dataset_cifar(dataset_train_loaded)

    cifar_train_datas.append(one_cifar_train_data)
    cifar_train_labels.append(one_cifar_train_label)

dataset_test_loaded = read_dataset(test_filename)
cifar_test_data, cifar_test_label = convert_loaded_dataset_cifar(dataset_test_loaded)
cifar_train_data = np.concatenate(cifar_train_datas)
cifar_train_label = np.concatenate(cifar_train_labels)
cifar_train_data, cifar_test_data = normalize_dataset(cifar_train_data, cifar_test_data)

print('Finished downloading')

x_input = tf.placeholder(tf.float32, [None, n_input])
true_class = tf.placeholder(tf.float32, [None, n_classes])

# Converted input images to be convolvable (sets it to 4d tensor)
x_input_4d = tf.reshape(x_input, [-1, 32, 32, 3])  # -1="adapt dimension", (28,28) is input_image, 1 is color_channel

#
#
# First convolutional layer, each input patch of 5x5 pixels with 1 channel (grey-scale) produce 32 features
#
#

weights_conv1 = initialize_variable_weights([5, 5, 3, 64], 25*3)
biases_conv1 = initialize_variable_biases([64])

# Computes first convolution and applies rectified linear unit on its output y = (max(0, x))
h_conv1 = tf.nn.relu(conv2d(x_input_4d, weights_conv1) + biases_conv1)

# Then apply max pool. Since pool is 2x2, output dimensions is halved => 14x14.
h_pool1 = max_pool_2x2(h_conv1)

#
#
# Second convolutional layer, each input patch of 5x5 with 32 channel produces 64 features
#
#

weights_conv2 = initialize_variable_weights([5, 5, 64, 64], 25*64)
biases_conv2 = initialize_variable_biases([64])

# Computes first convolution and applies rectified linear unit on its output y = (max(0, x))
h_conv2 = tf.nn.relu(conv2d(h_pool1, weights_conv2) + biases_conv2)

# Then apply max pool. Since pool is 2x2, output dimensions is halved => 7x7.
h_pool2 = max_pool_2x2(h_conv2)

#
#
# Third layer : densely connected with dropout. Each of the 64 features from the 7x7 "neurons" will map to 384 neurons
# with 1 feature
#
#

weights_dense1 = initialize_variable_weights([8*8*64, 384], 64*64)
biases_dense1 = initialize_variable_biases([384])

# Before we are able to nultiply h_pool2 output with weight_dense, we need to flatten its output into one vector
h_pool2_flattened = tf.reshape(h_pool2, [-1, 8*8*64])

# Computes values for next layer and apply rectified linear unit to output.
h_dense1 = tf.nn.relu(tf.matmul(h_pool2_flattened, weights_dense1) + biases_dense1)

# Applies dropout on output
keep_probability = tf.placeholder(tf.float32)
h_dense1_dropped = tf.nn.dropout(h_dense1, keep_probability)

#
#
# Fourth layer : densely connected with dropout. Each of 384 neurons will map to 192 neurons
#
#

weights_dense2 = initialize_variable_weights([384, 192], 384)
biases_dense2 = initialize_variable_biases([192])

# Computes values for next layer and apply rectified linear unit to output.
h_dense = tf.nn.relu(tf.matmul(h_dense1_dropped, weights_dense2) + biases_dense2)

# Applies dropout on output
h_dense2_dropped = tf.nn.dropout(h_dense, keep_probability)

#
#
# Fifth (fnal) layer : applies readout layer, to convert the 1024 neurons value to the 10 output classes possible.
#
#

weights_readout = initialize_variable_weights(([192, n_classes]), 192)
biases_readout = initialize_variable_biases([n_classes])

# Computes predicion for given class (not probability). Softmax has not been applied.
prediction = tf.matmul(h_dense2_dropped, weights_readout) + biases_readout

# Computes cross entropy and initialize optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=true_class))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Set up accuracy computation.
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(true_class, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('cross_entropy', cross_entropy)
if INSPECT:

    variable_summaries('weights_conv1', weights_conv1)
    variable_summaries('weights_conv2', weights_conv2)
    variable_summaries('weights_dense1', weights_dense1)
    variable_summaries('weights_dense2', weights_dense2)
    variable_summaries('weights_readout', weights_readout)

    variable_summaries('biases_conv1', biases_conv1)
    variable_summaries('biases_conv2', biases_conv2)
    variable_summaries('biases_dense1', biases_dense1)
    variable_summaries('biases_dense2', biases_dense2)
    variable_summaries('biases_readout', biases_readout)

    variable_summaries('prediction_conv1', tf.reshape(h_pool1, [-1, 16*16*32]))
    variable_summaries('prediction_conv2', tf.reshape(h_pool2, [-1, 8*8*64]))
    variable_summaries('prediction_dense1', h_dense1_dropped)
    variable_summaries('prediction_dense2', h_dense2_dropped)
    variable_summaries('prediction_readout', prediction)


print('Model set up')

session = tf.InteractiveSession()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('convNetSummary', session.graph)
tf.global_variables_initializer().run()

# Creates feed for test dataset
test_feed = {x_input: cifar_test_data, true_class: cifar_test_label, keep_probability: 1.0}

for index in range(10000):

    train_batch_data, train_batch_label = get_next_batch(cifar_train_data, cifar_train_label)
    train_feed = {x_input: train_batch_data, true_class: train_batch_label, keep_probability: 0.5}

    if index % 20 == 0:

        # Tests results on Test set
        print('%d, %g' % (index, session.run(accuracy, feed_dict=test_feed)))

    _, summary = session.run([train_step, merged], feed_dict=train_feed)

    train_writer.add_summary(summary, index)

print('Model finished running')
