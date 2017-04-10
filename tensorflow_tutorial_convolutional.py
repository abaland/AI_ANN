from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

print('Module loaded.')
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('Downloaded datasets.')


def make_weights(shape):

    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    ###############
    return weights
    ###############


def make_biases(shape):

    biases = tf.Variable(tf.constant(0.1, shape=shape))
    ##############
    return biases
    ##############


def convol_2d(x, weight):

    ################################################################
    return tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')
    ################################################################


def max_pool_2x2(x):

    pool_output = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    ###################
    return pool_output
    ###################

###############
# Model set up
###############

###################
# First layer
# Each input is a 28x28 pixel image, so flatten that into a list
x_input = tf.placeholder(tf.float32, [None, 784])

# Reshape x to a 4d tensor, to be used by the conv2d functions
x_4d = tf.reshape(x_input, [-1, 28, 28, 1])

# Each patch of 5x5 in the image will receive 1 channel (grey scale) and produce 32 outputs.
W_conv1 = make_weights([5, 5, 1, 32])
b_conv1 = make_biases([32])

# Applies the convulation, then applies the 2x2 pooling on the output (results in 14x14 "images")
h_conv1 = tf.nn.relu(convol_2d(x_4d, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

###################
# Second layer
# Each patch of 5x5 in the image will receive 32 channels and produce 64 outputs.
W_conv2 = make_weights([5, 5, 32, 64])
b_conv2 = make_biases([64])

# Applies the convulation, then applies the 2x2 pooling on the output (results in 7x7 "images")
h_conv2 = tf.nn.relu(convol_2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

###################
# Third layer (densely connector layer). Each of the 7x7 "aggregated pixels" maps to 64 outputs.
# Use that to obtain 1 "straight" layer of 1024 neurons.
W_fc1 = make_weights([7 * 7 * 64, 1024])
b_fc1 = make_biases([1024])

# reshapes the 4d tensor to a batch of vectors, to be usable with the weights
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# Applies weight to get all 1024 neurons for layer.
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

###################
# Apply dropout here to cancel some of the weights (to reduce overfitting)
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropped = tf.nn.dropout(h_fc1, keep_prob)

#################
# Final layer, convert the 1024 to a 10-way softmax.

# Creates weights for each input-pixel-node to output-digit-node
W_fc2 = tf.Variable(tf.zeros([1024, 10]))

# Creates biases for each output-digit-node
b_fc2 = tf.Variable(tf.zeros(10))

h_fc2 = tf.matmul(h_fc1_dropped, W_fc2) + b_fc2

#################
# Computes resultcross_entropy = tf.reduce_mean(
# True values for all the images (binary for each digit)
truth_value = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=truth_value, logits=h_fc2))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(h_fc2, 1), tf.argmax(truth_value, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print('Model set up.')

# Start tensorflow session and initialize global parameters
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

# Applies 1000 steps (1 step = 1 batches)
for index in range(20000):

    batch_input, batch_output = mnist.train.next_batch(50)

    if index % 100 == 0:

        train_accuracy = accuracy.eval(feed_dict={x_input: batch_input, truth_value: batch_output, keep_prob: 1.0})
        print("step %d, training accuracy %.3f" % (index, train_accuracy))

    # Computes resulting model on test set
    train_step.run(feed_dict={x_input: batch_input, truth_value: batch_output, keep_prob: 0.5})

print('Model finished running.')
accuracy_result = accuracy.eval(feed_dict={x_input: mnist.test.images, truth_value: mnist.test.labels, keep_prob: 1.0})
print("test accuracy %.3f" % (accuracy_result, ))
print('Done.')
