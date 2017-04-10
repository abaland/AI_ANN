from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

print('Module loaded.')
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('Downloaded datasets.')


###############
# Model set up
###############
# Each input is a 28x28 pixel image, so flatten that into a list
x_input = tf.placeholder(tf.float32, [None, 784])

# Creates weights for each input-pixel-node to output-digit-node
weights = tf.Variable(tf.zeros([784, 10]))

# Creates biases for each output-digit-node
bias = tf.Variable(tf.zeros(10))

# Creates prediction as Y = softmax(W*x+b)
prediction = tf.nn.softmax(tf.matmul(x_input, weights) + bias)

# True values for all the images (binary for each digit)
truth_value = tf.placeholder(tf.float32, [None, 10])

# Sets cross-entropy as cost : 1/n * (- sum y' * log(y))
cross_entroy = tf.reduce_mean(-tf.reduce_sum(truth_value * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)

print('Model set up.')

# Start tensorflow session and initialize global parameters
session = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Applies 1000 steps (1 step = 1 batches)
for index in range(1000):

    batch_input, batch_output = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x_input: batch_input, truth_value: batch_output})

print('Model finished running.')

# Computes resulting model on test set
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(truth_value, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(session.run(accuracy, feed_dict={x_input: mnist.test.images, truth_value: mnist.test.labels}))
print('Done.')
