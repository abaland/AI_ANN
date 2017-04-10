
import tensorflow as tf
import cifar10_datafeed
import cifar10_model

n_classes = cifar10_datafeed.n_classes  # Number of class in dataset
image_size = cifar10_datafeed.image_size  # Size of image sent to model for input
n_channels = cifar10_datafeed.n_channels  # Number of input chanells per pixel (RGB)
n_inputs = image_size * image_size * n_channels  # Total number of inputs per image (total pixel * channel per pixel)
image_size_pooled = int(image_size / 4)  # Total number of inputs per image after max pooling

total_test_examples = cifar10_datafeed.total_test_examples  # Total number of examples in test set


########################################################################################################################
# make_test_model
########################################################################################################################
def make_test_model():
    """
    Creates test model to that will take full test set of images as input, and returns accuracy computation model

    INPUT:
        add_summary (bool) whether to add model information monitoring or not

    RETURN:
        (tf.Tensor) method to measure accuracy of batch
    """

    # Gets relevant dataset based (full training dataset or full test set)
    test_data, test_labels = cifar10_datafeed.get_input(True, total_test_examples, shuffle=False)

    # Computes predicion for given class (not probability). Softmax has not been applied.
    prediction = cifar10_model.make_model(test_data)

    # Compares prediction class to actual class and computes accuracy
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(test_labels, 1))
    test_accuracy_model = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Adds summary values for accuracy
    tf.summary.scalar('test_accuracy', test_accuracy_model)

    ###########################
    return test_accuracy_model
    ###########################

######################
# END make_test_model
######################


print('Module loaded')

# Creates all relevant models to use in the algorithm
test_set_accuracy = make_test_model()

print('Model set up')

# Initializes all necessary parameters to run model
session = tf.InteractiveSession()  # Session to run model
train_writer = tf.summary.FileWriter('convNetSummary', session.graph)  # File to write reporting
saver = tf.train.Saver()  # Handler for simple initialization or restoration of save files
tf.train.start_queue_runners()  # Coordinator to enqueue records with FixedLengthRecordReader

# Initializes value of variable appropriately
cifar10_model.load_variable_value(saver, session)

# Runs one training/summary step
accuracy_value = session.run(test_set_accuracy)

print('Done. Accuracy is %0.3f' % (accuracy_value,))
