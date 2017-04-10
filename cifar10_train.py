import tensorflow as tf
import cifar10_datafeed
import cifar10_model
from time import time as now

progress_report_interval = 50  # Number of train steps between 2 progress log (train/test accuracy)

batch_size = 256  # SNumber of samples in mini-batch
n_classes = cifar10_datafeed.n_classes  # Number of class in dataset
image_size = cifar10_datafeed.image_size  # Size of image sent to model for input
n_channels = cifar10_datafeed.n_channels  # Number of input chanells per pixel (RGB)
n_inputs = image_size * image_size * n_channels  # Inputs per image (n pixel * channel per pixel)
image_size_pooled = int(image_size / 4)  # Inputs per image after max pooling

total_train_examples = 1000  # Total number of examples in training set
total_test_examples = cifar10_datafeed.total_test_examples  # Total number of examples in test set

reset_variables = False  # Attempts to restore a saved state of variables if the save data exists
variable_save_file = './test/saved_weights.cpkt'  # Location of file where weights will be stored


####################################################################################################
# get_training_step
####################################################################################################
def get_training_step(cost_function):
    """
    Creates optimizer based on cost function

    INPUT:
        cost_function (?) cost function that needs to be optimized

    RETURN:
        (tf.train.Optimizer) optimization step for training model
    """

    train_step_model = tf.train.AdamOptimizer(1e-4).minimize(cost_function)

    ########################
    return train_step_model
    ########################

########################
# END get_training_step
########################


####################################################################################################
# make_train_model
####################################################################################################
def make_train_model(add_summary=False):
    """
    Creates training model that will take mini-batches of images as input, and returns both an
        training algorithm step and an accuracy computation model

    INPUT:
        add_summary (bool) whether to add model information monitoring or not

    RETURN:
        (tf.train.AdamOptimizer) optimization step for training model
    """

    # Gets both training set images batches and labels
    train_batch_data, train_batch_label = cifar10_datafeed.get_input(False, batch_size,
                                                                     shuffle=True)

    # Computes predicion for given class (not probability). Softmax has not been applied.
    prediction = cifar10_model.make_model(train_batch_data, add_summary)

    # Computes cross entropy and initialize optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=train_batch_label))

    # Compares prediction class to actual class and computes accuracy
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(train_batch_label, 1))
    train_accuracy_model = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Adds summary values for cross_entropy and accuracy
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('train_accuracy', train_accuracy_model)

    # Adds weight decay to model if applicable
    all_weight_decays = tf.get_collection('weight_decay')
    if len(all_weight_decays) > 0:

        # Weight decay existed, adds it to totalcost function
        total_weight_decay = tf.add_n(all_weight_decays)
        train_step_model = get_training_step(cross_entropy + total_weight_decay)

        # Adds tracker of weight decay value
        tf.summary.scalar('total_weight_decay', total_weight_decay)

    else:

        # No weight decay, uses cross-entropy alone as cost function
        train_step_model = get_training_step(cross_entropy)

    #############################################################
    return cross_entropy, train_step_model, train_accuracy_model
    #############################################################

#######################
# END make_train_model
#######################


print('Module loaded')

# Creates all relevant models to use in the algorithm
train_cross_entropy, train_step, train_accuracy = make_train_model()

# Merges all information trackers set during the model creation (summary)
merged = tf.summary.merge_all()

print('Model set up')

# Initializes all necessary parameters to run model
session = tf.InteractiveSession()  # Session to run model
train_writer = tf.summary.FileWriter('convNetSummary', session.graph)  # File to write reporting
saver = tf.train.Saver()  # Handler for simple initialization or restoration of save files
tf.train.start_queue_runners()  # Coordinator to enqueue records with FixedLengthRecordReader

# Initializes value of variable appropriately
cifar10_model.load_variable_value(saver, session, reset_variables)

to_print_1 = '%d, train CE: %g, train ACC: %g.'
to_print_2 = '%d, train CE: %g, train ACC: %g. Inputs per second: %0.3f. Seconds per batch: %0.3f.'
start_time = now()
for index in range(100000):

    # Runs one training/summary step
    _, cross_entropy_value, accuracy_value, summary = session.run([train_step, train_cross_entropy,
                                                                   train_accuracy, merged])

    # Periodically tests evolution in accuracy for test set and training set
    if index % progress_report_interval == 0:

        # Saves current state of weights
        saver.save(session, variable_save_file)

        if index == 0:

            print(to_print_1 % (index, cross_entropy_value, accuracy_value))

        else:

            examples_per_seconds = batch_size * progress_report_interval / (now() - start_time)
            seconds_per_batch = (now() - start_time) / progress_report_interval
            print(to_print_2 % (index, cross_entropy_value, accuracy_value, examples_per_seconds,
                                seconds_per_batch))
            start_time = now()

        # Adds summary to file
        train_writer.add_summary(summary, index)

print('Done')
