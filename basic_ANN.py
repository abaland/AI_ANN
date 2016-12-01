"""
Applies the most basic type of ANN to the IRIS dataset using the Squared-error as a cost function and sigmoid as
activation.

Dataset taken from http://archive.ics.uci.edu/ml/datasets/Iris

NOTE : since the dataset only contains 150 examples, splitting the dataset to have training, validation and test dataset
is not so efficient. Because of this, only training and test dataset are taken, and a number of iterations is provided.
"""

##################
# Global Packages
##################

import math as m
import random
import numpy as np

__author__ = 'Adrien Baland'
__date__ = '2016.11.19'  # Latest revision date


##################
# Global Variables
##################

# Dataset specific pqrqmeters
filename = 'iris.data'  # Name of the file containing data, to load
all_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']  # List of all classes in this dataset
out_true_all = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]  # Alternate representation of classes
n_columns = 5  # Number of columns in dataset (including class (last column))

percentage_split = 80  # Percentage of data split into training data

learning_rate = 1.  # Learning value value
iterations = 2000  # Number of iterations for algorithm

# Number of layers. First is input layer, Last is output layer
n_layers = [n_columns-1, 9, len(all_classes)]
n_input = n_layers[0]
n_hidden = n_layers[1:-1]
n_output = n_layers[-1]


########################################################################################################################
# parse_line
########################################################################################################################
# Revision History:
#   15-11-2016 AB - Function created
########################################################################################################################
def parse_line(datapoint):
    """
    Converts one split line of data into an array of value.
    All columns are assumed to contain numerical values except class (no restriction).

    INPUT:
        datapoint (str[]) one row of data, with fields already split

    OUTPUT:
        ((float|int[])[]) list of all variable values as floats, followed by class value in alternate representation
    """

    try:

        # Converts all non-class input values to float
        all_entries = [float(datapoint[index_variable]) for index_variable in range(n_columns-1)]

        datapoint_class = all_classes.index(datapoint[n_columns-1])
        # Adds the class value (in alternate representation)
        all_entries.append(datapoint_class)

        # Returns converted values
        ###################
        return all_entries
        ###################

    # Ignore rows that couldn't be parsed
    except Exception as e:

        print('Could not parse following line %s : %s.' % (str(datapoint), str(e)))

        # Failed to convert, so return failed status
        ############
        return None
        ############

#################
# END parse_line
#################


########################################################################################################################
# read_data
########################################################################################################################
# Revision History:
#   15-11-2016 AB - Function created
########################################################################################################################
def read_data(file_url):
    """
    Reads and converts dataset to a list of list

    INPUT:
      file_url (str) : address for the dataset

    OUTPUT:
      (int|int[])[][] : converted data.
    """

    # Initializes the converted dataset to an empty list
    all_data = []

    with open(file_url, 'r') as input_file:

        for input_file_line in input_file:

            # Removes the \n at the end, and split each field
            one_data_row = input_file_line[:-1].split(',')

            # Convert splitted line to the data format
            parsed_point = parse_line(one_data_row)

            # Makes sure the conversion was successfull. It if was, add datapoint
            if parsed_point is not None:

                all_data.append(parsed_point)

    ################
    return all_data
    ################

################
# END read_data
################


########################################################################################################################
# split_train_test
########################################################################################################################
# Revision History:
#   15-11-2016 AB - Function created
########################################################################################################################
def split_train_test(data, percentage):
    """
    Separates full dataset into a training part and a test part.

    INPUT:
        data ((float|int[])[][]) the full dataset, as a list of list of field (float or class representation)
        percentage (int) percentage of the dataset to use for training data

    OUTPUT:
      (int|int[])[][] : training dataset.
      (int|int[])[][] : test dataset.
    """

    # Computes number of entries to use for training part
    i_split = int(m.floor(len(data) * percentage / 100))

    # Randomizes the data before split, to make sure no ordering adds biases to dataset
    random.shuffle(data)

    # Separates the dataset
    train = data[:i_split]
    test = data[i_split:]

    # Returns datasets into a np.array form.
    #######################################
    return np.array(train), np.array(test)
    #######################################

#######################
# END split_train_test
#######################


########################################################################################################################
# initialize_weights
########################################################################################################################
# Revision History:
#   15-11-2016 AB - Function created
########################################################################################################################
def initialize_weights():
    """
    Initializes weights between layers. Makes the assumption that successive layers are fully connected

    OUTPUT:
        (np.random.rand[]) for each pair of successive layer, matrix with weights between nodes
    """

    # Initializes weight list
    all_weights = []

    for index_layer in range(len(n_layers)-1):

        # Gets number of nodes in the current layer and the next one (dimensions of weight matrix)
        n_node_current = n_layers[index_layer]
        n_node_next = n_layers[index_layer+1]

        # Initializes weight matrix as random.
        layer_weights = np.random.rand(n_node_current, n_node_next)

        # Appends the weight matrix to the list
        all_weights.append(layer_weights)

    ###################
    return all_weights
    ###################

#########################
# END initialize_weights
#########################


########################################################################################################################
# activation
########################################################################################################################
# Revision History:
#   15-11-2016 AB - Function created
#   17-11-2016 AB - Added Exception Catch
########################################################################################################################
def activation(x):
    """
    Returns actvation function

    INPUT:
        x (float) : activation input

    OUTPUT:
        y (float) : activation value
    """

    # Computes sigmoid
    try:
        y = 1. / (1. + m.exp(-x))

    except OverflowError:

        y = 1 if x > 0 else 0

    #########
    return y
    #########

#################
# END activation
#################


########################################################################################################################
# activation_derivative
########################################################################################################################
# Revision History:
#   15-11-2016 AB - Function created
########################################################################################################################
def activation_derivative(x):
    """
    Returns derivative of activation function

    INPUT:
        x (float) : function input

    OUTPUT:
        y (float) : function value
    """

    y = activation(x) * (1. - activation(x))

    #########
    return y
    #########

############################
# END activation_derivative
############################


########################################################################################################################
# make_prediction
########################################################################################################################
# Revision History:
#   15-11-2016 AB - Function created
########################################################################################################################
def make_prediction(datapoint, all_weights):
    """
    Uses network to try predicting a value for the provided datapoint

    INPUT:
        datapoint (float[]) input value for one data row.
        all_weights (float[][][]) list of inter-layer weights for the network

    OUTPUT:
        (float[][]) list of all pre-activation values for all-successive layers
        (float[][]) list of all post-activation values for all-successive layers
    """

    ###########
    # Pre-loop
    ###########
    # Computes the pre-activation value for initial layer
    pre_activation_values = [np.dot(datapoint, all_weights[0])]
    post_activation_values = [np.copy(pre_activation_values[0])]

    # Computes post-activation value for initial layer
    for index_node in range(len(post_activation_values[0])):

        post_activation_values[0][index_node] = activation(post_activation_values[0][index_node])

    # Recursive over hidden layer / output layer
    for index_layer in range(1, len(all_weights)):

        # Gets previous layer output, which servers as input
        previous_layer_input = post_activation_values[index_layer-1]

        # Multiplies input by the weight, to get pre-activation output
        pre_activation_values.append(np.dot(previous_layer_input, all_weights[index_layer]))
        post_activation_values.append(np.copy(pre_activation_values[-1]))

        # Applies activation function to get post-activation output
        for index_node in range(len(post_activation_values[index_layer])):

            post_activation_values[index_layer][index_node] = \
                activation(post_activation_values[index_layer][index_node])

    #####################################################
    return pre_activation_values, post_activation_values
    #####################################################

######################
# END make_prediction
######################


########################################################################################################################
# apply_backpropagation
########################################################################################################################
# Revision History:
#   15-11-2016 AB - Function created
#   17-11-2016 AB - Fixed post-activation used instead of pre-activation
#   19-11-2016 AB - Changed for loops to np matrix products.
########################################################################################################################
def apply_backpropagation(datapoint, all_weights, all_weights_diff, _, post_activation_values, correct_class):
    """
    Computes the input-node-independent part of the weight-update in the backpropagation algorithm.

    Example:
        In logistic functions, where Dw_i = -e * x_i * (y* - y) * y * (1 - y), this returns (y* - y) * y * (1 - y)

    INPUT:
        datapoint (float[]) input value for one data row.
        all_weights (float[][][]) list of all weights for the successive layers
        all_weights (float[][][]) list of all weights updates to apply so far
        pre_activation_values (float[][]) list of all pre-activation values for all-successive layers
        post_activation_values (float[][]) list of all post-activation values for all-successive layers
        correct_class (int[]) the correct class in vector representation

    OUTPUT:
        (float[][]) input-node-independent part of weight update, for each successive layer
    """

    predicted_class = post_activation_values[-1]

    partial_de_dy = - (correct_class - predicted_class)

    # Comments below to explain indexing are made based on 1 input layer + 1 hidden + 1 output. More hidden layer does
    # not change the reasonning.
    # Layer_index in [2, 1]
    for layer_index in range(len(n_layers)-1, 0, -1):

        if layer_index == 1:

            # If layer_index is 1, weights are between "Input" And "Hidden" => so use datapoint as input vector
            input_vector = datapoint

        else:

            # If layer_index is 2, weights are between "Hidden" and "Output" => use network prediction for hidden layer
            # post_activation_values[0] => layer_index - 2
            input_vector = post_activation_values[layer_index-2]

        # Computes dE/dz_{k+1} as y_{k+1} * (1 - y_{k+1}) * dE/dy_{k+1}
        # Indexing is the one used for input_vector, incremented by 1 => layer_index - 1
        activation_derivative_value = \
            post_activation_values[layer_index-1] * (1 - post_activation_values[layer_index-1])
        partial_de_dz = activation_derivative_value * partial_de_dy

        # Weights updates between layer k and k+1 as dE/dw_{(k,i),(k+1,j)}
        all_weights_diff[layer_index-1] += np.ma.outerproduct(input_vector, partial_de_dz)

        # dE/dy update to prepare for previous layers
        partial_de_dy = np.dot(all_weights[layer_index-1], partial_de_dz)

    #######
    return
    #######

############################
# END apply_backpropagation
############################


########################################################################################################################
# do_one_epoch
########################################################################################################################
# Revision History:
#   15-11-2016 AB - Function created
########################################################################################################################
def do_one_epoch(data, all_weights, batch_size):
    """
    Makes one full pass through the dataset to update weights.

    INPUT:
        data (float[][]) : dataset to use for training
        all_weights (float[][][]) : all weights of the network
        batch_size (int) : batch-size to use to update weights
    """

    # Initializes the weight update values to 0
    all_weights_diffs = []
    for layer_index in range(len(all_weights)):

        all_weights_diffs.append(np.zeros_like(all_weights[layer_index]))

    batch_count = 0
    for i_sample in range(len(data)):

        # Makes prediction for the current example
        datapoint = data[i_sample][0:n_columns-1]
        correct_class = int(data[i_sample][4])
        correct_class_as_vector = out_true_all[correct_class]
        pre_activation_values, post_activation_values = make_prediction(datapoint, all_weights)

        # Applies the backpropagation algorithm using the prediction error, to compute example-contribution to gradient
        apply_backpropagation(datapoint,
                              all_weights,
                              all_weights_diffs,
                              pre_activation_values,
                              post_activation_values,
                              correct_class_as_vector)

        # If the batch size matched the batch count, updates the weights.
        batch_count += 1
        if batch_count == batch_size or i_sample == len(data)-1:

            for layer_index in range(len(all_weights)):

                all_weights[layer_index] -= learning_rate * all_weights_diffs[layer_index] / batch_size
                all_weights_diffs[layer_index].fill(0.)

            batch_count = 0

    #######
    return
    #######

###################
# END do_one_epoch
###################


########################################################################################################################
# test_model
########################################################################################################################
# Revision History:
#   17-11-2016 AB - Function created
########################################################################################################################
def test_model(data, all_weights):
    """
    Tests model performance of a given dataset

    INPUT:
        data (float[][]) : dataset to use for testing
        all_weights (float[][][]) : all weights of the network
    """

    total_error_value = 0.0
    total_correct_prediction = 0

    for i_sample in range(len(data)):

        # Makes prediction for the current example
        datapoint = data[i_sample][0:n_columns-1]
        correct_class = int(data[i_sample][4])
        correct_class_as_vector = out_true_all[correct_class]
        _, post_activation_values = make_prediction(datapoint, all_weights)
        output_class_prediction = post_activation_values[-1]

        # Checks if correct or not
        total_error_value += np.linalg.norm(correct_class_as_vector - output_class_prediction)

        if np.argmax(output_class_prediction) == correct_class:

            total_correct_prediction += 1

    # Prints test summary
    print("Correct prediction : %d/%d. Error : %0.4f\n" % (total_correct_prediction, len(data), total_error_value))

    #######
    return
    #######

#################
# END test_model
#################


########################################################################################################################
# main
########################################################################################################################
# Revision History:
#   15-11-2016 AB - Function created
#   17-11-2016 AB - Added decreasing learning rate
########################################################################################################################
def main():
    global learning_rate

    data = read_data(filename)

    train_data, test_data = split_train_test(data, percentage_split)

    all_weights = initialize_weights()

    for epoch in range(iterations):

        do_one_epoch(train_data, all_weights, 10)

        # Decreases learning rate every X iterations.
        if epoch % 20 == 19:

            learning_rate *= .99

            print('Train : '),
            test_model(train_data, all_weights)
            print('Test : '),
            test_model(test_data, all_weights)
            print('')

    print all_weights

    return train_data, test_data

###########
# END main
###########


if __name__ == '__main__':

    main()
