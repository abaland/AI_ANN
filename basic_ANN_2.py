"""
This version is very similar to the previous basic_ANN script, with the exception that the output layer uses a softmax
 activation function and cross-entropy cost function instead of the (squarred-error, sigmoid) combination. The other
 hidden layer still use the latter combination.

Dataset taken from http://archive.ics.uci.edu/ml/datasets/Iris
"""

##################
# Global Packages
##################

import math as m
import numpy as np

#################
# Local Packages
#################
from basic_ANN import read_data
from basic_ANN import split_train_test
from basic_ANN import initialize_weights

__author__ = 'Adrien Baland'
__date__ = '2016.11.17'  # Latest revision date


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
# activation
########################################################################################################################
# Revision History:
#   20-11-2016 AB - Function created
########################################################################################################################
def activation(pre_activation_values, is_last_layer):
    """
    Returns actvation function

    INPUT:
        x (float[]) : value vector before activation function is applied

    OUTPUT:
        y (float) : value vector before activation function is applied
    """

    post_activation_values = np.zeros_like(pre_activation_values)

    if is_last_layer:

        try:
            denominator_softmax = sum([m.exp(x) for x in pre_activation_values])

        except OverflowError:

            post_activation_values[np.argmax(pre_activation_values)] = 1.0

            ##############################
            return post_activation_values
            ##############################

        for node_index in range(len(post_activation_values)):

            # Computes sigmoid
            try:
                post_activation_values[node_index] = m.exp(pre_activation_values[node_index]) / denominator_softmax

            except OverflowError:

                post_activation_values[node_index] = 1 if pre_activation_values[node_index] > 0 else 0

    else:

        for node_index in range(len(post_activation_values)):

            # Computes sigmoid
            try:
                post_activation_values[node_index] = 1. / (1. + m.exp(-pre_activation_values[node_index]))

            except OverflowError:

                post_activation_values[node_index] = 1 if pre_activation_values[node_index] > 0 else 0

    ##############################
    return post_activation_values
    ##############################

#################
# END activation
#################


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
    # If only 2 layers, we go straight to output layer, so use softmax immediately
    is_last_layer = False
    if len(n_layers) == 2:

        is_last_layer = True

    post_activation_values = [activation(pre_activation_values[0], is_last_layer)]

    # Recursive over hidden layer / output layer
    for index_layer in range(1, len(all_weights)):

        if index_layer == len(all_weights) - 1:

            is_last_layer = True

        # Gets previous layer output, which servers as input
        previous_layer_input = post_activation_values[index_layer-1]

        # Multiplies input by the weight, to get pre-activation output
        pre_activation_values.append(np.dot(previous_layer_input, all_weights[index_layer]))
        post_activation_values.append(np.copy(pre_activation_values[-1]))

        post_activation_values[index_layer] = activation(pre_activation_values[index_layer], is_last_layer)

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

    # Declares dE/dy (only for IDE purpose)
    partial_de_dy = None

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

        # Tests if we are currently using the ouptput layer. If we are, use cross-entropy dC/dz
        if layer_index == len(n_layers) - 1:

            predicted_class = post_activation_values[-1]
            partial_de_dz = correct_class - predicted_class

        else:

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

                all_weights[layer_index] -= learning_rate * all_weights_diffs[layer_index]
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
        _, post_activation_values = make_prediction(datapoint, all_weights)
        output_class_prediction = post_activation_values[-1]

        # Checks if correct or not
        try:

            total_error_value -= m.log(output_class_prediction[correct_class])

        except ValueError:

            total_error_value = float('inf')

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
    print('Starting basic_ANN_2')

    data = read_data(filename)

    train_data, test_data = split_train_test(data, percentage_split)

    all_weights = initialize_weights()

    for epoch in range(iterations):

        do_one_epoch(train_data, all_weights, 1)

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
