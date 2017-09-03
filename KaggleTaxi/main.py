import pandas
import math
from sklearn.linear_model import LinearRegression
import numpy as np
import tensorflow as tf


# Data files information
# All headers : id, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude,
#   dropoff_latitude, trip_duration, weekday, hour
# with_validation, train_filename = True, 'trainSmallConverted.csv'
with_validation, train_filename = True, 'trainMediumConverted.csv'
# with_validation, train_filename = False, 'trainConverted.csv'

# Names of file containing validation dataset
validation_filename = 'trainValidationConverted.csv'

# Name of file containing testing set
test_filename = 'testConverted.csv'


####################################################################################################
# get_prediction_error
####################################################################################################
def get_prediction_error(predictions, true_values):
    """
    Computes the average prediction error by comparing the vector of true values and the vector of
        predictions.
    Uses sum of squared errors between the two vectors, normalized by the number of entries.

    INPUT:
        (float[]) list of all trip duration predictions
        (float[]) list of all actual values of trip durations.

    OUTPUT:
        (float) average prediction error
    """

    # Sums all errors between predictions and true values
    average_error = 0.
    for prediction_index in range(len(predictions)):

        prediction = predictions[prediction_index]
        true_value = true_values[prediction_index]
        average_error += (prediction - true_value) * (prediction - true_value)

    # Normalizes by number of predictions
    average_error = math.sqrt(average_error) / len(predictions)

    #####################
    return average_error
    #####################


####################################################################################################
# export_predictions
####################################################################################################
def export_predictions(trip_ids, predictions, output_filename):
    """
    Exports all trip duration predicted to a file.
    File starts by a header row "id,trip_duration".
    Then exports each prediction as a row coma-separated.

    INPUT:
        trip_ids (str[]) list of all trip ids
        predictions (float[]) list of all trip duration predicted
        output_filename (str) name of file where all predictions will be written
    """

    with open(output_filename, 'w') as predictor_file:

        # Exports header row
        predictor_file.write('id,trip_duration\n')

        # Exports each prediction as a row
        for prediction_index in range(len(predictions)):

            trip_id = trip_ids[prediction_index]
            prediction = predictions[prediction_index]

            predictor_file.write('%s,%s\n' % (trip_id, prediction))

    #######
    return
    #######


####################################################################################################
# make_neural_model
####################################################################################################
def make_neural_model(the_dataset, hidden_nodes1):
    """
    Creates model to predict the trip duration using a basic neural network.
    The inputs are the total of passengers, longitude/latitude for pickup/dropoff,
    the longitude/latitude deltas and the straightline delta.
    Only 1 hidden layer is used, with a softmax.
    No normalization occurs.
    Returns a dictionnary with the necessary elements to use created model later

    INPUT:
        the_dataset (pandas.DataFrame) dataset to analyze
        hidden_nodes1 (int) number of nodes in the first hidden layer

    OUTPUT:
        (dict) dictionanry with relevant objects to use created model. Entries are:
            all_inputs
            input_placeholder
            output_placerholder
            training_operator
            predictions
    """

    # Sets up all input vectors to be sent, part 1 : pre-existing input
    total_passengers = the_dataset.passenger_count.values
    pickup_longitude = the_dataset.pickup_longitude.values
    pickup_latitude = the_dataset.pickup_latitude.values
    dropoff_longitude = the_dataset.dropoff_longitude.values
    dropoff_latitude = the_dataset.dropff_latitude.values

    # Sets up all input vectors to be sent, part 2 : parts which must be computed
    delta_latitude = the_dataset.dropoff_latitude - the_dataset.pickup_latitude
    delta_longitude = the_dataset.dropoff_longitude - the_dataset.pickup_longitude
    straightline_delta = ((delta_longitude ** 2 - delta_latitude ** 2) ** .5).value
    delta_latitude = abs(delta_longitude).value
    delta_longitude = abs(delta_longitude).value

    # Sets up the input matrix
    all_inputs = np.array([total_passengers, pickup_longitude, pickup_latitude, dropoff_longitude,
                           dropoff_latitude, straightline_delta, delta_latitude, delta_longitude]).T

    # Placeholder for input vectors
    input_layer = tf.placeholder(tf.float32, [None, 8])

    # Sets up first layer
    layer1_weights = tf.Variable(tf.zeros([8, hidden_nodes1]))
    layer1_bias = tf.Variable(tf.zeros(hidden_nodes1))
    layer1_output = tf.nn.softmax(tf.matmul(input_layer, layer1_weights) + layer1_bias)

    # Sets up second (final) layer
    layer2_weights = tf.Variable(tf.zeros([hidden_nodes1, 1]))
    layer2_bias = tf.Variable(tf.zeros(1))
    predictions = tf.matmul(layer1_output, layer2_weights) + layer2_bias

    # Placeholder for trip duration
    truth_value = tf.placeholder(tf.float32, [None, 10])

    # Computes prediction average error from model
    prediction_average_error = tf.reduce_mean(tf.squared_difference(predictions, truth_value))

    # Uses average error to set up prediction
    train_step = tf.train.AdagradOptimizer(1e-4).minimize(prediction_average_error)

    all_model_information = {
        'all_inputs': all_inputs,
        'input_placeholder': input_layer,
        'output_placeholder': truth_value,
        'training_operator': train_step,
        'predictions': predictions
    }

    #############################
    return all_model_information
    #############################


#
#
#


####################################################################################################
# compute_average
####################################################################################################
def compute_average(array_to_average):
    """
    Computes average of an array of number.

    INPUT:
        array_to_average ((int|float)[]) array of values that must be averaged

    RETURN:
        (float) average value of the array
    """

    average_trip_length = 1.0 * sum(array_to_average) / len(array_to_average)

    ###########################
    return average_trip_length
    ###########################


####################################################################################################
# prediction_test_1
####################################################################################################
def prediction_test_1():
    """
    First predictor test.
    Computes the average trip length over full dataset and always predict that value.
    """

    # Each prediction is the average trip length
    def make_predictions(dataset, _mean_trip_length):

        return [_mean_trip_length for _ in range(len(dataset))]

    # Loads datasets
    train_set = pandas.read_csv(train_filename)
    validation_set = pandas.read_csv(validation_filename)
    test_set = pandas.read_csv(test_filename)

    # Compute average trip length (over full dataset)
    mean_trip_length = int(round(compute_average(train_set.trip_duration.values)))

    # Computes error for validation dataset
    if with_validation:
        validation_predictions = make_predictions(validation_set, mean_trip_length)
        validation_true_values = validation_set.trip_duration.values
        validation_error = get_prediction_error(validation_predictions, validation_true_values)
        print('First model. Error: %.4f.' % (validation_error,))

    # Computes prediction for test dataset, and exports them
    test_predictions = make_predictions(test_set, mean_trip_length)
    export_predictions(test_set.id.values, test_predictions, 'submission1.csv')

    #######
    return
    #######


#
#
#


####################################################################################################
# get_per_day_mean_trip_length
####################################################################################################
def get_per_day_mean_trip_length(the_dataset):
    """
    Computes the average trip length for each day separately, and returns them as a list

    INPUT:
        the_dataset (pandas.DataFrame) dataset to analyze

    OUTPUT:
        (float[]) average trip length for each day (with monday as day 0)
    """

    # Initializes per-day average trip length
    all_average_durations = [0 for _ in range(7)]

    for day in range(7):

        # Isolates part of dataset that is only relevant for that day
        data_subset = the_dataset.loc[the_dataset['weekday'] == day]

        # Computes average trip duration for that subset
        average_duration = int(round(compute_average(data_subset.trip_duration.values)))

        # Updates per day average trip duration with value for that day
        all_average_durations[day] = average_duration

    #############################
    return all_average_durations
    #############################


####################################################################################################
# prediction_test_2
####################################################################################################
def prediction_test_2():
    """
    Seconds predictor tested.
    Computes the average trip length for each day in dataset separately, and always predict relevant
        value.
    """

    # Each prediction is the average trip length for the corresponding day
    def make_prediction(dataset, _per_day_mean_length):

        # Initializes prediction
        predictions = [0 for _ in range(len(dataset))]

        # Gets weekday value for all data row
        weekdays = dataset.weekday.values

        for i in range(len(dataset)):

            # For each row, gets weekday and predicts corresponding trip average
            predictions[i] = _per_day_mean_length[weekdays[i]]

        ###################
        return predictions
        ###################

    # Loads datasets
    train_set = pandas.read_csv(train_filename)
    validation_set = pandas.read_csv(validation_filename)
    test_set = pandas.read_csv(test_filename)

    # Compute average trip length (per day)
    per_day_mean_trip_length = get_per_day_mean_trip_length(train_set)

    # Computes error for validation dataset
    if with_validation:
        validation_predictions = make_prediction(validation_set, per_day_mean_trip_length)
        validation_true_values = validation_set.trip_duration.values
        average_error = get_prediction_error(validation_predictions, validation_true_values)
        print('Second model. Error: %.4f.' % (average_error,))

    # Computes prediction for test dataset, and exports them
    test_predictions = make_prediction(test_set, per_day_mean_trip_length)
    export_predictions(test_set.id.values, test_predictions, 'submission2.csv')

    #######
    return
    #######


#
#
#


####################################################################################################
# get_per_day_per_block_mean_trip_length
####################################################################################################
def get_per_day_per_block_mean_trip_length(the_dataset, hours_per_block=4):
    """
    Computes the average trip length for each day and hour blocks separately, and returns them as a
        list of list

    INPUT:
        the_dataset (pandas.DataFrame) dataset to analyze
        hours_per_block (int) number of hours gathered to form an "hour block"

    OUTPUT:
        (float[][]) average trip length for each day (with monday as day 0) for each hour block
    """

    # Gets number of hours blocks that exist in one day (hours_per_block must be divisor of 24)
    tot_blocks_per_day = 24 / hours_per_block

    # Initializes list of average trip duration (per day, per hour block)
    all_average_durations = [[0 for _ in range(tot_blocks_per_day)] for _ in range(7)]

    for day in range(7):

        for hour_block_id in range(tot_blocks_per_day):

            # Gets min and max hour that compose the hour block
            hour_min = hour_block_id * hours_per_block
            hour_max = (hour_block_id + 1) * hours_per_block

            # Gets subset such that hour is within hour block and is on relevant day
            data_subset = the_dataset.loc[the_dataset['weekday'] == day]
            data_subset = data_subset.loc[hour_min <= the_dataset['hour']]
            data_subset = data_subset.loc[the_dataset['hour'] < hour_max]

            # Computes average trip length for (day, hour block) combination
            mean_trip_length = int(round(compute_average(data_subset.trip_duration.values)))

            # Updates list of average trip durations
            all_average_durations[day][hour_block_id] = mean_trip_length

    #############################
    return all_average_durations
    #############################


####################################################################################################
# prediction_test_3
####################################################################################################
def prediction_test_3():
    """
    Third predictor tested.
    Computes the average trip length for each (day, hour block) in dataset separately, and always
        predict relevant value.
    """

    # UPDATABLE: number of hours per block
    hours_per_block = 4

    # Each prediction is the average trip length for the corresponding (day, hour block)
    def make_prediction(dataset, _per_day_per_block_mean_length):

        # Initializes prediction
        predictions = [0 for _ in range(len(dataset))]

        # Gets weekday and hours value for all data rows
        weekdays = dataset.weekday.values
        hours = dataset.hour.values

        for i in range(len(dataset)):

            # For each row, gets weekday and hour block for that row
            weekday = weekdays[i]
            hour = hours[i]
            hour_block_id = hour / hours_per_block

            # Makes prediction for that (day, hour block) combination
            predictions[i] = _per_day_per_block_mean_length[weekday][hour_block_id]

        ###################
        return predictions
        ###################

    # Loads datasets
    train_set = pandas.read_csv(train_filename)
    validation_set = pandas.read_csv(validation_filename)
    test_set = pandas.read_csv(test_filename)

    # Compute average trip length (per day, per hour block)
    all_durations = get_per_day_per_block_mean_trip_length(train_set, hours_per_block)

    # Computes error for validation dataset
    if with_validation:
        validation_predictions = make_prediction(validation_set, all_durations)
        validation_true_values = validation_set.trip_duration.values
        average_error = get_prediction_error(validation_predictions, validation_true_values)
        print('Third model. Error: %.4f.' % (average_error,))

    # Computes prediction for test dataset, and exports them
    test_predictions = make_prediction(test_set, all_durations)
    export_predictions(test_set.id.values, test_predictions, 'submission3.csv')

    #######
    return
    #######


#
#
#


####################################################################################################
# get_per_day_per_block_regression
####################################################################################################
def get_per_day_per_block_regression(the_dataset, hours_per_block=4):
    """
    Makes a linear regression model for each day and hour blocks separately.
    The regression follows formula:
        a + b * delta_long + c * delta_lat + d * sqrt(delta_long ^ 2 + delta_lat ^ 2)
    Returns the list of [a, b, c, d] for each (day, hour block) combination.

    INPUT:
        the_dataset (pandas.DataFrame) dataset to analyze
        hours_per_block (int) number of hours gathered to form an "hour block"

    OUTPUT:
        (float[][][]) regression model parameters for each day for each hour block
    """

    # Gets number of hours blocks that exist in one day (hours_per_block must be divisor of 24)
    tot_blocks_per_day = 24 / hours_per_block

    # Initializes list of parameters for each linear regression model that will be created
    all_models = [[None for _ in range(tot_blocks_per_day)] for _ in range(7)]

    for day in range(7):

        for hour_block_id in range(tot_blocks_per_day):

            # Gets min and max hour that compose the hour block
            hour_min = hour_block_id * hours_per_block
            hour_max = (hour_block_id + 1) * hours_per_block

            # Gets subset such that hour is within hour block and is on relevant day
            data_subset = the_dataset.loc[the_dataset['weekday'] == day]
            data_subset = data_subset.loc[hour_min <= the_dataset['hour']]
            data_subset = data_subset.loc[the_dataset['hour'] < hour_max]

            # Gets the longitude difference and the latitude difference for all rows.
            delta_long = data_subset['pickup_longitude'] - data_subset['dropoff_longitude']
            delta_lat = data_subset['pickup_latitude'] - data_subset['dropoff_latitude']

            # Gets actual values for longitude differences, latitude differences, and straightline
            #   distance (still in degrees form) of trip
            delta_long_values = abs(delta_long).values
            delta_lat_values = abs(delta_lat).values
            delta_straightline_values = ((delta_lat ** 2 + delta_long ** 2) ** 0.5).values
            all_values = np.array([delta_long_values, delta_lat_values, delta_straightline_values])

            # Creates regression model and fits it to provided data
            the_model = LinearRegression(fit_intercept=True)
            the_model.fit(all_values.T, data_subset.trip_duration.values)

            # Replaces model by corresponding coefficients
            a = the_model.intercept_
            b, c, d = list(the_model.coef_)
            all_models[day][hour_block_id] = [a, b, c, d]

    ##################
    return all_models
    ##################


####################################################################################################
# prediction_test_4
####################################################################################################
def prediction_test_4():
    """
    Fourth predictor tested.
    This loads the training dataset, makes a linear regression model for each (day, hour block).
    The regression follows formula:
        a + b * delta_long + c * delta_lat + d * sqrt(delta_long ^ 2 + delta_lat ^ 2)
    """

    # UPDATABLE: number of hours per block
    hours_per_block = 1

    # Each prediction is the output from regression model for the corresponding (day, hour block)
    def make_prediction(dataset, _all_regressions, _hours_per_block):

        # Initializes prediction
        predictions = [0 for _ in range(len(dataset))]

        # Gets weekday and hours value for all data rows
        hours = dataset.hour.values
        weekdays = dataset.weekday.values

        # Gets longitude/latitude deltas and straightline variation values
        delta_long = (dataset.pickup_longitude - dataset.dropoff_longitude)
        delta_lat = (dataset.pickup_latitude - dataset.dropoff_latitude)
        delta_long_values = abs(delta_long).values
        delta_lat_values = abs(delta_lat).values
        delta_straightline_values = ((delta_lat ** 2 + delta_long ** 2) ** 0.5).values

        for i in range(len(dataset)):

            # For each row, gets weekday and hour block for that row
            weekday = weekdays[i]
            hour = hours[i]
            hour_block_id = hour / _hours_per_block

            # Gets coefficients from Linear Regression model
            inter_coef, long_coef, lat_coef, straightline_coef = \
                _all_regressions[weekday][hour_block_id]

            # Makes prediction using the linear regression model coefficients
            predictions[i] = inter_coef + long_coef * delta_long_values[i] + \
                lat_coef * delta_lat_values[i] + straightline_coef * delta_straightline_values[i]

        ###################
        return predictions
        ###################

    # Loads datasets
    train_set = pandas.read_csv(train_filename)
    validation_set = pandas.read_csv(validation_filename)
    test_set = pandas.read_csv(test_filename)

    # Makes regression models (per day, per hour block)
    all_regressions = get_per_day_per_block_regression(train_set, hours_per_block)

    # Computes error for validation dataset
    if with_validation:
        validation_predictions = make_prediction(validation_set, all_regressions, hours_per_block)
        validation_true_values = validation_set.trip_duration.values
        average_error = get_prediction_error(validation_predictions, validation_true_values)
        print('Fourth model. Error: %.4f.' % (average_error,))

    # Exports prediction.
    test_predictions = make_prediction(test_set, all_regressions, hours_per_block)
    export_predictions(test_set.id.values, test_predictions, 'submission4.csv')

    #######
    return
    #######


#
#
#


####################################################################################################
# prediction_test_5
####################################################################################################
def prediction_test_5():

    # UPDATABLE: number of hours per block
    hours_per_block = 1

    # Each prediction is the output from regression model for the corresponding (day, hour block)
    def make_prediction(dataset, _all_regressions, _hours_per_block):

        # Initializes prediction
        predictions = [0 for _ in range(len(dataset))]

        # Gets weekday and hours value for all data rows
        hours = dataset.hour.values
        weekdays = dataset.weekday.values

        # Gets longitude/latitude deltas and straightline variation values
        delta_long = (dataset.pickup_longitude - dataset.dropoff_longitude)
        delta_lat = (dataset.pickup_latitude - dataset.dropoff_latitude)
        delta_long_values = abs(delta_long).values
        delta_lat_values = abs(delta_lat).values
        delta_straightline_values = ((delta_lat ** 2 + delta_long ** 2) ** 0.5).values

        for i in range(len(dataset)):

            # For each row, gets weekday and hour block for that row
            weekday = weekdays[i]
            hour = hours[i]
            hour_block_id = hour / _hours_per_block

            # Gets coefficients from Linear Regression model
            inter_coef, long_coef, lat_coef, straightline_coef = \
                _all_regressions[weekday][hour_block_id]

            # Makes prediction using the linear regression model coefficients
            predictions[i] = inter_coef + long_coef * delta_long_values[i] + \
                lat_coef * delta_lat_values[i] + \
                straightline_coef * delta_straightline_values[i]

        ###################
        return predictions
        ###################

    # Loads datasets
    train_set = pandas.read_csv(train_filename)
    validation_set = pandas.read_csv(validation_filename)
    test_set = pandas.read_csv(test_filename)

    # Makes regression models (per day, per hour block)
    all_regressions = get_per_day_per_block_regression(train_set, hours_per_block)

    # Computes error for validation dataset
    if with_validation:
        validation_predictions = make_prediction(validation_set, all_regressions, hours_per_block)
        validation_true_values = validation_set.trip_duration.values
        average_error = get_prediction_error(validation_predictions, validation_true_values)
        print('Fourth model. Error: %.4f.' % (average_error,))

    # Exports prediction.
    test_predictions = make_prediction(test_set, all_regressions, hours_per_block)
    export_predictions(test_set.id.values, test_predictions, 'submission4.csv')

    #######
    return
    #######


#
#
#


####################################################################################################
# main
####################################################################################################
def main():

    # prediction_test_1()

    # prediction_test_2()

    # prediction_test_3()

    # prediction_test_4()

    prediction_test_5()

    #######
    return
    #######


if __name__ == '__main__':

    main()
