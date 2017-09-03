"""
Handles the conversion of the provided data files (train, trainSmall, test) into their alternate
representation, with outliers filtered out, the dropoff_datetime deleted, and the pickup_datetime
converted into a integer value (seconds since 2016-01-01 00:00:00).
"""

import pandas
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


# Data files information
# All headers : id, vendor_id, pickup_datetime, dropoff_datetime, passenger_count, pickup_longitude,
#   pickup_latitude, dropoff_longitude, dropoff_latitude, store_and_fwd_flag, trip_duration.
# Note : dropoff_datetime will never be used as it it absent from test data (it gives trip length)
train_small_filename = 'trainSmall.csv'
train_medium_filename = 'trainMedium.csv'
train_validation_filename = 'trainValidation.csv'
train_filename = 'train.csv'
test_filename = 'test.csv'

# Bounds computes by iteratively looking at a heatmat of the datapoints and updating the bounds
#   until there was not an obvious outlying point
min_longitude = -74.025
max_longitude = -73.76
min_latitude = 40.625
max_latitude = 40.875
min_trip_duration = 60
max_trip_duration = 10800  # Ignores trips above 3 hours


####################################################################################################
# remove_dropoff_time
####################################################################################################
def remove_dropoff_time(the_dataset):
    """
    """

    the_dataset = the_dataset.drop('dropoff_datetime', 1)

    ###################
    return the_dataset
    ###################


####################################################################################################
# remove_dropoff_time
####################################################################################################
def remove_useless_attributes(the_dataset):
    """
    """

    the_dataset = the_dataset.drop('store_and_fwd_flag', 1)
    the_dataset = the_dataset.drop('vendor_id', 1)

    ###################
    return the_dataset
    ###################


####################################################################################################
# convert_pickup_time
####################################################################################################
def convert_pickup_time(the_dataset):
    """
    """

    datetime_format = '%Y-%m-%d %H:%M:%S'

    all_pickup_times_str = the_dataset.pickup_datetime.values
    all_pickup_times = map(lambda x: datetime.strptime(x, datetime_format), all_pickup_times_str)

    all_weekdays = map(lambda x: x.weekday(), all_pickup_times)
    all_hours = map(lambda x: x.hour, all_pickup_times)

    the_dataset['weekday'] = pandas.Series(all_weekdays, index=the_dataset.index)
    the_dataset['hour'] = pandas.Series(all_hours, index=the_dataset.index)

    the_dataset = the_dataset.drop('pickup_datetime', 1)

    ###################
    return the_dataset
    ###################


####################################################################################################
# filter_longitude_latitde
####################################################################################################
def filter_longitude_latitde(the_dataset):
    """
    """

    initial_size = len(the_dataset)

    the_dataset = the_dataset.loc[the_dataset['pickup_longitude'] > min_longitude]
    the_dataset = the_dataset.loc[the_dataset['pickup_longitude'] < max_longitude]
    the_dataset = the_dataset.loc[the_dataset['dropoff_longitude'] > min_longitude]
    the_dataset = the_dataset.loc[the_dataset['dropoff_longitude'] < max_longitude]

    the_dataset = the_dataset.loc[the_dataset['pickup_latitude'] > min_latitude]
    the_dataset = the_dataset.loc[the_dataset['pickup_latitude'] < max_latitude]
    the_dataset = the_dataset.loc[the_dataset['dropoff_latitude'] > min_latitude]
    the_dataset = the_dataset.loc[the_dataset['dropoff_latitude'] < max_latitude]

    the_dataset = the_dataset.loc[the_dataset['trip_duration'] > min_trip_duration]
    the_dataset = the_dataset.loc[the_dataset['trip_duration'] < max_trip_duration]

    the_dataset = the_dataset.round(4)

    final_size = len(the_dataset)
    print('Number of entries deleted: %d.' % (initial_size - final_size,))

    ###################
    return the_dataset
    ###################


####################################################################################################
# show_latitude_longitudes
####################################################################################################
def show_latitude_longitudes(the_dataset):
    """
    """

    all_longitudes = the_dataset.dropoff_longitude.values
    all_latitudes = the_dataset.dropoff_latitude.values

    heatmap, xedges, yedges = np.histogram2d(all_longitudes, all_latitudes, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()

    #######
    return
    #######


####################################################################################################
# export_converted_dataset
####################################################################################################
def export_converted_dataset(the_dataset, original_filename):
    """
    """

    converted_filename = original_filename.replace('.csv', 'Converted.csv')
    the_dataset.to_csv(converted_filename, index=False)

    #######
    return
    #######


####################################################################################################
# convert_training_file
####################################################################################################
def convert_training_file(training_filename):

    train_dataset = pandas.read_csv(training_filename)
    train_dataset = remove_dropoff_time(train_dataset)
    train_dataset = remove_useless_attributes(train_dataset)
    train_dataset = convert_pickup_time(train_dataset)
    train_dataset = filter_longitude_latitde(train_dataset)
    export_converted_dataset(train_dataset, training_filename)

    #######
    return
    #######


####################################################################################################
# convert_testing_file
####################################################################################################
def convert_testing_file(testing_filename):

    test_dataset = pandas.read_csv(testing_filename)
    test_dataset = remove_useless_attributes(test_dataset)
    test_dataset = convert_pickup_time(test_dataset)
    export_converted_dataset(test_dataset, testing_filename)

    #######
    return
    #######


####################################################################################################
# main
####################################################################################################
def main():

    convert_training_file(train_medium_filename)
    convert_training_file(train_validation_filename)

    #######
    return
    #######


if __name__ == '__main__':

    main()
