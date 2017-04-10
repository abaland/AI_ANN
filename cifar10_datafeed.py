import tensorflow as tf

total_train_examples = 50000
total_test_examples = 10000

raw_image_size = 32
image_size = 24
n_classes = 10
n_channels = 3


########################################################################################################################
# filter_by_class
########################################################################################################################
def filter_by_class(image_batch, label_batch, label_as_vector):

    correct_boolean_list = tf.reduce_all(tf.equal(label_batch, label_as_vector), 1)

    batch_filtered = tf.gather(image_batch, tf.reshape(tf.where(correct_boolean_list), [-1]))

    ######################
    return batch_filtered
    ######################

######################
# END filter_by_class
######################


########################################################################################################################
# read_data_from_filename_list
########################################################################################################################
def read_data_from_filename_list(filename_list_tf, with_summary=False):

    # Sets number of byte for one record. Class fits in 1 byte, Each pixel (32x32) has 3 channel, each encoded in 1 byte
    class_bytes = 1
    data_bytes = raw_image_size * raw_image_size * n_channels
    one_record_bytes = class_bytes + data_bytes

    # Creates a Tensorflow reader to read record with a given size
    record_reader = tf.FixedLengthRecordReader(record_bytes=one_record_bytes)

    # Reads record from file, and parses the record (each element is a 8bit integer => obtain vector of 8-bit integers)
    _, record = record_reader.read(filename_list_tf)
    record_parsed = tf.decode_raw(record, tf.uint8)

    # Reads class (first byte of vector) (first slice the 1st 8bit value, and cast it to int32)
    class_parsed = tf.cast(tf.strided_slice(record_parsed, [0], [class_bytes]), tf.int32)

    # Reads the image data, cast it to int32, and reshape it to a 3x32x32 tensor (original cifar format)
    image_parsed = tf.cast(tf.strided_slice(record_parsed, [class_bytes], [one_record_bytes]), tf.float32)
    image_reshaped = tf.reshape(image_parsed, [n_channels, raw_image_size, raw_image_size])

    # Reshapes image again to be a 32x32x3 image (format for model)
    image_final = tf.transpose(image_reshaped, [1, 2, 0])

    # Adds summary export for images by class
    if with_summary:

        tf.summary.image('airplane', filter_by_class(image_final, class_parsed, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        tf.summary.image('automobile', filter_by_class(image_final, class_parsed, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
        tf.summary.image('bird', filter_by_class(image_final, class_parsed, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
        tf.summary.image('cat', filter_by_class(image_final, class_parsed, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
        tf.summary.image('deer', filter_by_class(image_final, class_parsed, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
        tf.summary.image('dog', filter_by_class(image_final, class_parsed, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
        tf.summary.image('frog', filter_by_class(image_final, class_parsed, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
        tf.summary.image('horse', filter_by_class(image_final, class_parsed, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
        tf.summary.image('ship', filter_by_class(image_final, class_parsed, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))
        tf.summary.image('truck', filter_by_class(image_final, class_parsed, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
        tf.summary.image('images', image_final)

    #################################
    return image_final, class_parsed
    #################################

###################################
# END read_data_from_filename_list
###################################


########################################################################################################################
# apply_random_transform
########################################################################################################################
def apply_random_transform(image_original):

    # Starts by cropping image (since image will be modified afterwards, better to crop it first)
    image_cropped_intact = tf.random_crop(image_original, [image_size, image_size, n_channels])

    # Randomly flips the image horizontally
    image_cropped_flipped = tf.image.random_flip_left_right(image_cropped_intact)

    # Applies noise to brightness and contrast of images
    image_cropped_altered = tf.image.random_brightness(image_cropped_flipped, max_delta=63)
    image_cropped_altered = tf.image.random_contrast(image_cropped_altered, lower=0.2, upper=0.8)

    #############################
    return image_cropped_altered
    #############################

#############################
# END apply_random_transform
#############################


########################################################################################################################
# _get_batch
########################################################################################################################
def _get_batch(image_read, class_read, min_queue_examples, batch_size, shuffle):

    num_preprocess_threads = 16

    if shuffle:

        image_batch, class_batch = tf.train.shuffle_batch([image_read, class_read], batch_size=batch_size,
                                                          num_threads=num_preprocess_threads,
                                                          capacity=min_queue_examples + 3 * batch_size,
                                                          min_after_dequeue=min_queue_examples)

    else:

        image_batch, class_batch = tf.train.batch([image_read, class_read], batch_size=batch_size,
                                                  num_threads=num_preprocess_threads,
                                                  capacity=min_queue_examples + 3 * batch_size)

    class_one_hot = tf.one_hot(class_batch, n_classes, on_value=1, off_value=0)

    #####################################################################
    return image_batch, tf.reshape(class_one_hot, [batch_size, n_classes])
    #####################################################################

#################
# END _get_batch
#################


########################################################################################################################
# get_input
########################################################################################################################
def get_input(is_eval_data, batch_size=256, shuffle=False):

    # Gets correct data file names depending on whether train or test sets are needed
    if is_eval_data:

        data_files = ['Data/Cifar/test_batch.bin']
        batch_size = total_test_examples
        total_input_dataset = total_test_examples
        shuffle = False

    else:

        data_files = ['Data/Cifar/data_batch_%d.bin' % (file_index,) for file_index in range(1, 6)]
        total_input_dataset = total_train_examples

    # Converts the data filenames list into its Tensorflow equivalent
    data_files_tf = tf.train.string_input_producer(data_files)

    # Gets method to read data and class from the files
    # NOTE : This only contains one image. Calling tf.train.batch with a [x, y, z] tensor will output a [n, x, y, z]
    image_read, class_read = read_data_from_filename_list(data_files_tf)

    # Applies transform operations if training set (includes random crop) or just crop center if test set.
    if is_eval_data:

        image_cropped = tf.image.resize_image_with_crop_or_pad(image_read, image_size, image_size)

    else:

        image_cropped = apply_random_transform(image_read)

    # Applies normalization to the image (normalization is here done on a per-image basis, and not full batch)
    normalized_image = tf.image.per_image_standardization(image_cropped)

    # Ensures the image and label have the correct tensor shape
    normalized_image.set_shape([image_size, image_size, n_channels])
    class_read.set_shape([1])

    # NOTE : Don't really understand that one. Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(total_input_dataset * min_fraction_of_examples_in_queue)

    #########################################################################################
    return _get_batch(normalized_image, class_read, min_queue_examples, batch_size, shuffle)
    #########################################################################################

################
# END get_input
################
