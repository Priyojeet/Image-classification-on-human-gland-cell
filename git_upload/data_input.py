from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import random
import pandas as pd
import numpy as np

# Process images of this size.
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 400

# Global constants describing the Dataset data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 320
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 20


def read_from_queue(path,num_channels):
    return tf.image.decode_png(path, channels=0, dtype=tf.uint16)


    class DatasetRecord(object):
        pass

    result = DatasetRecord()

    # Read a record, getting filenames from the filename_queue.
    text_reader = tf.TextLineReader()
    _, csv_content = text_reader.read(all_files_queue)

    i_path, s_path = tf.decode_csv(csv_content,
                                           record_defaults=[[""], [""]])

    result.uint8image = read_from_queue(tf.read_file(i_path),0)
    segment = read_from_queue(tf.read_file(s_path),0)

    result.label = segment
    result.i_path = i_path
    result.s_path = s_path
    result.csv = csv_content
    return result


    num_preprocess_threads = 16
    images, labels, i_paths = tf.train.batch(
        [image, label, i_path],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples+batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, labels, i_paths


def gen_csv_paths(data_dir, pref, sessid = None):
    filenames = get_png_files(os.path.join(data_dir, 'images_' + pref))
    filenames.sort()
    segments = []
    for filename in filenames:
        name = filename
        name = name.replace("/images_","/segments_")
        segments.append(name[:-4]+"_anno"+name[-4:])
    all_files = np.array([filenames, segments])

    if pref == 'train':
        indices = [random.randint(0,len(all_files[0])-1) for i in range(len(all_files[0]))]
        all_files = all_files[:,indices]
        pd_arr = pd.DataFrame(all_files).transpose()
        pd_arr.to_csv(pref + str(sessid) + '.csv', index=False, header=False)
    else:
        pd_arr = pd.DataFrame(all_files).transpose()
        pd_arr.to_csv(pref + '.csv', index=False, header=False)


def get_read_input(eval_data, sessid = None):
    if eval_data == 'train':
        all_files_queue = tf.train.string_input_producer([eval_data + str(sessid) + '.csv'])
    else:
        all_files_queue = tf.train.string_input_producer([eval_data + '.csv'])

    # Read examples from files in the filename queue.
    read_input = read_dataset(all_files_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    read_input.label = tf.cast(read_input.label, tf.int32)

    return read_input, reshaped_image


def get_png_files(dirname):
    return [dirname + '/' + f for f in os.listdir(dirname) if f.endswith('.png')]


def inputs(eval_data, batch_size, sessid):
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    read_input, reshaped_image = get_read_input(eval_data, sessid)
    float_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    float_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    read_input.label.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # Set max intensity to 1
    read_input.label = tf.cast(tf.divide(read_input.label, 255), tf.int32)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    shuffle = False
    # shuffle = False if eval_data == 'test' else True
    return _generate_image_and_label_batch(float_image, read_input.label, read_input.i_path,
                                           min_queue_examples, batch_size,
                                           shuffle=shuffle)
