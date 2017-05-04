import re

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from settings import *


class ADE20K:
    def __init__(self):
        training_indices = file('training_index_shuffled', 'r')
        self.image_full_paths = [''] * 20182
        self.segmentation_full_paths = [''] * 20182
        count = 0
        for line in training_indices:
            stripped_line = line.strip()
            self.image_full_paths[count] = DATA_DIRECTORY + '/' + stripped_line
            self.segmentation_full_paths[count] = DATA_DIRECTORY + '/' + \
                                                  self.image_filename_to_segmentation_filename(stripped_line)
            count += 1

        validation_indices = file('validation_index', 'r')
        self.validation_image_full_paths = [''] * 2000
        self.validation_segmentation_full_paths = [''] * 2000
        count = 0
        for line in validation_indices:
            stripped_line = line.strip()
            self.validation_image_full_paths[count] = DATA_DIRECTORY + '/' + stripped_line
            self.validation_segmentation_full_paths[count] = DATA_DIRECTORY + '/' + \
                                                             self.image_filename_to_segmentation_filename(stripped_line)
            count += 1

    @staticmethod
    def get_segmentation_filenames(full_paths):
        return map(ADE20K.image_filename_to_segmentation_filename, full_paths)

    @staticmethod
    def image_filename_to_segmentation_filename(image_filename):
        return re.sub(r'\.jpg$', '_seg.png', image_filename)


def read_and_decode_image_file(filename_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)
    shape = tf.shape(image)
    image = tf.cond(shape[1] < shape[0], lambda: tf.image.transpose_image(image), lambda: image)
    image = tf.image.resize_images(image, size=[IMAGE_STANDARDIZATION_HEIGHT, IMAGE_STANDARDIZATION_WIDTH],
                                   method=ResizeMethod.BILINEAR)
    image = tf.image.per_image_standardization(image)
    return tf.cast(image, dtype=tf.float32)


def read_and_decode_segmentation_file(filename_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_png(value)
    shape = tf.shape(image)
    image = tf.cond(shape[1] < shape[0], lambda: tf.image.transpose_image(image), lambda: image)
    image = tf.image.resize_images(image, size=[IMAGE_STANDARDIZATION_HEIGHT, IMAGE_STANDARDIZATION_WIDTH],
                                   method=ResizeMethod.NEAREST_NEIGHBOR)

    return tf.cast(image, dtype=tf.float32)


def decode_class_mask(im):
    labels = (im[:, :, 0] // 10) * 256 + im[:, :, 1]
    # tf.expand_dims(labels, axis=2)
    return tf.cast(labels, dtype=tf.int32)


def process_raw_input(input_map, image_dimensions, class_embeddings=None):
    input_image_data, segmentation_data = input_map
    input_image_data, segmentation_data = double_random_crop(input_image_data, segmentation_data, image_dimensions,
                                                             name='crop_image_with_labels')

    resized_segmentation_data = tf.image.resize_images(segmentation_data,
                                                       [image_dimensions[0] / 16, image_dimensions[1] / 16],
                                                       tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if class_embeddings is not None:
        resized_segmentation_data = tf.nn.embedding_lookup(class_embeddings, resized_segmentation_data)
    resized_segmentation_data = tf.squeeze(resized_segmentation_data)
    return [input_image_data, resized_segmentation_data]


def get_pipeline(batch_size, image_dimensions, class_embeddings, num_epochs=500):
    if class_embeddings is not None:
        class_embeddings = tf.constant(class_embeddings)
    input_map = raw_tf_record_reader("ade20k.tfrecords", num_epochs)
    input_tensors = process_raw_input(input_map,
                                      image_dimensions,
                                      class_embeddings=class_embeddings)
    min_after_dequeue = 20182 / 2
    capacity = 20182
    return tf.train.shuffle_batch(input_tensors,
                                  batch_size=batch_size,
                                  capacity=capacity,
                                  min_after_dequeue=min_after_dequeue,
                                  num_threads=4)


def double_random_crop(image, segmentation_image, size, seed=None, name=None):
    size = (size[0], size[1], 3)
    image_size = (size[0], size[1], 3)
    segmentation_size = (size[0], size[1], 1)
    with tf.variable_scope(name, "double_random_crop", [image, segmentation_image, size]):
        image_tensor = tf.convert_to_tensor(image, name="image_tensor")
        segmentation_image_tensor = tf.convert_to_tensor(segmentation_image, name="segmentation_image_tensor")
        image_size = tf.convert_to_tensor(image_size, dtype=tf.int32, name="size")
        segmentation_size = tf.convert_to_tensor(segmentation_size, dtype=tf.int32, name="size")
        shape_1 = tf.shape(image_tensor)
        shape_2 = tf.shape(segmentation_image_tensor)
        check_shape_0 = tf.assert_equal(shape_1[0], shape_2[0], ["Need same sized images"])
        check_shape_1 = tf.assert_equal(shape_1[1], shape_2[1], ["Need same sized images"])

        with tf.control_dependencies([check_shape_0, check_shape_1]):
            shape = shape_1
            limit = shape - size + 1
            offset = tf.random_uniform(
                [3],
                dtype=tf.int32,
                maxval=tf.int32.max,
                minval=0,
                seed=seed) % limit
            # offset = tf.Print(offset, [offset])
            sliced_image_1 = tf.slice(image_tensor,
                                      offset,
                                      image_size,
                                      name="double_random_crop_image")
            sliced_image_2 = tf.slice(segmentation_image_tensor,
                                      offset,
                                      segmentation_size,
                                      name="double_random_crop_segmentation")
            return sliced_image_1, sliced_image_2


def single_tf_record_reader(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'labels': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
        })

    labels = tf.decode_raw(features['labels'], tf.int32)
    labels = tf.reshape(labels, [IMAGE_STANDARDIZATION_HEIGHT, IMAGE_STANDARDIZATION_WIDTH, 1])

    image = tf.decode_raw(features['image'], tf.float32)
    image = tf.reshape(image, [IMAGE_STANDARDIZATION_HEIGHT, IMAGE_STANDARDIZATION_WIDTH, 3])

    return [image, labels]


def raw_tf_record_reader(filename, num_epochs):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)
    return single_tf_record_reader(filename_queue)


def raw_input_reader(ade20k, num_epochs=500):
    image_filename_queue = tf.train.string_input_producer(
        tf.constant(ade20k.image_full_paths), num_epochs=num_epochs, shuffle=False)
    segmentation_filename_queue = tf.train.string_input_producer(
        tf.constant(ade20k.segmentation_full_paths), num_epochs=num_epochs, shuffle=False)

    input_image_data = read_and_decode_image_file(image_filename_queue)
    segmentation_data = read_and_decode_segmentation_file(segmentation_filename_queue)
    decoded_segmentation_data = decode_class_mask(segmentation_data)
    decoded_segmentation_data = tf.expand_dims(decoded_segmentation_data, -1)
    return [input_image_data, decoded_segmentation_data]


def get_raw_pipeline(batch_size, num_epochs=500):
    ade20k = ADE20K()
    input_map = raw_input_reader(ade20k, num_epochs)
    return tf.train.batch(input_map,
                          batch_size=batch_size,
                          shapes=[[IMAGE_STANDARDIZATION_HEIGHT, IMAGE_STANDARDIZATION_WIDTH, 3],
                                  [IMAGE_STANDARDIZATION_HEIGHT, IMAGE_STANDARDIZATION_WIDTH, 1]],
                          num_threads=6)


def regularize_truth(truth):
    epsilon = tf.constant(0.1)
    ###
    # Label Smoothing Regularization from https://arxiv.org/pdf/1512.00567.pdf
    # q' = (1 - \epsilon) * q + \epsilon / K
    # where;
    #  q is the distribution to regularize,
    #  q' is the regularized label,
    #  K is the uniform prior distribution,
    #  \epsilon is the smoothing parameter.
    # we do this to fight imbalanced data
    ###
    # return (1 - epsilon) * truth + epsilon / MAX_CLASS_ID
    return truth
