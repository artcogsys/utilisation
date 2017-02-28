import re

import tensorflow as tf

from settings import *


class ADE20K:
    def __init__(self):
        training_indices = file('training_index', 'r')
        self.image_full_paths = [''] * 20182
        self.segmentation_full_paths = [''] * 20182
        count = 0
        for line in training_indices:
            stripped_line = line.strip()
            self.image_full_paths[count] = DATA_DIRECTORY + '/' + stripped_line
            self.segmentation_full_paths[count] = DATA_DIRECTORY + '/' + \
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
    return tf.cast(image, dtype=tf.float32)


def read_and_decode_segmentation_file(filename_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_png(value)
    shape = tf.shape(image)
    image = tf.cond(shape[1] < shape[0], lambda: tf.image.transpose_image(image), lambda: image)
    return tf.cast(image, dtype=tf.float32)


def decode_class_mask(im):
    labels = (im[:, :, 0] // 10) * 256 + im[:, :, 1]
    tf.expand_dims(labels, axis=2)
    return tf.cast(labels, dtype=tf.int32)


def input_pipeline(ade20k, image_dimensions, num_epochs=500, batch_size=2):
    image_filename_queue = tf.train.string_input_producer(
        tf.constant(ade20k.image_full_paths), num_epochs=num_epochs, shuffle=False)

    segmentation_filename_queue = tf.train.string_input_producer(
        tf.constant(ade20k.segmentation_full_paths), num_epochs=num_epochs, shuffle=False)

    input_image_data = read_and_decode_image_file(image_filename_queue)
    segmentation_data = read_and_decode_segmentation_file(segmentation_filename_queue)
    input_image_data, segmentation_data = double_random_crop(input_image_data, segmentation_data, image_dimensions,
                                                             name='crop_image_with_labels', )

    min_after_dequeue = batch_size * 2
    capacity = batch_size * 4
    decoded_segmentation_data = decode_class_mask(segmentation_data)
    input_batch, segment_batch = tf.train.shuffle_batch(
        [input_image_data, decoded_segmentation_data], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return input_batch, segment_batch


def get_pipeline(batch_size, image_dimensions):
    ade20k = ADE20K()
    return input_pipeline(ade20k, image_dimensions, batch_size=batch_size)


def double_random_crop(image, segmentation_image, size, seed=None, name=None):
    size = (size[0], size[1], 3)
    with tf.variable_scope(name, "double_random_crop", [image, segmentation_image, size]):
        image_tensor = tf.convert_to_tensor(image, name="image_tensor")
        segmentation_image_tensor = tf.convert_to_tensor(segmentation_image, name="segmentation_image_tensor")
        size = tf.convert_to_tensor(size, dtype=tf.int32, name="size")
        shape_1 = tf.shape(image_tensor)
        shape_2 = tf.shape(segmentation_image_tensor)
        check_shape = tf.assert_equal(shape_1, shape_2, ["Need same sized images"])
        check = tf.Assert(
            tf.reduce_all(shape_1 >= size),
            ["Need value.shape >= size, got ", shape_1, size])
        with tf.control_dependencies([check, check_shape]):
            shape = shape_1
            limit = shape - size + 1
            offset = tf.random_uniform(
                tf.shape(shape),
                dtype=size.dtype,
                maxval=size.dtype.max,
                seed=seed) % limit
            sliced_image_1 = tf.slice(image_tensor, offset, size, name="double_random_crop_image")
            sliced_image_2 = tf.slice(segmentation_image_tensor, offset, size, name="double_random_crop_segmentation")

            return sliced_image_1, sliced_image_2
