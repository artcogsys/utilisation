import numpy as np
import tensorflow as tf

from settings import *


class ADE20K:
    # 20.210 training
    # 2.000 validation

    file_types = {"jpg": 0, "png": 1, "txt": 2, "1.png": 3, "2.png": 4, "3.png": 5}

    def __init__(self, folder, number_of_scenes):
        file_list = self.get_file_list(folder)
        self.scenes = np.empty([number_of_scenes, len(self.file_types)], dtype=np.object)
        for file_name in file_list:
            self.set_file(file_name)

    def set_file(self, file_name):
        self.scenes[
            self.file_name_to_scene_index(file_name),
            self.file_name_to_file_type(file_name)
        ] = file_name

    @staticmethod
    def get_file_list(folder):
        raw_file_list = list(open(folder + INDEX_FILE, 'r'))
        file_list = map(lambda line: folder + line.rstrip('\n').lstrip('./'), raw_file_list)
        return file_list

    # def get_training_pipeline(self, batch_size, num_epochs=None):
    #     file_list = ADE20K.get_file_index(TRAINING_DATA_DIRECTORY)
    #     return input_pipeline(file_list, batch_size, num_epochs=num_epochs)

    @staticmethod
    def file_name_to_scene_index(file_name):
        if file_name.endswith("jpg"):
            idx_str = file_name[-12:-4]
        elif file_name.endswith("txt") or file_name.endswith("seg.png"):
            idx_str = file_name[-16:-8]
        else:
            idx_str = file_name[-20:-12]
        # to make them scale form 0 to (arraysize - 1)
        return int(idx_str) - 1

    def file_name_to_file_type(self, file_name):
        if file_name[-5:] in self.file_types:
            return self.file_types[file_name[-5:]]
        else:
            return self.file_types[file_name[-3:]]


training = ADE20K(TRAINING_DATA_DIRECTORY, 20210)
validation = ADE20K(VALIDATION_DATA_DIRECTORY, 2000)


def read_and_decode_jpg(filename_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)
    return tf.cast(image, dtype=tf.float32)


def read_and_decode_png(filename_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_png(value)
    return tf.cast(image, dtype=tf.float32)


def resize(image, input_size, scaling_factor):
    return tf.image.resize_images(image, input_size[0] / scaling_factor, input_size[1] / scaling_factor)


def input_pipeline(dataset, num_epochs=500, batch_size=100):
    input_filename_queue = tf.train.string_input_producer(
        dataset.scenes[:, 0], num_epochs=num_epochs)
    input_image_data = read_and_decode_jpg(input_filename_queue)

    segment_filename_queue = tf.train.string_input_producer(
        dataset.scenes[:, 1], num_epochs=num_epochs)
    segment_image_data = read_and_decode_jpg(segment_filename_queue)

    min_after_dequeue = 250
    capacity = min_after_dequeue + 3 * batch_size

    input_batch, segment_batch = tf.train.shuffle_batch(
        [input_image_data, segment_image_data], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return input_batch, segment_batch
