import tensorflow as tf

from pipeline import double_random_crop, decode_class_mask, ADE20K, read_and_decode_image_file, \
    read_and_decode_segmentation_file
from settings import IMAGE_STANDARDIZATION_HEIGHT, IMAGE_STANDARDIZATION_WIDTH


def input_evaluation_pipeline(ade20k):
    image_filename_queue = tf.train.string_input_producer(
        tf.constant(ade20k.validation_image_full_paths), num_epochs=1, shuffle=False)
    segmentation_filename_queue = tf.train.string_input_producer(
        tf.constant(ade20k.validation_segmentation_full_paths), num_epochs=1, shuffle=False)

    input_image_data = read_and_decode_image_file(image_filename_queue)
    segmentation_data = read_and_decode_segmentation_file(segmentation_filename_queue)
    shape = tf.shape(input_image_data)
    new_shape = [(shape[0] / 16) * 16, (shape[1] / 16) * 16]
    input_image_data, segmentation_data = double_random_crop(input_image_data, segmentation_data, new_shape,
                                                             name='crop_image_with_labels')

    resized_segmentation_data = tf.image.resize_images(segmentation_data,
                                                       [IMAGE_STANDARDIZATION_HEIGHT / 16,
                                                        IMAGE_STANDARDIZATION_WIDTH / 16],
                                                       tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    decoded_segmentation_data = decode_class_mask(resized_segmentation_data)

    input_image_data = tf.reshape(input_image_data, [IMAGE_STANDARDIZATION_HEIGHT, IMAGE_STANDARDIZATION_WIDTH, 3])
    return tf.train.batch([input_image_data, decoded_segmentation_data], batch_size=1)


def get_evaluation_pipeline():
    ade20k = ADE20K()
    return input_evaluation_pipeline(ade20k)
