import tensorflow as tf
from tensorflow.contrib import slim

from ade20k import get_train, get_validation


def provider_to_pipeline(provider, image_size):
    provider = slim.dataset_data_provider.DatasetDataProvider(
        provider,
        num_readers=20,
        common_queue_capacity=provider.num_samples,
        common_queue_min=100,
        shuffle=True)

    [image_id, image, label] = provider.get(['id', 'image', 'label'])

    image, label = crop(image, label, image_size)

    image = preprocess(image)

    label = tf.squeeze(label, 2)
    return image_id, image, label


def get_training_pipeline(image_size):
    return provider_to_pipeline(get_train(), image_size)


def get_validation_pipeline(image_size):
    return provider_to_pipeline(get_validation(), image_size)


def crop(image, label, size):
    size = tf.constant(size)
    shapes_equal = tf.Assert(tf.reduce_all(tf.equal(tf.shape(image)[0:2], tf.shape(label)[0:2])),
                             data=[tf.shape(image), tf.shape(label)])

    with tf.control_dependencies([shapes_equal]):
        crop_size = tf.cast(size, dtype=tf.float32)
        scale = tf.random_uniform(shape=[1], minval=0.7, maxval=1.3)

        # Data Augmentation, random scale between 0.7 - 1.3
        # also make the shapes divisable by 8, so that we can
        crop_size = tf.cast(crop_size * scale, dtype=tf.int32)

        image, label = double_random_crop(image, label, crop_size)

        image_crop = tf.expand_dims(image, axis=0)
        label_crop = tf.expand_dims(label, axis=0)

        image_part = tf.squeeze(tf.image.resize_bilinear(image_crop, size=size), 0)
        label_part = tf.squeeze(tf.image.resize_nearest_neighbor(label_crop, size=size / 8), 0)
        label_part = tf.cast(label_part, dtype=tf.int32)
    return image_part, label_part


def double_random_crop(image, label, size, seed=None, name=None):
    size = tf.convert_to_tensor((size[0], size[1], 3))

    with tf.variable_scope(name, "double_random_crop", [image, label]):
        image_tensor = tf.convert_to_tensor(image, name="image_tensor")
        label_tensor = tf.convert_to_tensor(label, name="segmentation_image_tensor")

        check_shapes = tf.assert_equal(tf.shape(image_tensor)[0:2],
                                       tf.shape(label_tensor)[0:2],
                                       ["image and label should have the same height and width"])

        with tf.control_dependencies([check_shapes]):
            input_shape = tf.shape(image_tensor)
            limit = input_shape - size + 1
            limit = [tf.maximum(limit[0], 1),
                     tf.maximum(limit[1], 1),
                     tf.maximum(limit[2], 1)]
            offset = tf.random_uniform(
                [3],
                dtype=tf.int32,
                maxval=tf.int32.max,
                minval=0,
                seed=seed) % limit

            slice_size = [tf.minimum(size[0], input_shape[0] - offset[0]),
                          tf.minimum(size[1], input_shape[1] - offset[1])]

            image_shape = tf.convert_to_tensor([slice_size[0], slice_size[1], 3], dtype=tf.int32, name="size")
            label_shape = tf.convert_to_tensor([slice_size[0], slice_size[1], 1], dtype=tf.int32, name="size")

            image_slice = tf.slice(image_tensor,
                                   offset,
                                   image_shape,
                                   name="double_random_crop_image")

            label_slice = tf.slice(label_tensor,
                                   offset,
                                   label_shape,
                                   name="double_random_crop_label")

            image_crop = tf.pad(image_slice, [[0, size[0] - slice_size[0]], [0, size[1] - slice_size[1]], [0, 0]])
            label_crop = tf.pad(label_slice, [[0, size[0] - slice_size[0]], [0, size[1] - slice_size[1]], [0, 0]])

            return image_crop, label_crop


def preprocess(image):
    return tf.image.per_image_standardization(image)
