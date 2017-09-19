import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from ade20k import get_train, get_validation
from preprocess import preprocess_for_train, preprocess_for_eval


def provider_to_pipeline(provider):
    provider = slim.dataset_data_provider.DatasetDataProvider(
        provider,
        num_readers=20,
        common_queue_capacity=provider.num_samples,
        common_queue_min=100,
        shuffle=True)

    [image_id, image, label] = provider.get(['id', 'image', 'label'])

    return image_id, image, label


def get_training_pipeline(image_size):
    image_id, image, label = provider_to_pipeline(get_train())
    preprocessed_images, preprocessed_labels = preprocess_for_train(image, label, image_size[0], image_size[1])
    preprocessed_labels = tf.image.resize_images(preprocessed_labels,
                                                 size=(image_size[0] / 8, image_size[1] / 8),
                                                 method=ResizeMethod.NEAREST_NEIGHBOR)
    preprocessed_labels = tf.squeeze(preprocessed_labels, 2)
    return image_id, preprocessed_images, preprocessed_labels


def get_validation_pipeline(image_size):
    image_id, image, label = provider_to_pipeline(get_validation())
    preprocessed_images, preprocessed_labels = preprocess_for_eval(image, label, image_size[0], image_size[1],
                                                                   image_size[0])
    preprocessed_labels = tf.image.resize_images(preprocessed_labels,
                                                 size=(image_size[0]/8, image_size[1]/8),
                                                 method=ResizeMethod.NEAREST_NEIGHBOR)
    preprocessed_labels = tf.squeeze(preprocessed_labels, 2)
    return image_id, preprocessed_images, preprocessed_labels

def get_raw_validation_pipeline():
    return provider_to_pipeline(get_validation())



#
# def crop(image, label, size):
#     size = tf.constant(size)
#     shapes_equal = tf.Assert(tf.reduce_all(tf.equal(tf.shape(image)[0:2], tf.shape(label)[0:2])),
#                              data=[tf.shape(image), tf.shape(label)])
#
#     with tf.control_dependencies([shapes_equal]):
#         crop_size = tf.cast(size, dtype=tf.float32)
#
#         smaller_edge = tf.cast(tf.reduce_min(tf.shape(image)[0:2]), dtype=tf.float32)
#         scale = (crop_size[0] / smaller_edge) * tf.random_uniform(shape=[1], minval=0.7, maxval=1.3)
#
#         new_size = tf.cast(crop_size * scale, tf.int32)
#
#         image = tf.expand_dims(image, axis=0)
#         tf.summary.image("image/original", image)
#         label = tf.expand_dims(label, axis=0)
#         tf.summary.image("label/original", label)
#         image_scaled = tf.squeeze(tf.image.resize_bicubic(image, size=new_size), 0)
#         label_scaled = tf.squeeze(tf.image.resize_nearest_neighbor(label, size=new_size), 0)
#
#         image, label = double_random_crop(image_scaled, label_scaled, size)
#         tf.summary.image("image/cropped", tf.expand_dims(image, axis=0))
#         label = tf.expand_dims(label, axis=0)
#         tf.summary.image("label/cropped", label)
#         label = tf.squeeze(tf.image.resize_nearest_neighbor(label, size=size / 8), 0)
#         tf.summary.image("label/cropped_resized", tf.expand_dims(label, axis=0))
#         label = tf.cast(label, dtype=tf.int32)
#     return image, label
#
#
# def double_random_crop(image, label, size, seed=None, name=None):
#     size = tf.convert_to_tensor((size[0], size[1], 3))
#
#     with tf.variable_scope(name, "double_random_crop", [image, label]):
#         image_tensor = tf.convert_to_tensor(image, name="image_tensor")
#         label_tensor = tf.convert_to_tensor(label, name="segmentation_image_tensor")
#
#         check_shapes = tf.assert_equal(tf.shape(image_tensor)[0:2],
#                                        tf.shape(label_tensor)[0:2],
#                                        ["image and label should have the same height and width"])
#
#         with tf.control_dependencies([check_shapes]):
#             input_shape = tf.shape(image_tensor)
#             limit = input_shape - size + 1
#             limit = [tf.maximum(limit[0], 1),
#                      tf.maximum(limit[1], 1),
#                      tf.maximum(limit[2], 1)]
#             offset = tf.random_uniform(
#                 [3],
#                 dtype=tf.int32,
#                 maxval=tf.int32.max,
#                 minval=0,
#                 seed=seed) % limit
#
#             slice_size = [tf.minimum(size[0], input_shape[0] - offset[0]),
#                           tf.minimum(size[1], input_shape[1] - offset[1])]
#
#             image_shape = tf.convert_to_tensor([slice_size[0], slice_size[1], 3], dtype=tf.int32, name="size")
#             label_shape = tf.convert_to_tensor([slice_size[0], slice_size[1], 1], dtype=tf.int32, name="size")
#
#             image_slice = tf.slice(image_tensor,
#                                    offset,
#                                    image_shape,
#                                    name="double_random_crop_image")
#
#             label_slice = tf.slice(label_tensor,
#                                    offset,
#                                    label_shape,
#                                    name="double_random_crop_label")
#
#             image_crop = tf.pad(image_slice, [[0, size[0] - slice_size[0]], [0, size[1] - slice_size[1]], [0, 0]])
#             label_crop = tf.pad(label_slice, [[0, size[0] - slice_size[0]], [0, size[1] - slice_size[1]], [0, 0]])
#
#             return image_crop, label_crop
#
#
# def preprocess(image):
#     return tf.image.per_image_standardization(image)
