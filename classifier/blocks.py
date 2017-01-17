import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


def create_weights(name, shape):
    return tf.get_variable(name, shape=shape, initializer=xavier_initializer())


def conv2d(input_layer, filter_shape, stride):
    filter = create_weights("filter_weights", filter_shape)
    return tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding="SAME")


def relu_conv2d(input_layer, filter_shape, stride=1, mcrelu=False):
    l = conv2d(input_layer, filter_shape, stride=stride)
    return normalized_relu_activation(l, filter_shape[3], negative_concatenation=mcrelu)


def normalized_relu_activation(input_layer, output_size, negative_concatenation=False):
    l = batch_normalization(input_layer)
    if negative_concatenation:
        l = tf.concat(3, [l, -1 * l])
        output_size *= 2
    l = add_bias(l, output_size)
    return tf.nn.relu(l)


def batch_normalization(layer):
    mean = tf.Variable(initial_value=.0, trainable=True)
    variance = tf.Variable(initial_value=.0, trainable=True)
    offset = tf.Variable(initial_value=.0, trainable=True)
    scale = tf.Variable(initial_value=.0, trainable=True)
    variance_epsilon = tf.Variable(initial_value=1e-7, trainable=True)
    # @TODO: check how to implement caffe's batch_norm.use_global_stats=True
    return tf.nn.batch_normalization(layer, mean, variance, offset, scale, variance_epsilon)


def add_bias(layer, number_of_channels):
    bias = create_weights("bias", number_of_channels)
    return tf.nn.bias_add(layer, bias)


def residual_bottleneck_mcrelu(input_layer, kernel_size, input_channels, output_channels_list,
                               stride=1):
    activated_input_layer = normalized_relu_activation(input_layer, input_channels)
    first = relu_conv2d(activated_input_layer, [1, 1, input_channels, output_channels_list[0]])
    conv = relu_conv2d(first, [kernel_size, kernel_size, output_channels_list[0], output_channels_list[1]],
                       stride=stride, mcrelu=True)
    pre_activation = conv2d(conv, filter_shape=[1, 1, output_channels_list[1], output_channels_list[2]], stride=1)

    scaled_input_layer = scale_residual_input(input_layer, input_channels, output_channels_list)

    return residual_connection(scaled_input_layer, pre_activation)


def residual_connection(parent, child):
    return parent + child


def scale_residual_input(input_layer, input_channels, output_channels_list, stride=1):
    if input_channels != output_channels_list[3]:
        return conv2d(input_layer, [1, 1, input_channels, output_channels_list[3]], stride=stride)
    else:
        return input_layer


def residual_inception(input_layer, input_channels, output_channels_list, stride=1, pooling=False):
    activated_input_layer = normalized_relu_activation(input_layer, input_channels)

    inception_1 = relu_conv2d(activated_input_layer, [1, 1, input_channels, output_channels_list[0]], stride=stride)

    inception_2_1 = relu_conv2d(activated_input_layer, [1, 1, input_channels, output_channels_list[1]], stride=stride)
    inception_2 = relu_conv2d(inception_2_1, [3, 3, input_channels, output_channels_list[2]], stride=1)

    inception_3_1 = relu_conv2d(activated_input_layer, [1, 1, input_channels, output_channels_list[3]], stride=stride)
    inception_3_2 = relu_conv2d(inception_3_1, [3, 3, input_channels, output_channels_list[4]], stride=1)
    inception_3 = relu_conv2d(inception_3_2, [3, 3, input_channels, output_channels_list[5]], stride=1)
    result = tf.concat(3, [inception_1, inception_2, inception_3])

    if pooling:
        inception_4_1 = tf.nn.max_pool(activated_input_layer, ksize=[1, 3, 3, 1], strides=[1, stride, stride, 1],
                                       padding="SAME")
        inception_4 = relu_conv2d(inception_4_1, [1, 1, input_channels, output_channels_list[6]], stride=1)
        result = tf.concat(3, [result, inception_4])

    total_number_of_channels = reduce(lambda x, y: x+y, output_channels_list[0:7])
    downsampled_result = conv2d(result, [1, 1, total_number_of_channels, output_channels_list[7]])

    scaled_input_layer = scale_residual_input(input_layer, input_channels, output_channels_list[7], stride=stride)
    return residual_connection(scaled_input_layer, downsampled_result)
