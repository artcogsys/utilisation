import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


def create_weights(name, shape):
    return tf.get_variable(name, shape=shape, initializer=xavier_initializer())


def conv2d(input_layer, filter_shape, stride=1):
    filter = create_weights("filter_weights", filter_shape)
    return tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding="SAME")


def relu_conv2d(input_layer, filter_shape, stride=1, mcrelu=False):
    l = conv2d(input_layer, filter_shape, stride=stride)
    return normalized_relu_activation(l, filter_shape[3], negative_concatenation=mcrelu)


def normalized_relu_activation(input_layer, output_size, negative_concatenation=False):
    if negative_concatenation:
        input_layer = tf.concat([input_layer, -1 * input_layer], 3)
        output_size *= 2
    input_layer = batch_normalization(input_layer)
    return tf.nn.relu(input_layer)


def batch_normalization(layer):
    mean = tf.Variable(initial_value=.0, trainable=True)
    variance = tf.Variable(initial_value=1., trainable=True)
    offset = tf.Variable(initial_value=.0, trainable=True)
    scale = tf.Variable(initial_value=1., trainable=True)
    variance_epsilon = tf.Variable(initial_value=1e-7, trainable=True)
    # @TODO: check how to implement caffe's batch_norm.use_global_stats=True
    return tf.nn.batch_normalization(layer, mean, variance, offset, scale, variance_epsilon)


def add_bias(layer, number_of_channels):
    bias = create_weights("bias", number_of_channels)
    return tf.nn.bias_add(layer, bias)


def residual_bottleneck_mcrelu(input_layer, kernel_size, input_channels, output_channels_list,
                               stride=1):
    with tf.variable_scope('input_activation'):
        activated_input_layer = normalized_relu_activation(input_layer, input_channels)
    with tf.variable_scope('bottleneck_1'):
        first = relu_conv2d(activated_input_layer, [1, 1, input_channels, output_channels_list[0]])
    with tf.variable_scope('bottleneck_2'):
        conv = relu_conv2d(first, [kernel_size, kernel_size, output_channels_list[0], output_channels_list[1]],
                           stride=stride, mcrelu=True)
    with tf.variable_scope('bottleneck_3'):
        pre_activation = conv2d(conv, filter_shape=[1, 1, 2 * output_channels_list[1], output_channels_list[2]],
                                stride=1)
    with tf.variable_scope('batch_normalization'):
        scaled_input_layer = scale_residual_input(input_layer, input_channels, output_channels_list[2], stride)
    with tf.variable_scope('residual_connection'):
        output_layer = residual_connection(scaled_input_layer, pre_activation)
    return output_layer


def residual_connection(parent, child):
    return parent + child


def scale_residual_input(input_layer, input_channels, output_channels, stride=1):
    if input_channels != output_channels:
        return conv2d(input_layer, [1, 1, input_channels, output_channels], stride=stride)
    else:
        return input_layer


def residual_inception(input_layer, input_channels, output_channels_list, stride=1, pooling=False):
    with tf.variable_scope('input_activation'):
        activated_input_layer = normalized_relu_activation(input_layer, input_channels)
    with tf.variable_scope('inception_1'):
        inception_1 = relu_conv2d(activated_input_layer, [1, 1, input_channels, output_channels_list[0]], stride=stride)
    with tf.variable_scope('inception_2_1'):
        inception_2_1 = relu_conv2d(activated_input_layer, [1, 1, input_channels, output_channels_list[1]],
                                    stride=stride)
    with tf.variable_scope('inception_2_2'):
        inception_2 = relu_conv2d(inception_2_1, [3, 3, output_channels_list[1], output_channels_list[2]], stride=1)
    with tf.variable_scope('inception_3_1'):
        inception_3_1 = relu_conv2d(activated_input_layer, [1, 1, input_channels, output_channels_list[3]],
                                    stride=stride)
    with tf.variable_scope('inception_3_2'):
        inception_3_2 = relu_conv2d(inception_3_1, [3, 3, output_channels_list[3], output_channels_list[4]], stride=1)
    with tf.variable_scope('inception_3_3'):
        inception_3 = relu_conv2d(inception_3_2, [3, 3, output_channels_list[4], output_channels_list[5]], stride=1)
    with tf.variable_scope('result'):
        result = tf.concat([inception_1, inception_2, inception_3], 3)

    total_number_of_channels = output_channels_list[0] + output_channels_list[2] + output_channels_list[5]

    if pooling:
        with tf.variable_scope('inception_4_1'):
            inception_4_1 = tf.nn.max_pool(activated_input_layer, ksize=[1, 3, 3, 1], strides=[1, stride, stride, 1],
                                           padding="SAME")
            inception_4_1 = batch_normalization(inception_4_1)
        with tf.variable_scope('inception_4_2'):
            inception_4 = relu_conv2d(inception_4_1, [1, 1, input_channels, output_channels_list[6]], stride=1)
        with tf.variable_scope('result_2'):
            result = tf.concat([result, inception_4], 3)
            total_number_of_channels += output_channels_list[6]
    result = batch_normalization(result)
    with tf.variable_scope('inception_downsample'):
        downsampled_result = conv2d(result, [1, 1, total_number_of_channels, output_channels_list[7]], stride=1)
    with tf.variable_scope('batch_normalization'):
        scaled_input_layer = scale_residual_input(input_layer, input_channels, output_channels_list[7], stride=stride)
        scaled_input_layer = batch_normalization(scaled_input_layer)
    with tf.variable_scope('residual_connetion'):
        output_layer = residual_connection(scaled_input_layer, downsampled_result)
    return output_layer


def global_avg_pool(input_layer):  # resnet method
    assert input_layer.get_shape().ndims == 4
    return tf.reduce_mean(input_layer, [1, 2])


def fc(input_layer, input_channels, output_channels):
    weights = create_weights('fc_weights', [input_channels, output_channels])
    shape = tf.shape(input_layer)
    flattened_input = tf.reshape(input_layer, [-1, input_channels])
    pre_bias = tf.matmul(flattened_input, weights)
    result = add_bias(pre_bias, output_channels)
    result = batch_normalization(result)
    new_shape = tf.stack([shape[0], shape[1], shape[2], output_channels])
    return tf.reshape(result, new_shape)


def relu_fc(input_layer, input_channels, output_channels):
    return tf.nn.relu(fc(input_layer, input_channels, output_channels))
