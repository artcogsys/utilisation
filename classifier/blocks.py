import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

MAX_POOLING = "max_pooling"
STRIDES = "strides"


class Blocks:
    def __init__(self):
        self.weights = []

    def create_weights(self, name, shape):
        w = tf.get_variable(name, shape=shape, initializer=xavier_initializer())
        self.weights.append(w)
        return w

    def create_weights_with_initializer(self, name, shape, initializer):
        w = tf.get_variable(name, shape=shape, initializer=initializer)
        self.weights.append(w)
        return w

    def conv2d(self, input_layer, filter_shape, stride=1, dilation=1):
        with tf.variable_scope("conv2d"):
            _filter = self.create_weights("filter_weights", filter_shape)
            return tf.nn.convolution(input_layer, _filter, strides=[stride, stride], padding="SAME",
                                     dilation_rate=[dilation, dilation])

    def relu_conv2d(self, input_layer, filter_shape, stride=1, mcrelu=False):
        with tf.variable_scope("relu_conv2d"):
            l = self.conv2d(input_layer, filter_shape, stride=stride)
            return self.normalized_relu_activation(l, negative_concatenation=mcrelu)

    def normalized_relu_activation(self, input_layer, negative_concatenation=False):
        if negative_concatenation:
            input_layer = tf.concat([input_layer, -1 * input_layer], 3)

        input_layer = self.batch_normalization(input_layer)
        return tf.nn.relu(input_layer)

    @staticmethod
    def batch_normalization(input_layer):
        with tf.variable_scope("batch_norm"):
            bn = tf.contrib.layers.batch_norm(input_layer, fused=True, trainable=False, scale=True)
            return bn

    def add_bias(self, layer, number_of_channels=None):
        return tf.contrib.layers.bias_add(layer)

    def residual_bottleneck_mcrelu(self, input_layer, kernel_size, input_channels, output_channels_list,
                                   stride=1):
        with tf.variable_scope('input_activation'):
            activated_input_layer = self.normalized_relu_activation(input_layer)
        with tf.variable_scope('bottleneck_1'):
            first = self.relu_conv2d(activated_input_layer, [1, 1, input_channels, output_channels_list[0]])
        with tf.variable_scope('bottleneck_2'):
            conv = self.relu_conv2d(first, [kernel_size, kernel_size, output_channels_list[0], output_channels_list[1]],
                                    stride=stride, mcrelu=True)
        with tf.variable_scope('bottleneck_3'):
            pre_activation = self.conv2d(conv,
                                         filter_shape=[1, 1, 2 * output_channels_list[1], output_channels_list[2]],
                                         stride=1)
        with tf.variable_scope('batch_normalization'):
            scaled_input_layer = self.scale_residual_input(input_layer, input_channels, output_channels_list[2], stride)
        with tf.variable_scope('residual_connection'):
            output_layer = self.residual_connection(scaled_input_layer, pre_activation)
        return output_layer

    @staticmethod
    def residual_connection(parent, child):
        return parent + child

    def scale_residual_input(self, input_layer, input_channels, output_channels, stride=1):
        if input_channels != output_channels:
            return self.conv2d(input_layer, [1, 1, input_channels, output_channels], stride=stride)
        else:
            return input_layer

    def residual_inception(self, input_layer, input_channels, output_channels_list, stride=1, pooling=False):
        with tf.variable_scope('input_activation'):
            activated_input_layer = self.normalized_relu_activation(input_layer)
        with tf.variable_scope('inception_1'):
            inception_1 = self.relu_conv2d(activated_input_layer, [1, 1, input_channels, output_channels_list[0]],
                                           stride=stride)
        with tf.variable_scope('inception_2_1'):
            inception_2_1 = self.relu_conv2d(activated_input_layer, [1, 1, input_channels, output_channels_list[1]],
                                             stride=stride)
        with tf.variable_scope('inception_2_2'):
            inception_2 = self.relu_conv2d(inception_2_1, [3, 3, output_channels_list[1], output_channels_list[2]],
                                           stride=1)
        with tf.variable_scope('inception_3_1'):
            inception_3_1 = self.relu_conv2d(activated_input_layer, [1, 1, input_channels, output_channels_list[3]],
                                             stride=stride)
        with tf.variable_scope('inception_3_2'):
            inception_3_2 = self.relu_conv2d(inception_3_1, [3, 3, output_channels_list[3], output_channels_list[4]],
                                             stride=1)
        with tf.variable_scope('inception_3_3'):
            inception_3 = self.relu_conv2d(inception_3_2, [3, 3, output_channels_list[4], output_channels_list[5]],
                                           stride=1)
        with tf.variable_scope('result'):
            result = tf.concat([inception_1, inception_2, inception_3], 3)

        total_number_of_channels = output_channels_list[0] + output_channels_list[2] + output_channels_list[5]

        if pooling:
            with tf.variable_scope('inception_4_1'):
                inception_4_1 = tf.nn.max_pool(activated_input_layer, ksize=[1, 3, 3, 1],
                                               strides=[1, stride, stride, 1],
                                               padding="SAME")
                inception_4_1 = self.batch_normalization(inception_4_1)
            with tf.variable_scope('inception_4_2'):
                inception_4 = self.relu_conv2d(inception_4_1, [1, 1, input_channels, output_channels_list[6]], stride=1)
            with tf.variable_scope('result_2'):
                result = tf.concat([result, inception_4], 3)
                total_number_of_channels += output_channels_list[6]
        result = self.batch_normalization(result)
        with tf.variable_scope('inception_downsample'):
            downsampled_result = self.conv2d(result, [1, 1, total_number_of_channels, output_channels_list[7]],
                                             stride=1)
        with tf.variable_scope('batch_normalization'):
            scaled_input_layer = self.scale_residual_input(input_layer, input_channels, output_channels_list[7],
                                                           stride=stride)
            scaled_input_layer = self.batch_normalization(scaled_input_layer)
        with tf.variable_scope('residual_connection'):
            output_layer = self.residual_connection(scaled_input_layer, downsampled_result)
        return output_layer

    @staticmethod
    def global_avg_pool(input_layer):  # resnet method
        assert input_layer.get_shape().ndims == 4
        return tf.reduce_mean(input_layer, [1, 2])

    def fc(self, input_layer, input_channels, output_channels):
        weights = self.create_weights('fc_weights', [input_channels, output_channels])
        flattened_input = tf.reshape(input_layer, [-1, input_channels])
        pre_bias = tf.matmul(flattened_input, weights)
        result = self.add_bias(pre_bias, output_channels)

        return result

    def relu_fc(self, input_layer, input_channels, output_channels):
        return tf.nn.relu(self.fc(input_layer, input_channels, output_channels))

    def block(self, layer, down_sampling, kernel_sizes, strides, dilation_rates, input_channel, output_channels):
        if down_sampling is MAX_POOLING:
            layer = tf.nn.max_pool(layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

        if input_channel != output_channels[-1]:
            layer = self.normalized_relu_activation(layer)
            with tf.variable_scope("identity_mapping"):
                residual = self.conv2d(layer,
                                       filter_shape=[1, 1, input_channel, output_channels[-1]],
                                       stride=2 if down_sampling is STRIDES else 1)
        else:
            residual = layer
            layer = self.normalized_relu_activation(layer)

        for c in range(0, len(kernel_sizes)):
            with tf.variable_scope("inner_convolution_%d" % c):
                if c is not 0:
                    layer = self.normalized_relu_activation(layer)
                layer = self.conv2d(layer, [kernel_sizes[c], kernel_sizes[c], input_channel, output_channels[c]],
                                    stride=strides[c], dilation=dilation_rates[c])
                input_channel = output_channels[c]

        layer = layer + residual
        return layer
