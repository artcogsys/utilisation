import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

MAX_POOLING = "max_pooling"
STRIDES = "strides"


class RelationalBlocks:
    def __init__(self, use_relational_conv=False):
        self.weights = []
        self.use_relational_conv = use_relational_conv

    def create_weights(self, name, shape):
        w = tf.get_variable(name, shape=shape, initializer=xavier_initializer())
        self.weights.append(w)
        return w

    def create_weights_with_initializer(self, name, shape, initializer):
        w = tf.get_variable(name, shape=shape, initializer=initializer)
        self.weights.append(w)
        return w

    def conv2d(self, input_layer, weight_shape, stride=1, dilation=1, previous_weights=None):
        with tf.variable_scope("conv2d"):
            _weight = self.create_weights("weights", weight_shape)
            return tf.nn.convolution(input_layer, _weight, strides=[stride, stride], padding="SAME",
                                     dilation_rate=[dilation, dilation]), previous_weights

    def relational_conv2d(self, input_layer, weight_shape, previous_weights=None, stride=1, dilation=1):
        with tf.variable_scope("conv2d"):
            if previous_weights is None:
                _weights = self.create_weights("new_weights", weight_shape)
            else:
                previous_shape = map(lambda x: x.value, previous_weights.get_shape())
                assert previous_shape[2] <= weight_shape[2] and previous_shape[3] <= weight_shape[3]
                if previous_shape[2] < weight_shape[2]:
                    new_inputs = self.create_weights("new_inputs",
                                                     [weight_shape[0],
                                                      weight_shape[1],
                                                      weight_shape[2] - previous_shape[2],
                                                      weight_shape[3]])
                    previous_weights = tf.concat([previous_weights, new_inputs], axis=2)
                if previous_shape[3] < weight_shape[3]:
                    new_outputs = self.create_weights("new_outputs",
                                                      [weight_shape[0],
                                                       weight_shape[1],
                                                       weight_shape[2],
                                                       weight_shape[3] - previous_shape[3]])
                    previous_weights = tf.concat([previous_weights, new_outputs], axis=3)
                _weights = previous_weights
            return tf.nn.convolution(input_layer, _weights, strides=[stride, stride], padding="SAME",
                                     dilation_rate=[dilation, dilation]), _weights

    def relu_conv2d(self, input_layer, weight_shape, stride=1, mcrelu=False):
        with tf.variable_scope("relu_conv2d"):
            l = self.conv2d(input_layer, weight_shape, stride=stride)
            return self.normalized_relu_activation(l, negative_concatenation=mcrelu)

    def normalized_relu_activation(self, input_layer, negative_concatenation=False):
        if negative_concatenation:
            input_layer = tf.concat([input_layer, -1 * input_layer], 3)

        input_layer = self.batch_normalization(input_layer)
        input_layer = self.add_bias(input_layer)
        return tf.nn.relu(input_layer)

    @staticmethod
    def batch_normalization(input_layer):
        with tf.variable_scope("batch_norm"):
            bn = tf.contrib.layers.batch_norm(input_layer, fused=True, trainable=False, scale=True)
            return bn

    def add_bias(self, layer):
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
                                         weight_shape=[1, 1, 2 * output_channels_list[1], output_channels_list[2]],
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

    @staticmethod
    def global_avg_pool(input_layer):  # resnet method
        assert input_layer.get_shape().ndims == 4
        return tf.reduce_mean(input_layer, [1, 2])

    def fc(self, input_layer, input_channels, output_channels):
        weights = self.create_weights('fc_weights', [input_channels, output_channels])
        flattened_input = tf.reshape(input_layer, [-1, input_channels])
        pre_bias = tf.matmul(flattened_input, weights)
        result = self.add_bias(pre_bias)

        return result

    def relu_fc(self, input_layer, input_channels, output_channels):
        return tf.nn.relu(self.fc(input_layer, input_channels, output_channels))

    def relational_block(self, layer, down_sampling, kernel_sizes, strides, dilation_rates,
                         input_channel, output_channels, previous_weights):

        if down_sampling is MAX_POOLING:
            layer = tf.nn.max_pool(layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

        if input_channel != output_channels[-1]:
            layer = self.normalized_relu_activation(layer)
            with tf.variable_scope("identity_mapping"):
                residual = tf.pad(layer, [[0, 0], [0, 0], [0, 0], [0, output_channels[-1] - input_channel]])
        else:
            residual = layer
            layer = self.normalized_relu_activation(layer)

        for c in range(0, len(kernel_sizes)):
            with tf.variable_scope("inner_convolution_%d" % c):
                if c is not 0:
                    layer = self.normalized_relu_activation(layer)
                if self.use_relational_conv:
                    layer, previous_weights = self.relational_conv2d(layer,
                                                                     [kernel_sizes[c], kernel_sizes[c], input_channel,
                                                                      output_channels[c]],
                                                                     previous_weights=previous_weights,
                                                                     stride=strides[c],
                                                                     dilation=dilation_rates[c])
                else:
                    layer, previous_weights = self.conv2d(layer,
                                                          [kernel_sizes[c], kernel_sizes[c], input_channel,
                                                           output_channels[c]],
                                                          previous_weights=previous_weights,
                                                          stride=strides[c],
                                                          dilation=dilation_rates[c])
                input_channel = output_channels[c]

        layer = layer + residual
        return layer, previous_weights
