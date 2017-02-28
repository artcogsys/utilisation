import tensorflow as tf

from blocks import relu_conv2d, residual_bottleneck_mcrelu, residual_inception, create_weights, fc, \
    relu_fc, batch_normalization
from pipeline import get_pipeline


class PVANet():
    def __init__(self, batch_size=2, image_dimensions=(256, 192), num_output_classes=3148, training=False):
        self.training = training
        self.num_output_classes = num_output_classes
        self.graph = tf.Graph()

        self.batch_size = batch_size
        self.image_dimensions = image_dimensions
        self.make_graph()

    def make_graph(self):
        with self.graph.as_default():
            if self.training:
                self.input, self.truth = get_pipeline(batch_size=self.batch_size,
                                                      image_dimensions=self.image_dimensions)
            else:
                self.input = tf.placeholder(tf.float32,
                                            shape=(None, self.image_dimensions[0], self.image_dimensions[1], 3))
                self.truth = tf.placeholder(tf.float32,
                                            shape=(None, self.image_dimensions[0], self.image_dimensions[1], 1))
                # tf.summary.image('input image', self.input)
                # tf.summary.image('truth image', self.truth)
            with tf.variable_scope('pre_processing'):
                pre_processed_input = batch_normalization(self.input)
            with tf.variable_scope('conv_1'):
                conv_1_1 = relu_conv2d(pre_processed_input, [7, 7, 3, 16], stride=2, mcrelu=True)
            with tf.variable_scope('pool_1'):
                pool_1_1 = tf.nn.max_pool(conv_1_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

            with tf.variable_scope('conv_2_1'):
                conv_2_1 = residual_bottleneck_mcrelu(pool_1_1, kernel_size=3, input_channels=32,
                                                      output_channels_list=[24, 24, 64])
            with tf.variable_scope('conv_2_2'):
                conv_2_2 = residual_bottleneck_mcrelu(conv_2_1, kernel_size=3, input_channels=64,
                                                      output_channels_list=[24, 24, 64])
            with tf.variable_scope('conv_2_3'):
                conv_2_3 = residual_bottleneck_mcrelu(conv_2_2, kernel_size=3, input_channels=64,
                                                      output_channels_list=[24, 24, 64])
            with tf.variable_scope('conv_3_1'):
                conv_3_1 = residual_bottleneck_mcrelu(conv_2_3, kernel_size=3, input_channels=64,
                                                      output_channels_list=[48, 48, 128], stride=2)

            with tf.variable_scope('conv_3_2'):
                conv_3_2 = residual_bottleneck_mcrelu(conv_3_1, kernel_size=3, input_channels=128,
                                                      output_channels_list=[48, 48, 128])
            with tf.variable_scope('conv_3_3'):
                conv_3_3 = residual_bottleneck_mcrelu(conv_3_2, kernel_size=3, input_channels=128,
                                                      output_channels_list=[48, 48, 128])
            with tf.variable_scope('conv_3_4'):
                conv_3_4 = residual_bottleneck_mcrelu(conv_3_3, kernel_size=3, input_channels=128,
                                                      output_channels_list=[48, 48, 128])

            with tf.variable_scope('conv_4_1'):
                conv_4_1 = residual_inception(conv_3_4, 128, [64, 48, 128, 24, 48, 48, 128, 256], stride=2,
                                              pooling=True)

            with tf.variable_scope('conv_4_2'):
                conv_4_2 = residual_inception(conv_4_1, 256, [64, 64, 128, 24, 48, 48, 0, 256])
            with tf.variable_scope('conv_4_3'):
                conv_4_3 = residual_inception(conv_4_2, 256, [64, 64, 128, 24, 48, 48, 0, 256])
            with tf.variable_scope('conv_4_4'):
                conv_4_4 = residual_inception(conv_4_3, 256, [64, 64, 128, 24, 48, 48, 0, 256])

            with tf.variable_scope('conv_5_1'):
                conv_5_1 = residual_inception(conv_4_4, 256, [64, 96, 192, 32, 64, 64, 128, 384], stride=2,
                                              pooling=True)

            with tf.variable_scope('conv_5_2'):
                conv_5_2 = residual_inception(conv_5_1, 384, [64, 96, 192, 32, 64, 64, 0, 384])
            with tf.variable_scope('conv_5_3'):
                conv_5_3 = residual_inception(conv_5_2, 384, [64, 96, 192, 32, 64, 64, 0, 384])
            with tf.variable_scope('conv_5_4'):
                conv_5_4 = residual_inception(conv_5_3, 384, [64, 96, 192, 32, 64, 64, 0, 384])

            with tf.variable_scope('conv_concat'):
                conv_3_4_scaled = tf.nn.max_pool(conv_3_4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
                deconvolution_filter = create_weights('deconvolution_filter', [4, 4, 384, 384])
                conv_5_4_scaled = tf.nn.conv2d_transpose(conv_5_4, deconvolution_filter,
                                                         [self.batch_size,
                                                          tf.shape(conv_5_4)[1] * 2,
                                                          tf.shape(conv_5_4)[2] * 2,
                                                          384], strides=[1, 2, 2, 1])
                conv_concat = tf.concat([conv_3_4_scaled, conv_5_4_scaled, conv_4_4], 3)
            with tf.variable_scope('feature_scale_1'):
                conv_concat = relu_fc(conv_concat, 768, self.num_output_classes)
            with tf.variable_scope('feature_scale_2'):
                self.results = fc(conv_concat, self.num_output_classes, self.num_output_classes)

            with tf.variable_scope('error'):
                one_hot_encoded_truth = tf.one_hot(tf.cast(self.truth, tf.int32), self.num_output_classes)
                one_hot_encoded_truth = tf.squeeze(one_hot_encoded_truth)

            with tf.variable_scope('gradient'):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdadeltaOptimizer().minimize(self.loss, global_step=self.global_step)
            tf.summary.scalar("loss", self.loss)
