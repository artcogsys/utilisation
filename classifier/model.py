import tensorflow as tf

from blocks import relu_conv2d, residual_bottleneck_mcrelu, residual_inception, create_weights, concat_features, \
    max_pool, conv2d, add_bias
from clean_pipeline import get_evaluation_pipeline
from pipeline import get_pipeline
from settings import DATA_FORMAT


class PVANet:
    def __init__(self, batch_size=2, image_dimensions=(256, 192), num_output_classes=3148, training=False,
                 evaluation=False, class_embeddings=None):
        self.training = training
        self.evaluation = evaluation
        self.num_output_classes = num_output_classes
        self.graph = tf.Graph()

        self.batch_size = batch_size
        self.image_dimensions = image_dimensions
        self.class_embeddings = class_embeddings

        with self.graph.as_default():
            if self.training:
                with tf.device('/cpu:0'):
                    self.input, self.truth = get_pipeline(batch_size=self.batch_size,
                                                          image_dimensions=self.image_dimensions,
                                                          class_embeddings=self.class_embeddings)
                # tf.summary.image('input image', self.input)
            elif self.evaluation:
                self.input, self.truth = get_evaluation_pipeline()
            else:
                if DATA_FORMAT is "NCHW":
                    self.input = tf.placeholder(tf.float32,
                                                shape=(None, 3, self.image_dimensions[0], self.image_dimensions[1]))
                else:
                    self.input = tf.placeholder(tf.float32,
                                                shape=(None, self.image_dimensions[0], self.image_dimensions[1], 3))
                self.truth = tf.placeholder(tf.int32, shape=(None, self.image_dimensions[0], self.image_dimensions[1]))
            with tf.variable_scope('conv_1'):
                conv_1_1 = relu_conv2d(self.input, [7, 7, 3, 16], stride=2, mcrelu=True)
                # tf.summary.histogram("conv_1_1", conv_1_1)
            with tf.variable_scope('pool_1'):
                pool_1_1 = max_pool(conv_1_1,
                                    kernel_size=3,
                                    stride=2,
                                    padding="SAME")
                # tf.summary.histogram("pool_1_1", pool_1_1)
            with tf.variable_scope('conv_2_1'):
                conv_2_1 = residual_bottleneck_mcrelu(pool_1_1, kernel_size=3, input_channels=32,
                                                      output_channels_list=[24, 24, 64])
                # tf.summary.histogram("conv_2_1", conv_2_1)
            with tf.variable_scope('conv_2_2'):
                conv_2_2 = residual_bottleneck_mcrelu(conv_2_1, kernel_size=3, input_channels=64,
                                                      output_channels_list=[24, 24, 64])
                # tf.summary.histogram("conv_2_2", conv_2_2)
            with tf.variable_scope('conv_2_3'):
                conv_2_3 = residual_bottleneck_mcrelu(conv_2_2, kernel_size=3, input_channels=64,
                                                      output_channels_list=[24, 24, 64])
                # tf.summary.histogram("conv_2_3", conv_2_3)
            with tf.variable_scope('conv_3_1'):
                conv_3_1 = residual_bottleneck_mcrelu(conv_2_3, kernel_size=3, input_channels=64,
                                                      output_channels_list=[48, 48, 128], stride=2)
                # tf.summary.histogram("conv_3_1", conv_3_1)
            with tf.variable_scope('conv_3_2'):
                conv_3_2 = residual_bottleneck_mcrelu(conv_3_1, kernel_size=3, input_channels=128,
                                                      output_channels_list=[48, 48, 128])
                # tf.summary.histogram("conv_3_2", conv_3_2)
            with tf.variable_scope('conv_3_3'):
                conv_3_3 = residual_bottleneck_mcrelu(conv_3_2, kernel_size=3, input_channels=128,
                                                      output_channels_list=[48, 48, 128])
                # tf.summary.histogram("conv_3_3", conv_3_3)
            with tf.variable_scope('conv_3_4'):
                conv_3_4 = residual_bottleneck_mcrelu(conv_3_3, kernel_size=3, input_channels=128,
                                                      output_channels_list=[48, 48, 128])
                # tf.summary.histogram("conv_3_4", conv_3_4)
            with tf.variable_scope('conv_4_1'):
                conv_4_1 = residual_inception(conv_3_4, 128, [64, 48, 128, 24, 48, 48, 128, 256], stride=2,
                                              pooling=True)
                # tf.summary.histogram("conv_4_1", conv_4_1)
            with tf.variable_scope('conv_4_2'):
                conv_4_2 = residual_inception(conv_4_1, 256, [64, 64, 128, 24, 48, 48, 0, 256])
                # tf.summary.histogram("conv_4_2", conv_4_2)
            with tf.variable_scope('conv_4_3'):
                conv_4_3 = residual_inception(conv_4_2, 256, [64, 64, 128, 24, 48, 48, 0, 256])
                # tf.summary.histogram("conv_4_3", conv_4_3)
            with tf.variable_scope('conv_4_4'):
                conv_4_4 = residual_inception(conv_4_3, 256, [64, 64, 128, 24, 48, 48, 0, 256])
                # tf.summary.histogram("conv_4_4", conv_4_4)
            with tf.variable_scope('conv_5_1'):
                conv_5_1 = residual_inception(conv_4_4, 256, [64, 96, 192, 32, 64, 64, 128, 384], stride=2,
                                              pooling=True)
                # tf.summary.histogram("conv_5_1", conv_5_1)
            with tf.variable_scope('conv_5_2'):
                conv_5_2 = residual_inception(conv_5_1, 384, [64, 96, 192, 32, 64, 64, 0, 384])
                # tf.summary.histogram("conv_5_2", conv_5_2)
            with tf.variable_scope('conv_5_3'):
                conv_5_3 = residual_inception(conv_5_2, 384, [64, 96, 192, 32, 64, 64, 0, 384])
                # tf.summary.histogram("conv_5_3", conv_5_3)
            with tf.variable_scope('conv_5_4'):
                conv_5_4 = residual_inception(conv_5_3, 384, [64, 96, 192, 32, 64, 64, 0, 384])
                # tf.summary.histogram("conv_5_4", conv_5_4)
            with tf.variable_scope('conv_concat'):
                conv_3_4_scaled = max_pool(conv_3_4, kernel_size=3, stride=2, padding="SAME")
                deconvolution_filter = create_weights('deconvolution_filter', [4, 4, 384, 384])
                output_shape = tf.unstack(tf.shape(conv_5_4))

                if DATA_FORMAT is "NCHW":
                    # output_shape = [output_shape[0], output_shape[3], output_shape[1], output_shape[2]]
                    output_shape[2] *= 2
                    output_shape[3] *= 2
                    strides = [1, 1, 2, 2]
                else:
                    output_shape[1] *= 2
                    output_shape[2] *= 2
                    strides = [1, 2, 2, 1]
                output_shape = tf.parallel_stack(output_shape)
                conv_5_4_scaled = tf.nn.conv2d_transpose(conv_5_4,
                                                         deconvolution_filter,
                                                         output_shape,
                                                         strides=strides,
                                                         data_format=DATA_FORMAT)
                conv_concat = tf.nn.relu(concat_features([conv_3_4_scaled, conv_5_4_scaled, conv_4_4]))
                # tf.summary.histogram("conv_concat", conv_concat)
            with tf.variable_scope('feature_scale_1'):
                feature_scale_1 = relu_conv2d(conv_concat, [1, 1, 768, self.num_output_classes * 2])
                # tf.summary.histogram("feature_scale_1", feature_scale_1)

            with tf.variable_scope('feature_scale_2'):
                feature_scale_2 = conv2d(feature_scale_1, [1, 1, self.num_output_classes * 2, self.num_output_classes])
                feature_scale_2 = add_bias(feature_scale_2, self.num_output_classes)

            if DATA_FORMAT is "NCHW":
                feature_scale_2 = tf.transpose(feature_scale_2, [0, 2, 3, 1])
            # DATA_FORMAT is "NHCW" after this moment

            with tf.variable_scope('classification_error'):
                self.logits = feature_scale_2
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                          labels=self.truth))

                # self.loss = tf.reduce_mean(tf.square(self.results - regularized_truth))
                # self.loss = tf.Print(self.loss, [self.loss], 'loss')
            # tf.summary.histogram('output histogram', self.results)
            # tf.summary.histogram('truth histogram', regularized_truth)
            tf.summary.scalar("loss", self.loss)
            with tf.variable_scope('classification_gradient'):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                # self.optimizer = tf.train.AdadeltaOptimizer().minimize(self.loss, global_step=self.global_step)
                # self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(self.loss,
                #                                                                         global_step=
                #                                                                         self.global_step)
                # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.05).minimize(self.loss,
                #                                                                                         global_step=
                #                                                                                         self.global_step)
                self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)
