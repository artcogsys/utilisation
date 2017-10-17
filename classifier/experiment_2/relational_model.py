import tensorflow as tf
import data as cifar10

from relational_blocks import RelationalBlocks, MAX_POOLING


class RelationalResNet:
    def __init__(self, batch_size, num_output_classes,
                 placeholder_inputs=False, relational=False):
        self.num_output_classes = num_output_classes

        self.batch_size = batch_size

        self.blocks = RelationalBlocks(use_relational_conv=relational)
        self.do_validate = tf.placeholder(dtype=tf.bool, shape=None)

        if not placeholder_inputs:
            with tf.device("/cpu:0"):
                self.training_input, self.training_label = cifar10.distorted_inputs("data/cifar-10-batches-bin",
                                                                                    self.batch_size)
                self.valid_images, self.valid_label = cifar10.inputs(True, "data/cifar-10-batches-bin",
                                                                     self.batch_size)
                self.input, self.label = tf.cond(self.do_validate,
                                                 lambda: (self.valid_images, self.valid_label),
                                                 lambda: (self.training_input, self.training_label))
        else:
            self.input = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None, None, 3))
            self.label = tf.placeholder(dtype=tf.int32, shape=(self.batch_size))

        m = 1

        with tf.variable_scope("conv_1"):
            conv_1, _ = self.blocks.conv2d(self.input, [3, 3, 3, 16 * m])

        with tf.variable_scope("conv_2_1"):
            conv_2_1, _w = self.blocks.relational_block(conv_1,
                                                        down_sampling=None,
                                                        kernel_sizes=[3, 3],
                                                        strides=[1, 1],
                                                        dilation_rates=[1, 1],
                                                        input_channel=16 * m,
                                                        output_channels=[32 * m, 32 * m],
                                                        previous_weights=None)
        with tf.variable_scope("conv_2_2"):
            conv_2_2, _w = self.blocks.relational_block(conv_2_1,
                                                        down_sampling=None,
                                                        kernel_sizes=[3, 3],
                                                        strides=[1, 1],
                                                        dilation_rates=[1, 1],
                                                        input_channel=32 * m,
                                                        output_channels=[40 * m, 40 * m],
                                                        previous_weights=None)
        with tf.variable_scope("conv_2_3"):
            conv_2_3, _w = self.blocks.relational_block(conv_2_2,
                                                        down_sampling=None,
                                                        kernel_sizes=[3, 3],
                                                        strides=[1, 1],
                                                        dilation_rates=[1, 1],
                                                        input_channel=40 * m,
                                                        output_channels=[48 * m, 48 * m],
                                                        previous_weights=None)

        with tf.variable_scope("conv_3_1"):
            conv_3_1, _w = self.blocks.relational_block(conv_2_3,
                                                        down_sampling=MAX_POOLING,
                                                        kernel_sizes=[3, 3],
                                                        strides=[1, 1],
                                                        dilation_rates=[1, 1],
                                                        input_channel=48 * m,
                                                        output_channels=[48 * m, 48 * m],
                                                        previous_weights=_w)

        with tf.variable_scope("conv_3_2"):
            conv_3_2, _w = self.blocks.relational_block(conv_3_1,
                                                        down_sampling=None,
                                                        kernel_sizes=[3, 3],
                                                        strides=[1, 1],
                                                        dilation_rates=[1, 1],
                                                        input_channel=48 * m,
                                                        output_channels=[64 * m, 64 * m],
                                                        previous_weights=_w)

        with tf.variable_scope("conv_3_3"):
            conv_3_3, _w = self.blocks.relational_block(conv_3_2,
                                                        down_sampling=None,
                                                        kernel_sizes=[3, 3],
                                                        strides=[1, 1],
                                                        dilation_rates=[1, 1],
                                                        input_channel=64 * m,
                                                        output_channels=[80 * m, 80 * m],
                                                        previous_weights=_w)

        with tf.variable_scope("conv_4_1"):
            conv_4_1, _w = self.blocks.relational_block(conv_3_3,
                                                        down_sampling=MAX_POOLING,
                                                        kernel_sizes=[3, 3],
                                                        strides=[1, 1],
                                                        dilation_rates=[1, 1],
                                                        input_channel=80 * m,
                                                        output_channels=[80 * m, 80 * m],
                                                        previous_weights=_w)

        with tf.variable_scope("conv_4_2"):
            conv_4_2, _w = self.blocks.relational_block(conv_4_1,
                                                        down_sampling=None,
                                                        kernel_sizes=[3, 3],
                                                        strides=[1, 1],
                                                        dilation_rates=[1, 1],
                                                        input_channel=80 * m,
                                                        output_channels=[104 * m, 104 * m],
                                                        previous_weights=_w)
        with tf.variable_scope("conv_4_3"):
            conv_4_3, _w = self.blocks.relational_block(conv_4_2,
                                                        down_sampling=None,
                                                        kernel_sizes=[3, 3],
                                                        strides=[1, 1],
                                                        dilation_rates=[1, 1],
                                                        input_channel=104 * m,
                                                        output_channels=[128 * m, 128 * m],
                                                        previous_weights=_w)

        with tf.variable_scope("global_average_pooling"):
            # global average pooling
            global_avg = tf.reduce_mean(conv_4_3, [1, 2])
        with tf.variable_scope("output"):
            self.logits = self.blocks.fc(global_avg,
                                         input_channels=128 * m,
                                         output_channels=10)
            self.freeze_layer = self.logits

        with tf.variable_scope("decay"):
            cost = []
            for weight in self.blocks.weights:
                cost.append(tf.nn.l2_loss(weight))

            decay = 0.001 * tf.reduce_sum(cost)
            tf.summary.scalar('decay', decay)

        self.one_hot_truth = tf.squeeze(tf.one_hot(self.label, 10))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.one_hot_truth)
        self.loss = tf.reduce_mean(cross_entropy)
        self.loss = self.loss + decay
        tf.add_to_collection('losses', self.loss)
        tf.summary.scalar('loss_total', self.loss)

        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), self.label)
            correct_prediction_2 = tf.nn.in_top_k(self.logits, self.label, 5, name=None)
            self.top_1_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            top_5_accuracy = tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))
            tf.summary.scalar('accuracy_top1', self.top_1_accuracy)
            tf.summary.scalar('accuracy_top5', top_5_accuracy)

        with tf.variable_scope('train'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            boundaries = [20000, 40000]
            values = [0.1, 0.01, 0.001]

            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
            tf.summary.scalar('learning_rate', self.learning_rate)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)
