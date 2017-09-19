import tensorflow as tf

from blocks import Blocks, MAX_POOLING

from pipeline import get_training_pipeline, get_validation_pipeline


class PVANet:
    def __init__(self, batch_size, image_size, num_output_classes, training=False,
                 evaluation=False):
        self.training = training
        self.evaluation = evaluation
        self.num_output_classes = num_output_classes
        self.graph = tf.Graph()

        self.batch_size = batch_size
        self.image_size = image_size
        self.blocks = Blocks()

        with self.graph.as_default():
            with tf.device("/cpu:0"):
                self.is_training = tf.placeholder_with_default(tf.constant(True), shape=[])

                training_image_id, training_image, training_label = get_training_pipeline(image_size=self.image_size)
                # self.all_losses_by_id = tf.Variable(tf.multiply(tf.ones([20214]), 10),
                #                                     dtype=tf.float32)
                # self.mean_loss = tf.reduce_mean(self.all_losses_by_id)
                # self.keep_condition_low_loss = tf.greater(self.all_losses_by_id[training_image_id],
                #                                           self.mean_loss * 0.9)
                # self.keep_condition_labels_exist = tf.greater(tf.reduce_sum(training_label), 0)
                # self.keep_condition = tf.logical_and(self.keep_condition_labels_exist, self.keep_condition_low_loss)
                # tf.summary.scalar('mean_loss', self.mean_loss)
                #
                # total_skipped = tf.Variable(tf.zeros([1]))
                # tf.summary.scalar('total_skipped_samples', total_skipped[0])
                # update_total_skipped = tf.cond(self.keep_condition_low_loss,
                #                                lambda: total_skipped,
                #                                lambda: tf.scatter_add(total_skipped, [0], [1]))
                # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_total_skipped)

                validation_image_id, validation_image, validation_label = get_validation_pipeline(
                    image_size=self.image_size)

                training_image_ids, training_images, training_labels = tf.train.batch(
                    [training_image_id, training_image, training_label],
                    # keep_input=self.keep_condition,
                    batch_size=self.batch_size,
                    shapes=[[], [image_size[0], image_size[1], 3], [image_size[0] / 8, image_size[1] / 8]],
                    capacity=3 * batch_size,
                    num_threads=4)

                validation_image_ids, validation_images, validation_labels = tf.train.batch(
                    [validation_image_id, validation_image, validation_label],
                    batch_size=self.batch_size,
                    shapes=[[], [image_size[0], image_size[1], 3], [image_size[0] / 8, image_size[1] / 8]],
                    capacity=3 * batch_size,
                    num_threads=2)

                self.input_id, self.input, self.truth = tf.cond(self.is_training,
                                                                lambda: [training_image_ids, training_images,
                                                                         training_labels],
                                                                lambda: [validation_image_ids, validation_images,
                                                                         validation_labels])

            with tf.variable_scope("conv_1"):
                conv_1 = self.blocks.conv2d(self.input, [3, 3, 3, 64])

            with tf.variable_scope("conv_2_1"):
                conv_2_1 = self.blocks.block(conv_1,
                                             down_sampling=MAX_POOLING,
                                             kernel_sizes=[3, 3],
                                             strides=[1, 1],
                                             dilation_rates=[1, 1],
                                             input_channel=64,
                                             output_channels=[128, 128])

            with tf.variable_scope("conv_3_1"):
                conv_3_1 = self.blocks.block(conv_2_1,
                                             down_sampling=MAX_POOLING,
                                             kernel_sizes=[3, 3],
                                             strides=[1, 1],
                                             dilation_rates=[1, 1],
                                             input_channel=128,
                                             output_channels=[256, 256])

            with tf.variable_scope("conv_4_1"):
                conv_4_1 = self.blocks.block(conv_3_1,
                                             down_sampling=MAX_POOLING,
                                             kernel_sizes=[3, 3],
                                             strides=[1, 1],
                                             dilation_rates=[1, 1],
                                             input_channel=256,
                                             output_channels=[512, 512])

            with tf.variable_scope("conv_5_1"):
                conv_5_1 = self.blocks.block(conv_4_1,
                                             down_sampling=None,
                                             kernel_sizes=[3, 3],
                                             strides=[1, 1],
                                             dilation_rates=[1, 2],
                                             input_channel=512,
                                             output_channels=[512, 512])

            with tf.variable_scope("conv_5_2"):
                conv_5_2 = self.blocks.block(conv_5_1,
                                             down_sampling=None,
                                             kernel_sizes=[3, 3],
                                             strides=[1, 1],
                                             dilation_rates=[2, 2],
                                             input_channel=512,
                                             output_channels=[512, 512])

            with tf.variable_scope("conv_5_3"):
                conv_5_3 = self.blocks.block(conv_5_2,
                                             down_sampling=None,
                                             kernel_sizes=[3, 3],
                                             strides=[1, 1],
                                             dilation_rates=[2, 2],
                                             input_channel=512,
                                             output_channels=[512, 512])

            with tf.variable_scope("conv_6"):
                conv_6 = self.blocks.block(conv_5_3,
                                           down_sampling=None,
                                           kernel_sizes=[1, 3, 1],
                                           strides=[1, 1, 1],
                                           dilation_rates=[1, 4, 1],
                                           input_channel=512,
                                           output_channels=[256, 256, 512])

            with tf.variable_scope("conv_7"):
                conv_7 = self.blocks.block(conv_6,
                                           down_sampling=None,
                                           kernel_sizes=[1, 3, 1],
                                           strides=[1, 1, 1],
                                           dilation_rates=[1, 4, 1],
                                           input_channel=512,
                                           output_channels=[256, 256, 512])

            with tf.variable_scope("classifier_1"):
                classifier_1 = self.blocks.normalized_relu_activation(conv_7)
                classifier_1 = self.blocks.conv2d(classifier_1, [3, 3, 512, 512], stride=1, dilation=12)

            with tf.variable_scope("classifier_2"):
                classifier_2 = tf.nn.relu(classifier_1)
                classifier_2 = self.blocks.conv2d(classifier_2, [3, 3, 512, self.num_output_classes], stride=1,
                                                  dilation=12)
            with tf.variable_scope('classification_error'):
                self.logits = classifier_2
                _max_logit_value, self.class_ids = tf.nn.top_k(self.logits)
                self.results = tf.nn.softmax(self.logits)
                one_hot_truth = tf.one_hot(self.truth, self.num_output_classes, on_value=0.85, off_value=0.001)

                self.per_pixel_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                              labels=one_hot_truth)

                non_zero_indices = tf.cast(tf.not_equal(self.truth, 0), dtype=tf.float32)

                valid_losses = tf.multiply(self.per_pixel_loss, non_zero_indices)

                sce_loss = tf.reduce_sum(valid_losses) / tf.reduce_sum(non_zero_indices)


            with tf.variable_scope('mean_iou'):
                one_hot_truth = tf.one_hot(self.truth, self.num_output_classes)
                intersection = tf.multiply(self.results, one_hot_truth)
                union = tf.subtract(tf.add(self.results, one_hot_truth), intersection)
                iou = tf.where(
                    tf.greater(union, 0),
                    tf.div(intersection, tf.where(
                        tf.equal(union, 0),
                        tf.ones_like(union), union)),
                    tf.zeros_like(intersection))
                self.mean_iou = tf.reduce_mean(tf.reduce_sum(iou[:, :, :, 0:], axis=3))
                inverse_mean_iou = tf.subtract(tf.constant(1, dtype=tf.float32), self.mean_iou)

            with tf.variable_scope('pixel_accuracy'):
                labeled_pixels = tf.cast(tf.greater(self.truth, 0), dtype=tf.float32)
                correct_pixels = tf.cast(tf.equal(tf.squeeze(self.class_ids, axis=3), self.truth), dtype=tf.float32)
                # labeled_pixels = tf.reduce_sum(labeled_pixels, axis=[1, 2])
                correct_labeled_pixels = tf.multiply(correct_pixels, labeled_pixels)
                pixel_accuracy = tf.reduce_sum(correct_labeled_pixels) / tf.reduce_sum(labeled_pixels)

            with tf.variable_scope("decay"):
                cost = []
                for weight in self.blocks.weights:
                    cost.append(tf.nn.l2_loss(weight))

                decay = 0.00003 * tf.reduce_sum(cost)

            with tf.variable_scope("all_losses"):
                tf.summary.scalar("mean_iou", self.mean_iou)
                tf.summary.scalar("pixel_accuracy", pixel_accuracy)
                self.competition_loss = (self.mean_iou + pixel_accuracy) / 2
                tf.summary.scalar("competition_metric", self.competition_loss)

                tf.summary.scalar("decay", decay)
                tf.summary.scalar("sce_loss", sce_loss)
                self.loss = decay + inverse_mean_iou + sce_loss * 0.25
                tf.summary.scalar("total_loss", self.loss)

            # costs = list()
            # for var in self.blocks.get_decayed_variables():
            #     costs.append(tf.nn.l2_loss(var))
            #
            # tf.multiply(0.0001, tf.reduce_sum(costs))

            # with tf.variable_scope('per_sample_losses'):
            #     self.per_sample_loss = tf.reduce_sum(valid_losses, axis=[1, 2]) / tf.reduce_sum(non_zero_indices,
            #                                                                                     axis=[1, 2])
            #     self.update_losses_op = tf.scatter_update(self.all_losses_by_id, self.input_id, self.per_sample_loss)
            #     tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.update_losses_op)

            with tf.variable_scope('classification_gradient'):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                boundaries = [30000, 80000]
                values = [0.1, 0.01, 0.001]

                self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
                tf.summary.scalar('learning_rate', self.learning_rate)

                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies(update_ops):
                    self.training_step = self.optimizer.minimize(self.loss, global_step=self.global_step)
