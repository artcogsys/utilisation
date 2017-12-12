import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from pipeline import get_training_pipeline, get_validation_pipeline


class InceptionV3ADE20K:
    def __init__(self, is_training=False, batch_size=64):
        self.is_training = is_training
        self.batch_size = batch_size
        self.image_size = (299, 299)

        self.graph = tf.Graph()
        self.trainable_weights = []
        with self.graph.as_default():
            # feed true to this to run a batch of validation data
            self.do_validate = tf.placeholder_with_default(tf.constant(False), shape=[])
            self.input_id, self.input, self.truth = self.get_training_inputs() if self.is_training else self.get_inference_inputs()

            read_graph("pretrained/inception_v3_pretrained.pb", input_tensor=self.input)

            self.spatial_features = self.graph.get_tensor_by_name('InceptionV3/InceptionV3/Mixed_7c/concat:0')
            self.outputs = self.graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
            self.softmax_outputs = tf.nn.softmax(self.outputs)
            self.aggregated_indices, self.aggregated_outputs = self.get_aggregated_category_results(self.softmax_outputs)
            self.spatial_logits, self.spatial_outputs = self.get_segmentation_tensor()
            self.flattened_spatial_outputs = self.get_flattened_segmentation_tensor()

            if self.is_training:
                self.segmentation_loss = self.get_segmentation_loss()
                self.weight_decay = self.get_weight_decay()
                self.global_step, self.training_step = self.get_train_ops()

    def get_inference_inputs(self):
        return None, tf.placeholder(tf.float32, [1, self.image_size[0], self.image_size[1], 3]), None

    def get_training_inputs(self):
        with tf.device("/cpu:0"):
            training_image_id, training_image, training_label = get_training_pipeline(
                image_size=self.image_size)

            validation_image_id, validation_image, validation_label = get_validation_pipeline(
                image_size=self.image_size)

            training_image_ids, training_images, training_labels = tf.train.batch(
                [training_image_id, training_image, training_label],
                batch_size=self.batch_size,
                shapes=[[], [self.image_size[0], self.image_size[1], 3],
                        [self.image_size[0], self.image_size[1]]],
                capacity=3 * self.batch_size,
                num_threads=8)

            validation_image_ids, validation_images, validation_labels = tf.train.batch(
                [validation_image_id, validation_image, validation_label],
                batch_size=self.batch_size,
                shapes=[[], [self.image_size[0], self.image_size[1], 3],
                        [self.image_size[0], self.image_size[1]]],
                capacity=3 * self.batch_size,
                num_threads=2)

            return tf.cond(self.do_validate,
                           lambda: [validation_image_ids, validation_images,
                                    validation_labels],
                           lambda: [training_image_ids, training_images,
                                    training_labels])

    def get_aggregated_category_results(self, softmax_outputs):
        with tf.variable_scope("aggregated_class_output"):
            aggregated_indices = tf.placeholder(dtype=tf.int32)
            aggregated_outputs = tf.reduce_sum(tf.gather(tf.squeeze(softmax_outputs), aggregated_indices))
            return aggregated_indices, aggregated_outputs

    def get_segmentation_tensor(self):
        with tf.variable_scope("segmentation"):
            reduced_spatial_features = tf.nn.avg_pool(self.spatial_features,
                                                      strides=[1, 2, 2, 1],
                                                      ksize=[1, 2, 2, 1],
                                                      padding="VALID")
            spatial_filter = tf.get_variable("spatial_filter", shape=[1, 1, 2048, 150],
                                             initializer=xavier_initializer())
            spatial_bias = tf.get_variable("spatial_bias", shape=[150], initializer=tf.zeros_initializer())
            spatial_logits = tf.nn.bias_add(
                tf.nn.conv2d(reduced_spatial_features, filter=spatial_filter, strides=[1, 1, 1, 1],
                             padding="VALID"), spatial_bias)
            self.trainable_weights.append(spatial_filter)
            self.trainable_weights.append(spatial_bias)
            return spatial_logits, tf.nn.sigmoid(spatial_logits)

    def get_segmentation_loss(self):
        with tf.variable_scope("segmentation_loss"):
            one_hot_existance = tf.one_hot(self.truth, depth=151)
            # from 299x299 to 300x300
            one_hot_existance = tf.pad(one_hot_existance, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]])
            reduced_truth = tf.nn.max_pool(one_hot_existance, ksize=[1, 75, 75, 1], strides=[1, 75, 75, 1],
                                           padding="VALID")[:, :, :, 1:]
            per_pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.spatial_logits,
                                                                     labels=reduced_truth * tf.constant(0.9))
            per_class_loss = tf.reduce_mean(per_pixel_loss, axis=[0, 1, 2])
            loss = tf.reduce_sum(per_class_loss)
            tf.summary.scalar('segmentation_loss', loss)

        with tf.variable_scope("accuracies"):
            for threshold in [0.04, 0.07, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9]:
                class_selections = tf.cast(self.spatial_outputs > threshold, dtype=tf.float32)
                intersection = class_selections * reduced_truth
                union = class_selections + reduced_truth - intersection
                mean_iou = tf.reduce_mean(tf.reduce_sum(intersection, axis=[0, 1, 2]) /
                                          (tf.reduce_sum(union, axis=[0, 1, 2]) + 1))
                tf.summary.scalar("mean_iou_%.2f" % threshold, mean_iou)

        return loss

    def get_weight_decay(self):
        with tf.variable_scope("decay"):
            costs = []
            for weight in self.trainable_weights:
                costs.append(tf.nn.l2_loss(weight))
            decay = tf.reduce_sum(costs) * 0.001
            tf.summary.scalar("decay", decay)
            return decay

    def get_train_ops(self):
        loss_with_decay = self.segmentation_loss + self.weight_decay
        global_step = tf.Variable(0, name='global_step', trainable=False)
        boundaries = [10000, 20000]
        values = [0.1, 0.01, 0.001]

        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, )

        training_step = optimizer.minimize(loss_with_decay,
                                           global_step=global_step,
                                           var_list=self.trainable_weights)
        return global_step, training_step

    def get_flattened_segmentation_tensor(self):
        with tf.variable_scope("flattened_result"):
            result_shape = tf.unstack(tf.shape(self.spatial_outputs))
            return tf.reshape(self.spatial_outputs,
                              [result_shape[0] * result_shape[1] * result_shape[2] * result_shape[3]],
                              name="outputs")


def read_graph(graph_file, input_tensor=None):
    if tf.gfile.Exists(graph_file):
        graph_def = tf.GraphDef()
        with open(graph_file, "rb") as f:
            graph_def.ParseFromString(f.read())
    else:
        raise Exception("Model file not found!")

    tf.import_graph_def(graph_def, input_map={"input:0": input_tensor} if input_tensor is not None else {}, name="")
