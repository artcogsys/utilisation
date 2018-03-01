import tensorflow as tf
from tensorflow.python.framework import graph_util


def read_graph(graph_file):
    if tf.gfile.Exists(graph_file):
        graph_def = tf.GraphDef()
        with open(graph_file, "rb") as f:
            graph_def.ParseFromString(f.read())
    else:
        raise Exception("Model file not found!")
    graph = tf.Graph()
    with graph.as_default():

        tf.import_graph_def(graph_def, input_map={}, name="")
    return graph


def remove_background_classes():
    graph = read_graph("./multiscale_1.0_224.pb")
    with tf.Session(graph=graph) as sess:
        before_softmax = graph.get_tensor_by_name(u'final_result:0')
        object_indices = tf.convert_to_tensor([0, 1, 2, 3, 5, 7, 8])
        final_result = tf.gather(before_softmax, object_indices, axis=1, name="final_object_result")

        constant_graph = graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [final_result.op.name])
        with tf.gfile.GFile("./multiscale_sigmoid.pb", "wb") as f:
            f.write(constant_graph.SerializeToString())


def remove_background_classes_from_multiscale():
    graph = read_graph("./multiscale_1.0_224.pb")
    with tf.Session(graph=graph) as sess:
        gap_result = graph.get_tensor_by_name(u'global_average_pooling_result/Wx_plus_b/add:0')
        nms_result = graph.get_tensor_by_name(u'non-max_suppression_result/Wx_plus_b/add:0')
        object_indices = tf.convert_to_tensor([0, 1, 2, 3, 5, 7, 8])

        gap_softmax = tf.squeeze(tf.nn.softmax(tf.gather(gap_result, object_indices, axis=-1)), axis=[1, 2])
        nms_softmax = tf.reduce_max(tf.nn.softmax(tf.gather(nms_result, object_indices, axis=-1)), axis=[1, 2])

        logits = tf.reduce_max(tf.stack([gap_softmax, nms_softmax], axis=1), axis=1, name="final_object_result")

        constant_graph = graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [logits.op.name])
        with tf.gfile.GFile("./only_objects_multiscale.pb", "wb") as f:
            f.write(constant_graph.SerializeToString())


def grid_outputs():
    graph_file = "multiclass.pb"
    if tf.gfile.Exists(graph_file):
        graph_def = tf.GraphDef()
        with open(graph_file, "rb") as f:
            graph_def.ParseFromString(f.read())
    else:
        raise Exception("Model file not found!")
    graph = tf.Graph()
    with graph.as_default():
        large_input_image = tf.placeholder(tf.float32, shape=[1, 480, 480, 3])
        tf.import_graph_def(graph_def, input_map={"input": large_input_image},
                            return_elements=["MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6"], name="")

    with tf.Session(graph=graph) as sess:
        spatial_output = graph.get_tensor_by_name(u'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6:0')
        conv_weights = graph.get_tensor_by_name(u'MobilenetV1/Logits/Conv2d_1c_1x1/weights:0')
        conv_biases = graph.get_tensor_by_name(u'MobilenetV1/Logits/Conv2d_1c_1x1/biases:0')
        final_weights = graph.get_tensor_by_name(u'final_training_ops/weights/final_weights:0')
        final_biases = graph.get_tensor_by_name(u'final_training_ops/biases/final_biases:0')

        final_weights = tf.reshape(final_weights, [1, 1, 1001, 10])

        # reduced_spatial_output = tf.nn.avg_pool(spatial_output, [1, 4, 4, 1], strides=[1, 1, 1, 1], padding="VALID")
        reduced_output = tf.nn.avg_pool(spatial_output, [1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID")

        def apply_classifier(l):
            l = tf.nn.conv2d(l, conv_weights, strides=[1, 1, 1, 1], padding="VALID")
            l = tf.nn.bias_add(l, conv_biases)
            l = tf.nn.conv2d(l, final_weights, strides=[1, 1, 1, 1], padding="VALID")
            l = tf.nn.bias_add(l, final_biases)

            object_indices = tf.convert_to_tensor([0, 1, 2, 3, 5, 7, 8])
            return tf.nn.softmax(tf.gather(l, object_indices, axis=-1))
            # return tf.reduce_max(l, axis=[0, 1, 2], name="final_object_result")

        # most_distant = apply_classifier(spatial_output)
        # distant = apply_classifier(reduced_spatial_output)
        final_result = apply_classifier(reduced_output)

        # final_result = tf.reduce_max(tf.stack([distant, closeup]), axis=0, name="final_object_result")

        constant_graph = graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [final_result.op.name])
        with tf.gfile.GFile("./grid_outputs.pb", "wb") as f:
            f.write(constant_graph.SerializeToString())


# def increase_input_size():


# with tf.Session(graph=graph) as sess:
#     softmax = graph.get_tensor_by_name(u'Softmax:0')
#     final_result = tf.reduce_max(softmax, axis=[0, 1, 2], name="final_object_result")
#
#     constant_graph = graph_util.convert_variables_to_constants(
#         sess, graph.as_graph_def(), [final_result.op.name])
#
#     with tf.gfile.GFile("./grid_outputs_ext.pb", "wb") as f:
#         f.write(constant_graph.SerializeToString())

remove_background_classes()
# increase_input_size()
# import_to_tensorboard("./grid_outputs.pb", "mc")
