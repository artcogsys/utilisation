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


graph = read_graph("./multiclass.pb")
with tf.Session(graph=graph) as sess:
    before_softmax = graph.get_tensor_by_name(u'final_training_ops/Wx_plus_b/add:0')
    object_indices = tf.convert_to_tensor([0, 1, 2, 3, 5, 7, 8])
    final_result = tf.nn.softmax(tf.gather(before_softmax, object_indices, axis=1), name="final_object_result")

    constant_graph = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [final_result.op.name])
    with tf.gfile.GFile("./only_object_multiclass.pb", "wb") as f:
        f.write(constant_graph.SerializeToString())
