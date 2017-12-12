import tensorflow as tf

from pretrained.model import InceptionV3ADE20K

model = InceptionV3ADE20K()

with tf.Session(graph=model.graph) as sess:
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint("pretrained/checkpoints")
    saver.restore(sess, latest_checkpoint)
    # sess.run(tf.variables_initializer(tf.global_variables()))

    graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, [model.spatial_outputs.op.name, model.outputs.op.name, model.softmax_outputs.op.name,
                               model.aggregated_outputs.op.name])

    print(model.spatial_outputs.op.name)
    print(model.outputs.op.name)
    print(model.softmax_outputs.op.name)
    print(model.input.op.name)
    print(model.aggregated_indices.op.name)
    print(model.aggregated_outputs.op.name)

    with tf.gfile.GFile("pretrained/inception_v3_ade.pb", "wb") as f:
        f.write(graph_def.SerializeToString())
    f.close()
    sess.close()
