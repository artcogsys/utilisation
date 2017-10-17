import tensorflow as tf

from model import ADEResNet

model = ADEResNet(batch_size=1, image_size=(256, 256), num_output_classes=151, placeholder_inputs=True)

with tf.Session(graph=model.graph) as sess:
    saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=3)
    latest_checkpoint = tf.train.latest_checkpoint("checkpoints")
    saver.restore(sess, latest_checkpoint)
    # sess.run(tf.variables_initializer(tf.global_variables()))

    graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, [model.flattened_class_ids.op.name])

    print(model.flattened_class_ids.op.name)
    print(model.input.op.name)

    with tf.gfile.GFile("ade_resnet.pb", "wb") as f:
        f.write(graph_def.SerializeToString())
    f.close()
    sess.close()
