import tensorflow as tf

from pipeline import input_pipeline, training


def conv2d(tensor, filter_size, input_channels, output_channels):
    kernel = tf.Variable(
        tf.truncated_normal([filter_size, filter_size, input_channels, output_channels], dtype=tf.float32, stddev=1e-1),
        name='filter_weights')
    conv_2d = tf.nn.conv2d(tensor, kernel, [1, 1, 1, 1], padding="SAME")

    return conv_2d


graph = tf.Graph()

#  not really a network, but enough to make a start.

with graph.as_default():
    input_image, real_image = input_pipeline(training, num_epochs=4, batch_size=5)
    a = conv2d(input_image, 3, 3, 256)
    a = tf.Print(a, [a])

with tf.Session(graph=graph) as sess:
    sess.run(tf.variables_initializer(tf.local_variables()))
    sess.run(tf.variables_initializer(tf.global_variables()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop():
            _a, = sess.run([a])
    except tf.errors.OutOfRangeError as e:
        print e.message
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
