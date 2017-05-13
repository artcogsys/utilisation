import os

import tensorflow as tf

from pipeline import get_raw_pipeline
from settings import DATA_FORMAT, DATA_DIRECTORY


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


with tf.Session() as sess:
    global_step = tf.Variable(initial_value=0, dtype=tf.int32)

    increment_global_step_op = tf.assign(global_step, tf.add(global_step, 1))

    image_tensor, label_tensor = get_raw_pipeline(batch_size=1, num_epochs=1)
    sess.run(tf.variables_initializer(tf.local_variables()))
    sess.run(tf.variables_initializer(tf.global_variables()))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    if DATA_FORMAT is "NCHW":
        filename = DATA_DIRECTORY + '/ade20k-nchw.tfrecords'
    else:
        filename = DATA_DIRECTORY + '/ade20k.tfrecords'

    writer = tf.python_io.TFRecordWriter(filename)

    try:
        while not coord.should_stop():
            _, count, image, labels, = sess.run([increment_global_step_op, global_step, image_tensor, label_tensor])
            example = tf.train.Example(features=tf.train.Features(feature={
                'labels': _bytes_feature(labels.tostring()),
                'image': _bytes_feature(image.tostring())}))
            writer.write(example.SerializeToString())
            print count
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    writer.close()
    sess.close()
