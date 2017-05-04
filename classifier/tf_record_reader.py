import tensorflow as tf

from pipeline import raw_tf_record_reader

with tf.Session() as sess:
    image_tensor, label_tensor = raw_tf_record_reader('ade20k.tfrecords', 1)
    sess.run(tf.variables_initializer(tf.local_variables()))
    sess.run(tf.variables_initializer(tf.global_variables()))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop():
            image, labels, = sess.run([image_tensor, label_tensor])
            print image.shape
            print labels.shape
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
