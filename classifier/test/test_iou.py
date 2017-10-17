import tensorflow as tf

num_output_classes = 3
with tf.Session() as sess:
    truth = tf.Variable([[1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 2, 2, 2, 2],
                         [0, 2, 2, 2, 2]])

    prediction = tf.Variable([[1, 1, 1, 0, 0],
                              [1, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0],
                              [2, 2, 2, 2, 2],
                              [0, 2, 2, 2, 2]])

    # truth = tf.Variable(tf.ones([10, 10], dtype=tf.int32))
    # prediction = tf.Variable(tf.ones([10, 10], dtype=tf.int32))
    one_hot_truth = tf.one_hot(truth, num_output_classes)
    one_hot_prediction = tf.one_hot(prediction, num_output_classes)
    intersection = tf.multiply(one_hot_prediction, one_hot_truth)
    union = tf.subtract(tf.add(one_hot_prediction, one_hot_truth), intersection)
    iou = tf.where(
        tf.greater(union, 0),
        tf.div(intersection, tf.where(
            tf.equal(union, 0),
            tf.ones_like(union), union)),
        tf.zeros_like(intersection))
    mean_iou = tf.reduce_mean(tf.reduce_sum(iou[:, :, 0:], axis=2))

    sess.run(tf.variables_initializer(tf.global_variables()))
    print sess.run([mean_iou])
