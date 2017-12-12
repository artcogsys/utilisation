import tensorflow as tf
import numpy as np
with tf.Session() as sess:
    indices = tf.placeholder(tf.int32)
    values = tf.range(0, 1000, dtype=tf.float32) + 1.
    aggregated_values = tf.reduce_sum(tf.gather(values, indices))
    sess.run(tf.global_variables_initializer())
    aggregate_indices = [1, 2, 3, 4, 5, 6, 7, 8, 0, 12, 123, 412, 124]
    result, = sess.run([aggregated_values], feed_dict={indices: aggregate_indices})
    assert result == (len(aggregate_indices) + np.sum(aggregate_indices))
