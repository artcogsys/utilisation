import tensorflow as tf

import time

inp = tf.placeholder(tf.int32)
b = tf.ones([5])

c = tf.Variable(tf.zeros(15000))
reassign_c = tf.assign(c, c - c)
with tf.control_dependencies([reassign_c]):
    a = tf.scatter_update(c, inp, b)

with tf.Session() as s:
    s.run(tf.variables_initializer(tf.global_variables()))
    started_at = time.time()
    for i in range(0, 100):
        s.run(a, feed_dict={inp: [6, 7, 8, 9, 10]})
    ended_at = time.time()
    print "Execution took %.2f seconds" % (ended_at - started_at)


