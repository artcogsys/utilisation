import tensorflow as tf
import time

labels = tf.random_uniform([64, 256, 256, 1], minval=0, maxval=151, dtype=tf.int32)
one_hot_labels = tf.one_hot(labels, 152)

s = tf.Session()
s.run(tf.variables_initializer(tf.global_variables()))
start = time.time()
for i in range(0, 200):
    s.run(one_hot_labels)
end = time.time()
print "200 batches takes: %d" % (end - start)
