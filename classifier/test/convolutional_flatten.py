import tensorflow as tf

# kernel_size = 64
# class_count = 151
# batch_size = 64
# image_size = 256

kernel_size = 2
class_count = 6
batch_size = 1
image_size = 4

flattening_kernel = tf.reshape(tf.diag(tf.ones([kernel_size * kernel_size])),
                               [kernel_size, kernel_size, 1, kernel_size * kernel_size])

data = tf.random_uniform(minval=0,
                         maxval=class_count - 1,
                         shape=[batch_size, image_size, image_size, 1],
                         dtype=tf.int32)

flattened_data = tf.cast(tf.nn.conv2d(tf.cast(data, dtype=tf.float32),
                                      flattening_kernel,
                                      [1, kernel_size, kernel_size, 1],
                                      padding="SAME"), dtype=tf.int32)


s = tf.Session()
s.run(tf.variables_initializer(tf.global_variables()))
print s.run([flattened_data, data])