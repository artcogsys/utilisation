import tensorflow as tf

indices = tf.constant([[0, 1, 0, 0], [2, 0, 0, 1], [1, 4, 2, 0], [0, 0, 1, 0]])

expected_output = tf.constant([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 0, 1], [1, 1, 0, 0, 0]])

long_ones = tf.ones([5])
one_hot_slot = tf.Variable(tf.zeros([4, 5], dtype=tf.int32))


reset_op = tf.assign(one_hot_slot, one_hot_slot - one_hot_slot)
with tf.control_dependencies([reset_op]):
    update_ops = []
    for i in range(0, 4):
        unique_indices, _ = tf.unique(indices[i, :])  # to be able to do that, we need to work per sliding window
        unique_values = tf.gather(long_ones, unique_indices)
        update_ops.append(tf.scatter_update(one_hot_slot[i, :], unique_indices, unique_values))

with tf.control_dependencies(update_ops):
    assert_op = tf.assert_equal(one_hot_slot, expected_output)

s = tf.Session()
s.run(tf.variables_initializer(tf.global_variables()))

print s.run(assert_op)