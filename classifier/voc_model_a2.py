import tensorflow as tf
from blocks import relu_conv2d, conv2d, batch_normalization, normalized_relu_activation, block, MAX_POOLING, STRIDES

with tf.Session() as session:
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 504, 504, 3])

    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(input_placeholder, [3, 3, 3, 64])

    with tf.variable_scope("conv_2_1"):
        conv_2_1 = block(conv_1,
                         down_sampling=MAX_POOLING,
                         kernel_sizes=[3, 3],
                         strides=[1, 1],
                         dilation_rates=[1, 1],
                         input_channel=64,
                         output_channels=[128, 128])

    with tf.variable_scope("conv_3_1"):
        conv_3_1 = block(conv_2_1,
                         down_sampling=MAX_POOLING,
                         kernel_sizes=[3, 3],
                         strides=[1, 1],
                         dilation_rates=[1, 1],
                         input_channel=128,
                         output_channels=[256, 256])

    with tf.variable_scope("conv_4_1"):
        conv_4_1 = block(conv_3_1,
                         down_sampling=MAX_POOLING,
                         kernel_sizes=[3, 3],
                         strides=[1, 1],
                         dilation_rates=[1, 1],
                         input_channel=256,
                         output_channels=[512, 512])

    with tf.variable_scope("conv_5_1"):
        conv_5_1 = block(conv_4_1,
                         down_sampling=None,
                         kernel_sizes=[3, 3],
                         strides=[1, 1],
                         dilation_rates=[1, 2],
                         input_channel=512,
                         output_channels=[512, 1024])

    with tf.variable_scope("conv_5_2"):
        conv_5_2 = block(conv_5_1,
                         down_sampling=None,
                         kernel_sizes=[3, 3],
                         strides=[1, 1],
                         dilation_rates=[2, 2],
                         input_channel=1024,
                         output_channels=[512, 1024])

    with tf.variable_scope("conv_5_3"):
        conv_5_3 = block(conv_5_2,
                         down_sampling=None,
                         kernel_sizes=[3, 3],
                         strides=[1, 1],
                         dilation_rates=[2, 2],
                         input_channel=1024,
                         output_channels=[512, 1024])

    with tf.variable_scope("conv_6"):
        conv_6 = block(conv_5_3,
                       down_sampling=None,
                       kernel_sizes=[1, 3, 1],
                       strides=[1, 1, 1],
                       dilation_rates=[1, 4, 1],
                       input_channel=1024,
                       output_channels=[512, 1024, 2048])

    with tf.variable_scope("conv_7"):
        conv_7 = block(conv_6,
                       down_sampling=None,
                       kernel_sizes=[1, 3, 1],
                       strides=[1, 1, 1],
                       dilation_rates=[1, 4, 1],
                       input_channel=2048,
                       output_channels=[1024, 2048, 4096])

    with tf.variable_scope("classifier_1"):
        classifier_1 = normalized_relu_activation(conv_7)
        classifier_1 = conv2d(classifier_1, [3, 3, 4096, 150], stride=1, dilation=12)

    # with tf.variable_scope("classifier_2"):
    #     classifier_2 = tf.nn.relu(classifier_1)
    #     classifier_2 = conv2d(classifier_2, [3, 3, 512, 150], stride=1, dilation=12)

    output = classifier_1

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("summaries", session.graph, flush_secs=1)

    session.run(tf.variables_initializer(tf.global_variables()))

    graph_def = tf.graph_util.convert_variables_to_constants(
        session, session.graph_def, [output.op.name])

    print(input_placeholder.op.name)
    print(output.op.name)

    with tf.gfile.GFile("model_a2_conv2.pb", "wb") as f:
        f.write(graph_def.SerializeToString())
    f.close()
    train_writer.flush()
    train_writer.close()
    session.close()
