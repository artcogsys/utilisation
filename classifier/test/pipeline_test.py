import tensorflow as tf

from pipeline import double_random_crop


class PipelineTest(tf.test.TestCase):
    def testPipeline(self):
        with self.test_session() as ts:

            image = tf.reshape(tf.range(0, 75), [5, 5, 3])
            label = image[:, :, 0:1]
            image, label = double_random_crop(image, label, [1, 6], seed=121)
            real_image, real_label, = ts.run([image, label])

            print real_image
            print real_label
            self.assertAllEqual(real_image.shape, [1, 6, 3])
            self.assertAllEqual(real_label.shape, [1, 6, 1])


if __name__ == '__main__':
    tf.test.main()
